import re
import sqlite3
import sys
from functools import partial
from multiprocessing import Pool, Manager

from dask import bag
from dask.distributed import Lock as daskLock
from meeko import MoleculePreparation
from meeko import obutils
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina


def mk_prepare_ligand_string(molecule_string, build_macrocycle=True, add_water=False, merge_hydrogen=True,
                             add_hydrogen=False, pH_value=None, verbose=False, mol_format='SDF'):

    mol = obutils.load_molecule_from_string(molecule_string, molecule_format=mol_format)

    if pH_value is not None:
        mol.CorrectForPH(float(pH_value))

    if add_hydrogen:
        mol.AddHydrogens()
        charge_model = ob.OBChargeModel.FindType("Gasteiger")
        charge_model.ComputeCharges(mol)

    m = Chem.MolFromMolBlock(molecule_string)
    amide_rigid = len(m.GetSubstructMatch(Chem.MolFromSmarts('[C;!R](=O)[#7]([!#1])[!#1]'))) == 0

    preparator = MoleculePreparation(merge_hydrogens=merge_hydrogen, macrocycle=build_macrocycle,
                                     hydrate=add_water, amide_rigid=amide_rigid)
                                     #additional parametrs
                                     #rigidify_bonds_smarts=[], rigidify_bonds_indices=[])
    preparator.prepare(mol)
    if verbose:
        preparator.show_setup()

    return preparator.write_pdbqt_string()


def ligand_preparation(smi):

    def convert2mol(m):

        def gen_conf(mol, useRandomCoords, randomSeed):
            params = AllChem.ETKDGv3()
            params.useRandomCoords = useRandomCoords
            params.randomSeed = randomSeed
            conf_stat = AllChem.EmbedMolecule(mol, params)
            return mol, conf_stat

        if not m:
            return None
        m = Chem.AddHs(m, addCoords=True)
        m, conf_stat = gen_conf(m, useRandomCoords=False, randomSeed=1024)
        if conf_stat == -1:
            # if molecule is big enough and rdkit cannot generate a conformation - use params.useRandomCoords = True
            m, conf_stat = gen_conf(m, useRandomCoords=True, randomSeed=1024)
            if conf_stat == -1:
                return None
        AllChem.UFFOptimizeMolecule(m, maxIters=100)
        return Chem.MolToMolBlock(m)

    try:
        mol = Chem.MolFromSmiles(smi)
        mol_conf_sdf = convert2mol(mol)
    except TypeError:
        sys.stderr.write(f'incorrect SMILES {smi} for converting to molecule\n')
        return None

    mol_conf_pdbqt = mk_prepare_ligand_string(mol_conf_sdf,
                                              build_macrocycle=False,
                                              # can do it True, but there is some problem with >=7-chains mols
                                              add_water=False, merge_hydrogen=True, add_hydrogen=False,
                                              # pH_value=7.4, can use this opt but some results are different in comparison to chemaxon
                                              verbose=False, mol_format='SDF')
    return mol_conf_pdbqt


def fix_pdbqt(pdbqt_block):
    pdbqt_fixed = []
    for line in pdbqt_block.split('\n'):
        if not line.startswith('HETATM') and not line.startswith('ATOM'):
            pdbqt_fixed.append(line)
            continue
        atom_type = line[12:16].strip()
        # autodock vina types
        if 'CA' in line[77:79]: #Calcium is exception
            atom_pdbqt_type = 'CA'
        else:
            atom_pdbqt_type = re.sub('D|A', '', line[77:79]).strip() # can add meeko macrocycle types (G and \d (CG0 etc) in the sub expression if will be going to use it

        if re.search('\d', atom_type[0]) or len(atom_pdbqt_type) == 2: #1HG or two-letter atom names such as CL,FE starts with 13
            atom_format_type = '{:<4s}'.format(atom_type)
        else: # starts with 14
            atom_format_type = ' {:<3s}'.format(atom_type)
        line = line[:12] + atom_format_type + line[16:]
        pdbqt_fixed.append(line)

    return '\n'.join(pdbqt_fixed)


def docking(ligands_pdbqt_string, receptor_pdbqt_fname, center, box_size, ncpu):
    '''
    :param ligands_pdbqt_string: str or list of strings
    :param receptor_pdbqt_fname:
    :param center: (x_float,y_float,z_float)
    :param box_size: (size_x_int, size_y_int, size_z_int)
    :param ncpu: int
    :return: (score_top, pdbqt_string_block)
    '''
    v = Vina(sf_name='vina', cpu=ncpu, seed=1024, no_refine=False, verbosity=0)
    v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt_fname)
    v.set_ligand_from_string(ligands_pdbqt_string)
    v.compute_vina_maps(center=center, box_size=box_size, spacing=1)
    #change n_poses
    v.dock(exhaustiveness=8, n_poses=9)

    return v.energies(n_poses=1)[0][0], v.poses(n_poses=1)


def pdbqt2molblock(pdbqt_block, smi, mol_id):
    mol_block = None
    mol = Chem.MolFromPDBBlock('\n'.join([i[:66] for i in pdbqt_block.split('MODEL')[1].split('\n')]), removeHs=False, sanitize=False)
    if mol:
        try:
            template_mol = Chem.MolFromSmiles(smi)
            # explicit hydrogends are removed from carbon atoms (chiral hydrogens) to match pdbqt mol,
            # e.g. [NH3+][C@H](C)C(=O)[O-]
            template_mol = Chem.AddHs(template_mol, explicitOnly=True,
                                      onlyOnAtoms=[a.GetIdx() for a in template_mol.GetAtoms() if
                                                   a.GetAtomicNum() != 6])
            mol = AllChem.AssignBondOrdersFromTemplate(template_mol, mol)
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
            mol.SetProp('_Name', mol_id)
            mol_block = Chem.MolToMolBlock(mol)
        except Exception:
            sys.stderr.write(f'Could not assign bond orders while parsing PDB: {mol_id}\n')
    return mol_block


def process_mol_docking(mol_id, smi, receptor_pdbqt_fname, center, box_size, dbname, table_name, ncpu, lock=None):

    def insert_data(dbname, table_name, pdbqt_out, score, mol_block, mol_id):
        with sqlite3.connect(dbname) as conn:
            conn.execute(f"""UPDATE {table_name}
                               SET pdb_block = ?,
                                   docking_score = ?,
                                   mol_block = ?,
                                   time = CURRENT_TIMESTAMP
                               WHERE
                                   id = ?
                            """, (pdbqt_out, score, mol_block, mol_id))

    ligand_pdbqt = ligand_preparation(smi)
    if ligand_pdbqt is None:
        return mol_id
    score, pdbqt_out = docking(ligand_pdbqt, receptor_pdbqt_fname, center, box_size, ncpu)
    mol_block = pdbqt2molblock(pdbqt_out, smi, mol_id)
    if mol_block is None:
        pdbqt_out = fix_pdbqt(pdbqt_out)
        mol_block = pdbqt2molblock(pdbqt_out, smi, mol_id)
        if mol_block:
            sys.stderr.write('PDBQT was fixed\n')

    if lock is not None:  # multiprocessing
        with lock:
            insert_data(dbname, table_name, pdbqt_out, score, mol_block, mol_id)
    else:  # dask
        with daskLock(dbname):
            insert_data(dbname, table_name, pdbqt_out, score, mol_block, mol_id)

    return mol_id


def iter_docking(dbname, table_name, receptor_pdbqt_fname, protein_setup, protonation, ncpu, use_dask):
    '''
    This function should update output db with docked poses and scores. Docked poses should be stored as pdbqt (source)
    and mol block. All other post-processing will be performed separately.
    :param dbname:
    :param table_name:
    :param receptor_pdbqt_fname:
    :param protein_setup:
    :param protonation: True or False
    :param ncpu: int
    :param use_dask: indicate whether or not using dask cluster
    :type use_dask: bool
    :return:
    '''

    def get_param_from_config(config_fname):
        config = {}
        with open(config_fname) as inp:
            for line in inp:
                if not line.strip():
                    continue
                param_name, value = line.replace(' ', '').split('=')
                config[param_name] = float(value)
        center, box_size = (config['center_x'], config['center_y'], config['center_z']),\
                           (config['size_x'], config['size_y'], config['size_z'])
        return center, box_size

    with sqlite3.connect(dbname) as conn:
        cur = conn.cursor()
        smi_field_name = 'smi_protonated' if protonation else 'smi'
        if table_name == 'mols':
            iteration = list(cur.execute("SELECT max(iteration) FROM mols"))[0][0]
            smiles_dict = dict(cur.execute(f"SELECT id, {smi_field_name} "
                                           f"FROM mols "
                                           f"WHERE iteration = {iteration} AND docking_score IS NULL AND {smi_field_name} !=''"))
        else:
            smiles_dict = dict(cur.execute(f"SELECT id, {smi_field_name} "
                                           f"FROM {table_name} "
                                           f"WHERE docking_score IS NULL"))
    if not smiles_dict:
        return

    center, box_size = get_param_from_config(protein_setup)

    if use_dask:
        i = 0
        # npart = max(len(smiles_dict) // 5, 1000)
        b = bag.from_sequence(smiles_dict.items(), npartitions=len(smiles_dict))
        for i, mol_id in enumerate(b.starmap(process_mol_docking,
                                             receptor_pdbqt_fname=receptor_pdbqt_fname,
                                             center=center, box_size=box_size, dbname=dbname, table_name=table_name, ncpu=1).compute(),
                                   1):
            if i % 100 == 0:
                sys.stderr.write(f'\r{i} molecules were docked')
        sys.stderr.write(f'\r{i} molecules were docked\n')

    else:
        pool = Pool(ncpu)
        manager = Manager()
        lock = manager.Lock()
        i = 0
        for i, mol_id in enumerate(pool.starmap(partial(process_mol_docking,
                                                        receptor_pdbqt_fname=receptor_pdbqt_fname,
                                                        center=center, box_size=box_size, dbname=dbname, table_name=table_name,
                                                        ncpu=1, lock=lock),
                                                smiles_dict.items()), 1):
            if i % 100 == 0:
                sys.stderr.write(f'\r{i} molecules were docked')
        sys.stderr.write(f'\r{i} molecules were docked\n')
