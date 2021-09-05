import re
import sqlite3
import sys
from functools import partial
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina

from scripts import mk_prepare_ligand_string


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

    mol = Chem.MolFromSmiles(smi)
    mol_conf_sdf = convert2mol(mol)
    mol_conf_pdbqt = mk_prepare_ligand_string.main(mol_conf_sdf,
                                                   build_macrocycle=False, # can do it True, but there is some problem with >=7-chains mols
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
    v.dock(exhaustiveness=32, n_poses=9)

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
            mol.SetProp('_Name', mol_id)
            mol_block = Chem.MolToMolBlock(mol)
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        except Exception:
            sys.stderr.write(f'Could not assign bond orders while parsing PDB: {mol_id}\n')
    return mol_block


def process_mol_docking(mol_id, smi, receptor_pdbqt_fname, center, box_size, dbname, ncpu):
    ligand_pdbqt = ligand_preparation(smi)
    score, pdbqt_out = docking(ligand_pdbqt, receptor_pdbqt_fname, center, box_size, ncpu)
    mol_block = pdbqt2molblock(pdbqt_out, smi, mol_id)
    if mol_block is None:
        pdbqt_out = fix_pdbqt(pdbqt_out)
        mol_block = pdbqt2molblock(pdbqt_out, smi, mol_id)
        if mol_block:
            sys.stderr.write('PDBQT was fixed\n')

    with sqlite3.connect(dbname) as conn:
        conn.execute("""UPDATE mols
                           SET pdb_block = ?,
                               docking_score = ?,
                               mol_block = ?
                           WHERE
                               id = ?
                        """, (pdbqt_out, score, mol_block, mol_id))
    return mol_id


def iter_docking(dbname, receptor_pdbqt_fname, protein_setup, protonation, iteration, ncpu):
    '''
    This function should update output db with docked poses and scores. Docked poses should be stored as pdbqt (source)
    and mol block. All other post-processing will be performed separately.
    :param conn:
    :param receptor_pdbqt_fname:
    :param protein_setup:
    :param protonation: True or False
    :param iteration: int
    :param ncpu: int
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
        smiles_dict = dict(cur.execute(f"SELECT id, {smi_field_name} FROM mols WHERE iteration = '{iteration - 1}'"))

    center, box_size = get_param_from_config(protein_setup)

    pool = Pool(ncpu)
    for i, mol_id in enumerate(pool.starmap(partial(process_mol_docking, dbname=dbname,
                                                    receptor_pdbqt_fname=receptor_pdbqt_fname,
                                                    center=center, box_size=box_size,
                                                    ncpu=ncpu),
                                            smiles_dict.items()), 1):
        if i % 100 == 0:
            sys.stderr.write(f'\r{i} molecules were docked')
    sys.stderr.write(f'\r{i} molecules were docked\n')
