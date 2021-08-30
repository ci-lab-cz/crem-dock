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


def docking(ligands_pdbqt_string, receptor_pdbqt_fname, center, box_size, ncpu):
    '''
    :param ligands_pdbqt_string: str or list of strings
    :param receptor_pdbqt_fname:
    :param center: (x_float,y_float,z_float)
    :param box_size: (size_x_int, size_y_int, size_z_int)
    :param ncpu: int
    :return: (score_top, pdb_string_block)
    '''
    v = Vina(sf_name='vina', cpu=ncpu, seed=1024, no_refine=False, verbosity=0)
    v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt_fname)
    v.set_ligand_from_string(ligands_pdbqt_string)
    v.compute_vina_maps(center=center, box_size=box_size, spacing=1)
    #change n_poses
    v.dock(exhaustiveness=32, n_poses=9)

    score_top = v.energies(n_poses=1)[0][0]
    pdb_top_block = '\n'.join([i[:66] for i in v.poses(n_poses=1).split('MODEL')[1].split('\n')])

    return score_top, pdb_top_block


def iter_docking(conn, receptor_pdbqt_fname, protein_setup, protonation, iteration, ncpu):
    '''

    :param conn:
    :param receptor_pdbqt_fname:
    :param protein_setup:
    :param protonation: True or False
    :param iteration: int
    :param ncpu: int
    :return: dict(mol_id:(energy_float, pdb_string_block),...)
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

    cur = conn.cursor()
    if protonation:
        smiles_dict = cur.execute(f"SELECT id, smi_protonated FROM mols WHERE iteration = '{iteration - 1}'")
    else:
        smiles_dict = cur.execute(f"SELECT id, smi FROM mols WHERE iteration = '{iteration - 1}'")

    mol_ids, smiles = zip(*smiles_dict)
    center, box_size = get_param_from_config(protein_setup)

    pool = Pool(ncpu)
    ligands_pdbqt_string = pool.map(ligand_preparation, smiles)
    dock_result = pool.map(partial(docking, receptor_pdbqt_fname=receptor_pdbqt_fname, center=center,
                                   box_size=box_size, ncpu=ncpu), iterable=ligands_pdbqt_string)

    return {i: k for i, k in zip(mol_ids, dock_result)}
