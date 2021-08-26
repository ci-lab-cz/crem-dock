import subprocess
import tempfile
from functools import partial
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina

from scripts import mk_prepare_ligand_string


def ligand_preparation(smi):
    def convert2mol(m):
        def gen_conf(mol, useRandomCoords, randomSeed):
            params = AllChem.ETKDG()
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
    :param receptor_pdbqt_fname:
    :param ligands_pdbqt_string: str or list of strs
    :param center: list [x,y,z]
    :param box_size: list [size_x, size_y, size_z]
    :param ncpu: int
    :return:
    '''
    v = Vina(sf_name='vina', cpu=ncpu, seed=1024, no_refine=False, verbosity=0)
    v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt_fname)
    v.set_ligand_from_string(ligands_pdbqt_string)
    v.compute_vina_maps(center=center, box_size=box_size, spacing=1)
    v.dock(exhaustiveness=32, n_poses=9)

    return [v.energies(n_poses=1)[0][0], v.poses(n_poses=1)]


def iter_docking(conn, receptor_pdbqt_fname, protein_setup, protonation, iteration, ncpu):
    '''

    :param conn:
    :param receptor_pdbqt_fname:
    :param protein_setup:
    :param protonation:
    :param iteration:
    :param ncpu:
    :return: [[energy_lig1, pose_lig1]...[energy_ligN, pose_ligN]]
    '''

    def get_param_from_config(config_fname):
        vina_config_dict = {}
        with open(config_fname) as inp:
            for line in inp:
                if not line.strip():
                    continue
                param_name, value = line.replace(' ', '').split('=')
                vina_config_dict[param_name] = float(value)
        return vina_config_dict

    cur = conn.cursor()
    smiles = cur.execute(f"SELECT smi, id FROM mols WHERE iteration = '{iteration - 1}'")
    smiles, mol_ids = zip(*smiles)
    if protonation:
        try:
            fp = tempfile.NamedTemporaryFile(suffix='.smi')
            fp.writelines('{0}\t{1}\n'.format(item[0], item[1]).encode('utf-8') for item in zip(smiles, mol_ids))
            cmd_run = f"cxcalc majormicrospecies -H 7.4 -f smiles -M -K '{fp.name}'"
            smiles = subprocess.check_output(cmd_run, shell=True).decode().split()
            for mol_id, smi_protonated in zip(mol_ids, smiles):
                cur.execute("""UPDATE mols
                                   SET smi_protonated = ? 
                                   WHERE
                                       id = ?
                                """,
                            (Chem.MolToSmiles(Chem.MolFromSmiles(smi_protonated), isomericSmiles=True), mol_id))
            conn.commit()
        finally:
            fp.close()

    pool = Pool(ncpu)
    ligands_pdbqt_string = pool.map(ligand_preparation, smiles)
    config = get_param_from_config(protein_setup)
    center, box_size = [config['center_x'], config['center_y'], config['center_z']], [config['size_x'],
                                                                                      config['size_y'],
                                                                                      config['size_z']]
    dock_result = pool.map(partial(docking, receptor_pdbqt_fname=receptor_pdbqt_fname, center=center,
                                   box_size=box_size, ncpu=ncpu), iterable=ligands_pdbqt_string)

    return dock_result, mol_ids
