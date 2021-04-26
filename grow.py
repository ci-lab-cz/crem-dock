import glob
import os
import shutil
import argparse
import random
import string
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.ML.Cluster import Butina
from rdkit.Chem.Descriptors import MolWt
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
from scripts import Docking, Smi2PDB
from crem.crem import grow_mol
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from itertools import combinations


def cpu_type(x):
    return max(1, min(int(x), cpu_count()))


def sort_two_lists(primary, secondary):
    # sort two lists by order of elements of the primary list
    paired_sorted = sorted(zip(primary, secondary), key=lambda x: x[0])
    return map(list, zip(*paired_sorted))  # two lists


def get_mol_ids(mols):
    return [mol.GetProp('_Name') for mol in mols]


def set_common_atoms(mol_name, child_mol, parent_mol, conn):
    '''

    :param mol_name:
    :param child_mol:
    :param parent_mol:
    :param conn:
    :return:
    '''
    cur = conn.cursor()

    ids = child_mol.GetSubstructMatch(parent_mol)
    atoms = []
    for i, j in combinations(ids, 2):
        bond = child_mol.GetBondBetweenAtoms(i, j)
        if bond is not None:
            atoms.append(f'{i}_{j}')
    atoms = '_'.join(atoms)
    cur.execute("""UPDATE mols
                       SET atoms = ?
                       WHERE
                           id = ?
                    """, (atoms, mol_name))
    conn.commit()


def save_smi_to_pdb(conn, iteration, tmpdir, ncpu):
    '''
    Creates file with SMILES which is supplied to Chemaxon cxcalc utility to get molecule ionization states at pH 7.4.
    Parse output and generate PDB files stored in the specified directory. Updates the common atoms field in DB
    (if iteration > 1)
    :param conn:
    :param iteration:
    :param tmpdir:
    :param ncpu:
    :return:
    '''
    fname = os.path.join(tmpdir, '1.smi')
    cur = conn.cursor()
    smiles = list(cur.execute(f"SELECT smi, id FROM mols WHERE iteration = '{iteration - 1}'"))
    mol_ids = [i for smi, i in smiles]
    with open(fname, 'wt') as f:
        f.writelines('%s\t%s\n' % item for item in smiles)
    cmd_run = f"cxcalc majormicrospecies -H 7.4 -f smiles -M -K '{fname}'"
    smi_output = subprocess.check_output(cmd_run, shell=True).decode().split()

    pool = Pool(ncpu)

    if iteration == 1:
        pool.imap_unordered(Smi2PDB.save_to_pdb_mp, [(i, os.path.join(tmpdir, f'{j}.pdb')) for i, j in zip(smi_output, mol_ids)])
        pool.close()
        pool.join()

    else:
        pars = dict(cur.execute(f"SELECT id, parent_id FROM mols WHERE iteration = '{iteration - 1}'"))
        parent_ids = list(set(pars.values()))
        sql = f'SELECT id, mol_block FROM mols WHERE id IN ({",".join("?" * len(parent_ids))})'
        parent_mols = {i: Chem.MolFromMolBlock(j) for i, j in cur.execute(sql, parent_ids)}
        mols = [Chem.MolFromSmiles(i) for i in smi_output]
        pool.imap_unordered(Smi2PDB.save_to_pdb2_mp, [(mol, parent_mols[pars[i]], os.path.join(tmpdir, f'{i}.pdb'))
                                                      for i, mol in zip(mol_ids, mols)])
        pool.close()
        pool.join()

        for name, child_mol in zip(mol_ids, mols):
            set_common_atoms(name, child_mol, parent_mols[pars[name]], conn)


def prep_ligands(conn, dname, python_path, vina_script_dir, ncpu):

    def supply_lig_prep(conn, dname, python_path, vina_script_dir):
        cur = conn.cursor()
        for fname in glob.glob(os.path.join(dname, '*.pdb')):
            mol_id = os.path.basename(os.path.splitext(fname)[0])
            atoms = list(cur.execute(f"SELECT atoms FROM mols WHERE id = '{mol_id}'"))[0][0]
            if not atoms:
                atoms = ''
            yield fname, os.path.abspath(
                os.path.splitext(fname)[0] + '.pdbqt'), python_path, vina_script_dir + 'prepare_ligand4.py', atoms

    pool = Pool(ncpu)
    pool.imap_unordered(Docking.prepare_ligands_mp, list(supply_lig_prep(conn, dname, python_path, vina_script_dir)))
    pool.close()
    pool.join()


def dock_ligands(dname, target_fname_pdbqt, target_setup_fname, vina_path, ncpu):
    def supply_lig_dock_data(dname, target_fname_pdbqt, target_setup_fname):
        for fname in glob.glob(os.path.join(dname, '*.pdbqt')):
            yield fname, fname.rsplit('.', 1)[0] + '_dock.pdbqt', target_fname_pdbqt, target_setup_fname, vina_path

    Parallel(n_jobs=ncpu, verbose=8)(delayed(Docking.run_docking)(i_fname, o_fname, target, setup, vina_script)
                                     for i_fname, o_fname, target, setup, vina_script in supply_lig_dock_data(dname,
                                                                                                              target_fname_pdbqt,
                                                                                                              target_setup_fname))


def get_score(pdb_block):
    """
    Return correct docking score
    :param pdb_block:
    :return:
    """
    score = float(pdb_block.split()[5])
    active_torsions = int(pdb_block.split('active torsions')[0][-2])
    all_torsions = int(pdb_block.split('TORSDOF')[1].split('\n')[0])
    score_correct = round(score * (1 + 0.0585 * active_torsions) / (1 + 0.0585 * all_torsions), 2)
    return score_correct


def update_db(conn, dname):
    """
    Insert score, rmsd, fixed bonds string, pdb and mol blocks of molecules having a corresponding pdbqt file in the
    temporary dir in docking DB
    :param conn: connection to docking DB
    :param dname: path to temp directory with files after docking. Those files should have names <mol_id>_dock.pdbqt
    :return:
    """

    for fname in glob.glob(os.path.join(dname, '*_dock.pdbqt')):
        cur = conn.cursor()

        # get pdb block
        pdb_block = open(fname).read()

        # get score
        score = get_score(pdb_block)

        # get mol block for the first pose
        mol_id = os.path.basename(fname).replace('_dock.pdbqt', '')
        smi = list(cur.execute(f"SELECT smi FROM mols WHERE id = '{mol_id}'"))[0][0]

        mol_block = None
        mol = Chem.MolFromPDBBlock('\n'.join([i[:66] for i in pdb_block.split('MODEL')[1].split('\n')]),
                                   removeHs=False)

        if mol:
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smi), mol)
                mol.SetProp('_Name', mol_id)
                mol_block = Chem.MolToMolBlock(mol)
            except:
                mol = None
            # get atoms
            parent_id = list(cur.execute(f"SELECT parent_id FROM mols WHERE id = '{mol_id}'"))[0][0]
            if parent_id and mol:
                parent_mol_block = list(cur.execute(f"SELECT mol_block FROM mols WHERE id = '{parent_id}'"))[0][0]
                parent_mol = Chem.MolFromMolBlock(parent_mol_block)
                rms = get_rmsd(mol, parent_mol)
            else:
                rms = None
        else:
            rms = None

        cur.execute("""UPDATE mols
                           SET pdb_block = ?,
                               mol_block = ?,
                               docking_score = ?,
                               rmsd = ? 
                           WHERE
                               id = ?
                        """, (pdb_block, mol_block, score, rms, mol_id))
    conn.commit()


def get_rmsd(child_mol, parent_mol):
    """
    Returns best fit rmsd between a common part of child and parent molecules taking into account symmetry of molecules
    :param child_mol: Mol
    :param parent_mol: Mol
    :return:
    """
    match_ids = child_mol.GetSubstructMatches(parent_mol, uniquify=False, useChirality=True)
    best_rms = float('inf')
    for ids in match_ids:
        diff = np.array(child_mol.GetConformer().GetPositions()[ids,]) - np.array(
            parent_mol.GetConformer().GetPositions())
        rms = np.sqrt((diff ** 2).sum() / len(diff))
        if rms < best_rms:
            best_rms = rms
    return best_rms


def get_docked_mol_ids(conn, iteration):
    """
    Returns mol_ids for molecules which where docked at the given iteration and conversion to mol block was successful
    :param conn:
    :param iteration:
    :return:
    """
    cur = conn.cursor()
    res = cur.execute(f"SELECT id FROM mols WHERE iteration = '{iteration - 1}' AND mol_block IS NOT NULL")
    return [i[0] for i in res]


def get_mols(conn, mol_ids):
    """
    Returns list of Mol objects from docking DB, order is arbitrary
    :param conn: connection to docking DB
    :param mol_ids: list of molecules to retrieve
    :return:
    """
    cur = conn.cursor()
    sql = f'SELECT mol_block FROM mols WHERE id IN ({",".join("?" * len(mol_ids))})'
    return [Chem.MolFromMolBlock(items[0]) for items in cur.execute(sql, mol_ids)]


def get_mol_scores(conn, mol_ids):
    """
    Return dict of mol_id: score
    :param conn: connection to docking DB
    :param mol_ids: list of mol ids
    :return:
    """
    cur = conn.cursor()
    sql = f'SELECT id, docking_score FROM mols WHERE id IN ({",".join("?" * len(mol_ids))})'
    return dict(cur.execute(sql, mol_ids))


def get_mol_rms(conn, mol_ids):
    """
    Return dict of mol_id: rmsd
    :param conn: connection to docking DB
    :param mol_ids: list of mol ids
    :return:
    """
    cur = conn.cursor()
    sql = f'SELECT id, rmsd FROM mols WHERE id IN ({",".join("?" * len(mol_ids))})'
    return dict(cur.execute(sql, mol_ids))


def filter_mols(mols, mw=None, rtb=None):
    """
    Returns list of molecules satisfying given conditions
    :param mols: list of molecules
    :param mw: maximum MW (optional)
    :param rtb: maximum number of rotatable bonds (optional)
    :return: list of molecules
    """
    output = []
    for mol in mols:
        if (mw is None or MolWt(mol) <= mw) and (rtb is None or CalcNumRotatableBonds(mol) <= rtb):
            output.append(mol)
    return output


def filter_mols_by_rms(mols, conn, rms):
    """
    Remove molecules with rmsd greater than the threshold
    :param mols: list of molecules
    :param conn: connection to docking DB
    :param rms: rmsd threshold
    :return: list of molecules
    """
    output = []
    mol_ids = get_mol_ids(mols)
    rmsd = get_mol_rms(conn, mol_ids)
    for mol in mols:
        if rmsd[mol.GetProp('_Name')] <= rms:
            output.append(mol)
    return output


def select_top_mols(mols, conn, ntop):
    """
    Returns list of ntop molecules with the highest score
    :param mols: list of molecules
    :param conn: connection to docking DB
    :param ntop: number of top scored molecules to select
    :return:
    """
    mol_ids = get_mol_ids(mols)
    scores = get_mol_scores(conn, mol_ids)
    scores, mol_ids = sort_two_lists([scores[mol_id] for mol_id in mol_ids], mol_ids)
    mol_ids = set(mol_ids[:ntop])
    mols = [mol for mol in mols if mol.GetProp('_Name') in mol_ids]
    return mols


def sort_clusters(conn, clusters):
    """
    Returns clusters with molecules filtered by properties and reordered according to docking scores
    :param conn: connection to docking DB
    :param clusters: tuple of tuples with mol ids in each cluster
    :return: list of lists with mol ids
    """
    scores = get_mol_scores(conn, [mol_id for cluster in clusters for mol_id in cluster])
    output = []
    for cluster in clusters:
        s, mol_ids = sort_two_lists([scores[mol_id] for mol_id in cluster], cluster)
        output.append(mol_ids)
    return output


def gen_cluster_subset_algButina(mols, tanimoto):
    """
    Returns tuple of tuples with mol ids in each cluster
    :param mols: list of molecules
    :param tanimoto: tanimoto threshold for clustering
    :return:
    """
    dict_index = {}
    fps = []
    for i, mol in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)  ### why so few?
        fps.append(fp)
        dict_index[i] = mol.GetProp('_Name')
    dists = []
    for i, fp in enumerate(fps):
        distance_matrix = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in distance_matrix])
    # returns tuple of tuples with sequential numbers of compounds in each cluster
    cs = Butina.ClusterData(dists, len(fps), 1 - tanimoto, isDistData=True)
    # replace sequential ids with actual mol ids
    output = tuple(tuple((dict_index[y] for y in x)) for x in cs)
    return output


def get_protected_ids(mol, protein_file, dist_threshold):
    """
    Returns list of atoms ids which have hydrogen atoms close to the protein
    :param mol: molecule
    :param protein_file: protein pdbqt file
    :param dist_threshold: minimum distance to hydrogen atoms
    :return:
    """
    pdb_block = open(protein_file).readlines()
    protein = Chem.MolFromPDBBlock('\n'.join([line[:66] for line in pdb_block]), sanitize=False)
    if protein is None:
        raise ValueError("Protein structure is incorrect. Please check protein pdbqt file.")
    protein_cord = protein.GetConformer().GetPositions()
    ids = set()
    for atom, cord in zip(mol.GetAtoms(), mol.GetConformer().GetPositions()):
        b = False
        if atom.GetAtomicNum() == 1:
            for i in protein_cord:
                if euclidean(cord, i) <= dist_threshold:
                    b = True
                    break
            if b:
                ids.add(atom.GetIdx())
    return sorted(ids)


def __grow_mol(mol, protein_pdbqt, h_dist_threshold=2, ncpu=1, **kwargs):
    mol = Chem.AddHs(mol, addCoords=True)
    protected_ids = get_protected_ids(mol, protein_pdbqt, h_dist_threshold)
    return list(grow_mol(mol, protected_ids=protected_ids, return_rxn=False, return_mol=True, ncores=ncpu, **kwargs))


def __grow_mols(mols, protein_pdbqt, h_dist_threshold=2, ncpu=1, **kwargs):
    """

    :param mols: list of molecules
    :param protein_pdbqt: protein pdbqt file
    :param h_dist_threshold: maximum distance from H atoms to the protein to mark them as protected from further grow
    :param ncpu: number of cpu
    :param kwargs: arguments passed to crem function grow_mol
    :return: dict of parent ids and lists of corresponding generated mols
    """
    res = dict()
    for mol in mols:
        tmp = __grow_mol(mol, protein_pdbqt, h_dist_threshold=h_dist_threshold, ncpu=ncpu, **kwargs)
        if tmp:
            res[mol.GetProp('_Name')] = tmp
    return res


def insert_db(conn, data):
    cur = conn.cursor()
    cur.executemany("""INSERT INTO mols VAlUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""", data)
    conn.commit()


def create_db(fname):
    conn = sqlite3.connect(fname)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS mols")
    cur.execute("""CREATE TABLE IF NOT EXISTS mols
            (
             id TEXT PRIMARY KEY,
             iteration INTEGER,
             smi TEXT,
             parent_id TEXT,
             docking_score REAL,
             atoms TEXT,
             rmsd REAL,
             pdb_block TEXT,
             mol_block TEXT
            )""")
    conn.commit()
    conn.close()


def insert_starting_structures_to_db(fname, db_fname):
    """

    :param fname: SMILES or SDF with 3D coordinates
    :param db_fname: output DB
    :return:
    """
    data = []
    make_docking = True
    conn = sqlite3.connect(db_fname)
    try:
        if fname.lower().endswith('.smi') or fname.lower().endswith('.smiles'):
            with open(fname) as f:
                for i, line in enumerate(f):
                    tmp = line.strip().split()
                    smi = tmp[0]
                    name = tmp[1] if len(tmp) > 1 else '000-' + str(i).zfill(6)
                    data.append((name, 0, smi, None, None, None, None, None, None))
        elif fname.lower().endswith('.sdf'):
            make_docking = False
            for i, mol in enumerate(Chem.SDMolSupplier(fname)):
                if mol:
                    name = mol.GetProp('_Name')
                    if not name:
                        name = '000-' + str(i).zfill(6)
                        mol.SetProp('_Name', name)
                    data.append((name, 0, Chem.MolToSmiles(mol, isomericSmiles=True), None, None, None, None, None,
                                 Chem.MolToMolBlock(mol)))
        else:
            raise ValueError('input file with fragments has unrecognizable extension. '
                             'Only SMI, SMILES and SDF are allowed.')
        insert_db(conn, data)
    finally:
        conn.close()
    return make_docking


def get_last_iter_from_db(db_fname):
    """
    Returns last successful iteration number (with non-NULL docking scores)
    :param db_fname:
    :return: iteration number
    """
    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()
        res = list(cur.execute("SELECT iteration, MIN(docking_score) FROM mols GROUP BY iteration ORDER BY iteration"))
        for iteration, score in reversed(res):
            if score is not None:
                return iteration


def selection_grow_greedy(mols, conn, protein_pdbqt, ntop, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param protein_pdbqt:
    :param ntop:
    :param ncpu:
    :param kwargs:
    :return: dict of parent ids and lists of corresponding generated mols
    """
    selected_mols = select_top_mols(mols, conn, ntop)
    res = __grow_mols(selected_mols, protein_pdbqt, ncpu=ncpu, **kwargs)
    return res


def selection_grow_clust(mols, conn, tanimoto, protein_pdbqt, ntop, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param tanimoto:
    :param protein_pdbqt:
    :param ntop:
    :param ncpu:
    :param kwargs:
    :return: dict of parent ids and lists of corresponding generated mols
    """
    clusters = gen_cluster_subset_algButina(mols, tanimoto)
    sorted_clusters = sort_clusters(conn, clusters)
    # select top n mols from each cluster
    selected_mols = []
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))  # {mol_id: mol, ...}
    for cluster in sorted_clusters:
        for i in cluster[:ntop]:
            selected_mols.append(mol_dict[i])
    # grow selected mols
    res = __grow_mols(selected_mols, protein_pdbqt, ncpu=ncpu, **kwargs)
    return res


def selection_grow_clust_deep(mols, conn, tanimoto, protein_pdbqt, ntop, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param tanimoto:
    :param protein_pdbqt:
    :param ntop:
    :param ncpu:
    :param kwargs:
    :return: dict of parent ids and lists of corresponding generated mols
    """
    res = dict()
    clusters = gen_cluster_subset_algButina(mols, tanimoto)
    sorted_clusters = sort_clusters(conn, clusters)
    # create dict of named mols
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))  # {mol_id: mol, ...}
    # grow up to N top scored mols from each cluster
    for cluster in sorted_clusters:
        processed_mols = 0
        for mol_id in cluster:
            tmp = __grow_mol(mol_dict[mol_id], protein_pdbqt, ncpu=ncpu, **kwargs)
            if tmp:
                res[mol_id] = tmp
                processed_mols += 1
            if processed_mols == ntop:
                break
    return res


def identify_pareto(df, tmpdir):
    """
    Return ids of mols on pareto front
    :param df:
    :param tmpdir:
    :param iteration:
    :return:
    """
    df.sort_values(0, inplace=True)
    scores = df.values
    population_size = scores.shape[0]
    population_ids = df.index
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        for j in range(population_size):
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                pareto_front[i] = 0
                break
    pareto = df.loc[pareto_front]
    x_all, y_all = df[0], df[1]
    x_pareto, y_pareto = pareto[0], pareto[1]
    plt.figure(figsize=(10, 10))
    plt.scatter(x_all, y_all)
    plt.plot(x_pareto, y_pareto, color='r')
    plt.xlabel('Docking_score')
    plt.ylabel('Mol_wweight')
    plt.savefig(tmpdir + '.jpg')
    return population_ids[pareto_front].tolist()


def selection_by_pareto(mols, conn, mw, rtb, protein_pdbqt, ncpu, tmpdir, iteration, **kwargs):
    """

    :param mols:
    :param conn:
    :param mw:
    :param rtb:
    :param protein_pdbqt:
    :param ncpu:
    :param tmpdir:
    :param iteration:
    :param kwargs:
    :return: dict of parent ids and lists of corresponding generated mols
    """
    mols = [mol for mol in mols if MolWt(mol) <= mw - 50 and CalcNumRotatableBonds(mol) <= rtb - 1]
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))
    scores = get_mol_scores(conn, mol_ids)
    scores_mw = {mol_id: [score, MolWt(mol_dict[mol_id])] for mol_id, score in scores.items() if score is not None}
    pareto_front_df = pd.DataFrame.from_dict(scores_mw, orient='index')
    mols_pareto = identify_pareto(pareto_front_df, tmpdir)
    mols = get_mols(conn, mols_pareto)
    res = __grow_mols(mols, protein_pdbqt, ncpu=ncpu, **kwargs)
    return res


def make_iteration(conn, iteration, protein_pdbqt, protein_setup, ntop, tanimoto, mw, rmsd, rtb, alg_type,
                   ncpu, tmpdir, vina_path, python_path, vina_script_dir, make_docking=True, make_selection=True,
                   **kwargs):
    if make_docking:
        tmpdir = os.path.abspath(os.path.join(tmpdir, f'iter{iteration}'))
        os.makedirs(tmpdir)
        save_smi_to_pdb(conn, iteration, tmpdir, ncpu)
        prep_ligands(conn, tmpdir, python_path, vina_script_dir, ncpu)
        dock_ligands(tmpdir, protein_pdbqt, protein_setup, vina_path, ncpu)
        update_db(conn, tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)

    mol_ids = get_docked_mol_ids(conn, iteration)
    mols = get_mols(conn, mol_ids)

    res = []

    if make_selection:

        mols = filter_mols(mols, mw, rtb)
        if not mols:
            print(f'iteration{iteration}: no molecule was selected by MW and RTB')
        if iteration != 1:
            mols = filter_mols_by_rms(mols, conn, rmsd)
            if not mols:
                print(f'iteration{iteration}: no molecule was selected by rmsd')
        if mols:
            if alg_type == 1:
                res = selection_grow_greedy(mols=mols, conn=conn, protein_pdbqt=protein_pdbqt, ntop=ntop, ncpu=ncpu,
                                            **kwargs)
            elif alg_type == 2:
                res = selection_grow_clust_deep(mols=mols, conn=conn, tanimoto=tanimoto, protein_pdbqt=protein_pdbqt,
                                                ntop=ntop, ncpu=ncpu, **kwargs)
            elif alg_type == 3:
                res = selection_grow_clust(mols=mols, conn=conn, tanimoto=tanimoto, protein_pdbqt=protein_pdbqt,
                                           ntop=ntop, ncpu=ncpu, **kwargs)
            elif alg_type == 4:
                res = selection_by_pareto(mols=mols, conn=conn, mw=mw, rtb=rtb, protein_pdbqt=protein_pdbqt,
                                          ncpu=ncpu, tmpdir=tmpdir, iteration=iteration, **kwargs)

    else:
        res = __grow_mols(mols, protein_pdbqt, ncpu=ncpu, **kwargs)

    if res:

        data = []
        opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)
        nmols = -1
        for parent_id, mols in res.items():
            for mol in mols:
                nmols += 1
                # this is a workaround for rdkit issue - if a double bond has STEREOANY it will cause errors at
                # stereoisomer enumeration, we replace STEREOANY with STEREONONE in these cases
                try:
                    isomers = tuple(EnumerateStereoisomers(mol[1], options=opts))
                except RuntimeError:
                    for bond in mol[1].GetBonds():
                        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
                            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
                    isomers = tuple(EnumerateStereoisomers(mol[1], options=opts))
                for i, m in enumerate(isomers):
                    smi = Chem.MolToSmiles(m, isomericSmiles=True)
                    mol_id = str(iteration).zfill(3) + '-' + str(nmols).zfill(6) + '-' + str(i).zfill(2)
                    data.append((mol_id, iteration, smi, parent_id, None, None, None, None, None))

        insert_db(conn, data)

        return True

    else:
        print('Growth has stopped')
        return False


def main():
    parser = argparse.ArgumentParser(description='Fragment growing within binding pocket with Autodock Vina.')
    parser.add_argument('-i', '--input_frags', metavar='FILENAME', required=False,
                        help='SMILES file with input fragments or SDF file with 3D coordinates of pre-aligned input '
                             'fragments (e.g. from PDB complexes). Optional argument.')
    parser.add_argument('-o', '--output', metavar='FILENAME', required=True,
                        help='SQLite DB with docking results. If an existed DB was supplied input fragments will be '
                             'ignored if any and the program will continue docking from the last successful iteration.')
    parser.add_argument('-d', '--db', metavar='fragments.db', required=True,
                        help='SQLite DB with fragment replacements.')
    parser.add_argument('-r', '--radius', default=1, type=int,
                        help='context radius for replacement.')
    parser.add_argument('-m', '--min_freq', default=0, type=int,
                        help='the frequency of occurrence of the fragment in the source database. Default: 0.')
    parser.add_argument('--max_replacements', type=int, required=False, default=None,
                        help='the maximum number of randomly chosen replacements. Default: None (all replacements).')
    parser.add_argument('-p', '--protein', metavar='protein.pdbqt', required=True,
                        help='input PDBQT file with a prepared protein.')
    parser.add_argument('-s', '--protein_setup', metavar='protein.log', required=True,
                        help='input text file with Vina docking setup.')
    parser.add_argument('--mgl_install_dir', metavar='DIRNAME', required=True,
                        help='path to the dir with installed MGLtools.')
    parser.add_argument('--vina', metavar='vina_path', required=True,
                        help='path to the vina executable.')
    parser.add_argument('-t', '--algorithm', default=1, type=int,
                        help='the number of the search algorithm: 1 - greedy search, 2 - deep clustering, '
                             '3 - clustering, 4 - Pareto front.')
    parser.add_argument('-mw', '--mol_weight', default=500, type=int,
                        help='maximum ligand weight')
    parser.add_argument('-nt', '--ntop', type=int, required=False,
                        help='the number of the best molecules')
    parser.add_argument('-rm', '--rmsd', type=float, required=True,
                        help='ligand movement')
    parser.add_argument('-b', '--rotatable_bonds', type=int, required=True,
                        help='the number of rotatable bonds in ligand')
    parser.add_argument('--tmpdir', metavar='DIRNAME', default=None,
                        help='directory where temporary files will be stored. If omitted atmp dir will be created in '
                             'the same location as output DB.')
    parser.add_argument('-n', '--ncpu', default=1, type=cpu_type,
                        help='number of cpus. Default: 1.')

    args = parser.parse_args()

    python_path = os.path.join(args.mgl_install_dir, 'bin/python')
    vina_script_dir = os.path.join(args.mgl_install_dir, 'MGLToolsPckgs/AutoDockTools/Utilities24/')

    if args.tmpdir is None:
        tmpdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(args.output)),
                                              ''.join(random.sample(string.ascii_lowercase, 6))))
    else:
        tmpdir = args.tmpdir

    os.makedirs(tmpdir)
    iteration = 1

    # depending on input setup operations applied on the first iteration
    # input      make_docking   make_selection
    # SMILES             True             True
    # 3D SDF            False            False
    # existed DB        False             True
    try:
        if os.path.exists(args.output):
            make_docking = False
            make_selection = True
            iteration = get_last_iter_from_db(args.output)
            if iteration is None:
                raise FileExistsError("The data was not found in the existing database. Please check the database")
        else:
            create_db(args.output)
            make_docking = insert_starting_structures_to_db(args.input_frags, args.output)
            make_selection = make_docking

        conn = sqlite3.connect(args.output)

        while True:
            index_tanimoto = 0.9  # required for alg 2 and 3
            res = make_iteration(conn=conn, iteration=iteration, protein_pdbqt=args.protein,
                                 protein_setup=args.protein_setup, ntop=args.ntop, tanimoto=index_tanimoto,
                                 mw=args.mol_weight, rmsd=args.rmsd, rtb=args.rotatable_bonds, alg_type=args.algorithm,
                                 ncpu=args.ncpu, tmpdir=tmpdir, vina_path=args.vina, python_path=python_path,
                                 vina_script_dir=vina_script_dir, make_docking=make_docking,
                                 make_selection=make_selection,
                                 db_name=args.db, radius=args.radius, min_freq=args.min_freq, min_atoms=1, max_atoms=10,
                                 max_replacements=args.max_replacements)
            make_docking = True
            make_selection = True

            if res:
                iteration += 1
                if args.algorithm in [2, 3]:
                    index_tanimoto -= 0.05
            else:
                if iteration == 1:
                    # 0 succesfull iteration for finally printing
                    iteration = 0
                break
    finally:
        if args.tmpdir is None:
            shutil.rmtree(tmpdir, ignore_errors=True)
        print(f'{iteration} iterations were completed successfully')


if __name__ == '__main__':
    main()





