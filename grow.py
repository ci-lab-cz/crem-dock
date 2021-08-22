import argparse
import glob
import json
import os
import random
import shutil
import sqlite3
import string
import subprocess
import sys
import traceback
from itertools import combinations
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from crem.crem import grow_mol
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.ML.Cluster import Butina
from scipy.spatial import distance_matrix

from scripts import Docking, Smi2PDB


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


def save_smi_to_pdb(conn, iteration, tmpdir, protonation, ncpu):
    '''
    Creates file with SMILES which is supplied to Chemaxon cxcalc utility to get molecule ionization states at pH 7.4.
    Parse output and generate PDB files stored in the specified directory.
    :param conn:
    :param iteration:
    :param tmpdir:
    :param ncpu:
    :return:
    '''
    cur = conn.cursor()
    smiles = cur.execute(f"SELECT smi, id FROM mols WHERE iteration = '{iteration - 1}'")
    smiles, mol_ids = zip(*smiles)
    if protonation:
        fname = os.path.join(tmpdir, '1.smi')
        with open(fname, 'wt') as f:
            f.writelines('%s\t%s\n' % item for item in zip(smiles, mol_ids))
        cmd_run = f"cxcalc majormicrospecies -H 7.4 -f smiles '{fname}'"
        smiles = subprocess.check_output(cmd_run, shell=True).decode().split()
        for mol_id, smi_protonated in zip(mol_ids, smiles):
            cur.execute("""UPDATE mols
                       SET smi_protonated = ? 
                       WHERE
                           id = ?
                    """, (smi_protonated, mol_id))
        conn.commit()

    pool = Pool(ncpu)

    pool.imap_unordered(Smi2PDB.save_to_pdb_mp,
                        ((smi, os.path.join(tmpdir, f'{mol_id}.pdb')) for smi, mol_id in zip(smiles,mol_ids)))
    pool.close()
    pool.join()


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


def update_db(conn, dname, protonation):
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
        if protonation:
            smi = list(cur.execute(f"SELECT smi_protonated FROM mols WHERE id = '{mol_id}'"))[0][0]
        else:
            smi = list(cur.execute(f"SELECT smi FROM mols WHERE id = '{mol_id}'"))[0][0]

        mol_block = None
        mol = Chem.MolFromPDBBlock('\n'.join([i[:66] for i in pdb_block.split('MODEL')[1].split('\n')]),
                                   removeHs=False)

        if mol:
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(Chem.AddHs(Chem.MolFromSmiles(smi), explicitOnly=True), mol)
                mol.SetProp('_Name', mol_id)
                mol_block = Chem.MolToMolBlock(mol)
            except:
                sys.stderr.write(f'Could not assign bond orders while parsing PDB: {fname}\n')
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
            sys.stderr.write(f'Could not read PDB: {fname}\n')
            rms = None

        cur.execute("""UPDATE mols
                           SET pdb_block = ?,
                               mol_block = ?,
                               docking_score = ?,
                               rmsd = ? 
                           WHERE
                               id = ?
                        """, (pdb_block.split('MODEL 2')[0], mol_block, score, rms, mol_id))
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
    return round(best_rms, 3)


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
    sql = f'SELECT mol_block, protected_user_canon_ids FROM mols WHERE id IN ({",".join("?" * len(mol_ids))})'
    mols = []
    for items in cur.execute(sql, mol_ids):
        m = Chem.MolFromMolBlock(items[0], removeHs=False)
        if not m:
            continue
        if items[1] is not None:
            m.SetProp('protected_user_canon_ids', items[1])
        mols.append(m)
    cur.close()
    return mols


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


def get_protected_ids(mol, protein_xyz, dist_threshold):
    """
    Returns list of ids of heavy atoms ids which have ALL hydrogen atoms close to the protein
    :param mol: molecule
    :param protein_file: protein pdbqt file
    :param dist_threshold: minimum distance to hydrogen atoms
    :return:
    """
    hids = np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1])
    xyz = mol.GetConformer().GetPositions()[hids]
    min_xyz = xyz.min(axis=0) - dist_threshold
    max_xyz = xyz.max(axis=0) + dist_threshold
    # select protein atoms which are within a box of min-max coordinates of ligand hydrogen atoms
    pids = (protein_xyz >= min_xyz).any(axis=1) & (protein_xyz <= max_xyz).any(axis=1)
    pxyz = protein_xyz[pids]
    m = distance_matrix(xyz, pxyz)  # get matrix (ligandH x protein)
    ids = set(hids[(m <= dist_threshold).any(axis=1)].tolist())  # ids of H atoms close to a protein

    output_ids = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() > 1:
            if not (set(n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() == 1) - ids):  # all hydrogens of a heavy atom are close to protein
                output_ids.append(a.GetIdx())

    return output_ids


def get_protein_heavy_atom_xyz(protein_pdbqt):
    """
    Returns coordinates of heavy atoms
    :param protein_pdbqt:
    :return: 2darray (natoms x 3)
    """
    pdb_block = open(protein_pdbqt).readlines()
    protein = Chem.MolFromPDBBlock('\n'.join([line[:66] for line in pdb_block]), sanitize=False)
    if protein is None:
        raise ValueError("Protein structure is incorrect. Please check protein pdbqt file.")
    xyz = protein.GetConformer().GetPositions()
    xyz = xyz[[a.GetAtomicNum() > 1 for a in protein.GetAtoms()], ]
    return xyz


def __grow_mol(conn, mol, protein_xyz, protonation, h_dist_threshold=2, ncpu=1, **kwargs):

    def find_protected_ids(protected_ids, mol1, mol2):
        """
        Find a correspondence between protonated and non-protonated structures to transfer prpotected ids
        to non-protonated molecule
        :param protected_ids: ids of heavy atoms protected in protonated mol1
        :param mol1: protonated mol
        :param mol2: non-protonated mol
        :return: set of ids of heavy atoms which should be protected in non-protonated mol2
        """
        mcs = rdFMCS.FindMCS((mol1, mol2)).queryMol
        mcs1, mcs2 = mol1.GetSubstructMatches(mcs), mol2.GetSubstructMatches(mcs)
        # mcs1 = list(set(frozenset(i) for i in mcs1))
        # mcs2 = list(set(frozenset(i) for i in mcs2))
        if len(mcs1) > 1 or len(mcs2) > 1:
            sys.stderr.write(f'MCS has multiple mappings in one of these structures: protonated smi '
                             f'{Chem.MolToSmiles(mol1)} or non-protonated smi {Chem.MolToSmiles(mol2)}. '
                             f'One randomly choosing mapping will be used to determine protected atoms.\n')
        mcs1 = mcs1[0]
        mcs2 = mcs2[0]
        # we protect the same atoms which were protected in a protonated mol. Atoms which lost H after protonation
        # will never be selected as protected by the algorithm (only one exception if this heavy atoms bears more
        # than one H), so there is no need to specifically process them. Atoms, to which H were added after protonation,
        # will be protected only if all H atoms are close to a protein
        ids = [j for i, j in zip(mcs1, mcs2) if i in protected_ids]
        return ids

    mol = Chem.AddHs(mol, addCoords=True)
    _protected_user_ids = set()
    if mol.HasProp('protected_user_canon_ids'):
        _protected_user_ids = set(get_atom_idxs_for_canon(mol, [int(i) for i in mol.GetProp('protected_user_canon_ids').split(',')]))
    _protected_alg_ids = set(get_protected_ids(mol, protein_xyz, h_dist_threshold))
    protected_ids = _protected_alg_ids & _protected_user_ids

    if protonation and protected_ids:
        mol_id = mol.GetProp('_Name')
        cur = conn.cursor()
        mol2 = Chem.MolFromSmiles(list(cur.execute(f"SELECT smi FROM mols WHERE id = '{mol_id}'"))[0][0])  # non-protonated mol
        mol2.SetProp('_Name', mol_id)
        protected_ids = find_protected_ids(protected_ids, mol, mol2)
        mol = mol2

    try:
        res = list(grow_mol(mol, protected_ids=protected_ids, return_rxn=False, return_mol=True, ncores=ncpu, **kwargs))
    except Exception:
        error_message = traceback.format_exc()
        sys.stderr.write(f'Grow error.\n'
                         f'{error_message}\n'
                         f'{mol.GetProp("_Name")}\n'
                         f'{Chem.MolToSmiles(mol)}')
        res = []
    return res


def __grow_mols(conn, mols, protein_pdbqt, protonation, h_dist_threshold=2, ncpu=1, **kwargs):
    """

    :param mols: list of molecules
    :param protein_pdbqt: protein pdbqt file
    :param h_dist_threshold: maximum distance from H atoms to the protein to mark them as protected from further grow
    :param ncpu: number of cpu
    :param kwargs: arguments passed to crem function grow_mol
    :return: dict of parent mols and lists of corresponding generated mols
    """
    res = dict()
    protein_xyz = get_protein_heavy_atom_xyz(protein_pdbqt)
    for mol in mols:
        tmp = __grow_mol(conn, mol, protein_xyz, protonation, h_dist_threshold=h_dist_threshold, ncpu=ncpu, **kwargs)
        if tmp:
            res[mol] = tmp
    return res


def insert_db(conn, data):
    cur = conn.cursor()
    cur.executemany("""INSERT INTO mols VAlUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", data)
    conn.commit()


def create_db(fname):
    p = os.path.dirname(fname)
    if p:  # if p is "" (current dir) the error will occur
        os.makedirs(os.path.dirname(fname), exist_ok=True)
    conn = sqlite3.connect(fname)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS mols")
    cur.execute("""CREATE TABLE IF NOT EXISTS mols
            (
             id TEXT PRIMARY KEY,
             iteration INTEGER,
             smi TEXT,
             smi_protonated TEXT,
             parent_id TEXT,
             docking_score REAL,
             atoms TEXT,
             rmsd REAL,
             pdb_block TEXT,
             mol_block TEXT,
             protected_user_canon_ids TEXT DEFAULT NULL
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
                    data.append((name, 0, smi, None, None, None, None, None, None, None, None))
        elif fname.lower().endswith('.sdf'):
            make_docking = False
            for i, mol in enumerate(Chem.SDMolSupplier(fname)):
                if mol:
                    name = mol.GetProp('_Name')
                    if not name:
                        name = '000-' + str(i).zfill(6)
                        mol.SetProp('_Name', name)
                    mol = Chem.AddHs(mol, addCoords=True)
                    protected_user_canon_ids = None
                    if mol.HasProp('protected_user_ids'):
                        # rdkit numeration starts with 0 and sdf numeration starts with 1
                        protected_user_ids = [int(idx) - 1 for idx in mol.GetProp('protected_user_ids').split(',')]
                        protected_user_canon_ids = ','.join([str(canon_idx) for canon_idx in
                                                                    get_canon_for_atom_idx(mol, protected_user_ids)])

                    data.append((name, 0, Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True), None, None, None, None, None, None,
                                 Chem.MolToMolBlock(mol), protected_user_canon_ids))
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
                return iteration + 1


def selection_grow_greedy(mols, conn, protein_pdbqt, protonation, ntop, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param protein_pdbqt:
    :param ntop:
    :param ncpu:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols
    """
    selected_mols = select_top_mols(mols, conn, ntop)
    res = __grow_mols(conn, selected_mols, protein_pdbqt, protonation, ncpu=ncpu, **kwargs)
    return res


def selection_grow_clust(mols, conn, tanimoto, protein_pdbqt, protonation, ntop, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param tanimoto:
    :param protein_pdbqt:
    :param ntop:
    :param ncpu:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols
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
    res = __grow_mols(conn, selected_mols, protein_pdbqt, protonation, ncpu=ncpu, **kwargs)
    return res


def selection_grow_clust_deep(mols, conn, tanimoto, protein_pdbqt, protonation, ntop, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param tanimoto:
    :param protein_pdbqt:
    :param ntop:
    :param ncpu:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols
    """
    res = dict()
    clusters = gen_cluster_subset_algButina(mols, tanimoto)
    sorted_clusters = sort_clusters(conn, clusters)
    # create dict of named mols
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))  # {mol_id: mol, ...}
    protein_xyz = get_protein_heavy_atom_xyz(protein_pdbqt)
    # grow up to N top scored mols from each cluster
    for cluster in sorted_clusters:
        processed_mols = 0
        for mol_id in cluster:
            tmp = __grow_mol(conn, mol_dict[mol_id], protein_xyz, protonation, ncpu=ncpu, **kwargs)
            if tmp:
                res[mol_dict[mol_id]] = tmp
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


def selection_by_pareto(mols, conn, mw, rtb, protein_pdbqt, protonation, ncpu, tmpdir, iteration, **kwargs):
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
    :return: dict of parent mol and lists of corresponding generated mols
    """
    mols = [mol for mol in mols if MolWt(mol) <= mw - 50 and CalcNumRotatableBonds(mol) <= rtb - 1]
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))
    scores = get_mol_scores(conn, mol_ids)
    scores_mw = {mol_id: [score, MolWt(mol_dict[mol_id])] for mol_id, score in scores.items() if score is not None}
    pareto_front_df = pd.DataFrame.from_dict(scores_mw, orient='index')
    mols_pareto = identify_pareto(pareto_front_df, tmpdir)
    mols = get_mols(conn, mols_pareto)
    res = __grow_mols(conn, mols, protein_pdbqt, protonation, ncpu=ncpu, **kwargs)
    return res


def get_product_atom_protected(mol, protected_parent_ids):
    '''

    :param mol:
    :param protected_parent_ids: list[int]
    :type   protected_parent_ids: list[int]
    :return: sorted list of integers
    '''
    # After RDKit reaction procedure there is a field <react_atom_idx> with initial parent atom idx in product mol
    protected_product_ids = []
    for a in mol.GetAtoms():
        if a.HasProp('react_atom_idx') and int(a.GetProp('react_atom_idx')) in protected_parent_ids:
            protected_product_ids.append(a.GetIdx())
    return sorted(protected_product_ids)


def get_atom_idxs_for_canon(mol, canon_idxs):
    '''
    get the rdkit current indices for the canonical indices of the molecule
    :param mol:
    :param canon_idxs: list[int]
    :return:  sorted list of integers
    '''
    canon_ranks = np.array(Chem.CanonicalRankAtoms(mol))
    return sorted(np.where(np.isin(canon_ranks, canon_idxs))[0].tolist())


def get_canon_for_atom_idx(mol, idx):
    '''
    get the canonical numeration of the current molecule indices
    :param mol:
    :param idx: list[int]
    :return: sorted list of integers
    '''
    canon_ranks = np.array(Chem.CanonicalRankAtoms(mol))
    return sorted(canon_ranks[idx].tolist())


def make_iteration(conn, iteration, protein_pdbqt, protein_setup, ntop, tanimoto, mw, rmsd, rtb, alg_type,
                   ncpu, tmpdir, vina_path, python_path, vina_script_dir, protonation, debug,
                   make_docking=True, make_selection=True, **kwargs):
    if make_docking:
        tmpdir = os.path.abspath(os.path.join(tmpdir, f'iter{iteration}'))
        os.makedirs(tmpdir)
        save_smi_to_pdb(conn, iteration, tmpdir, protonation, ncpu)
        prep_ligands(conn, tmpdir, python_path, vina_script_dir, ncpu)
        dock_ligands(tmpdir, protein_pdbqt, protein_setup, vina_path, ncpu)
        update_db(conn, tmpdir, protonation)
        if not debug:
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
                res = selection_grow_greedy(mols=mols, conn=conn, protein_pdbqt=protein_pdbqt, protonation=protonation, ntop=ntop, ncpu=ncpu,
                                            **kwargs)
            elif alg_type == 2:
                res = selection_grow_clust_deep(mols=mols, conn=conn, tanimoto=tanimoto, protein_pdbqt=protein_pdbqt, protonation=protonation,
                                                ntop=ntop, ncpu=ncpu, **kwargs)
            elif alg_type == 3:
                res = selection_grow_clust(mols=mols, conn=conn, tanimoto=tanimoto, protein_pdbqt=protein_pdbqt, protonation=protonation,
                                           ntop=ntop, ncpu=ncpu, **kwargs)
            elif alg_type == 4:
                res = selection_by_pareto(mols=mols, conn=conn, mw=mw, rtb=rtb, protein_pdbqt=protein_pdbqt, protonation=protonation,
                                          ncpu=ncpu, tmpdir=tmpdir, iteration=iteration, **kwargs)

    else:
        res = __grow_mols(mols, protein_pdbqt, ncpu=ncpu, **kwargs)

    if res:
        data = []
        opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)
        nmols = -1

        for parent_mol, product_mols in res.items():
            for mol in product_mols:
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
                    m = Chem.AddHs(m)
                    parent_mol = Chem.AddHs(parent_mol)
                    mol_id = str(iteration).zfill(3) + '-' + str(nmols).zfill(6) + '-' + str(i).zfill(2)
                    product_protected_canon_user_id = None
                    if parent_mol.HasProp('protected_user_canon_ids') and parent_mol.GetProp('protected_user_canon_ids'):
                        parent_protected_user_ids = get_atom_idxs_for_canon(parent_mol,
                                                                            [int(idx) for idx in parent_mol.GetProp('protected_user_canon_ids').split(',')])
                        product_protected_user_id = get_product_atom_protected(m, parent_protected_user_ids)
                        product_protected_canon_user_id = ','.join([str(canon_idx) for canon_idx in get_canon_for_atom_idx(m, product_protected_user_id)])

                    data.append((mol_id, iteration, Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True), None, parent_mol.GetProp('_Name'), None, None, None, None, None,
                                 product_protected_canon_user_id))

        insert_db(conn, data)
        return True

    else:
        print('Growth has stopped')
        return False


def main():
    parser = argparse.ArgumentParser(description='Fragment growing within binding pocket with Autodock Vina.')
    parser.add_argument('-i', '--input_frags', metavar='FILENAME', required=False,
                        help='SMILES file with input fragments or SDF file with 3D coordinates of pre-aligned input '
                             'fragments (e.g. from PDB complexes). '
                             'SDF also may contain <protected_user_ids> filed where are atoms ids which are protected to grow (comma-sep).'
                             ' Optional argument.')
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
    parser.add_argument('--min_atoms', default=1, type=int,
                        help='the minimum number of atoms in the fragment which will replace H')
    parser.add_argument('--max_atoms', default=10, type=int,
                        help='the maximum number of atoms in the fragment which will replace H')
    parser.add_argument('-p', '--protein', metavar='protein.pdbqt', required=True,
                        help='input PDBQT file with a prepared protein.')
    parser.add_argument('-s', '--protein_setup', metavar='protein.log', required=True,
                        help='input text file with Vina docking setup.')
    parser.add_argument('--no_protonation', action='store_true', default=False,
                        help='disable protonation of molecules before docking. Protonation requires installed '
                             'cxcalc chemaxon utility.')
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
    parser.add_argument('--debug', action='store_true', default=False,
                        help='enable debug mode; all tmp files will not be erased.')
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

    os.makedirs(tmpdir, exist_ok=True)
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

        with open(os.path.splitext(args.output)[0] + '.json', 'wt') as f:
            json.dump(vars(args), f, sort_keys=True, indent=2)

        conn = sqlite3.connect(args.output)

        while True:
            index_tanimoto = 0.9  # required for alg 2 and 3
            res = make_iteration(conn=conn, iteration=iteration, protein_pdbqt=args.protein,
                                 protein_setup=args.protein_setup, ntop=args.ntop, tanimoto=index_tanimoto,
                                 mw=args.mol_weight, rmsd=args.rmsd, rtb=args.rotatable_bonds, alg_type=args.algorithm,
                                 ncpu=args.ncpu, tmpdir=tmpdir, vina_path=args.vina, python_path=python_path,
                                 vina_script_dir=vina_script_dir, make_docking=make_docking,
                                 make_selection=make_selection,
                                 db_name=args.db, radius=args.radius, min_freq=args.min_freq,
                                 min_atoms=args.min_atoms, max_atoms=args.max_atoms,
                                 max_replacements=args.max_replacements, protonation=not args.no_protonation,
                                 debug=args.debug)
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