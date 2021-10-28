import argparse
import json
import os
import random
import shutil
import sqlite3
import string
import subprocess
import sys
import tempfile
import traceback
from multiprocessing import cpu_count

import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from crem.crem import grow_mol
from dask.distributed import Client
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED
from rdkit.Chem import rdFMCS
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcCrippenDescriptors
from rdkit.ML.Cluster import Butina
from scipy.spatial import distance_matrix

from scripts import Docking


def cpu_type(x):
    return max(1, min(int(x), cpu_count()))


def filepath_type(x):
    if x:
        return os.path.abspath(x)
    else:
        return x


def sort_two_lists(primary, secondary):
    # sort two lists by order of elements of the primary list
    paired_sorted = sorted(zip(primary, secondary), key=lambda x: x[0])
    return map(list, zip(*paired_sorted))  # two lists


def neutralize_atoms(mol):
    # https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def get_mol_ids(mols):
    return [mol.GetProp('_Name') for mol in mols]


def add_protonation(conn, iteration):
    '''
    Protonate SMILES by Chemaxon cxcalc utility to get molecule ionization states at pH 7.4.
    Parse output and update db.
    :param conn:
    :param iteration:
    :return:
    '''
    cur = conn.cursor()
    smiles_dict = cur.execute(f"SELECT smi, id FROM mols WHERE iteration = '{iteration - 1}'")

    if not smiles_dict:
        sys.stderr.write(f'no molecules to protonate in iteration {iteration}\n')
        return

    smiles, mol_ids = zip(*smiles_dict)

    with tempfile.NamedTemporaryFile(suffix='.smi', mode='w', encoding='utf-8') as tmp:
        tmp.writelines(['\n'.join(smiles)])
        tmp.seek(0)
        cmd_run = f"cxcalc majormicrospecies -H 7.4 -f smiles -M -K '{tmp.name}'"
        smiles_protonated = subprocess.check_output(cmd_run, shell=True).decode().split()

    for mol_id, smi_protonated in zip(mol_ids, smiles_protonated):
        cur.execute("""UPDATE mols
                       SET smi_protonated = ?
                       WHERE
                           id = ?
                    """, (Chem.MolToSmiles(Chem.MolFromSmiles(smi_protonated), isomericSmiles=True), mol_id))
    conn.commit()


def update_db(conn, iteration):
    """
    Post-process all docked molecules from an individual iteration.
    Calculate rmsd of a molecule to a parent mol. Insert rmsd in output db.
    :param conn: connection to docking DB
    :param iteration: current iteration
    :return:
    """
    cur = conn.cursor()

    mol_ids = get_docked_mol_ids(conn, iteration)
    mols = get_mols(conn, mol_ids)
    # parent_ids and parent_mols can be empty if all compounds do not have parents
    parent_ids = dict(cur.execute(f"SELECT id, parent_id "
                                  f"FROM mols "
                                  f"WHERE id IN ({','.join('?' * len(mol_ids))}) AND "
                                  f"parent_id iS NOT NULL", mol_ids))
    uniq_parent_ids = list(set(parent_ids.values()))
    parent_mols = get_mols(conn, uniq_parent_ids)
    parent_mols = {m.GetProp('_Name'): m for m in parent_mols}

    for mol in mols:
        rms = None
        mol_id = mol.GetProp('_Name')
        try:
            parent_mol = parent_mols[parent_ids[mol_id]]
            rms = get_rmsd(mol, parent_mol)
        except KeyError:  # missing parent mol
            pass

        cur.execute("""UPDATE mols
                           SET 
                               rmsd = ? 
                           WHERE
                               id = ?
                        """, (rms, mol_id))
    conn.commit()


def get_rmsd(child_mol, parent_mol):
    """
    Returns best fit rmsd between a common part of child and parent molecules taking into account symmetry of molecules
    :param child_mol: Mol
    :param parent_mol: Mol
    :return:
    """
    child_mol = neutralize_atoms(Chem.RemoveHs(child_mol))
    parent_mol = neutralize_atoms(Chem.RemoveHs(parent_mol))
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


def get_docked_mol_data(conn, iteration):
    """
    Returns mol_ids, RMSD for molecules which where docked at the given iteration and conversion
    to mol block was successful
    :param conn:
    :param iteration:
    :return: DataFrame with columns RMSD and mol_id as index
    """
    cur = conn.cursor()
    res = tuple(cur.execute(f"SELECT id, rmsd FROM mols WHERE iteration = '{iteration - 1}' AND mol_block IS NOT NULL"))
    df = pd.DataFrame(res, columns=['id', 'rmsd']).set_index('id')
    return df


def get_mols(conn, mol_ids):
    """
    Returns list of Mol objects from docking DB, order is arbitrary, molecules with errors will be silently skipped
    :param conn: connection to docking DB
    :param mol_ids: list of molecules to retrieve
    :return:
    """
    cur = conn.cursor()
    sql = f'SELECT mol_block, protected_user_canon_ids FROM mols WHERE id IN ({",".join("?" * len(mol_ids))})'
    mols = []
    for items in cur.execute(sql, mol_ids):
        m = Chem.MolFromMolBlock(items[0], removeHs=False)
        Chem.AssignAtomChiralTagsFromStructure(m)
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

    # def find_protected_ids(protected_ids, mol1, mol2):
    #     """
    #     Find a correspondence between protonated and non-protonated structures to transfer prpotected ids
    #     to non-protonated molecule
    #     :param protected_ids: ids of heavy atoms protected in protonated mol1
    #     :param mol1: protonated mol
    #     :param mol2: non-protonated mol
    #     :return: set of ids of heavy atoms which should be protected in non-protonated mol2
    #     """
    #     mcs = rdFMCS.FindMCS((mol1, mol2)).queryMol
    #     mcs1, mcs2 = mol1.GetSubstructMatches(mcs), mol2.GetSubstructMatches(mcs)
    #     # mcs1 = list(set(frozenset(i) for i in mcs1))
    #     # mcs2 = list(set(frozenset(i) for i in mcs2))
    #     if len(mcs1) > 1 or len(mcs2) > 1:
    #         sys.stderr.write(f'MCS has multiple mappings in one of these structures: protonated smi '
    #                          f'{Chem.MolToSmiles(mol1)} or non-protonated smi {Chem.MolToSmiles(mol2)}. '
    #                          f'One randomly choosing mapping will be used to determine protected atoms.\n')
    #     mcs1 = mcs1[0]
    #     mcs2 = mcs2[0]
    #     # we protect the same atoms which were protected in a protonated mol. Atoms which lost H after protonation
    #     # will never be selected as protected by the algorithm (only one exception if this heavy atoms bears more
    #     # than one H), so there is no need to specifically process them. Atoms, to which H were added after protonation,
    #     # will be protected only if all H atoms are close to a protein
    #     ids = [j for i, j in zip(mcs1, mcs2) if i in protected_ids]
    #     return ids

    mol = Chem.AddHs(mol, addCoords=True)
    _protected_user_ids = set()
    if mol.HasProp('protected_user_canon_ids'):
        _protected_user_ids = set(get_atom_idxs_for_canon(mol, list(map(int, mol.GetProp('protected_user_canon_ids').split(',')))))
    _protected_alg_ids = set(get_protected_ids(mol, protein_xyz, h_dist_threshold))
    protected_ids = _protected_alg_ids | _protected_user_ids

    # remove explicit hydrogen and charges and redefine protected atom ids
    for i in protected_ids:
        mol.GetAtomWithIdx(i).SetIntProp('__tmp', 1)
    mol = neutralize_atoms(Chem.RemoveHs(mol))
    protected_ids = []
    for a in mol.GetAtoms():
        if a.HasProp('_tmp') and a.GetIntProp('_tmp'):
            protected_ids.append(a.GetIdx())

    # if protonation:
    #     mol_id = mol.GetProp('_Name')
    #     cur = conn.cursor()
    #     mol2 = Chem.MolFromSmiles(list(cur.execute(f"SELECT smi FROM mols WHERE id = '{mol_id}'"))[0][0])  # non-protonated mol
    #     mol2.SetProp('_Name', mol_id)
    #     if protected_ids:
    #         protected_ids = find_protected_ids(protected_ids, mol, mol2)
    #     mol = mol2

    try:
        res = list(grow_mol(mol, protected_ids=protected_ids, return_rxn=False, return_mol=True, ncores=ncpu, **kwargs))
    except Exception:
        error_message = traceback.format_exc()
        sys.stderr.write(f'Grow error.\n'
                         f'{error_message}\n'
                         f'{mol.GetProp("_Name")}\n'
                         f'{Chem.MolToSmiles(mol)}')
        res = []

    res = tuple(m for smi, m in res)

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
    cur.executemany("""INSERT INTO mols VAlUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", data)
    conn.commit()


def create_db(fname):
    p = os.path.dirname(fname)
    if p:  # if p is "" (current dir) the error will occur
        os.makedirs(os.path.dirname(fname), exist_ok=True)
    conn = sqlite3.connect(fname)
    cur = conn.cursor()
    # cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("DROP TABLE IF EXISTS mols")
    cur.execute("""CREATE TABLE IF NOT EXISTS mols
            (
             id TEXT PRIMARY KEY,
             iteration INTEGER,
             smi TEXT,
             smi_protonated TEXT,
             parent_id TEXT,
             docking_score REAL,
             mw REAL,
             rtb INTEGER,
             logp REAL,
             qed REAL,
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
                    mol_mw, mol_rtb, mol_logp, mol_qed = calc_properties(Chem.MolFromSmiles(smi))
                    data.append((name, 0, smi, None, None, None, mol_mw, mol_rtb, mol_logp, mol_qed, None, None, None, None))
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
                        protected_user_canon_ids = ','.join(map(str, get_canon_for_atom_idx(mol, protected_user_ids)))
                    mol_mw, mol_rtb, mol_logp, mol_qed = calc_properties(mol)

                    data.append((name, 0, Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True), None, None, None,
                                 mol_mw, mol_rtb, mol_logp, mol_qed, None, None, Chem.MolToMolBlock(mol),
                                 protected_user_canon_ids))
        else:
            raise ValueError('input file with fragments has unrecognizable extension. '
                             'Only SMI, SMILES and SDF are allowed.')
        insert_db(conn, data)
    finally:
        conn.close()
    return make_docking


def get_last_iter_from_db(db_fname):
    """
    Returns last iteration number
    :param db_fname:
    :return: iteration number
    """
    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()
        res = list(cur.execute("SELECT max(iteration) FROM mols"))[0][0]
        return res + 1


def selection_grow_greedy(mols, conn, protein_pdbqt, protonation, ntop, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param protein_pdbqt:
    :param protonation:
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
    :param protonation:
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
    :param protonation:
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


def selection_by_pareto(mols, conn, mw, rtb, protein_pdbqt, protonation, ncpu, tmpdir, **kwargs):
    """

    :param mols:
    :param conn:
    :param mw:
    :param rtb:
    :param protein_pdbqt:
    :param protonation:
    :param ncpu:
    :param tmpdir:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols
    """
    mols = [mol for mol in mols if MolWt(mol) <= mw - 50 and CalcNumRotatableBonds(mol) <= rtb - 1]
    if not mols:
        return None
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))
    scores = get_mol_scores(conn, mol_ids)
    scores_mw = {mol_id: [score, MolWt(mol_dict[mol_id])] for mol_id, score in scores.items() if score is not None}
    pareto_front_df = pd.DataFrame.from_dict(scores_mw, orient='index')
    mols_pareto = identify_pareto(pareto_front_df, tmpdir)
    mols = get_mols(conn, mols_pareto)
    res = __grow_mols(conn, mols, protein_pdbqt, protonation, ncpu=ncpu, **kwargs)
    return res


def get_child_protected_atom_ids(mol, protected_parent_ids):
    '''

    :param mol:
    :param protected_parent_ids: list[int]
    :type  protected_parent_ids: list[int]
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
    :return: sorted list of integers
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


def get_isomers(mol):
    opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)
    # this is a workaround for rdkit issue - if a double bond has STEREOANY it will cause errors at
    # stereoisomer enumeration, we replace STEREOANY with STEREONONE in these cases
    try:
        isomers = tuple(EnumerateStereoisomers(mol, options=opts))
    except RuntimeError:
        for bond in mol[1].GetBonds():
            if bond.GetStereo() == Chem.BondStereo.STEREOANY:
                bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
        isomers = tuple(EnumerateStereoisomers(mol, options=opts))
    return isomers


def calc_properties(mol):
    mw = round(MolWt(mol), 2)
    rtb = CalcNumRotatableBonds(mol)
    logp = round(CalcCrippenDescriptors(mol)[0], 2)
    qed = round(QED.qed(mol), 3)
    return mw, rtb, logp, qed


def make_iteration(dbname, iteration, protein_pdbqt, protein_setup, ntop, tanimoto, mw, rmsd, rtb, alg_type,
                   ncpu, tmpdir, protonation, continuation=True, make_docking=True, use_dask=False, **kwargs):

    sys.stderr.write(f'iteration {iteration} started\n')
    conn = sqlite3.connect(dbname)
    if protonation and not continuation:
        add_protonation(conn, iteration)
    if make_docking:
        Docking.iter_docking(dbname=dbname, receptor_pdbqt_fname=protein_pdbqt, protein_setup=protein_setup,
                             protonation=protonation, iteration=iteration, use_dask=use_dask, ncpu=ncpu)
        update_db(conn, iteration)

        res = []
        mol_data = get_docked_mol_data(conn, iteration)
        # mol_data = mol_data.loc[(mol_data['mw'] <= mw) & (mol_data['rtb'] <= rtb)]  # filter by MW and RTB
        if iteration != 1:
            mol_data = mol_data.loc[mol_data['rmsd'] <= rmsd]  # filter by RMSD
        if len(mol_data.index) == 0:
            sys.stderr.write(f'iteration {iteration}: no molecules were selected for growing.\n')
        else:
            mols = get_mols(conn, mol_data.index)
            if alg_type == 1:
                res = selection_grow_greedy(mols=mols, conn=conn, protein_pdbqt=protein_pdbqt, protonation=protonation,
                                            ntop=ntop, ncpu=ncpu, **kwargs)
            elif alg_type == 2:
                res = selection_grow_clust_deep(mols=mols, conn=conn, tanimoto=tanimoto, protein_pdbqt=protein_pdbqt,
                                                protonation=protonation, ntop=ntop, ncpu=ncpu, **kwargs)
            elif alg_type == 3:
                res = selection_grow_clust(mols=mols, conn=conn, tanimoto=tanimoto, protein_pdbqt=protein_pdbqt,
                                           protonation=protonation, ntop=ntop, ncpu=ncpu, **kwargs)
            elif alg_type == 4:
                res = selection_by_pareto(mols=mols, conn=conn, mw=mw, rtb=rtb, protein_pdbqt=protein_pdbqt,
                                          protonation=protonation, ncpu=ncpu, tmpdir=tmpdir, **kwargs)

    else:
        mols = get_mols(conn, get_docked_mol_ids(conn, iteration))
        res = __grow_mols(conn, mols=mols, protein_pdbqt=protein_pdbqt, protonation=protonation, ncpu=ncpu, **kwargs)

    if res:
        data = []
        nmols = -1
        for parent_mol, child_mols in res.items():
            parent_mol = Chem.AddHs(parent_mol)
            for mol in child_mols:
                mol_mw, mol_rtb, mol_logp, mol_qed = calc_properties(mol)
                if mol_mw <= mw and mol_rtb <= rtb:
                    nmols += 1
                    isomers = get_isomers(mol)
                    for i, m in enumerate(isomers):
                        m = Chem.AddHs(m)
                        mol_id = str(iteration).zfill(3) + '-' + str(nmols).zfill(6) + '-' + str(i).zfill(2)
                        # save canonical protected atom ids because we store mols as SMILES and lost original atom enumeraion
                        child_protected_canon_user_id = None
                        if parent_mol.HasProp('protected_user_canon_ids'):
                            parent_protected_user_ids = get_atom_idxs_for_canon(parent_mol, list(map(int, parent_mol.GetProp('protected_user_canon_ids').split(','))))
                            child_protected_user_id = get_child_protected_atom_ids(m, parent_protected_user_ids)
                            child_protected_canon_user_id = ','.join(map(str, get_canon_for_atom_idx(m, child_protected_user_id)))

                        data.append((mol_id, iteration, Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True), None,
                                     parent_mol.GetProp('_Name'), None, mol_mw, mol_rtb, mol_logp, mol_qed, None, None,
                                     None, child_protected_canon_user_id))

        insert_db(conn, data)
        return True

    else:
        sys.stderr.write('Growth has stopped\n')
        return False


def main():
    parser = argparse.ArgumentParser(description='Fragment growing within binding pocket with Autodock Vina.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_frags', metavar='FILENAME', required=False, type=filepath_type,
                        help='SMILES file with input fragments or SDF file with 3D coordinates of pre-aligned input '
                             'fragments (e.g. from PDB complexes). '
                             'If SDF contain <protected_user_ids> field (comma-separated 1-based indices) '
                             'these atoms will be protected from growing. This argument can be omitted if an existed '
                             'output DB is specified, then docking will be continued from the last successful '
                             'iteration. Optional.')
    parser.add_argument('-o', '--output', metavar='FILENAME', required=True, type=filepath_type,
                        help='SQLite DB with docking results. If an existed DB was supplied input fragments will be '
                             'ignored if any and the program will continue docking from the last successful iteration.')
    parser.add_argument('-d', '--db', metavar='fragments.db', required=True, type=filepath_type,
                        help='SQLite DB with fragment replacements.')
    parser.add_argument('-r', '--radius', default=1, type=int,
                        help='context radius for replacement.')
    parser.add_argument('--min_freq', default=0, type=int,
                        help='the frequency of occurrence of the fragment in the source database.')
    parser.add_argument('--max_replacements', type=int, required=False, default=None,
                        help='the maximum number of randomly chosen replacements. Default: None (all replacements).')
    parser.add_argument('--min_atoms', default=1, type=int,
                        help='the minimum number of atoms in the fragment which will replace H')
    parser.add_argument('--max_atoms', default=10, type=int,
                        help='the maximum number of atoms in the fragment which will replace H')
    parser.add_argument('-p', '--protein', metavar='protein.pdbqt', required=True, type=filepath_type,
                        help='input PDBQT file with a prepared protein.')
    parser.add_argument('-s', '--protein_setup', metavar='protein.log', required=True, type=filepath_type,
                        help='input text file with Vina docking setup.')
    parser.add_argument('--no_protonation', action='store_true', default=False,
                        help='disable protonation of molecules before docking. Protonation requires installed '
                             'cxcalc chemaxon utility.')
    parser.add_argument('-t', '--algorithm', default=1, type=int,
                        help='the number of the search algorithm: 1 - greedy search, 2 - deep clustering, '
                             '3 - clustering, 4 - Pareto front.')
    parser.add_argument('--mw', default=500, type=int,
                        help='maximum ligand weight to pass on the next iteration.')
    parser.add_argument('--ntop', type=int, default=20, required=False,
                        help='the number of the best molecules to select for the next iteration.')
    parser.add_argument('--rmsd', type=float, default=2, required=False,
                        help='maximum allowed RMSD value relative to a parent compound to pass on the next iteration.')
    parser.add_argument('--rotatable_bonds', type=int, default=5, required=False,
                        help='maximum allowed number of rotatable bonds in a compound to pass on the next iteration.')
    parser.add_argument('--tmpdir', metavar='DIRNAME', default=None, type=filepath_type,
                        help='directory where temporary files will be stored. If omitted tmp dir will be created in '
                             'the same location as output DB.')
    # parser.add_argument('--debug', action='store_true', default=False,
    #                     help='enable debug mode; all tmp files will not be erased.')
    parser.add_argument('--hostfile', metavar='FILENAME', required=False, type=str, default=None,
                        help='text file with addresses of nodes of dask SSH cluster. The most typical, it can be '
                             'passed as $PBS_NODEFILE variable from inside a PBS script. The first line in this file '
                             'will be the address of the scheduler running on the standard port 8786. If omitted, '
                             'calculations will run on a single machine as usual.')
    parser.add_argument('-c', '--ncpu', default=1, type=cpu_type,
                        help='number of cpus.')

    args = parser.parse_args()

    if args.hostfile is not None:
        dask.config.set({'distributed.scheduler.allowed-failures': 30})
        dask_client = Client(open(args.hostfile).readline().strip() + ':8786')
        tmpdir = tempfile.mkdtemp()
        try:
            tmparchive = os.path.join(tmpdir, 'archive')
            shutil.make_archive(tmparchive, 'zip',
                                root_dir=os.path.dirname(os.path.abspath(__file__)),
                                base_dir='scripts')
            dask_client.upload_file(tmparchive + '.zip')
        finally:
            shutil.rmtree(tmpdir)

    if args.tmpdir is None:
        tmpdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(args.output)),
                                              ''.join(random.sample(string.ascii_lowercase, 6))))
    else:
        tmpdir = args.tmpdir

    os.makedirs(tmpdir, exist_ok=True)
    iteration = 1

    # depending on input setup operations applied on the first iteration
    # input      make_docking & make_selection   continuation (to avoid protonation on the first step)
    # SMILES                              True          False
    # 3D SDF                             False          False
    # existed DB                          True           True
    try:
        if os.path.isfile(args.output):
            make_docking = True
            continuation = True
            iteration = get_last_iter_from_db(args.output)
            # if iteration is None:
            #     raise FileExistsError("The data was not found in the existing database. Please check the database")
        else:
            continuation = False
            create_db(args.output)
            make_docking = insert_starting_structures_to_db(args.input_frags, args.output)

        with open(os.path.splitext(args.output)[0] + '.json', 'wt') as f:
            json.dump(vars(args), f, sort_keys=True, indent=2)

        while True:
            index_tanimoto = 0.9  # required for alg 2 and 3
            res = make_iteration(dbname=args.output, iteration=iteration, protein_pdbqt=args.protein,
                                 protein_setup=args.protein_setup, ntop=args.ntop, tanimoto=index_tanimoto,
                                 mw=args.mw, rmsd=args.rmsd, rtb=args.rotatable_bonds, alg_type=args.algorithm,
                                 ncpu=args.ncpu, tmpdir=tmpdir, continuation=continuation, make_docking=make_docking,
                                 db_name=args.db, radius=args.radius, min_freq=args.min_freq,
                                 min_atoms=args.min_atoms, max_atoms=args.max_atoms,
                                 max_replacements=args.max_replacements, protonation=not args.no_protonation,
                                 use_dask=args.hostfile is not None)
            make_docking = True
            continuation = False

            if res:
                iteration += 1
                if args.algorithm in [2, 3]:
                    index_tanimoto -= 0.05
            else:
                if iteration == 1:
                    # 0 successful iteration for finally printing
                    iteration = 0
                break

    finally:
        if args.tmpdir is None:
            shutil.rmtree(tmpdir, ignore_errors=True)
        sys.stderr.write(f'{iteration} iterations were completed successfully\n')


if __name__ == '__main__':
    main()
