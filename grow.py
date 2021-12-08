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
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count, Pool

import dask
import numpy as np
import pandas as pd
from crem.crem import grow_mol
from dask.distributed import Client
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans

from scripts import Docking, plif


def cpu_type(x):
    return max(1, min(int(x), cpu_count()))


def filepath_type(x):
    if x:
        return os.path.abspath(x)
    else:
        return x


def str_lower_type(x):
    if x:
        return x.lower()
    else:
        return x


def similarity_value_type(x):
    return max(0, min(1, float(x)))


def sort_two_lists(primary, secondary, reverse=False):
    # sort two lists by order of elements of the primary list
    paired_sorted = sorted(zip(primary, secondary), key=lambda x: x[0], reverse=reverse)
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
    smiles_list = list(cur.execute(f"SELECT smi, id FROM mols WHERE iteration = {iteration - 1} AND "
                                   f"smi_protonated is NULL"))
    if not smiles_list:
        sys.stderr.write(f'no molecules to protonate in iteration {iteration}\n')
        return

    smiles, mol_ids = zip(*smiles_list)

    with tempfile.NamedTemporaryFile(suffix='.smi', mode='w', encoding='utf-8') as tmp:
        fd, output = tempfile.mkstemp()   # use output file to avoid overflow of stdout is extreme cases
        try:
            tmp.writelines(['\n'.join(smiles)])
            tmp.flush()
            cmd_run = f"cxcalc majormicrospecies -H 7.4 -f smiles -M -K '{tmp.name}' > '{output}'"
            subprocess.call(cmd_run, shell=True)
            smiles_protonated = open(output).read().split('\n')
        finally:
            os.remove(output)

    for mol_id, smi_protonated in zip(mol_ids, smiles_protonated):
        cur.execute("""UPDATE mols
                       SET smi_protonated = ?
                       WHERE
                           id = ?
                    """, (Chem.MolToSmiles(Chem.MolFromSmiles(smi_protonated), isomericSmiles=True), mol_id))
    conn.commit()


def update_db(conn, iteration, plif_ref=None, plif_protein_fname=None, ncpu=1):
    """
    Post-process all docked molecules from an individual iteration.
    Calculate rmsd of a molecule to a parent mol. Insert rmsd in output db.
    :param conn: connection to docking DB
    :param iteration: current iteration
    :param plif_ref: list of reference interactions (str)
    :param plif_protein_fname: PDB file with a protein containing all hydrogens to calc plif
    :param ncpu: number of cpu cores
    :return:
    """
    cur = conn.cursor()

    mol_ids = get_docked_mol_ids(conn, iteration)
    mols = get_mols(conn, mol_ids)
    # parent_ids and parent_mols can be empty if all compounds do not have parents
    parent_ids = dict(cur.execute(f"SELECT id, parent_id "
                                  f"FROM mols "
                                  f"WHERE id IN ({','.join('?' * len(mol_ids))}) AND "
                                  f"parent_id IS NOT NULL", mol_ids))
    uniq_parent_ids = list(set(parent_ids.values()))
    parent_mols = get_mols(conn, uniq_parent_ids)
    parent_mols = {m.GetProp('_Name'): m for m in parent_mols}

    # update rmsd
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

    # update plif
    if plif_ref is not None:
        pool = Pool(ncpu)
        # prot = plf.Molecule(Chem.MolFromPDBFile(plif_protein_fname, removeHs=False))   # seems plf.Molecule is unpicklable
        ref_df = pd.DataFrame(data={item: True for item in plif_ref}, index=['reference'])
        for mol_id, sim in pool.imap_unordered(partial(plif.plif_similarity,
                                                       plif_protein_fname=plif_protein_fname,
                                                       plif_ref_df=ref_df),
                                               mols):
            cur.execute("""UPDATE mols
                               SET 
                                   plif_sim = ? 
                               WHERE
                                   id = ?
                            """, (sim, mol_id))
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
    res = tuple(cur.execute(f"SELECT id, rmsd, plif_sim "
                            f"FROM mols "
                            f"WHERE iteration = '{iteration - 1}' AND mol_block IS NOT NULL"))
    df = pd.DataFrame(res, columns=['id', 'rmsd', 'plif_sim']).set_index('id')
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


def get_mol_qeds(conn, mol_ids):
    """
        Return dict of mol_id: qed
        :param conn: connection to docking DB
        :param mol_ids: list of mol ids
        :return:
        """
    cur = conn.cursor()
    sql = f'SELECT id, qed FROM mols WHERE id IN ({",".join("?" * len(mol_ids))})'
    return dict(cur.execute(sql, mol_ids))


def select_top_mols(mols, conn, ntop, ranking_func):
    """
    Returns list of ntop molecules with the highest score
    :param mols: list of molecules
    :param conn: connection to docking DB
    :param ntop: number of top scored molecules to select
    :return:
    """
    mol_ids = get_mol_ids(mols)
    scores = ranking_func(conn, mol_ids)
    scores, mol_ids = sort_two_lists([scores[mol_id] for mol_id in mol_ids], mol_ids, reverse=True)
    mol_ids = set(mol_ids[:ntop])
    mols = [mol for mol in mols if mol.GetProp('_Name') in mol_ids]
    return mols


def sort_clusters(conn, clusters, ranking_func):
    """
    Returns clusters with molecules filtered by properties and reordered according to docking scores
    :param conn: connection to docking DB
    :param clusters: tuple of tuples with mol ids in each cluster
    :return: list of lists with mol ids
    """
    scores = ranking_func(conn, [mol_id for cluster in clusters for mol_id in cluster])
    output = []
    for cluster in clusters:
        s, mol_ids = sort_two_lists([scores[mol_id] for mol_id in cluster], cluster, reverse=True)
        output.append(mol_ids)
    return output


def get_clusters_by_KMeans(mols, nclust):
    """
    Returns tuple of tuples with mol ids in each cluster
    :param mols: list of molecules
    :param tanimoto: tanimoto threshold for clustering
    :return:
    """
    clusters = defaultdict(list)
    fps = []
    idx_mols = []
    for mol in mols:
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)) ### why so few?
        idx_mols.append(mol.GetProp('_Name'))
    X = np.array(fps)
    labels = KMeans(n_clusters=nclust, random_state=0).fit_predict(X).tolist()
    for idx, cluster in zip(idx_mols, labels):
        clusters[cluster].append(idx)
    return tuple(tuple(x) for x in clusters.values())


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


def __grow_mol(mol, protein_xyz, max_mw, max_rtb, max_logp, h_dist_threshold=2, ncpu=1, **kwargs):

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

    mw = max_mw - Chem.Descriptors.MolWt(mol)
    if mw <= 0:
        return []
    rtb = max_rtb - CalcNumRotatableBonds(mol) - 1 # it is necessary to take into account the formation of bonds during the growth of the molecule
    if rtb == -1:
        rtb = 0
    logp = max_logp - MolLogP(mol) + 0.5

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
        res = list(grow_mol(mol, protected_ids=protected_ids, return_rxn=False, return_mol=True, ncores=ncpu,
                            symmetry_fixes=True, mw=(1, mw), rtb=(0, rtb), logp=(-100, logp), **kwargs))
    except Exception:
        error_message = traceback.format_exc()
        sys.stderr.write(f'Grow error.\n'
                         f'{error_message}\n'
                         f'{mol.GetProp("_Name")}\n'
                         f'{Chem.MolToSmiles(mol)}\n')
        res = []

    res = tuple(m for smi, m in res)

    return res


def __grow_mols(mols, protein_pdbqt, max_mw, max_rtb, max_logp, h_dist_threshold=2, ncpu=1, **kwargs):
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
        tmp = __grow_mol(mol, protein_xyz, max_mw=max_mw, max_rtb=max_rtb, max_logp=max_logp,
                         h_dist_threshold=h_dist_threshold, ncpu=ncpu, **kwargs)
        if tmp:
            res[mol] = tmp
    return res


def insert_db(conn, data, cols=None):
    if data:
        cur = conn.cursor()
        ncols = len(data[0])
        if cols is None:
            cur.executemany(f"INSERT OR IGNORE INTO mols VAlUES({','.join('?' * ncols)})", data)
        else:
            cols = ', '.join(cols)
            cur.executemany(f"INSERT OR IGNORE INTO mols ({cols}) VAlUES({','.join('?' * ncols)})", data)
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
             smi TEXT NOT NULL UNIQUE,
             smi_protonated TEXT,
             parent_id TEXT,
             docking_score REAL,
             mw REAL,
             rtb INTEGER,
             logp REAL,
             qed REAL,
             rmsd REAL,
             plif_sim REAL,
             pdb_block TEXT,
             mol_block TEXT,
             protected_user_canon_ids TEXT DEFAULT NULL,
             time TEXT
            )""")
    conn.commit()
    conn.close()


def insert_starting_structures_to_db(fname, db_fname, prefix):
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
                    data.append((f'{prefix}-{name}' if prefix else name,
                                 0,
                                 smi,
                                 mol_mw,
                                 mol_rtb,
                                 mol_logp,
                                 mol_qed))
            cols = ['id', 'iteration', 'smi', 'mw', 'rtb', 'logp', 'qed']
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
                    data.append((f'{prefix}-{name}' if prefix else name,
                                 0,
                                 Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True),
                                 mol_mw,
                                 mol_rtb,
                                 mol_logp,
                                 mol_qed,
                                 Chem.MolToMolBlock(mol),
                                 protected_user_canon_ids))
            cols = ['id', 'iteration', 'smi', 'mw', 'rtb', 'logp', 'qed', 'mol_block', 'protected_user_canon_ids']
        else:
            raise ValueError('input file with fragments has unrecognizable extension. '
                             'Only SMI, SMILES and SDF are allowed.')
        insert_db(conn, data, cols)
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


def selection_grow_greedy(mols, conn, protein_pdbqt, max_mw, max_rtb, max_logp, ntop, ranking_func, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param protein_pdbqt:
    :param protonation:
    :param ntop:
    :param max_mw:
    :param max_rtb:
    :param max_logp:
    :param ncpu:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols
    """
    if len(mols) == 0:
        return []
    selected_mols = select_top_mols(mols, conn, ntop, ranking_func)
    res = __grow_mols(selected_mols, protein_pdbqt, max_mw=max_mw, max_rtb=max_rtb, max_logp=max_logp, ncpu=ncpu, **kwargs)
    return res


def selection_grow_clust(mols, conn, nclust, protein_pdbqt, max_mw, max_rtb, max_logp, ntop, ranking_func, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param tanimoto:
    :param protein_pdbqt:
    :param protonation:
    :param ntop:
    :param max_mw:
    :param max_rtb:
    :param max_logp:
    :param ncpu:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols
    """
    if len(mols) == 0:
        return []
    clusters = get_clusters_by_KMeans(mols, nclust)
    sorted_clusters = sort_clusters(conn, clusters, ranking_func)
    # select top n mols from each cluster
    selected_mols = []
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))  # {mol_id: mol, ...}
    for cluster in sorted_clusters:
        for i in cluster[:ntop]:
            selected_mols.append(mol_dict[i])
    res = __grow_mols(selected_mols, protein_pdbqt, max_mw=max_mw, max_rtb=max_rtb, max_logp=max_logp, ncpu=ncpu, **kwargs)
    return res


def selection_grow_clust_deep(mols, conn, nclust, protein_pdbqt, ntop, max_mw, max_rtb, max_logp, ranking_func, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param tanimoto:
    :param protein_pdbqt:
    :param protonation:
    :param ntop:
    :param max_mw:
    :param max_rtb:
    :param max_logp:
    :param ncpu:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols
    """
    if len(mols) == 0:
        return []
    res = dict()
    clusters = get_clusters_by_KMeans(mols, nclust)
    sorted_clusters = sort_clusters(conn, clusters, ranking_func)
    # create dict of named mols
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))  # {mol_id: mol, ...}
    protein_xyz = get_protein_heavy_atom_xyz(protein_pdbqt)
    # grow up to N top scored mols from each cluster
    for cluster in sorted_clusters:
        processed_mols = 0
        for mol_id in cluster:
            tmp = __grow_mol(mol_dict[mol_id], protein_xyz, max_mw=max_mw, max_rtb=max_rtb, max_logp=max_logp,
                             ncpu=ncpu, **kwargs)
            if tmp:
                res[mol_dict[mol_id]] = tmp
                processed_mols += 1
            if processed_mols == ntop:
                break
    return res


def identify_pareto(df):
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
    return population_ids[pareto_front].tolist()


def selection_by_pareto(mols, conn, mw, rtb, logp, protein_pdbqt, ranking_func, ncpu, **kwargs):
    """

    :param mols:
    :param conn:
    :param mw:
    :param rtb:
    :param logp:
    :param protein_pdbqt:
    :param protonation:
    :param ncpu:
    :param tmpdir:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols
    """
    if not mols:
        return []
    mols = [mol for mol in mols if MolWt(mol) <= mw - 50 and CalcNumRotatableBonds(mol) <= rtb - 1 and MolLogP(mol) < logp]
    if not mols:
        return None
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))
    scores = ranking_func(conn, mol_ids)
    # needed for inverting X-axis values, so the values are arranged from largest to smallest with a minus sign
    scores = {i: j * (-1) for i, j in scores.items()}
    scores_mw = {mol_id: [score, MolWt(mol_dict[mol_id])] for mol_id, score in scores.items() if score is not None}
    pareto_front_df = pd.DataFrame.from_dict(scores_mw, orient='index')
    mols_pareto = identify_pareto(pareto_front_df)
    mols = get_mols(conn, mols_pareto)
    res = __grow_mols(mols, protein_pdbqt, max_mw=mw, max_rtb=rtb, max_logp=logp, ncpu=ncpu, **kwargs)
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
    rtb = CalcNumRotatableBonds(Chem.RemoveHs(mol)) # does not count things like amide or ester bonds
    logp = round(MolLogP(mol), 2)
    qed = round(QED.qed(mol), 3)
    return mw, rtb, logp, qed


def prep_data_for_insert(parent_mol, mol, n, iteration, rtb, mw, logp, prefix):
    """

    :param parent_mol:
    :param mol:
    :param n: sequential number
    :param iteration: iteration number
    :param rtb: maximum allowed number of RTB
    :param mw: maximum allowed MW
    :return:
    """
    data = []
    mol_mw, mol_rtb, mol_logp, mol_qed = calc_properties(mol)
    if mol_mw <= mw and mol_rtb <= rtb and mol_logp <= logp:
        isomers = get_isomers(mol)
        for i, m in enumerate(isomers):
            m = Chem.AddHs(m)
            mol_id = str(iteration).zfill(3) + '-' + str(n).zfill(6) + '-' + str(i).zfill(2)
            if prefix:
                mol_id = f'{prefix}-{mol_id}'
            # save canonical protected atom ids because we store mols as SMILES and lost original atom enumeraion
            child_protected_canon_user_id = None
            if parent_mol.HasProp('protected_user_canon_ids'):
                parent_protected_user_ids = get_atom_idxs_for_canon(parent_mol, list(
                    map(int, parent_mol.GetProp('protected_user_canon_ids').split(','))))
                child_protected_user_id = get_child_protected_atom_ids(m, parent_protected_user_ids)
                child_protected_canon_user_id = ','.join(map(str, get_canon_for_atom_idx(m, child_protected_user_id)))

            data.append((mol_id, iteration, Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True), None,
                         parent_mol.GetProp('_Name'), None, mol_mw, mol_rtb, mol_logp, mol_qed, None, None,
                         None, None, child_protected_canon_user_id, None))
    return data


def supply_parent_child_mols(d):
    # d - {parent_mol: [child_mol1, child_mol2, ...], ...}
    n = 0
    for parent_mol, child_mols in d.items():
        for child_mol in child_mols:
            yield parent_mol, child_mol, n
            n += 1


def ranking_by_docking_score(conn, mol_ids):
    """

    :param conn:
    :param mol_ids:
    :return: {mol_id: score}
    """
    scores = get_mol_scores(conn, mol_ids)
    scores = {i: j * (-1) for i, j in scores.items()}
    return scores


def ranking_by_docking_score_qed(conn, mol_ids):
    """

    :param conn:
    :param mol_ids:
    :return: dict {mol_id: score}
    """
    scores = get_mol_scores(conn, mol_ids)
    qeds = get_mol_qeds(conn, mol_ids)
    scores = {i: j * (-1) for i, j in scores.items()}
    min_score, max_score = min(list(scores.values())), max(list(scores.values()))
    stat_scores = {mol_id: ((scores[mol_id] - min_score) / (max_score - min_score) * qeds[mol_id]) for mol_id in
                   mol_ids}
    return stat_scores


def make_iteration(dbname, iteration, protein_pdbqt, protein_setup, ntop, nclust, mw, rmsd, rtb, logp, alg_type,
                   ncpu, protonation, make_docking=True, use_dask=False, plif_list=None, plif_protein=None,
                   plif_cutoff=1, prefix=None, ranking='1', **kwargs):

    ranking_type = {'1': ranking_by_docking_score, '2': ranking_by_docking_score_qed}
    ranking_func = ranking_type[ranking]

    sys.stderr.write(f'iteration {iteration} started\n')
    conn = sqlite3.connect(dbname)
    if protonation:
        add_protonation(conn, iteration)
    if make_docking:
        Docking.iter_docking(dbname=dbname, receptor_pdbqt_fname=protein_pdbqt, protein_setup=protein_setup,
                             protonation=protonation, iteration=iteration, use_dask=use_dask, ncpu=ncpu)
        update_db(conn, iteration, plif_ref=plif_list, plif_protein_fname=plif_protein, ncpu=ncpu)

        res = []
        mol_data = get_docked_mol_data(conn, iteration)
        # mol_data = mol_data.loc[(mol_data['mw'] <= mw) & (mol_data['rtb'] <= rtb)]  # filter by MW and RTB
        if iteration != 1:
            mol_data = mol_data.loc[mol_data['rmsd'] <= rmsd]  # filter by RMSD
        if plif_list and len(mol_data.index) > 0:
            mol_data = mol_data.loc[mol_data['plif_sim'] >= plif_cutoff]  # filter by PLIF
        if len(mol_data.index) == 0:
            sys.stderr.write(f'iteration {iteration}: no molecules were selected for growing.\n')
        else:
            mols = get_mols(conn, mol_data.index)
            if alg_type == 1:
                res = selection_grow_greedy(mols=mols, conn=conn, protein_pdbqt=protein_pdbqt,
                                            ntop=ntop, max_mw=mw, max_rtb=rtb, max_logp=logp, ranking_func=ranking_func,
                                            ncpu=ncpu, **kwargs)
            elif alg_type in [2, 3] and len(mols) <= nclust:    # if number of mols is lower than nclust grow all mols
                res = __grow_mols(mols=mols, protein_pdbqt=protein_pdbqt, max_mw=mw, max_rtb=rtb, max_logp=logp,
                                  ncpu=ncpu, **kwargs)
            elif alg_type == 2:
                res = selection_grow_clust_deep(mols=mols, conn=conn, nclust=nclust, protein_pdbqt=protein_pdbqt,
                                                ntop=ntop, max_mw=mw, max_rtb=rtb, max_logp=logp, ranking_func=ranking_func,
                                                ncpu=ncpu, **kwargs)
            elif alg_type == 3:
                res = selection_grow_clust(mols=mols, conn=conn, nclust=nclust, protein_pdbqt=protein_pdbqt,
                                           ntop=ntop, max_mw=mw, max_rtb=rtb, max_logp=logp, ranking_func=ranking_func,
                                           ncpu=ncpu, **kwargs)
            elif alg_type == 4:
                res = selection_by_pareto(mols=mols, conn=conn, mw=mw, rtb=rtb, logp=logp, protein_pdbqt=protein_pdbqt,
                                          ranking_func=ranking_func, ncpu=ncpu, **kwargs)

    else:
        mols = get_mols(conn, get_docked_mol_ids(conn, iteration))
        res = __grow_mols(mols=mols, protein_pdbqt=protein_pdbqt, max_mw=mw, max_rtb=rtb, max_logp=logp, ncpu=ncpu, **kwargs)

    if res:
        data = []
        p = Pool(ncpu)
        for d in p.starmap(partial(prep_data_for_insert, iteration=iteration, rtb=rtb, mw=mw, logp=logp, prefix=prefix),
                           supply_parent_child_mols(res)):
            data.extend(d)
        p.close()
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
                        help='the number of the search algorithm: 1 - greedy search, 2 - deep clustering (if some '
                             'molecules from a cluster cannot be grown they will be replaced with new lower scored '
                             'ones), 3 - clustering, 4 - Pareto front (MW vs. docking score).')
    parser.add_argument('--ntop', type=int, default=20, required=False,
                        help='the number of the best molecules to select for the next iteration in the case of greedy '
                             'search (algorithm 1) or the number of molecules from each cluster in the case of '
                             'clustering (algorithms 2 and 3).')
    parser.add_argument('--nclust', type=int, default=20, required=False,
                        help='the number of KMeans clusters to consider for molecule selection.')
    parser.add_argument('--rmsd', type=float, default=2, required=False,
                        help='maximum allowed RMSD value relative to a parent compound to pass on the next iteration.')
    parser.add_argument('--mw', default=450, type=float,
                        help='maximum ligand molecular weight to pass on the next iteration.')
    parser.add_argument('--rtb', type=int, default=5, required=False,
                        help='maximum allowed number of rotatable bonds in a compound.')
    parser.add_argument('--logp', type=float, default=4, required=False,
                        help='maximum allowed logP of a compound.')
    parser.add_argument('--plif', default=None, required=False, nargs='*', type=str_lower_type,
                        help='list of protein-ligand interactions compatible with ProLIF. Dot-separated names of each '
                             'interaction which should be observed for a ligand to pass to the next iteration. Derive '
                             'these names from a reference ligand. Example: ASP115.HBDonor or ARG.A.Hydrophobic.')
    parser.add_argument('--plif_protein', metavar='protein.pdb', default=None, required=False, type=filepath_type,
                        help='PDB file with the same protein as for docking but containing all hydrogens. Required to '
                             'identify protein-ligand interaction fingerprints.')
    parser.add_argument('--plif_cutoff', metavar='NUMERIC', default=1, required=False, type=similarity_value_type,
                        help='cutoff of Tversky similarity, value between 0 and 1.')
    parser.add_argument('--tmpdir', metavar='DIRNAME', default=None, type=filepath_type,
                        help='directory where temporary files will be stored. If omitted tmp dir will be created in '
                             'the same location as output DB.')
    parser.add_argument('--hostfile', metavar='FILENAME', required=False, type=str, default=None,
                        help='text file with addresses of nodes of dask SSH cluster. The most typical, it can be '
                             'passed as $PBS_NODEFILE variable from inside a PBS script. The first line in this file '
                             'will be the address of the scheduler running on the standard port 8786. If omitted, '
                             'calculations will run on a single machine as usual.')
    parser.add_argument('--prefix', metavar='STRING', required=False, type=str, default=None,
                        help='prefix which will be added to all names. This might be useful if multiple runs are made '
                             'which will be analyzed together.')
    parser.add_argument('--ranking', choices=['1', '2'], required=True,
                        help='the number of the algorithm for ranking molecules: 1 - ranking based on docking scores, '
                             '2 - ranking based on docking scores and QED.')
    parser.add_argument('-c', '--ncpu', default=1, type=cpu_type,
                        help='number of cpus.')


    args = parser.parse_args()


    if args.algorithm in [2, 3] and (args.nclust * args.ntop > 20):
        sys.stderr.write('The number of clusters (nclust) and top scored molecules selected from each cluster (ntop) '
                         'will result in selection on each iteration more than 20 molecules that may slower '
                         'computations.\n')
        sys.stderr.flush()

    if args.plif is not None and (args.plif_protein is None or not os.path.isfile(args.plif_protein)):
        raise FileNotFoundError('PLIF pattern was specified but the protein file is missing or was not supplied. '
                                'Calculation was aborted.')

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
    # input      make_docking & make_selection
    # SMILES                              True
    # 3D SDF                             False
    # existed DB                          True
    try:
        if os.path.isfile(args.output):
            make_docking = True
            iteration = get_last_iter_from_db(args.output)
            if iteration is None:
                raise IOError("The last iteration could not be retrieved from the database. Please check it.")
        else:
            create_db(args.output)
            make_docking = insert_starting_structures_to_db(args.input_frags, args.output, args.prefix)

        with open(os.path.splitext(args.output)[0] + '.json', 'wt') as f:
            json.dump(vars(args), f, sort_keys=True, indent=2)

        while True:
            res = make_iteration(dbname=args.output, iteration=iteration, protein_pdbqt=args.protein,
                                 protein_setup=args.protein_setup, ntop=args.ntop, nclust=args.nclust,
                                 mw=args.mw, rmsd=args.rmsd, rtb=args.rtb, logp=args.logp, alg_type=args.algorithm,
                                 ncpu=args.ncpu, make_docking=make_docking, db_name=args.db, radius=args.radius,
                                 min_freq=args.min_freq, min_atoms=args.min_atoms, max_atoms=args.max_atoms,
                                 max_replacements=args.max_replacements, protonation=not args.no_protonation,
                                 use_dask=args.hostfile is not None, plif_list=args.plif,
                                 plif_protein=args.plif_protein, plif_cutoff=args.plif_cutoff, prefix=args.prefix,
                                 ranking=args.ranking)
            make_docking = True

            if res:
                iteration += 1
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
