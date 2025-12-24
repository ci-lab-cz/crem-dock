import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, RDConfig
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import DataStructs
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from sklearn.cluster import AgglomerativeClustering

from cremdock.database import get_mols
from cremdock.auxiliary import sort_two_lists, calc_rtb
from cremdock.crem_grow import grow_mol_crem, grow_mols_crem
from cremdock.molecules import get_mol_ids, neutralize_atoms

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def selection_and_grow_greedy(mols, conn, protein_xyz, max_mw, max_rtb, max_logp, max_tpsa, ntop, ranking_func, ncpu=1,
                              **kwargs):
    """

    :param mols:
    :param conn:
    :param protein_xyz:
    :param max_mw:
    :param max_rtb:
    :param max_logp:
    :param max_tpsa:
    :param ntop:
    :param ranking_func:
    :param ncpu:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols, {parent_mol: [child_mol1, child_mol2, ...], ...}
    """
    if len(mols) == 0:
        return []
    selected_mols = select_top_mols(mols, conn, ntop, ranking_func)
    res = grow_mols_crem(selected_mols, protein_xyz, max_mw=max_mw, max_rtb=max_rtb, max_logp=max_logp, max_tpsa=max_tpsa,
                         ncpu=ncpu, **kwargs)
    return res


def selection_and_grow_clust(mols, conn, nclust, protein_xyz, max_mw, max_rtb, max_logp, max_tpsa, ntop,
                             ranking_func, use_murcko=False, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param nclust:
    :param protein_xyz:
    :param max_mw:
    :param max_rtb:
    :param max_logp:
    :param max_tpsa:
    :param ntop:
    :param ranking_func:
    :param use_murcko:
    :param ncpu:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols, {parent_mol: [child_mol1, child_mol2, ...], ...}
    """
    if len(mols) == 0:
        return []
    clusters = get_clusters(mols, nclust, use_murcko=use_murcko)
    sorted_clusters = sort_clusters(conn, clusters, ranking_func)
    # select top n mols from each cluster
    selected_mols = []
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))  # {mol_id: mol, ...}
    for cluster in sorted_clusters:
        for i in cluster[:ntop]:
            selected_mols.append(mol_dict[i])
    res = grow_mols_crem(selected_mols, protein_xyz, max_mw=max_mw, max_rtb=max_rtb, max_logp=max_logp, max_tpsa=max_tpsa,
                         ncpu=ncpu, **kwargs)
    return res


def selection_and_grow_clust_deep(mols, conn, nclust, protein_xyz, max_mw, max_rtb, max_logp, max_tpsa, ntop,
                                  ranking_func, use_murcko=False, ncpu=1, **kwargs):
    """

    :param mols:
    :param conn:
    :param nclust:
    :param protein_xyz:
    :param max_mw:
    :param max_rtb:
    :param max_logp:
    :param max_tpsa:
    :param ntop:
    :param ranking_func:
    :param use_murcko:
    :param ncpu:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols, {parent_mol: [child_mol1, child_mol2, ...], ...}
    """
    if len(mols) == 0:
        return []
    res = dict()
    clusters = get_clusters(mols, nclust, use_murcko=use_murcko)
    sorted_clusters = sort_clusters(conn, clusters, ranking_func)
    # create dict of named mols
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))  # {mol_id: mol, ...}
    # grow up to N top scored mols from each cluster
    for cluster in sorted_clusters:
        processed_mols = 0
        for mol_id in cluster:
            tmp = grow_mol_crem(mol_dict[mol_id], protein_xyz, max_mw=max_mw, max_rtb=max_rtb, max_logp=max_logp,
                                max_tpsa=max_tpsa, ncpu=ncpu, **kwargs)
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


def selection_and_grow_pareto(mols, conn, max_mw, max_rtb, max_logp, max_tpsa, protein_xyz, ranking_func,
                              pareto_property, ncpu, **kwargs):
    """

    :param mols:
    :param conn:
    :param max_mw:
    :param max_rtb:
    :param max_logp:
    :param max_tpsa:
    :param protein_xyz:
    :param ranking_func:
    :param pareto_property: str value from "mw" or "sa"
    :param ncpu:
    :param kwargs:
    :return: dict of parent mol and lists of corresponding generated mols, {parent_mol: [child_mol1, child_mol2, ...], ...}
    """
    if not mols:
        return []
    mols = [mol for mol in mols if MolWt(mol) <= max_mw - 50 and calc_rtb(mol) <= max_rtb - 1 and
            MolLogP(mol) < max_logp and CalcTPSA(mol) < max_tpsa]
    if not mols:
        return None
    mol_ids = get_mol_ids(mols)
    mol_dict = dict(zip(mol_ids, mols))
    scores = ranking_func(conn, mol_ids)
    pareto_mol_ids = get_pareto_front(mol_dict, scores, parameter=pareto_property, ncpu=ncpu)
    mols = get_mols(conn, pareto_mol_ids)
    res = grow_mols_crem(mols, protein_xyz, max_mw=max_mw, max_rtb=max_rtb, max_logp=max_logp, max_tpsa=max_tpsa, ncpu=ncpu, **kwargs)
    return res


def get_clusters(mols, nclust, use_murcko=False):
    """
    Returns tuple of tuples with mol ids in each cluster
    :param mols: list of molecules
    :param nclust: number of clusters for clustering
    :return:
    """
    clusters = defaultdict(list)
    fps = []
    idx_mols = []
    for mol in mols:
        mol = neutralize_atoms(mol)
        if use_murcko:
            mol = GetScaffoldForMol(mol)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        idx_mols.append(mol.GetProp('_Name'))
    X = np.zeros((len(fps), len(fps)), dtype=float)
    for i in range(len(fps)):
        dist = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:], returnDistance=True)
        X[i, i+1:] = dist
        X[i+1:, i] = dist
    labels = AgglomerativeClustering(n_clusters=nclust,
                                     metric='precomputed',
                                     linkage='average').fit_predict(X).tolist()
    for idx, cluster in zip(idx_mols, labels):
        clusters[cluster].append(idx)
    return tuple(tuple(x) for x in clusters.values())


def select_top_mols(mols, conn, ntop, ranking_func):
    """
    Returns list of ntop molecules with the highest score
    :param mols: list of molecules
    :param conn: connection to docking DB
    :param ntop: number of top scored molecules to select
    :param ranking_func:
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
    :param ranking_func:
    :return: list of lists with mol ids
    """
    scores = ranking_func(conn, [mol_id for cluster in clusters for mol_id in cluster])
    output = []
    for cluster in clusters:
        s, mol_ids = sort_two_lists([scores[mol_id] for mol_id in cluster], cluster, reverse=True)
        output.append(mol_ids)
    return output


def calc_sa_score(mol: Mol, mol_id: str = None) -> tuple[str, float]:
    sa_score = sascorer.calculateScore(mol)
    return mol_id, sa_score


def get_pareto_front(mol_dict: dict[str, Mol],
                     scores: dict[str, float],
                     parameter: str,
                     ncpu: int = None) -> list[Mol]:
    """

    :param mol_dict: dict of Mols
    :param scores: dict of docking scores
    :param parameter: can be "mw" os "sa"
    :param ncpu: number of cpu
    :return:
    """
    mol_dict = {mol_id: neutralize_atoms(mol) for mol_id, mol in mol_dict.items()}
    # -score: needed for inverting X-axis values, so the values are arranged from largest to smallest with a minus sign
    if parameter.lower() == 'mw':
        scores_2 = {mol_id: [-score, MolWt(mol_dict[mol_id])] for mol_id, score in scores.items() if score is not None}
    elif parameter.lower() == 'sa':
        if ncpu is None or ncpu <= 1:
            scores_2 = {mol_id: [-score, sascorer.calculateScore(mol_dict[mol_id])] for mol_id, score in scores.items() if score is not None}
        else:
            scores_2 = dict()
            with Pool(ncpu) as p:
                for mol_id, sa_score in p.starmap(calc_sa_score, [(v, k) for k, v in mol_dict.items()]):
                    scores_2[mol_id] = [-scores[mol_id], sa_score]
    pareto_front_df = pd.DataFrame.from_dict(scores_2, orient='index')
    mol_ids_pareto = identify_pareto(pareto_front_df)
    return mol_ids_pareto
