import logging
import yaml

from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.rdMolDescriptors import CalcFractionCSP3

from cremdock.database import get_mols, get_mol_qeds, get_mol_scores


"""
All score* functions should return a dict with mol_id and a score (the higher the better) for further 
ranking and selection
"""


def check_score_order(conn):
    """
    Return True or False for further inverting mol docking scores.
    :param conn:
    :return:
    """
    programs = {
        'vina': {
            'scoring': True,
            },
        'gnina': {
            'scoring': {
                'vinardo': True,
                'default': False,
                #'ad4_scoring': False,
                #'dkoes_fast': False,
                #'dkoes_scoring': False,
                #'dkoes_scoring_old': False,
                'vina': True
            }
        }
    }
    cur = conn.cursor()
    program_from_db = yaml.safe_load(cur.execute("SELECT yaml FROM setup").fetchone()[0])['program']
    if program_from_db not in programs:
        raise KeyError(f"Program '{program_from_db}' not found in the dictionary.")
    if program_from_db == 'vina':
        return programs[program_from_db]['scoring']

    scoring_from_db = yaml.safe_load(cur.execute("SELECT config FROM setup").fetchone()[0])['scoring']
    if scoring_from_db not in programs[program_from_db]['scoring']:
        raise KeyError(f"Scoring '{scoring_from_db}' not found for program '{program_from_db}'.")

    return programs[program_from_db]['scoring'][scoring_from_db]


def get_inverted_mol_scores(conn, mol_ids):
    """
    Returns dict of mol_id: score, where docking scores are multiplied by -1 (since all implemented docking methods
    return negative scores for the best molecules). This is necessary for further ranking
    :param conn:
    :param mol_ids:
    :return:
    """
    scores = get_mol_scores(conn, mol_ids)
    if check_score_order(conn):
        scores = {i: j * (-1) for i, j in scores.items()}
    return scores


def score_by_docking_score(conn, mol_ids):
    """
    invert docking scores of molecules
    :param conn:
    :param mol_ids:
    :return: {mol_id: score}
    """
    scores = get_inverted_mol_scores(conn, mol_ids)
    return scores


def score_by_docking_score_qed(conn, mol_ids):
    """
    scoring for molecule is calculated by the formula: docking score after scaling * QED
    :param conn:
    :param mol_ids:
    :return: dict {mol_id: score}
    """
    scores = get_inverted_mol_scores(conn, mol_ids)
    qeds = get_mol_qeds(conn, mol_ids)
    scale_scores = scale_min_max(scores)
    stat_scores = {mol_id: scale_scores[mol_id] * qeds[mol_id] for mol_id in scale_scores.keys()}
    return stat_scores


def score_by_fcsp3_bm(conn, mol_ids):
    """
    scoring is calculated by the formula: docking score after scaling * FCsp3_BM after scaling at 0.3
    :param conn:
    :param mol_ids:
    :return:
    """
    scores = get_inverted_mol_scores(conn, mol_ids)
    scale_scores = scale_min_max(scores)
    mol_dict = dict(zip(mol_ids, get_mols(conn, mol_ids)))
    fcsp3_bm = {mol_id: CalcFractionCSP3(GetScaffoldForMol(m)) for mol_id, m in mol_dict.items()}
    fcsp3_scale = {mol_id: fcsp3 / 0.3 if fcsp3 <= 0.3 else 1 for mol_id, fcsp3 in fcsp3_bm.items()}
    stat_scores = {mol_id: (scale_scores[mol_id] * fcsp3_scale[mol_id]) for mol_id in mol_ids}
    return stat_scores


def score_by_num_heavy_atoms(conn, mol_ids):
    """
    scoring for molecule is calculated by the formula: docking score / number heavy atoms
    :param conn:
    :param mol_ids:
    :return: dict {mol_id: score}
    """
    scores = get_inverted_mol_scores(conn, mol_ids)
    mol_dict = dict(zip(mol_ids, get_mols(conn, mol_ids)))
    stat_scores = {mol_id: scores[mol_id] / mol_dict[mol_id].GetNumHeavyAtoms() for mol_id in mol_ids}
    return stat_scores


def score_by_num_heavy_atoms_fcsp3_bm(conn, mol_ids):
    """
    scoring is calculated by the formula: docking score / number heavy atoms * FCsp3_BM after scaling at 0.3
    :param conn:
    :param mol_ids:
    :return:
    """
    scores = score_by_num_heavy_atoms(conn, mol_ids)
    scale_scores = scale_min_max(scores)
    mol_dict = dict(zip(mol_ids, get_mols(conn, mol_ids)))
    fcsp3_bm = {mol_id: CalcFractionCSP3(GetScaffoldForMol(m)) for mol_id, m in mol_dict.items()}
    fcsp3_scale = {mol_id: fcsp3 / 0.3 if fcsp3 <= 0.3 else 1 for mol_id, fcsp3 in fcsp3_bm.items()}
    stat_scores = {mol_id: (scale_scores[mol_id] * fcsp3_scale[mol_id]) for mol_id in mol_ids}
    return stat_scores


def score_by_fcsp3_bm_squared(conn, mol_ids):
    """
    scoring is calculated by the formula: docking score after scaling * FCsp3_BMc ** 2 after scaling at 0.3
    :param conn:
    :param mol_ids:
    :return:
    """
    scores = get_inverted_mol_scores(conn, mol_ids)
    scale_scores = scale_min_max(scores)
    mol_dict = dict(zip(mol_ids, get_mols(conn, mol_ids)))
    fcsp3_bm = {mol_id: CalcFractionCSP3(GetScaffoldForMol(m)) for mol_id, m in mol_dict.items()}
    fcsp3_scale = {mol_id: fcsp3 / 0.3 if fcsp3 <= 0.3 else 1 for mol_id, fcsp3 in fcsp3_bm.items()}
    stat_scores = {mol_id: (scale_scores[mol_id] * fcsp3_scale[mol_id] ** 2) for mol_id in mol_ids}
    return stat_scores


def score_by_num_heavy_atoms_qed(conn, mol_ids):
    """
    scoring is calculated by the formula: docking score / number heavy atoms * QED
    :param conn:
    :param mol_ids:
    :return: dict {mol_id: score}
    """
    qeds = get_mol_qeds(conn, mol_ids)
    scores = score_by_num_heavy_atoms(conn, mol_ids)
    scale_scores = scale_min_max(scores)
    stat_scores = {mol_id: (scale_scores[mol_id] * qeds[mol_id]) for mol_id in mol_ids}
    return stat_scores


def ranking_score(x):
    ranking_types = {1: score_by_docking_score,
                     2: score_by_docking_score_qed,
                     3: score_by_num_heavy_atoms,
                     4: score_by_num_heavy_atoms_qed,
                     5: score_by_fcsp3_bm,
                     6: score_by_num_heavy_atoms_fcsp3_bm,
                     7: score_by_fcsp3_bm_squared}
    try:
        return ranking_types[x]
    except KeyError:
        logging.error(f'Wrong type of a ranking function was passed: {x}. Should be within 1-7.')
        raise


def scale_min_max(scores):
    """
    translation of values into a range from 0 to 1
    :param scores:
    :return:
    """
    min_score, max_score = min(scores.values()), max(scores.values())
    scale_scores = {mol_id: (scores[mol_id] - min_score) / (max_score - min_score) for mol_id in scores.keys()}
    return scale_scores


