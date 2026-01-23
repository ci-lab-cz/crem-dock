import numpy as np

def sample_score(row_ids, cur, radius, n, degree):
    """
    Performs random selection of fragments proportionally to a docking score (Vina).
    :param row_ids: the list of row ids of fragments to consider
    :param cur: cursor to the fragment database
    :param radius: context radius
    :param n: the number of fragments to select
    :param degree:
    :return: the list of row ids of selected fragments
    """
    if n >= len(row_ids):
        return row_ids

    sql = f"""SELECT rowid, vina_score
                  FROM radius{radius}
                  WHERE rowid IN ({','.join(map(str, row_ids))})"""
    cur.execute(sql)
    row_score = cur.fetchall()
    x_zeroed = [(i, v-0.01 if v > 0 else v) for i, v in row_score]
    values = np.array([v for _, v in x_zeroed])
    row_ids_ = [_ for _, v in x_zeroed]
    p = values / values.sum()
    if degree:
        p = p ** degree
        p = p / p.sum()
    selected_rowids = np.random.choice(row_ids_, n, replace=False, p=p).tolist()
    return selected_rowids


def sample_score_exp(row_ids, cur, radius, n):
    """
    Performs random selection of fragments proportionally to the exponential of their docking score (Vina).
    :param row_ids: the list of row ids of fragments to consider
    :param cur: cursor to the fragment database
    :param radius: context radius
    :param n: the number of fragments to select
    :return: the list of row ids of selected fragments
    """
    if n >= len(row_ids):
        return row_ids

    sql = f"""SELECT rowid, vina_score
                  FROM radius{radius}
                  WHERE rowid IN ({','.join(map(str, row_ids))})"""
    cur.execute(sql)
    row_score = cur.fetchall()
    x_zeroed = [(i, v-0.01 if v > 0 else v) for i, v in row_score]
    values = np.array([v for _, v in x_zeroed])
    row_ids_ = [_ for _, v in x_zeroed]
    alpha = 1
    p = np.exp(alpha * values) / np.exp(alpha * values).sum()
    selected_rowids = np.random.choice(row_ids_, n, replace=False, p=p).tolist()
    return selected_rowids


def sample_HAC(row_ids, cur, radius, n):
    """
    Performs random selection of fragments proportionally to the number of heavy atoms.
    :param row_ids: the list of row ids of fragments to consider
    :param cur: cursor to the fragment database
    :param radius: context radius
    :param n: the number of fragments to select
    :return: the list of row ids of selected fragments
    """
    if n >= len(row_ids):
        return row_ids

    sql = f"""SELECT rowid, core_num_atoms
                  FROM radius{radius}
                  WHERE rowid IN ({','.join(map(str, row_ids))})"""
    cur.execute(sql)
    row_hac = cur.fetchall()
    weights = np.array([v for _, v in row_hac])
    row_ids_ = [_ for _, v in row_hac]
    p = weights / weights.sum()
    selected_rowids = np.random.choice(row_ids_, n, replace=False, p=p).tolist()
    return selected_rowids





