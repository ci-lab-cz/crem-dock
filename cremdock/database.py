import sqlite3
from functools import partial
from multiprocessing import Pool

import pandas as pd
from easydock import database as eadb
from easydock.auxiliary import mol_name_split
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcTPSA

from cremdock.auxiliary import calc_rtb
from cremdock.crem_grow import get_protein_heavy_atoms_xyz_from_string
from cremdock.molecules import get_isomers, get_rmsd
from cremdock.user_protected_atoms import get_canon_for_atom_idx, get_protected_canon_ids
from cremdock.scripts import plif


def create_db(fname, args, args_to_save):
    """
    Creates a DB using the corresponding function from moldock and adds some new columns and a table to it
    :param fname: file name of output DB
    :param args: argparse namespace
    :param args_to_save: list of arg names which values are file names which content should be stored as separate
                         fields in setup table
    :return:
    """
    eadb.create_db(fname, args, args_to_save, ('protein', 'protein_setup'), unique_smi=True)
    eadb.populate_setup_db(fname, args, args_to_save, ('protein', 'protein_setup'))
    with sqlite3.connect(fname, timeout=90) as conn:
        cur = conn.cursor()
        # cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("ALTER TABLE mols ADD iteration INTEGER")
        cur.execute("ALTER TABLE mols ADD parent_id TEXT")
        cur.execute("ALTER TABLE mols ADD mw REAL")
        cur.execute("ALTER TABLE mols ADD rtb INTEGER")
        cur.execute("ALTER TABLE mols ADD logp REAL")
        cur.execute("ALTER TABLE mols ADD tpsa REAL")
        cur.execute("ALTER TABLE mols ADD qed REAL")
        cur.execute("ALTER TABLE mols ADD rmsd REAL")
        cur.execute("ALTER TABLE mols ADD plif_sim REAL")
        cur.execute("ALTER TABLE mols ADD protected_user_canon_ids TEXT DEFAULT NULL")
        conn.commit()


def insert_starting_structures_to_db(fname, db_fname, prefix):
    """

    :param fname: SMILES or SDF with 3D coordinates
    :param db_fname: output DB
    :param prefix: string which will be added to all names
    :return:
    """
    data = []
    make_docking = True
    if fname.lower().endswith('.smi') or fname.lower().endswith('.smiles'):
        with open(fname) as f:
            for i, line in enumerate(f):
                tmp = line.strip().split()
                smi = Chem.CanonSmiles(tmp[0])
                name = tmp[1] if len(tmp) > 1 else '000-' + str(i).zfill(6)
                mol_mw, mol_rtb, mol_logp, mol_qed, mol_tpsa = calc_properties(Chem.MolFromSmiles(smi))
                data.append((f'{prefix}-{name}' if prefix else name,
                             0,
                             smi,
                             mol_mw,
                             mol_rtb,
                             mol_logp,
                             mol_qed,
                             mol_tpsa))
        cols = ['id', 'iteration', 'smi', 'mw', 'rtb', 'logp', 'qed', 'tpsa']
    elif fname.lower().endswith('.sdf'):
        make_docking = False
        for i, mol in enumerate(Chem.SDMolSupplier(fname, removeHs=False)):
            if mol:
                name = mol.GetProp('_Name')
                if not name:
                    name = '000-' + str(i).zfill(6)
                mol.SetProp('_Name', name + '_0')
                mol = Chem.AddHs(mol, addCoords=True)
                protected_user_canon_ids = None
                if mol.HasProp('protected_user_ids'):
                    # rdkit numeration starts with 0 and sdf numeration starts with 1
                    protected_user_ids = [int(idx) - 1 for idx in mol.GetProp('protected_user_ids').split(',')]
                    protected_user_canon_ids = ','.join(map(str, get_canon_for_atom_idx(mol, protected_user_ids)))
                mol_mw, mol_rtb, mol_logp, mol_qed, mol_tpsa = calc_properties(mol)
                data.append((f'{prefix}-{name}' if prefix else name,
                             0,
                             Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True),
                             mol_mw,
                             mol_rtb,
                             mol_logp,
                             mol_qed,
                             mol_tpsa,
                             Chem.MolToMolBlock(mol),
                             protected_user_canon_ids))
        cols = ['id', 'iteration', 'smi', 'mw', 'rtb', 'logp', 'qed', 'tpsa', 'mol_block', 'protected_user_canon_ids']
    else:
        raise ValueError('input file with fragments has unrecognizable extension. '
                         'Only SMI, SMILES and SDF are allowed.')
    eadb.insert_db(db_fname, data=data, cols=cols)
    return make_docking


def update_db(conn, iteration, plif_ref=None, plif_protein_fname=None, ncpu=1):
    """
    Post-process all docked molecules from an individual iteration.
    Calculate rmsd of a molecule to a parent mol. Insert rmsd in output db.
    :param conn: connection to docking DB
    :param plif_ref: list of reference interactions (str)
    :param plif_protein_fname: PDB file with a protein containing all hydrogens to calc plif
    :param ncpu: number of cpu cores
    :return:
    """
    cur = conn.cursor()
    mol_ids = get_docked_mol_ids(conn, iteration)
    mols = get_mols(conn, mol_ids)
    # parent_ids and parent_mols can be empty if all compounds do not have parents
    parent_ids = dict(eadb.select_from_db(cur,
                                          "SELECT id, parent_id FROM mols WHERE id IN (?) AND parent_id IS NOT NULL",
                                          mol_ids))
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
        try:
            ref_df = pd.DataFrame(data={item: True for item in plif_ref}, index=['reference'])
            for mol_id, sim in pool.imap_unordered(partial(plif.plif_similarity,
                                                           plif_protein_fname=plif_protein_fname,
                                                           plif_ref_df=ref_df),
                                                   mols):
                cur.execute(f"""UPDATE mols
                                   SET 
                                       plif_sim = ? 
                                   WHERE
                                       id = ?
                                """, (sim, mol_id))
            conn.commit()
        finally:
            pool.close()
            pool.join()


def calc_properties(mol):
    mw = round(MolWt(mol), 2)
    rtb = calc_rtb(mol)
    logp = round(MolLogP(mol), 2)
    qed = round(QED.qed(mol), 3)
    tpsa = round(CalcTPSA(mol), 2)
    return mw, rtb, logp, qed, tpsa


def prep_data_for_insert(parent_mol, mol, n, iteration, max_rtb, max_mw, max_logp, max_tpsa, prefix):
    """

    :param parent_mol:
    :param mol:
    :param n: sequential number
    :param iteration: iteration number
    :param max_rtb: maximum allowed number of RTB
    :param max_mw: maximum allowed MW
    :param max_logp: maximum allowed logP
    :param max_tpsa: maximum allowed TPSA
    :param prefix: string which will be added to all names
    :return:
    """
    data = []
    mol_mw, mol_rtb, mol_logp, mol_qed, mol_tpsa = calc_properties(mol)
    if mol_mw <= max_mw and mol_rtb <= max_rtb and mol_logp <= max_logp and mol_tpsa <= max_tpsa:
        isomers = get_isomers(mol)
        for i, m in enumerate(isomers):
            mol_id = str(iteration).zfill(3) + '-' + str(n).zfill(6) + '-' + str(i).zfill(2)
            if prefix:
                mol_id = f'{prefix}-{mol_id}'
            # save canonical protected atom ids because we store mols as SMILES and lost original atom enumeration
            # canonical ids are computed for fully hydrogenized molecules
            child_protected_canon_user_id = None
            if parent_mol.HasProp('protected_user_canon_ids'):
                child_protected_canon_user_id = get_protected_canon_ids(m)
                child_protected_canon_user_id = ','.join(map(str, child_protected_canon_user_id))
            data.append((mol_id, iteration, Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True),
                         parent_mol.GetProp('_Name'), mol_mw, mol_rtb, mol_logp, mol_qed, mol_tpsa,
                         child_protected_canon_user_id))
    return data


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
                            f"WHERE iteration = '{iteration}' AND mol_block IS NOT NULL"))
    df = pd.DataFrame(res, columns=['id', 'rmsd', 'plif_sim']).set_index('id')
    return df


def get_docked_mol_ids(conn, iteration):
    """
    Returns mol_ids for molecules which where docked at the given iteration and conversion to mol block was successful
    :param conn:
    :param iteration:
    :return:
    """
    cur = conn.cursor()
    res = cur.execute(f"SELECT id FROM mols WHERE iteration = '{iteration}' AND mol_block IS NOT NULL")  # TODO: use docking_score instead of mol_block
    return [i[0] for i in res]


def get_mol_qeds(conn, mol_ids):
    """
    Returns dict of mol_id: qed
    :param conn: connection to docking DB
    :param mol_ids: list of mol ids
    :return:
    """
    cur = conn.cursor()
    sql = 'SELECT id, qed FROM mols WHERE id IN (?)'
    return dict(eadb.select_from_db(cur, sql, mol_ids))


def get_mol_scores(conn, mol_ids):
    """
    Return dict of mol_id: score
    :param conn: connection to docking DB
    :param mol_ids: list of mol ids
    :return:
    """
    cur = conn.cursor()
    sql = 'SELECT id, docking_score FROM mols WHERE id IN (?)'
    return dict(eadb.select_from_db(cur, sql, mol_ids))


def get_mols(conn, mol_ids):
    """
    Returns list of Mol objects from docking DB, order is arbitrary, molecules with errors will be silently skipped
    :param conn: connection to docking DB
    :param mol_ids: list of molecules to retrieve
    :return:
    """
    cur = conn.cursor()
    sql = 'SELECT mol_block, protected_user_canon_ids FROM mols WHERE id IN (?) AND mol_block IS NOT NULL'

    mols = []
    for items in eadb.select_from_db(cur, sql, mol_ids):
        m = Chem.MolFromMolBlock(items[0], removeHs=False)
        Chem.AssignStereochemistryFrom3D(m)
        if not m:
            continue
        if len(items) > 1 and items[1] is not None:
            m.SetProp('protected_user_canon_ids', items[1])
        mol_id, stereo_id = mol_name_split(m.GetProp('_Name'))
        m.SetProp('_Name', mol_id)
        mols.append(m)
    cur.close()
    return mols


def get_last_iter_from_db(db_fname):
    """
    Returns last iteration number
    :param db_fname:
    :return: iteration number
    """
    with sqlite3.connect(db_fname, timeout=90) as conn:
        cur = conn.cursor()
        res = list(cur.execute("SELECT max(iteration) FROM mols"))[0][0]
        return res


def check_any_molblock_isnull(dbname):
    with sqlite3.connect(dbname, timeout=90) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM mols WHERE mol_block IS NULL")
        result = cur.fetchone()[0]
        if result == 0:
            return False
        else:
            return True


def get_protein_heavy_atom_xyz(dbname):
    with sqlite3.connect(dbname, timeout=90) as conn:
        cur = conn.cursor()
        cur.execute("SELECT protein FROM setup")
        pdb_block = cur.fetchone()[0]
        return get_protein_heavy_atoms_xyz_from_string(pdb_block)
