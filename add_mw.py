#!/usr/bin/env python3

import argparse
import sqlite3
import sys
from multiprocessing import Pool

from grow import filepath_type, cpu_type
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt


def calc_mw(items):
    # items - (rowid, smi)
    rowid, smi = items
    mw = None
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mw = round(MolWt(mol), 2)
    return rowid, mw


def main():
    parser = argparse.ArgumentParser(description='Add column "mw" to CReM database with MW of fragments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='FILENAME', required=True, type=filepath_type,
                        help='SQLite DB with CReM fragments.')
    parser.add_argument('-c', '--ncpu', default=1, type=cpu_type,
                        help='number of cpus.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print progress to STDERR.')

    args = parser.parse_args()

    pool = Pool(args.ncpu)

    with sqlite3.connect(args.input) as conn:
        cur = conn.cursor()
        tables = cur.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'radius%'")
        tables = [i[0] for i in tables]
        for table in tables:
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN mw NUMERIC DEFAULT NULL")
                conn.commit()
            except sqlite3.OperationalError as e:
                sys.stderr.write(str(e) + '\n')
            sys.stderr.write(f'\ntable {table} processed\n')
            cur.execute(f"SELECT rowid, core_smi FROM {table} WHERE mw IS NULL")
            res = cur.fetchall()
            for i, (rowid, mw) in enumerate(pool.imap_unordered(calc_mw, res), 1):
                if mw is not None:
                    cur.execute(f"UPDATE {table} SET mw = {mw} WHERE rowid = '{rowid}'")
                if args.verbose and i % 1000 == 0:
                    sys.stderr.write(f'\r{i} fragments processed')
            conn.commit()


if __name__ == '__main__':
    main()
