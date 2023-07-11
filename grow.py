#!/usr/bin/env python3
import argparse
import os
import sqlite3
import sys
from functools import partial
from multiprocessing import Pool
import datetime

from easydock import database as eadb
from easydock.run_dock import get_supplied_args, docking

import database
import user_protected_atoms
from arg_types import cpu_type, filepath_type, similarity_value_type, str_lower_type
from crem_grow import grow_mols_crem
from molecules import get_major_tautomer
from ranking import ranking_score
from selection import selection_and_grow_greedy, selection_and_grow_clust, selection_and_grow_clust_deep, selection_and_grow_pareto


def supply_parent_child_mols(d):
    # d - {parent_mol: [child_mol1, child_mol2, ...], ...}
    n = 0
    for parent_mol, child_mols in d.items():
        for child_mol in child_mols:
            yield parent_mol, child_mol, n
            n += 1


def make_iteration(dbname, iteration, config, mol_dock_func, priority_func, ntop, nclust, mw, rmsd, rtb, logp, tpsa,
                   alg_type, ranking_score_func, ncpu, protonation, make_docking=True, dask_client=None, plif_list=None,
                   protein_h=None, plif_cutoff=1, prefix=None, **kwargs):
    sys.stderr.write(f'iteration {iteration} started\n')
    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; start protonation\n')
    if protonation:
        eadb.add_protonation(dbname, tautomerize=False, add_sql='AND iteration=(SELECT MAX(iteration) from mols)')
    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; end protonation\n')
    conn = sqlite3.connect(dbname)
    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; make_doking {make_docking}\n')
    if make_docking:
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; mols selection for dock\n')
        mols = eadb.select_mols_to_dock(conn, add_sql='AND iteration=(SELECT MAX(iteration) from mols)')
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; start docking\n')
        for mol_id, res in docking(mols,
                                   dock_func=mol_dock_func,
                                   dock_config=config,
                                   priority_func=priority_func,
                                   ncpu=ncpu,
                                   dask_client=dask_client):
            if res:
                eadb.update_db(conn, mol_id, res)
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; end docking\n')
        database.update_db(conn, plif_ref=plif_list, plif_protein_fname=protein_h, ncpu=ncpu)
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; end update, calc rmsd and plif\n')

        res = dict()
        mol_data = database.get_docked_mol_data(conn, iteration)
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; docked mols count {mol_data.shape}\n')
        if iteration != 1:
            mol_data = mol_data.loc[mol_data['rmsd'] <= rmsd]  # filter by RMSD
        if plif_list and len(mol_data.index) > 0:
            mol_data = mol_data.loc[mol_data['plif_sim'] >= plif_cutoff]  # filter by PLIF
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; docked mols count after rmsd/plif filteration {mol_data.shape}\n')
        if len(mol_data.index) == 0:
            sys.stderr.write(f'iteration {iteration}: no molecules were selected for growing.\n')
        else:
            sys.stderr.write(
                f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; start selection and growing\n')
            mols = database.get_mols(conn, mol_data.index)
            if alg_type == 1:
                res = selection_and_grow_greedy(mols=mols, conn=conn, protein=protein_h,
                                                ntop=ntop, max_mw=mw, max_rtb=rtb, max_logp=logp, max_tpsa=tpsa,
                                                ranking_func=ranking_score_func, ncpu=ncpu, **kwargs)
            elif alg_type in [2, 3] and len(mols) <= nclust:  # if number of mols is lower than nclust grow all mols
                res = grow_mols_crem(mols=mols, protein=protein_h, max_mw=mw, max_rtb=rtb, max_logp=logp,
                                     max_tpsa=tpsa, ncpu=ncpu, **kwargs)
            elif alg_type == 2:
                res = selection_and_grow_clust_deep(mols=mols, conn=conn, nclust=nclust, protein=protein_h,
                                                    ntop=ntop, max_mw=mw, max_rtb=rtb, max_logp=logp, max_tpsa=tpsa,
                                                    ranking_func=ranking_score_func, ncpu=ncpu, **kwargs)
            elif alg_type == 3:
                res = selection_and_grow_clust(mols=mols, conn=conn, nclust=nclust, protein=protein_h,
                                               ntop=ntop, max_mw=mw, max_rtb=rtb, max_logp=logp, max_tpsa=tpsa,
                                               ranking_func=ranking_score_func, ncpu=ncpu, **kwargs)
            elif alg_type == 4:
                res = selection_and_grow_pareto(mols=mols, conn=conn, max_mw=mw, max_rtb=rtb, max_logp=logp, max_tpsa=tpsa,
                                                protein=protein_h, ranking_func=ranking_score_func, ncpu=ncpu, **kwargs)
            sys.stderr.write(
                f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; end selection and growing\n')

    else:
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; docking omitted, all mols are grown\n')
        mols = database.get_mols(conn, database.get_docked_mol_ids(conn, iteration))
        res = grow_mols_crem(mols=mols, protein=protein_h, max_mw=mw, max_rtb=rtb, max_logp=logp, max_tpsa=tpsa,
                             ncpu=ncpu, **kwargs)
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; docking omitted, all mols were grown\n')

    sys.stderr.write(
        f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; n mols after grow {sum(len(v)for v in res.values())}\n')

    if res:
        res = user_protected_atoms.assign_protected_ids(res)
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; end assign_protected_ids\n')
        res = user_protected_atoms.set_isotope_to_parent_protected_atoms(res)
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; end set_isotope_to_parent_protected_atoms\n')
        res = get_major_tautomer(res)
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; end get_major_tautomer\n')
        res = user_protected_atoms.assign_protected_ids_from_isotope(res)
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; end assign_protected_ids_from_isotope\n')
        data = []
        p = Pool(ncpu)
        try:
            for d in p.starmap(partial(database.prep_data_for_insert, iteration=iteration, max_rtb=rtb, max_mw=mw,
                                       max_logp=logp, max_tpsa=tpsa, prefix=prefix), supply_parent_child_mols(res)):
                data.extend(d)
        finally:
            p.close()
            p.join()
        cols = ['id', 'iteration', 'smi', 'parent_id', 'mw', 'rtb', 'logp', 'qed', 'tpsa', 'protected_user_canon_ids']
        eadb.insert_db(dbname, data=data, cols=cols)
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; new mols were inserted in DB\n')
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
    parser.add_argument('-d', '--db', metavar='fragments.db', required=False, type=filepath_type, default=None,
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
    parser.add_argument('--no_protonation', action='store_true', default=False,
                        help='disable protonation of molecules before docking. Protonation requires installed '
                             'cxcalc chemaxon utility.')
    parser.add_argument('--n_iterations', default=None, type=int,
                        help='maximum number of iterations.')
    parser.add_argument('-t', '--algorithm', default=2, type=int, choices=[1, 2, 3, 4],
                        help='the number of the search algorithm: 1 - greedy search, 2 - deep clustering (if some '
                             'molecules from a cluster cannot be grown they will be replaced with new lower scored '
                             'ones), 3 - clustering, 4 - Pareto front (MW vs. docking score).')
    parser.add_argument('--ntop', type=int, default=20, required=False,
                        help='the number of the best molecules to select for the next iteration in the case of greedy '
                             'search (algorithm 1) or the number of molecules from each cluster in the case of '
                             'clustering (algorithms 2 and 3).')
    parser.add_argument('--nclust', type=int, default=20, required=False,
                        help='the number of KMeans clusters to consider for molecule selection.')
    parser.add_argument('--ranking', required=False, type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help='the number of the algorithm for ranking molecules: 1 - ranking based on docking scores, '
                             '2 - ranking based on docking scores and QED, '
                             '3 - ranking based on docking score/number heavy atoms of molecule,'
                             '4 - raking based on docking score/number heavy atoms of molecule and QED,'
                             '5 - ranking based on docking score and FCsp3_BM,'
                             '6 - ranking based docking score/number heavy atoms of molecule and FCsp3_BM.')
    parser.add_argument('--rmsd', type=float, default=2, required=False,
                        help='maximum allowed RMSD value relative to a parent compound to pass on the next iteration.')
    parser.add_argument('--mw', default=450, type=float,
                        help='maximum ligand molecular weight to pass on the next iteration.')
    parser.add_argument('--rtb', type=int, default=5, required=False,
                        help='maximum allowed number of rotatable bonds in a compound.')
    parser.add_argument('--logp', type=float, default=4, required=False,
                        help='maximum allowed logP of a compound.')
    parser.add_argument('--tpsa', type=float, default=120, required=False,
                        help='maximum allowed TPSA of a compound.')
    parser.add_argument('--protein_h', metavar='protein.pdb', required=False, type=filepath_type,
                        help='PDB file with the same protein as for docking, but it should have all hydrogens explicit.'
                             'Required for determination of growing points in molecules and PLIF detection.')
    parser.add_argument('--program', metavar='STRING', default='vina', required=False, choices=['vina', 'gnina'],
                        help='name of a docking program. Choices: vina (default), gnina.')
    parser.add_argument('--config', metavar='FILENAME', required=False,
                        help='YAML file with parameters used by docking program.\n'
                             'vina.yml\n'
                             'protein: path to pdbqt file with a protein\n'
                             'protein_setup: path to a text file with coordinates of a binding site\n'
                             'exhaustiveness: 8\n'
                             'n_poses: 10\n'
                             'seed: -1\n'
                             'gnina.yml\n')
    parser.add_argument('--plif', default=None, required=False, nargs='*', type=str_lower_type,
                        help='list of protein-ligand interactions compatible with ProLIF. Dot-separated names of each '
                             'interaction which should be observed for a ligand to pass to the next iteration. Derive '
                             'these names from a reference ligand. Example: ASP115.HBDonor or ARG34.A.Hydrophobic.')
    parser.add_argument('--plif_cutoff', metavar='NUMERIC', default=1, required=False, type=similarity_value_type,
                        help='cutoff of Tversky similarity, value between 0 and 1.')
    parser.add_argument('--hostfile', metavar='FILENAME', required=False, type=str, default=None,
                        help='text file with addresses of nodes of dask SSH cluster. The most typical, it can be '
                             'passed as $PBS_NODEFILE variable from inside a PBS script. The first line in this file '
                             'will be the address of the scheduler running on the standard port 8786. If omitted, '
                             'calculations will run on a single machine as usual.')
    parser.add_argument('--prefix', metavar='STRING', required=False, type=str, default=None,
                        help='prefix which will be added to all names. This might be useful if multiple runs are made '
                             'which will be analyzed together.')
    parser.add_argument('-c', '--ncpu', default=1, type=cpu_type,
                        help='number of cpus.')

    args = parser.parse_args()

    # depending on input setup operations applied on the first iteration
    # input      make_docking & make_selection
    # SMILES                              True
    # 3D SDF                             False
    # existed DB                          True
    if os.path.isfile(args.output):
        args_dict, tmpfiles = eadb.restore_setup_from_db(args.output)
        # this will ignore stored values of those args which were supplied via command line;
        # allowed command line args have precedence over stored ones, others will be ignored
        supplied_args = get_supplied_args(parser)
        # allow update of only given arguments
        supplied_args = tuple(arg for arg in supplied_args if arg in ['output', 'db', 'hostfile', 'ncpu'])
        for arg in supplied_args:
            del args_dict[arg]
        args.__dict__.update(args_dict)
        iteration = database.get_last_iter_from_db(args.output)
        if iteration is None:
            raise IOError("The last iteration could not be retrieved from the database. Please check it.")
        make_docking = True

    else:
        database.create_db(args.output, args, args_to_save=['protein_h'])
        make_docking = database.insert_starting_structures_to_db(args.input_frags, args.output, args.prefix)
        iteration = 1

    if args.algorithm in [2, 3] and (args.nclust * args.ntop > 20):
        sys.stderr.write('The number of clusters (nclust) and top scored molecules selected from each cluster (ntop) '
                         'will result in selection on each iteration more than 20 molecules that may slower '
                         'computations.\n')
        sys.stderr.flush()

    if args.plif is not None and (args.protein_h is None or not os.path.isfile(args.protein_h)):
        raise FileNotFoundError('PLIF pattern was specified but the protein file is missing or was not supplied. '
                                'Calculation was aborted.')

    if args.hostfile is not None:
        from dask.distributed import Client

        with open(args.hostfile) as f:
            hosts = [line.strip() for line in f]
        dask_client = Client(hosts[0] + ':8786', connection_limit=2048)
        # dask_client = Client()   # to test dask locally
    else:
        dask_client = None

    if args.program == 'vina':
        from easydock.vina_dock import mol_dock, pred_dock_time
    elif args.program == 'gnina':
        from easydock.gnina_dock import mol_dock
        from easydock.vina_dock import pred_dock_time
    else:
        raise ValueError(f'Illegal program argument was supplied: {args.program}')

    try:
        while True:
            res = make_iteration(dbname=args.output, iteration=iteration, config=args.config, mol_dock_func=mol_dock,
                                 priority_func=pred_dock_time, ntop=args.ntop, nclust=args.nclust,
                                 mw=args.mw, rmsd=args.rmsd, rtb=args.rtb, logp=args.logp, tpsa=args.tpsa,
                                 alg_type=args.algorithm, ranking_score_func=ranking_score(args.ranking), ncpu=args.ncpu,
                                 protonation=not args.no_protonation, make_docking=make_docking,
                                 dask_client=dask_client, plif_list=args.plif, protein_h=args.protein_h,
                                 plif_cutoff=args.plif_cutoff, prefix=args.prefix, db_name=args.db, radius=args.radius,
                                 min_freq=args.min_freq, min_atoms=args.min_atoms, max_atoms=args.max_atoms,
                                 max_replacements=args.max_replacements)
            make_docking = True

            if res:
                iteration += 1
                if args.n_iterations and iteration == args.n_iterations:
                    break
            else:
                if iteration == 1:
                    # 0 successful iteration for finally printing
                    iteration = 0
                break

    except Exception as e:
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; make_iteration error: {e}\n')

    finally:
        sys.stderr.write(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; iteration {iteration}; pid {os.getpid()}; how many iterations completed succesfully\n')
        sys.stderr.write(f'{iteration} iterations were completed successfully\n')


if __name__ == '__main__':
    main()
