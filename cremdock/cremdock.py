#!/usr/bin/env python3
import argparse
import logging
import os
import sqlite3
from functools import partial
from multiprocessing import Pool

from crem.utils import sample_csp3, filter_max_ring_size

from easydock import database as eadb
from easydock.run_dock import get_supplied_args, docking

from cremdock import database
from cremdock import user_protected_atoms
from cremdock.arg_types import cpu_type, filepath_type, similarity_value_type, str_lower_type
from cremdock.crem_grow import grow_mols_crem
from cremdock.database import get_protein_heavy_atom_xyz
from cremdock.molecules import get_major_tautomer
from cremdock.ranking import ranking_score
from cremdock.selection import selection_and_grow_greedy, selection_and_grow_clust, selection_and_grow_clust_deep, \
    selection_and_grow_pareto

sample_functions = {'sample_csp3': sample_csp3}

filter_functions = {'filter_max_ring_size': filter_max_ring_size}


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
    logging.info(f'iteration {iteration} started')
    conn = sqlite3.connect(dbname)
    logging.debug(f'iteration {iteration}, make_docking={make_docking}')
    protein_xyz = get_protein_heavy_atom_xyz(dbname)
    if make_docking:
        if protonation:
            logging.debug(f'iteration {iteration}, start protonation')
            eadb.add_protonation(dbname, program=protonation, tautomerize=False,
                                 add_sql='AND iteration=(SELECT MAX(iteration) from mols)')
            logging.debug(f'iteration {iteration}, end protonation')
        logging.debug(f'iteration {iteration}, start mols selection for docking')
        mols = eadb.select_mols_to_dock(conn, add_sql='AND iteration=(SELECT MAX(iteration) from mols)')
        logging.debug(f'iteration {iteration}, start docking')
        for mol_id, res in docking(mols,
                                   dock_func=mol_dock_func,
                                   dock_config=config,
                                   priority_func=priority_func,
                                   ncpu=ncpu,
                                   dask_client=dask_client):
            if res:
                eadb.update_db(conn, mol_id, res)
        logging.debug(f'iteration {iteration}, end docking')
        database.update_db(conn, plif_ref=plif_list, plif_protein_fname=protein_h, ncpu=ncpu)
        logging.debug(f'iteration {iteration}, DB was updated, rmsd and plif were calculated')

        res = dict()
        mol_data = database.get_docked_mol_data(conn, iteration)
        logging.debug(f'iteration {iteration}, docked mols count: {mol_data.shape[0]}')

        if iteration != 1 and rmsd is not None:
            mol_data = mol_data.loc[mol_data['rmsd'] <= rmsd]  # filter by RMSD
        if plif_list and len(mol_data.index) > 0:
            mol_data = mol_data.loc[mol_data['plif_sim'] >= plif_cutoff]  # filter by PLIF
        logging.debug(f'iteration {iteration}, docked mols count after rmsd/plif filters: {mol_data.shape[0]}')
        if len(mol_data.index) == 0:
            logging.info(f'iteration {iteration}, no molecules were selected for growing')
        else:
            logging.debug(f'iteration {iteration}, start selection and growing')
            mols = database.get_mols(conn, mol_data.index)
            if alg_type == 1:
                res = selection_and_grow_greedy(mols=mols, conn=conn, protein_xyz=protein_xyz,
                                                ntop=ntop, max_mw=mw, max_rtb=rtb, max_logp=logp, max_tpsa=tpsa,
                                                ranking_func=ranking_score_func, ncpu=ncpu, **kwargs)
            elif alg_type in [2, 3] and len(mols) <= nclust:  # if number of mols is lower than nclust grow all mols
                res = grow_mols_crem(mols=mols, protein_xyz=protein_xyz, max_mw=mw, max_rtb=rtb, max_logp=logp,
                                     max_tpsa=tpsa, ncpu=ncpu, **kwargs)
            elif alg_type == 2:
                res = selection_and_grow_clust_deep(mols=mols, conn=conn, nclust=nclust, protein_xyz=protein_xyz,
                                                    ntop=ntop, max_mw=mw, max_rtb=rtb, max_logp=logp, max_tpsa=tpsa,
                                                    ranking_func=ranking_score_func, ncpu=ncpu, **kwargs)
            elif alg_type == 3:
                res = selection_and_grow_clust(mols=mols, conn=conn, nclust=nclust, protein_xyz=protein_xyz,
                                               ntop=ntop, max_mw=mw, max_rtb=rtb, max_logp=logp, max_tpsa=tpsa,
                                               ranking_func=ranking_score_func, ncpu=ncpu, **kwargs)
            elif alg_type == 4:
                res = selection_and_grow_pareto(mols=mols, conn=conn, max_mw=mw, max_rtb=rtb, max_logp=logp,
                                                max_tpsa=tpsa, protein_xyz=protein_xyz,
                                                ranking_func=ranking_score_func, ncpu=ncpu, **kwargs)
            logging.debug(f'iteration {iteration}, end selection and growing')

    else:
        logging.debug(f'iteration {iteration}, docking was omitted, all mols are grown')
        mols = database.get_mols(conn, database.get_docked_mol_ids(conn, iteration))
        res = grow_mols_crem(mols=mols, protein_xyz=protein_xyz, max_mw=mw, max_rtb=rtb, max_logp=logp, max_tpsa=tpsa,
                             ncpu=ncpu, **kwargs)
        logging.debug(f'iteration {iteration}, docking was omitted, all mols were grown')

    logging.debug(f'iteration {iteration}, number of mols after growing: {sum(len(v)for v in res.values())}')

    if res:
        res = user_protected_atoms.assign_protected_ids(res)
        logging.debug(f'iteration {iteration}, end assign_protected_ids')
        res = user_protected_atoms.set_isotope_to_parent_protected_atoms(res)
        logging.debug(f'iteration {iteration}, end set_isotope_to_parent_protected_atoms')
        res = get_major_tautomer(res)
        logging.debug(f'iteration {iteration}, end get_major_tautomer')
        res = user_protected_atoms.assign_protected_ids_from_isotope(res)
        logging.debug(f'iteration {iteration}, end assign_protected_ids_from_isotope')
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
        logging.debug(f'iteration {iteration}, {len(data)} new mols were inserted in DB')
        return True

    else:
        logging.info(f'iteration {iteration}, growth was stopped')
        return False


def entry_point():
    parser = argparse.ArgumentParser(description='Fragment growing within a binding pocket guided by molecular docking.',
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=80))

    group1 = parser.add_argument_group('Input/output files')
    group1.add_argument('-i', '--input_frags', metavar='FILENAME', required=False, type=filepath_type,
                        help='SMILES file with input fragments or SDF file with 3D coordinates of pre-aligned input '
                             'fragments (e.g. from PDB complexes). '
                             'If SDF contain <protected_user_ids> field (comma-separated 1-based indices) '
                             'these atoms will be protected from growing. This argument can be omitted if an existed '
                             'output DB is specified, then docking will be continued from the last successful '
                             'iteration. Optional.')
    group1.add_argument('-o', '--output', metavar='FILENAME', required=True, type=filepath_type,
                        help='SQLite DB with docking results. If an existed DB was supplied input fragments will be '
                             'ignored if any and the program will continue docking from the last successful iteration.')

    group3 = parser.add_argument_group('Generation parameters')
    group3.add_argument('--n_iterations', metavar='INTEGER', default=None, type=int,
                        help='maximum number of iterations.')
    group3.add_argument('-t', '--algorithm', metavar='INTEGER', default=2, type=int, choices=[1, 2, 3, 4],
                        help='the number of the search algorithm: 1 - greedy search, 2 - deep clustering (if some '
                             'molecules from a cluster cannot be grown they will be replaced with other lower scored '
                             'ones), 3 - clustering (fixed number of molecules is selected irrespective their ability '
                             'to be grown), 4 - Pareto front (MW vs. docking score).')
    group3.add_argument('--ntop', metavar='INTEGER', type=int, default=2, required=False,
                        help='the number of the best molecules to select for the next iteration in the case of greedy '
                             'search (algorithm 1) or the number of molecules from each cluster in the case of '
                             'clustering (algorithms 2 and 3).')
    group3.add_argument('--nclust', metavar='INTEGER', type=int, default=20, required=False,
                        help='the number of KMeans clusters to consider for molecule selection.')
    group3.add_argument('--ranking', metavar='INTEGER', required=False, type=int, default=1,
                        choices=[1, 2, 3, 4, 5, 6, 7],
                        help='the number of the algorithm for ranking molecules:\n'
                             '1 - ranking based on docking scores,\n'
                             '2 - ranking based on docking scores and QED,\n'
                             '3 - ranking based on docking score/number heavy atoms of molecule,\n'
                             '4 - raking based on docking score/number heavy atoms of molecule and QED,\n'
                             '5 - ranking based on docking score and FCsp3_BM,\n'
                             '6 - ranking based docking score/number heavy atoms of molecule and FCsp3_BM,\n'
                             '7 - ranking based on docking score and FCsp3_BM**2.')

    group2 = parser.add_argument_group('CReM parameters')
    group2.add_argument('-d', '--db', metavar='FILENAME', required=False, type=filepath_type, default=None,
                        help='CReM fragment DB.')
    group2.add_argument('-r', '--radius', metavar='INTEGER', default=1, type=int,
                        help='context radius for replacement.')
    group2.add_argument('--min_freq', metavar='INTGER', default=0, type=int,
                        help='the frequency of occurrence of the fragment in the source database.')
    group2.add_argument('--max_replacements', metavar='INTEGER', type=int, required=False, default=None,
                        help='the maximum number of randomly chosen replacements. Default: None (all replacements).')
    group2.add_argument('--min_atoms', metavar='INTEGER', default=1, type=int,
                        help='the minimum number of atoms in the fragment which will replace H')
    group2.add_argument('--max_atoms', metavar='INTEGER', default=10, type=int,
                         help='the maximum number of atoms in the fragment which will replace H')
    group2.add_argument('--sample_func', default=None, required=False, choices=sample_functions.keys(),
                        help='Choose a function to randomly sample fragments for growing (if max_replacements is '
                             'given). Otherwise uniform sampling will be used.')
    group2.add_argument('--filter_func', default=None, required=False, choices=filter_functions.keys(),
                        help='Choose a function to pre-filter fragments for growing.'
                             'By default no pre-filtering will be applied.')

    group4 = parser.add_argument_group('Filters')
    group4.add_argument('--rmsd', metavar='NUMERIC', type=float, default=None, required=False,
                        help='maximum allowed RMSD value relative to a parent compound to pass on the next iteration.')
    group4.add_argument('--mw', metavar='NUMERIC', default=450, type=float,
                        help='maximum ligand molecular weight to pass on the next iteration.')
    group4.add_argument('--rtb', metavar='INTEGER', type=int, default=5, required=False,
                        help='maximum allowed number of rotatable bonds in a compound.')
    group4.add_argument('--logp', metavar='NUMERIC', type=float, default=4, required=False,
                        help='maximum allowed logP of a compound.')
    group4.add_argument('--tpsa', metavar='NUMERIC', type=float, default=120, required=False,
                        help='maximum allowed TPSA of a compound.')

    group6 = parser.add_argument_group('PLIF filters')
    group6.add_argument('--protein_h', metavar='protein.pdb', required=False, type=filepath_type,
                        help='PDB file with the same protein as for docking, but it should have all hydrogens '
                             'explicit. Required for correct PLIF detection.')
    group6.add_argument('--plif', metavar='STRING', default=None, required=False, nargs='*',
                        type=str_lower_type,
                        help='list of protein-ligand interactions compatible with ProLIF. Dot-separated names of each '
                             'interaction which should be observed for a ligand to pass to the next iteration. Derive '
                             'these names from a reference ligand. Example: ASP115.HBDonor or ARG34.A.Hydrophobic.')
    group6.add_argument('--plif_cutoff', metavar='NUMERIC', default=1, required=False, type=similarity_value_type,
                        help='cutoff of Tversky similarity, value between 0 and 1.')

    group5 = parser.add_argument_group('Docking parameters')
    group5.add_argument('--protonation', default=None, required=False, choices=['chemaxon', 'pkasolver'],
                        help='choose a protonation program supported by EasyDock.')
    group5.add_argument('--program', default='vina', required=False, choices=['vina', 'gnina'],
                        help='name of a docking program.')
    group5.add_argument('--config', metavar='FILENAME', required=False,
                        help='YAML file with parameters used by docking program.\n'
                             'vina.yml\n'
                             'protein: path to pdbqt file with a protein\n'
                             'protein_setup: path to a text file with coordinates of a binding site\n'
                             'exhaustiveness: 8\n'
                             'n_poses: 10\n'
                             'seed: -1\n'
                             'gnina.yml\n')
    group5.add_argument('--hostfile', metavar='FILENAME', required=False, type=str, default=None,
                        help='text file with addresses of nodes of dask SSH cluster. The most typical, it can be '
                             'passed as $PBS_NODEFILE variable from inside a PBS script. The first line in this file '
                             'will be the address of the scheduler running on the standard port 8786. If omitted, '
                             'calculations will run on a single machine as usual.')

    group7 = parser.add_argument_group('Auxiliary parameters')
    group7.add_argument('--log', metavar='FILENAME', required=False, type=str, default=None,
                        help='log file to collect progress and debug messages. If omitted, the log file with the same '
                             'name as output DB will be created.')
    group7.add_argument('--prefix', metavar='STRING', required=False, type=str, default=None,
                        help='prefix which will be added to all names. This might be useful if multiple runs are made '
                             'which will be analyzed together.')
    group7.add_argument('-c', '--ncpu', metavar='INTEGER', default=1, type=cpu_type,
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

    if not args.log:
        args.log = os.path.splitext(os.path.abspath(args.output))[0] + '.log'
    logging.basicConfig(filename=args.log, encoding='utf-8', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S',
                        format='[%(asctime)s] %(levelname)s: (PID:%(process)d) %(message)s')

    if args.algorithm in [2, 3] and (args.nclust * args.ntop > 20):
        logging.warning('The number of clusters (nclust) and top scored molecules selected from each cluster (ntop) '
                        'will result in selection on each iteration more than 20 molecules that may slower '
                        'computations.')

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

    sample_func = sample_functions[args.sample_func] if args.sample_func else None
    filter_func = filter_functions[args.filter_func] if args.filter_func else None

    try:
        while True:
            res = make_iteration(dbname=args.output, iteration=iteration, config=args.config, mol_dock_func=mol_dock,
                                 priority_func=pred_dock_time, ntop=args.ntop, nclust=args.nclust,
                                 mw=args.mw, rmsd=args.rmsd, rtb=args.rtb, logp=args.logp, tpsa=args.tpsa,
                                 alg_type=args.algorithm, ranking_score_func=ranking_score(args.ranking), ncpu=args.ncpu,
                                 protonation=args.protonation, make_docking=make_docking,
                                 dask_client=dask_client, plif_list=args.plif, protein_h=args.protein_h,
                                 plif_cutoff=args.plif_cutoff, prefix=args.prefix, db_name=args.db, radius=args.radius,
                                 min_freq=args.min_freq, min_atoms=args.min_atoms, max_atoms=args.max_atoms,
                                 max_replacements=args.max_replacements, sample_func=sample_func, filter_func=filter_func)
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
        logging.exception(e, stack_info=True)

    finally:
        logging.info(f'{iteration} iterations were completed')


if __name__ == '__main__':
    entry_point()