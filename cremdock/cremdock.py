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


def make_iteration(dbname, config, mol_dock_func, priority_func, ntop, nclust, mw, rmsd, rtb, logp, tpsa,
                   alg_type, ranking_score_func, ncpu, protonation, ring_sample, make_docking=True, tautomerize=False,
                   dask_client=None, plif_list=None, plif_protein=None, plif_cutoff=1, prefix=None,
                   n_iterations=None, **kwargs):
    iteration = database.get_last_iter_from_db(dbname)
    if n_iterations and n_iterations == iteration:
        final_iteration = True
    else:
        final_iteration = False
    logging.info(f'iteration {iteration} started')  # supress logging on the final iteration where only docking is occurred
    with sqlite3.connect(dbname, timeout=90) as conn:
        logging.debug(f'iteration {iteration}, make_docking={make_docking}')
        protein_xyz = get_protein_heavy_atom_xyz(dbname)
        if make_docking:
            if protonation:
                logging.debug(f'iteration {iteration}, start protonation')
                eadb.add_protonation(dbname, program=protonation, tautomerize=False,
                                     add_sql=f' AND iteration={iteration}')
                logging.debug(f'iteration {iteration}, end protonation')
            logging.debug(f'iteration {iteration}, start mols selection for docking')
            mols = eadb.select_mols_to_dock(conn, add_sql=f' AND iteration={iteration}')
            logging.debug(f'iteration {iteration}, start docking')
            for mol_id, res in docking(mols,
                                       dock_func=mol_dock_func,
                                       dock_config=config,
                                       priority_func=priority_func,
                                       ncpu=ncpu,
                                       dask_client=dask_client,
                                       ring_sample=ring_sample):
                if res:
                    eadb.update_db(conn, mol_id, res)
            logging.debug(f'iteration {iteration}, end docking')
            database.update_db(conn, iteration, plif_ref=plif_list, plif_protein_fname=plif_protein, ncpu=ncpu)
            logging.debug(f'iteration {iteration}, DB was updated (including rmsd and plif if set)')

            res = dict()
            mol_data = database.get_docked_mol_data(conn, iteration)
            logging.info(f'iteration {iteration}, docked mols count: {mol_data.shape[0]}')

            if final_iteration:  # make only docking
                return iteration, False

            rmsd_plif_flag = False
            if iteration > 0 and rmsd is not None:
                mol_data = mol_data.loc[mol_data['rmsd'] <= rmsd]  # filter by RMSD
                rmsd_plif_flag = True
            if plif_list and len(mol_data.index) > 0:
                mol_data = mol_data.loc[mol_data['plif_sim'] >= plif_cutoff]  # filter by PLIF
                rmsd_plif_flag = True
            if rmsd_plif_flag:
                logging.info(f'iteration {iteration}, docked mols count after rmsd/plif filters: {mol_data.shape[0]}')

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

        logging.info(f'iteration {iteration}, number of mols after growing: {sum(len(v)for v in res.values())}')

        if res:
            # res may containg duplicated molecules between different parent molecules
            # they will be discarded dirung insert
            res = user_protected_atoms.assign_protected_ids(res)
            logging.debug(f'iteration {iteration}, end assign_protected_ids')
            res = user_protected_atoms.set_isotope_to_parent_protected_atoms(res)
            logging.debug(f'iteration {iteration}, end set_isotope_to_parent_protected_atoms')
            if tautomerize:
                res = get_major_tautomer(res)
                logging.debug(f'iteration {iteration}, end get_major_tautomer')
            res = user_protected_atoms.assign_protected_ids_from_isotope(res)
            logging.debug(f'iteration {iteration}, end assign_protected_ids_from_isotope')
            data = []
            p = Pool(ncpu)
            try:
                for d in p.starmap(partial(database.prep_data_for_insert, iteration=iteration + 1, max_rtb=rtb, max_mw=mw,
                                           max_logp=logp, max_tpsa=tpsa, prefix=prefix), supply_parent_child_mols(res)):
                    data.extend(d)
            finally:
                p.close()
                p.join()
            cols = ['id', 'iteration', 'smi', 'parent_id', 'mw', 'rtb', 'logp', 'qed', 'tpsa', 'protected_user_canon_ids']
            inserted_row_count = eadb.insert_db(dbname, data=data, cols=cols)
            logging.info(f'iteration {iteration}, {inserted_row_count} new mols were inserted in DB after filtering by '
                         f'physicochemical properties')
            if data:
                return iteration, True
            else:
                return iteration, False  # if data is empty

        else:
            logging.info(f'iteration {iteration}, growth was stopped')
            return iteration, False


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
    group3.add_argument('-t', '--search', metavar='INTEGER', default=2, type=int, choices=[1, 2, 3, 4],
                        help='the number of the search algorithm: 1 - greedy search, 2 - deep clustering (if some '
                             'molecules from a cluster cannot be grown they will be replaced with other lower scored '
                             'ones), 3 - clustering (fixed number of molecules is selected irrespective their ability '
                             'to be grown), 4 - Pareto front (MW vs. docking score).')
    group3.add_argument('--ntop', metavar='INTEGER', type=int, default=2, required=False,
                        help='the number of the best molecules to select for the next iteration in the case of greedy '
                             'search (1) or the number of molecules from each cluster in the case of '
                             'clustering (search 2 and 3).')
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
    group3.add_argument('--tautomerize', action='store_true', default=False,
                        help='if set, for generated molecules a major tautomer will be generated by Chemaxon (require '
                             'a license).')

    group2 = parser.add_argument_group('CReM parameters')
    group2.add_argument('-d', '--db', metavar='FILENAME', required=False, type=filepath_type, default=None,
                        help='CReM fragment DB.')
    group2.add_argument('-r', '--radius', metavar='INTEGER', default=3, type=int,
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
    group6.add_argument('--plif', metavar='STRING', default=None, required=False, nargs='*',
                        type=str_lower_type,
                        help='list of protein-ligand interactions compatible with ProLIF. Dot-separated names of each '
                             'interaction which should be observed for a ligand to pass to the next iteration. Derive '
                             'these names from a reference ligand. Example: ASP115.HBDonor or ARG34.A.Hydrophobic.')
    group6.add_argument('--plif_cutoff', metavar='NUMERIC', default=1, required=False, type=similarity_value_type,
                        help='cutoff of Tversky similarity, value between 0 and 1.')
    group6.add_argument('--plif_protein', metavar='protein.pdb', required=False, type=filepath_type,
                        help='PDB file with the same protein as for docking, but it should have all hydrogens '
                             'explicit. Required for correct PLIF detection.')

    group5 = parser.add_argument_group('Docking parameters')
    group5.add_argument('--protonation', default=None, required=False, choices=['chemaxon', 'pkasolver', 'molgpka'],
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
    group5.add_argument('--ring_sample', action='store_true', default=False,
                               help='sample conformations of saturated rings. Multiple starting conformers will be docked and '
                                    'the best one will be stored. Otherwise a single random ring conformer will be used.')
    group5.add_argument('--hostfile', metavar='FILENAME', required=False, type=str, default=None,
                        help='text file with addresses of nodes of dask SSH cluster. The most typical, it can be '
                             'passed as $PBS_NODEFILE variable from inside a PBS script. The first line in this file '
                             'will be the address of the scheduler running on the standard port 8786. If omitted, '
                             'calculations will run on a single machine as usual.')

    group7 = parser.add_argument_group('Auxiliary parameters')
    group7.add_argument('--log', metavar='FILENAME', required=False, type=str, default=None,
                        help='log file to collect progress and debug messages. If omitted, the log file with the same '
                             'name as output DB will be created.')
    group7.add_argument('--log_level', metavar='STRING', required=False, type=int, default=2,
                        choices=list(range(6)),
                        help='the level of logging: 0 - NOTSET, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, '
                             '5 - CRITICAL.')
    group7.add_argument('--prefix', metavar='STRING', required=False, type=str, default=None,
                        help='prefix which will be added to all names. This might be useful if multiple runs are made '
                             'which will be analyzed together.')
    group7.add_argument('-c', '--ncpu', metavar='INTEGER', default=1, type=cpu_type,
                        help='number of cpus.')

    args = parser.parse_args()

    if not args.log:
        args.log = os.path.splitext(os.path.abspath(args.output))[0] + '.log'
    logging.basicConfig(filename=args.log, encoding='utf-8', level=args.log_level * 10, datefmt='%Y-%m-%d %H:%M:%S',
                        format='[%(asctime)s] %(levelname)s: (PID:%(process)d) %(message)s')

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
        make_docking = True

    else:
        database.create_db(args.output, args, args_to_save=['plif_protein'])
        make_docking = database.insert_starting_structures_to_db(args.input_frags, args.output, args.prefix)

    if args.search in [2, 3] and (args.nclust * args.ntop > 20):
        logging.warning('The number of clusters (nclust) and top scored molecules selected from each cluster (ntop) '
                        'will result in selection on each iteration more than 20 molecules that may slower '
                        'computations.')

    if args.plif is not None and (args.plif_protein is None or not os.path.isfile(args.plif_protein)):
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
        final_iteration = False
        while True:
            iteration, res = make_iteration(dbname=args.output, config=args.config, mol_dock_func=mol_dock,
                                            priority_func=pred_dock_time, ntop=args.ntop, nclust=args.nclust,
                                            mw=args.mw, rmsd=args.rmsd, rtb=args.rtb, logp=args.logp, tpsa=args.tpsa,
                                            alg_type=args.search, ranking_score_func=ranking_score(args.ranking),
                                            ncpu=args.ncpu, protonation=args.protonation, ring_sample=args.ring_sample,
                                            make_docking=make_docking, dask_client=dask_client, plif_list=args.plif,
                                            plif_protein=args.plif_protein, plif_cutoff=args.plif_cutoff,
                                            prefix=args.prefix, db_name=args.db, radius=args.radius,
                                            min_freq=args.min_freq, min_atoms=args.min_atoms, max_atoms=args.max_atoms,
                                            max_replacements=args.max_replacements, sample_func=sample_func,
                                            filter_func=filter_func, tautomerize=args.tautomerize,
                                            n_iterations=args.n_iterations)
            make_docking = True

            if not res:
                break

    except Exception as e:
        logging.exception(e, stack_info=True)

    finally:
        logging.info(f'{iteration} iterations were completed')


if __name__ == '__main__':
    entry_point()
