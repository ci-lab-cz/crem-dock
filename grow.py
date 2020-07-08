import glob
import os
import shutil
import argparse
import operator
import re
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from rdkit.Chem.Descriptors import MolWt
from scipy.spatial.distance import euclidean
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from scripts import Docking, Smi2PDB
from crem.crem import grow_mol
import sqlite3
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


def cpu_type(x):
    return max(1, min(int(x), cpu_count()))


def __get_first_mol_from_pdbqt(fname: str) -> Chem.Mol:
    '''
    :param fname:
    :return: list of rdkit MOL from pdbqt
    '''

    def create_mol(fname):
        with open(fname.split('_dock')[0] + '.pdb') as f:
            f1 = f.read()
            m = Chem.MolFromPDBBlock(f1, removeHs=False)

        with open(fname) as f:
            pdb_block = f.read().split('MODEL ')[1]
            mvina = Chem.MolFromPDBBlock('\n'.join([i[:66] for i in pdb_block.split('\n')]), removeHs=False)
        Editable_m = Chem.EditableMol(m)
        for bond in m.GetBonds():
            ind_1, ind_2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            Editable_m.RemoveBond(ind_1, ind_2)
            Editable_m.AddBond(ind_1, ind_2, order=Chem.rdchem.BondType.SINGLE)
        m_new = Editable_m.GetMol()
        match_ids = m_new.GetSubstructMatches(mvina)
        print(match_ids)
        for mvina_atom_id, m_atom_id in enumerate(match_ids[0]):
            pos = mvina.GetConformer().GetAtomPosition(mvina_atom_id)
            m.GetConformer().SetAtomPosition(m_atom_id, pos)

        ind_list = []
        for atom in m.GetAtoms():
            idx = atom.GetIdx()
            if idx not in match_ids[0]:
                ind_list.append(idx)

        for x in sorted(ind_list, reverse=True):
            nbrs = m.GetAtomWithIdx(x).GetNeighbors()
            if len(nbrs) == 1 and nbrs[0].GetAtomicNum() > 1:
                nbrs[0].SetNumExplicitHs(nbrs[0].GetNumExplicitHs() + 1)

        Editable_m_H = Chem.EditableMol(m)
        for i in sorted(ind_list, reverse=True):
            Editable_m_H.RemoveAtom(i)
        mm = Editable_m_H.GetMol()
        m_H = Chem.AddHs(mm, addCoords=True)
        print(Chem.MolToSmiles(m_H))
        m_H.UpdatePropertyCache()
        Chem.GetSymmSSSR(m_H)
        m_H.GetRingInfo().NumRings()
        return m_H

    #     with open(fname) as f:
    #         pdb_block = f.read().split('MODEL ')[1]
    mol = create_mol(fname)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    #         print(Chem.MolToSmiles(mol, isomericSmiles=True))
    return mol


def convert_smi_to_pdb(smi_fname, output_dname):
    with open(smi_fname) as f:
        for i, line in enumerate(f):
            tmp = line.strip().split()
            smi = tmp[0]
            name = tmp[1] if len(tmp) > 1 else str(i)
            Smi2PDB.save_to_pdb(smi, os.path.join(output_dname, name + '.pdb'))


def convert_smi_to_pdb2(smi_data, output_dname):
    """

    :param smi_data: dict {mol_id: smi}
    :param output_dname:
    :return:
    """
    for mol_id, smi in smi_data.items():
        Smi2PDB.save_to_pdb(smi, os.path.join(output_dname, mol_id + '.pdb'))


def prep_ligands(dname, python_path, vina_script_dir, ncpu):

    def supply_lig_prep(dname, python_path, vina_script_dir):
        for fname in glob.glob(os.path.join(dname, '*.pdb')):
            yield fname, fname.rsplit('.', 1)[0] + '.pdbqt', python_path, vina_script_dir + 'prepare_ligand4.py'

    Parallel(n_jobs=ncpu, verbose=8)(delayed(Docking.prepare_ligand)(i_fname, o_fname, python_path, vina_script)
                                     for i_fname, o_fname, python_path, vina_script in supply_lig_prep(dname,
                                                                                                       python_path,
                                                                                                       vina_script_dir))


def dock_ligands(dname, target_fname_pdbqt, target_setup_fname, vina_path, ncpu):

    def supply_lig_dock_data(dname, target_fname_pdbqt, target_setup_fname):
        for fname in glob.glob(os.path.join(dname, '*.pdbqt')):
            yield fname, fname.rsplit('.', 1)[0] + '_dock.pdbqt', target_fname_pdbqt, target_setup_fname, vina_path

    Parallel(n_jobs=ncpu, verbose=8)(delayed(Docking.run_docking)(i_fname, o_fname, target, setup, vina_script)
                                     for i_fname, o_fname, target, setup, vina_script in supply_lig_dock_data(dname,
                                                                                                              target_fname_pdbqt,
                                                                                                              target_setup_fname))


def get_mol_scores(dname):
    # return dict {mol_id: score}
    d = {}
    for fname in glob.glob(os.path.join(dname, '*_dock.pdbqt')):
        print(fname)
        with open(fname) as f:
            score = float(f.read().split()[5])
        d[re.sub('^.*/(.*)_dock.pdbqt', '\\1', fname)] = score
    return d


def select_mols(mol_score_dict, smi_dict, ntop, mw):
    sel = tuple((score, mol_id) for (mol_id, score) in mol_score_dict.items() if MolWt(Chem.MolFromSmiles(smi_dict[mol_id])) < mw)
    sel = sorted(sel)[:ntop]
    return tuple(mol_id for score, mol_id in sel)


def select_mols_from_clust(all_dict, d, ntop, mw):
    res = []
    for i in all_dict:
        s = select_mols(i, d, ntop, mw)
        res.append(s)
    r = tuple(y for x in res for y in x)
    return r   # return tuple of molecules


def sort_clusters(mol_score_dict, smi_dict, mw):
    sel = tuple((score, mol_id) for (mol_id, score) in mol_score_dict.items() if MolWt(Chem.MolFromSmiles(smi_dict[mol_id])) < mw)
    sel = sorted(sel)
    sel = tuple(mol_id for score, mol_id in sel)
    return sel


def select_clust(all_dict, d, mw):
    res = []
    for i in all_dict:
        s = sort_clusters(i, d, mw)
        res.append(s)
    return res


def get_simple_dict(smi_fname):
    with open(smi_fname) as f:
        d = {}
        for i, line in enumerate(f):
            tmp = line.strip().split()
            smi = tmp[0]
            name = tmp[1] if len(tmp) > 1 else '000-' + str(i).zfill(6)
            d[name] = smi
    return d


def gen_cluster_subset_algButina(smi_fname, index_tanimoto, mol_scores):
    with open(smi_fname) as f:
        dict_index = {}
        fps = []
        for i, line in enumerate(f):
            tmp = line.strip().split()
            smi = tmp[0]
            name = tmp[1] if len(tmp) > 1 else '000-' + str(i).zfill(6)
            mol = Chem.MolFromSmiles(smi)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            fps.append(fingerprint)
            dict_index[i] = name
    dists = []
    for i, fp in enumerate(fps):
        distance_matrix = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in distance_matrix])
    cs = Butina.ClusterData(dists, len(fps), index_tanimoto, isDistData=True)  # returns  tuple of tuples with sequential numbers of compounds in each cluster
    al = []
    for i in cs:
        data = []
        for number in i:
            data.append(dict_index[number])
        al.append(data)
    all_dict = []
    for i in al:
        dict_scores = {}
        for j in i:
            try:
                dict_scores[j] = mol_scores[j]
            except:
                continue  #
        all_dict.append(dict_scores)
    return all_dict


def get_protected_ids(best_conf_H, protein_file, dist_threshold):
    pdb_block = open(protein_file).readlines()
    protein = Chem.MolFromPDBBlock('\n'.join([line[:66] for line in pdb_block]))
    protein_cord = protein.GetConformer().GetPositions()
    ids = set()
    for atom, cord in zip(best_conf_H.GetAtoms(), best_conf_H.GetConformer().GetPositions()):
        b = False
        if atom.GetAtomicNum() == 1:
            for i in protein_cord:
                if euclidean(cord, i) <= dist_threshold:
                    b = True
                    break
            if b:
                ids.add(atom.GetIdx())

    return sorted(ids)


def grow_mols(fnames, target_fname_pdbqt, h_dist_threshold=2, **kwargs):

    new_mols = dict()
    for fname in fnames:

        parent_id = re.sub('^.*/(.*)_dock.pdbqt', '\\1', fname)
        best_conf_H = __get_first_mol_from_pdbqt(fname)
        protected_ids = get_protected_ids(best_conf_H, target_fname_pdbqt, h_dist_threshold)

        for smi in grow_mol(best_conf_H, protected_ids=protected_ids, return_rxn=False, **kwargs):
            if smi not in new_mols:
                new_mols[smi] = parent_id

    return sorted(new_mols.items(), key=operator.itemgetter(1))


def grow_mols_deep(all_clust, dname, target_fname_pdbqt, ntop, h_dist_threshold=2, **kwargs):

    new_mols = dict()
    for number, clust in enumerate(all_clust):
        # print(number)
        n_selected = 0
        for num, mol_id in enumerate(clust):
            # print(number, num, mol_id)
            parent_id = mol_id
            fn = os.path.join(dname, mol_id + '_dock.pdbqt')
            best_conf_H = __get_first_mol_from_pdbqt(fn)
            protected_ids = get_protected_ids(best_conf_H, target_fname_pdbqt, h_dist_threshold)
            new_molecules = list(grow_mol(best_conf_H, protected_ids=protected_ids, return_rxn=False, **kwargs))
            # print(new_molecules)
            if new_molecules:
                n_selected += 1
                for smi in new_molecules:
                    if smi not in new_mols:
                        new_mols[smi] = parent_id
                if n_selected == ntop:
                    break

    return sorted(new_mols.items(), key=operator.itemgetter(1))


def update_res_db(conn, values):
    """

    :param conn:
    :param values: dict(id: score)
    :return:
    """
    cur = conn.cursor()
    cur.executemany("""UPDATE mols
                       SET docking_score = ? 
                       WHERE
                           id = ?
                    """, [(v, k) for k, v in values.items()])
    conn.commit()


def insert_db(conn, data):
    cur = conn.cursor()
    cur.executemany("""INSERT INTO mols VAlUES(?, ?, ?, ?, ?)""", data)
    conn.commit()


def create_db(conn):
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS mols")
    cur.execute("""CREATE TABLE IF NOT EXISTS mols
            (
             id TEXT PRIMARY KEY,
             iteration INTEGER,
             smi TEXT,
             parent_id TEXT,
             docking_score REAL
            )""")
    conn.commit()


def get_smi_data(conn, iteration):
    """

    :param conn:
    :param iteration:
    :return: dict {mol_id: smi}
    """
    cur = conn.cursor()
    cur.execute("SELECT id, smi FROM mols WHERE iteration = %i" % iteration)
    return dict(cur.fetchall())


def insert_starting_smi_to_db(smi_fname, conn):
    with open(smi_fname) as f:
        data = []
        for i, line in enumerate(f):
            tmp = line.strip().split()
            smi = tmp[0]
            name = tmp[1] if len(tmp) > 1 else '000-' + str(i).zfill(6)
            data.append((name, 0, smi, None, None))
        insert_db(conn, data)


def selection_grow_greedy(mol_scores, smi_data, ntop, mw, target_fname_pdbqt, dname, **kwargs):
    selected_mols = select_mols(mol_scores, smi_data, ntop, mw)
    res = None
    if selected_mols:
        selected_mol_fnames = [os.path.join(dname, mol_id + '_dock.pdbqt') for mol_id in selected_mols]
        res = grow_mols(selected_mol_fnames, target_fname_pdbqt, **kwargs)
    return res


def selection_grow_clust(input_smi_fname, index_tanimoto, mol_scores, ntop, mw, target_fname_pdbqt, dname, **kwargs):
    d = get_simple_dict(input_smi_fname)
    clusters = gen_cluster_subset_algButina(input_smi_fname, index_tanimoto, mol_scores)
    selected_mols = select_mols_from_clust(clusters, d, ntop, mw)
    res = None
    if selected_mols:
        selected_mol_fnames = [os.path.join(dname, mol_id + '_dock.pdbqt') for mol_id in selected_mols]
        res = grow_mols(selected_mol_fnames, target_fname_pdbqt, **kwargs)
    return res


def selection_grow_clust_deep(input_smi_fname, index_tanimoto, mol_scores, dname, target_fname_pdbqt, ntop,
                              mw, **kwargs):
    d = get_simple_dict(input_smi_fname)
    clusters = gen_cluster_subset_algButina(input_smi_fname, index_tanimoto, mol_scores)
    sorted_clust = select_clust(clusters, d, mw)
    res = None
    if sorted_clust:
        res = grow_mols_deep(sorted_clust, dname, target_fname_pdbqt, ntop, **kwargs)
    return res


def make_iteration(input_smi_fname, output_smi_fname, iteration, target_fname_pdbqt, output_db_connection,
                   ntop, alg_type, docking_dir, protein_setup, ncpu, vina_path, python_path, vina_script_dir,
                   index_tanimoto=None, mw=500, **kwargs):

    dname = os.path.dirname(input_smi_fname)

    smi_data = get_smi_data(output_db_connection, iteration - 1)
    if iteration == 1 and docking_dir is not None:
        all_files = os.listdir(docking_dir)
        for file in all_files:
            shutil.copy(os.path.join(docking_dir, file), os.path.join(dname, file))
    else:
        convert_smi_to_pdb2(smi_data, dname)
        prep_ligands(dname, python_path, vina_script_dir, ncpu)
        dock_ligands(dname, target_fname_pdbqt, protein_setup, vina_path, ncpu)

    mol_scores = get_mol_scores(dname)
    update_res_db(output_db_connection, mol_scores)

    if alg_type == 1:
        res = selection_grow_greedy(mol_scores=mol_scores, smi_data=smi_data, ntop=ntop, mw=mw,
                                    target_fname_pdbqt=target_fname_pdbqt, dname=dname, ncores=ncpu, **kwargs)
    elif alg_type == 2:
        res = selection_grow_clust_deep(input_smi_fname=input_smi_fname, index_tanimoto=index_tanimoto,
                                        mol_scores=mol_scores, dname=dname, mw=mw,
                                        target_fname_pdbqt=target_fname_pdbqt, ntop=ntop, nncores=ncpu, **kwargs)
    elif alg_type == 3:
        res = selection_grow_clust(input_smi_fname=input_smi_fname, index_tanimoto=index_tanimoto,
                                   mol_scores=mol_scores, ntop=ntop, mw=mw, target_fname_pdbqt=target_fname_pdbqt,
                                   dname=dname, ncores=ncpu, **kwargs)
    else:
        res = []

    if res:

        data = []
        opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)
        for i, (smi, parent_id) in enumerate(res):
            m = Chem.MolFromSmiles(smi)
            isomers = tuple(EnumerateStereoisomers(m, options=opts))

            for st_numb, st_mol in enumerate(isomers):
                st_smi = Chem.MolToSmiles(st_mol, isomericSmiles=True)
                mol_id = str(iteration).zfill(3) + '-' + str(i).zfill(6) + '-' + str(st_numb).zfill(2)
                data.append((mol_id, iteration, st_smi, parent_id, None))
        insert_db(output_db_connection, data)

        with open(output_smi_fname, 'wt') as f:
            for item in data:
                f.write('%s\t%s\n' % (item[2], item[0]))
        return True

    else:
        return False


def main():
    parser = argparse.ArgumentParser(description='Fragment growing within binding pocket with Autodock Vina.')
    parser.add_argument('-i', '--input_frags', metavar='frags.smi', required=True,
                        help='SMILES file with input fragments.')
    parser.add_argument('-d', '--db', metavar='fragments.db', required=True,
                        help='SQLite DB file with fragment replacements.')
    parser.add_argument('-p', '--protein', metavar='protein.pdbqt', required=True,
                        help='input PDBQT file with a prepared protein.')
    parser.add_argument('-s', '--protein_setup', metavar='protein.log', required=True,
                        help='input text file with docking setup.')
    parser.add_argument('-o', '--output', metavar='DIR_NAME', required=True,
                        help='output directory. It should not exist.')
    parser.add_argument('--mgl_install_dir', metavar='DIR_NAME', required=True,
                        help='path to the dir with installed MGLtools.')
    parser.add_argument('--vina', metavar='vina_path', required=True,
                        help='path to the vina executable.')
    parser.add_argument('-n', '--ncpu', default=1, type=cpu_type,
                        help='number of cpus. Default: 1.')
    parser.add_argument('--docking_dir', metavar='DIR_NAME', required=False, default=None,
                        help='path to the dir with files after docking.')
    parser.add_argument('-r', '--radius', default=1, type=int,
                        help='context radius for replacement.')
    parser.add_argument('-m', '--min_freq', default=0, type=int,
                        help='the frequency of occurrence of the fragment in the source database.')
    parser.add_argument('-nt', '--ntop', type=int, required=False,
                        help='the number of the best molecules')
    parser.add_argument('--max_replacements', type=int, required=False, default=None,
                        help='the number of randomly chosen replacements')
    parser.add_argument('-t', '--algorithm', default=1, type=int,
                        help='the number of algorithm: 1 - greedy search, 2 - deep clustering, 3 - clustering.')

    args = parser.parse_args()

    #vina_path = './bin/autodock_vina_1_1_2_linux_x86/bin/vina'
    python_path = os.path.join(args.mgl_install_dir, 'bin/python')
    vina_script_dir = os.path.join(args.mgl_install_dir, 'MGLToolsPckgs/AutoDockTools/Utilities24/')

    if os.path.exists(args.output):
        raise ValueError('Output dir already exists. Please specify non-existing dir to store output.')

    os.makedirs(args.output)
    dname = os.path.join(args.output, 'iter_1')
    os.makedirs(dname)
    input_smi_fname = os.path.join(dname, 'input.smi')
    shutil.copyfile(args.input_frags, input_smi_fname)

    output_db = os.path.join(args.output, 'output.db')

    conn = sqlite3.connect(output_db)
    create_db(conn)

    iteration = 1

    insert_starting_smi_to_db(input_smi_fname, conn)

    while True:

        output_dname = os.path.abspath(os.path.join(os.path.dirname(input_smi_fname), '..', 'iter_%i' % (iteration + 1)))
        os.makedirs(output_dname)
        output_smi_fname = os.path.join(output_dname, 'input.smi')
        index_tanimoto = 0.9  # required for alg 2 and 3

        res = make_iteration(input_smi_fname=input_smi_fname, output_smi_fname=output_smi_fname,
                             iteration=iteration, target_fname_pdbqt=args.protein,
                             output_db_connection=conn, ntop=args.ntop, alg_type=args.algorithm,
                             index_tanimoto=index_tanimoto, mw=500, docking_dir=args.docking_dir,
                             protein_setup=args.protein_setup, ncpu=args.ncpu, vina_path=args.vina,
                             python_path=python_path, vina_script_dir=vina_script_dir, db_name=args.db,
                             radius=args.radius, min_freq=args.min_freq, min_atoms=1, max_atoms=10,
                             max_replacements=args.max_replacements)

        if res:
            iteration += 1
            input_smi_fname = output_smi_fname
            if args.algorithm in [2, 3]:
                index_tanimoto -= 0.05
        else:
            break


if __name__ == '__main__':

    main()
