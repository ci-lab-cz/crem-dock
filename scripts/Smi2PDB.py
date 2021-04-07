from os.path import join, exists
from os import makedirs
from functools import partial
import argparse
from sys import exc_info
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem


def save_to_pdb(smi, fname):
    '''
    Convert smi to PDB and save to file
    :param smi: smiles
    :param fname: filename
    :return: None
    '''
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        mol = Chem.AddHs(mol)
        try:
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if res == 0:
                pdb = Chem.MolToPDBBlock(mol)
                with open(fname, 'wt') as f:
                    f.write(pdb)
        except:
            print('Embedding problem of', smi)
    else:
        print('Conversion problem of', smi)


def save_to_pdb_mp(items):
    return save_to_pdb(*items)


def save_to_pdb2(child_mol, parent_mol, fname):
    '''
    Convert smi to PDB and save pathdir/id_frag.pdb
    :param smi: str(smiles)
    :param fname: str(filename)
    :return: None
    '''
    # convert to 3D coord
    try:
        mol = AllChem.ConstrainedEmbed(Chem.AddHs(child_mol), parent_mol)
        mol = Chem.MolToPDBBlock(mol)
    except ValueError as e:
        print("Unexpected error1:", e)
        try:
            mol = AllChem.ConstrainedEmbed(child_mol, parent_mol)
            mol = Chem.AddHs(mol, addCoords=True)
            mol = Chem.MolToPDBBlock(mol)
        except ValueError as e:
            print("Unexpected error2:", e)
            return None
    with open(fname, 'wt') as pdb:
        pdb.write(mol)
        print('Done')


def save_to_pdb2_mp(items):
    return save_to_pdb2(*items)


def main(input_fname, output_dname, ncpu):
    ncpu = min(cpu_count(), ncpu)
    p = Pool(ncpu)

    with open(input_fname) as data:
        smi_list = data.readlines()

    p.map(partial(save_to_pdb, path_to_save=output_dname),
          ([id_smi, smi.strip()] for id_smi, smi in enumerate(smi_list)))

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DockFragment compounds SmiToPDBconverter .')
    parser.add_argument('-i', '--input', metavar='input.smi', required=True,
                        help='input smi file')
    parser.add_argument('-o', '--output', metavar='Dir', required=False, default=None,
                        help='output dir for saving file pdb')
    parser.add_argument('-c', '--ncpu', metavar='NUMBER', type=int, required=False, default=1,
                        help='number of cpus used for computation')

    args = parser.parse_args()

    input_fname = args.input
    output_dname = args.output
    ncpu = args.ncpu

    main(input_fname=input_fname,
         output_dname=output_dname,
         ncpu=ncpu
         )
