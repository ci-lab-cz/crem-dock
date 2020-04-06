from os.path import join, exists
from os import makedirs
from functools import partial
import argparse
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem


def save_to_pdb(smi, fname):
    '''
    Convert smi to PDB and save pathdir/id_frag.pdb
    :param smi: str(smiles)
    :param fname: str(filename)
    :return: None
    '''
    # convert to 3D coord
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        # compute coord
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        mol = Chem.MolToPDBBlock(mol)
        # print(id_frag, 'Done')
    except:
        print('Problem convertation of', smi)
        return None

    with open(fname, 'wt') as pdb:
        pdb.write(mol)


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
