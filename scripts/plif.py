import argparse
import pandas as pd
import prolif as plf
from rdkit import Chem, DataStructs
from arg_types import filepath_type


def filter_by_plif(mols, plif_ref, protein_fname, threshold=1):
    """
    Calculate Tversky similarity between reference plif and plif of molecules. For Tversky index alpha was set 1 and
    beta to 0. This means that all bits not present in the reference plif will be ignored. In the combination with
    cutoff value 1 this results in a strict presence of all reference contacts in tested molecules.
    May be changed in future.

    :param mols: list of Mol objects
    :param plif_ref: list of strings representing required protein-ligand interactions
    :param protein_fname: path to a PDB file with all hydrogens attached to calculate plif
    :param threshold: numeric cutoff value to pass molecules by plif similarity
    :return: list of Mol objects passed the filter
    """
    if len(mols) == 0:
        return []
    prot = plf.Molecule(Chem.MolFromPDBFile(protein_fname, removeHs=False, sanitize=False))
    fp = plf.Fingerprint()
    fp.run_from_iterable((plf.Molecule.from_rdkit(mol) for mol in mols), prot)   # danger, hope it will always keep the order of molecules
    df = fp.to_dataframe()
    df.columns = [''.join(item.strip().lower() for item in items[1:]) for items in df.columns]
    df.index = [mol.GetProp('_Name') for mol in mols]
    ref_df = pd.DataFrame(data={item: True for item in plif_ref}, index=['reference'])
    df = pd.concat([ref_df, df]).fillna(False)
    b = plf.to_bitvectors(df)
    res = DataStructs.BulkTverskySimilarity(b[0], b[1:], 1, 0)
    mols = [mol for mol, v in zip(mols, res) if v >= threshold]
    return mols


def plif_similarity(mol, plif_protein_fname, plif_ref_df):
    """
    Calculate Tversky similarity between reference plif and plif of molecules. For Tversky index alpha was set 1 and
    beta to 0. This means that all bits not present in the reference plif will be ignored.
    May be changed in future.

    :param mol: RDKit Mol
    :param plif_protein_fname: PDB file with a protein containing all hydrogens
    :param plif_ref_df: pandas.DataFrame of reference interactions (with a single row simplified header, dot-separated)
    :return:
    """
    plf_prot = plf.Molecule(Chem.MolFromPDBFile(plif_protein_fname, removeHs=False, sanitize=False))
    fp = plf.Fingerprint()
    fp.run_from_iterable([plf.Molecule.from_rdkit(mol)], plf_prot)   # danger, hope it will always keep the order of molecules
    df = fp.to_dataframe()
    df.columns = [''.join(item.strip().lower() for item in items[1:]) for items in df.columns]
    df = pd.concat([plif_ref_df, df]).fillna(False)
    b = plf.to_bitvectors(df)
    sim = DataStructs.TverskySimilarity(b[0], b[1], 1, 0)
    return mol.GetProp('_Name'), round(sim, 3)


def calc_plif(protein_fname, ligand_fname, sanitize_protein):
    """
    Calculate PLIF for multiple molecules in input SDF file.
    :param protein_fname: protein PDB
    :param ligand_fname: SDF with ligands
    :param sanitize_protein: (bool) whether or not sanitize protein structure
    :return: pandas DataFrame with
    """
    mols = [mol for mol in Chem.SDMolSupplier(ligand_fname, removeHs=False) if mol is not None]
    mol_names = [mol.GetProp('_Name') for mol in mols]
    plf_prot = plf.Molecule(Chem.MolFromPDBFile(protein_fname, removeHs=False, sanitize=sanitize_protein))
    fp = plf.Fingerprint()
    fp.run_from_iterable([plf.Molecule.from_rdkit(mol) for mol in mols], plf_prot)   # danger, hope it will always keep the order of molecules
    df = fp.to_dataframe()
    df.columns = [''.join(item.strip().lower() for item in items[1:]) for items in df.columns]
    df.index = mol_names
    return df


def main():
    parser = argparse.ArgumentParser(description='Calculate PLIF for an input protein and molecules.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--protein', metavar='FILENAME', required=True, type=filepath_type,
                        help='PDB file of a protein.')
    parser.add_argument('-l', '--ligands', metavar='FILENAME', required=True, type=filepath_type,
                        help='SDF file of ligands.')
    parser.add_argument('-x', '--no_protein_sanitization', action='store_true', default=False,
                        help='sanitize input protein molecule.')
    parser.add_argument('-o', '--output', metavar='FILENAME', required=True, type=filepath_type,
                        help='output file with compute PLIF.')

    args = parser.parse_args()
    df = calc_plif(protein_fname=args.protein,
                   ligand_fname=args.ligands,
                   sanitize_protein=not args.no_protein_sanitization)
    df.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()
