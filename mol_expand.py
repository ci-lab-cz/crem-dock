#!/usr/bin/env python

import argparse
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from scipy.spatial.distance import cdist
from multiprocessing import cpu_count
from crem.crem import grow_mol
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


def cpu_type(x):
    return max(1, min(int(x), cpu_count()))


def get_mols(pdb_fname, lig_id, lig_smi):
    lig_lines = []
    prot_lines = []
    with open(pdb_fname) as f:
        for line in f:
            if line[17:20] == 'HOH' or line.startswith('CONECT') or (line.startswith('HETATM') and line[17:20] != lig_id):
                continue
            if (line.startswith('ATOM') or line.startswith('HETATM')) and line[17:20] == lig_id:
                lig_lines.append(line)
            else:
                prot_lines.append(line)

    prot = Chem.MolFromPDBBlock(''.join(prot_lines), sanitize=False)

    lig = Chem.MolFromPDBBlock(''.join(lig_lines))
    lig = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(lig_smi), lig)
    lig = Chem.AddHs(lig, addCoords=True)
    return prot, lig


def get_protected_ids(prot, lig, threshold):
    prot_xyz = prot.GetConformer().GetPositions()
    lig_xyz = [(a.GetIdx(), xyz) for a, xyz in zip(lig.GetAtoms(), lig.GetConformer().GetPositions()) if a.GetAtomicNum() == 1]
    min_dist = np.min(cdist(prot_xyz, np.array([c for i, c in lig_xyz])), axis=0)
    protected_ids = np.argwhere(min_dist <= threshold).flatten().tolist()
    return protected_ids if protected_ids else None


def expand_mol(lig, db_fname, protected_ids, ncpu):
    stereo_opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)
    new_smi = []
    for new in grow_mol(lig, db_name=db_fname, min_atoms=1, max_atoms=10, protected_ids=protected_ids,
                        return_rxn=False, return_mol=True, max_replacements=None, ncores=ncpu):
        for b in new[1].GetBonds():
            if b.GetStereo() == Chem.rdchem.BondStereo.STEREOANY:
                b.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
        isomers = tuple(EnumerateStereoisomers(new[1], options=stereo_opts))
        for isomer in isomers:
            new_smi.append(Chem.MolToSmiles(Chem.RemoveHs(isomer), isomericSmiles=True))
    return new_smi


def main():
    parser = argparse.ArgumentParser(description='Grow molecule by replacement of replaceable hydrogens.')
    parser.add_argument('-i', '--input', metavar='input.pdb', required=True, type=str,
                        help='PDB file with a protein-ligand complex.')
    parser.add_argument('-s', '--lig_smi', metavar='SMILES', required=True, type=str,
                        help='file with SMILES or a SMILES string of a ligand to correctly parse its '
                             'connectivity in PDB.')
    parser.add_argument('-d', '--db', metavar='fragments.db', required=True, type=str,
                        help='SQLite DB file with fragment replacements.')
    parser.add_argument('-g', '--lig_id', metavar='STRING', required=True, type=str,
                        help='ID of a ligand in the input file. All other atoms will be considered as protein side.')
    parser.add_argument('-o', '--output', metavar='output.smi', required=True, type=str,
                        help='output file with SMILES of newly generated molecules.')
    parser.add_argument('-l', '--lig_output', metavar='output.mol', required=False, default=None, type=str,
                        help='output file with 3D coordinates of a parent ligand.')
    parser.add_argument('-t', '--threshold', metavar='NUMERIC', required=False, default=2, type=float,
                        help='minimum distance from hydrogen atom to protein atoms to protect the hydrogen '
                             'from replacement.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, default=1, type=cpu_type,
                        help='number of cores.')

    args = parser.parse_args()
    if os.path.isfile(args.lig_smi):
        lig_smi = open(args.lig_smi).readline().strip().split()[0]
    else:
        lig_smi = args.lig_smi
    prot, lig = get_mols(args.input, args.lig_id, lig_smi)
    protected_ids = get_protected_ids(prot, lig, args.threshold)
    new_smi = expand_mol(lig, args.db, protected_ids, args.ncpu)

    with open(args.output, 'wt') as f:
        for i, smi in enumerate(new_smi, 1):
            f.write(f'{smi}\t{args.lig_id}-{str(i).zfill(5)}\n')

    if args.lig_output:
        Chem.MolToMolFile(lig, args.lig_output)


if __name__ == '__main__':
    main()
