#!/usr/bin/env python

import argparse
import glob
import os
import re
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def read_pdbqt(fname, mol):
    with open(fname) as f:
        pdb_block = f.read().split('MODEL ')[1]
        m = Chem.MolFromPDBBlock('\n'.join([i[:66] for i in pdb_block.split('\n')]), removeHs=True)
        if m:
            try:
                m = AllChem.AssignBondOrdersFromTemplate(mol, m)
            except Exception as e:
                sys.stdout.write('failed to AssignBondOrdersFromTemplate')
                sys.stdout.write(str(e))
                m = None
    with open(fname) as f:
        f.readline()
        line = f.readline()
        if line.startswith('REMARK minimizedAffinity'):
            score = float(line.strip().split()[2])   # smina ad4 score
        elif line.startswith('REMARK VINA RESULT:'):
            score = float(line.strip().split()[3])   # vina score
        else:
            score = float('inf')
    return m, score


def get_rmsd(mol, ref):
    mol = Chem.RemoveHs(mol)
    ref = Chem.RemoveHs(ref)
    mol_xyz = mol.GetConformer().GetPositions()
    ref_xyz = ref.GetConformer().GetPositions()
    rms = float('inf')
    for ids in mol.GetSubstructMatches(ref):
        res = np.sqrt(np.mean(np.sum((mol_xyz[ids, ] - ref_xyz) ** 2, axis=1)))
        print(ids, res)
        if res < rms:
            rms = res
        # (mol_xyz[ids,:] - ref_xyz) ** 2
    return round(rms, 2)


def main():
    parser = argparse.ArgumentParser(description='Extract docking scores from the list of pdbqt files and calculate '
                                                 'their rmsd relative to the parent molecule.')
    parser.add_argument('-i', '--input', metavar='input.mol', required=True, type=str,
                        help='file with 3D coordinates of a parent ligand.')
    parser.add_argument('-d', '--dock_dir', metavar='DIRNAME', required=True, type=str,
                        help='name of the directory containing docking results')
    parser.add_argument('-s', '--suffix', metavar='STRING', required=False, type=str, default='_dock.pdbqt',
                        help='constant suffix of compounds with docked poses. Default: _dock.pdbqt.')
    parser.add_argument('-m', '--smiles', metavar='docking.smi', required=True, type=str,
                        help='SMILES of docked compounds with names. Tab-separated file.')
    parser.add_argument('-o', '--output', metavar='output.txt', required=True, type=str,
                        help='text file with docking scores and rmsd relative to parent compound')

    args = parser.parse_args()
    lig = Chem.MolFromMolFile(args.input)

    dock_mols = {}
    with open(args.smiles) as f:
        for line in f:
            smi, name = line.strip().split()[:2]
            dock_mols[name] = Chem.MolFromSmiles(smi)

    with open(args.output, 'wt') as f:
        for fname in glob.glob(os.path.join(args.dock_dir, '*' + args.suffix)):
            mol_name = re.sub('^.*/(.*)' + args.suffix, '\\1', fname)
            mol, score = read_pdbqt(fname, dock_mols[mol_name])
            if mol is None:
                print(f'{mol_name} cannot be parsed from PDBQT file')
                continue
            rms = get_rmsd(mol, lig)
            f.write('\t'.join(map(str, (mol_name, score, rms))) + '\n')


if __name__ == '__main__':
    main()
