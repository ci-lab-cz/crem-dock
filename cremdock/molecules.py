import os
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions, EnumerateStereoisomers


def get_isomers(mol):
    opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)
    # this is a workaround for rdkit issue - if a double bond has STEREOANY it will cause errors at
    # stereoisomer enumeration, we replace STEREOANY with STEREONONE in these cases
    try:
        isomers = tuple(EnumerateStereoisomers(mol, options=opts))
    except RuntimeError:
        for bond in mol[1].GetBonds():
            if bond.GetStereo() == Chem.BondStereo.STEREOANY:
                bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
        isomers = tuple(EnumerateStereoisomers(mol, options=opts))
    return isomers


def get_major_tautomer(mol_dict):
    """
    convert child molecules with parent mol names to smiles file, tautomerize and return the same data structure
    as input
    :param mol_dict: {parent_mol: [child_mol1, child_mol2, ...], ...}
    :return:
    """
    data = defaultdict(list)
    parent_mols = {mol.GetProp("_Name"): mol for mol in mol_dict.keys()}
    with tempfile.NamedTemporaryFile(suffix='.smi', mode='w', encoding='utf-8') as tmp:
        fd, output = tempfile.mkstemp()
        try:
            # remove H to avoid problem with kekulization. Ex: "Can't kekulize mol.  Unkekulized atoms: 4 5 7"
            smiles = [f'{Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)}\t{parent_mol.GetProp("_Name")}\n'
                      for parent_mol, mols in mol_dict.items() for mol in mols]
            tmp.writelines([''.join(smiles)])
            tmp.flush()
            cmd_run = ['cxcalc', '-S', '--ignore-error', 'majortautomer', '-f', 'smiles', '-a', 'false', tmp.name]
            with open(output, 'w') as file:
                subprocess.run(cmd_run, stdout=file, text=True)
            for mol in Chem.SDMolSupplier(output):
                if mol:
                    mol_name = mol.GetProp('_Name')
                    stable_tautomer_smi = mol.GetPropsAsDict().get('MAJOR_TAUTOMER', None)
                    if stable_tautomer_smi is not None:
                        data[parent_mols[mol_name]].append(Chem.MolFromSmiles(stable_tautomer_smi))
        finally:
            os.close(fd)
            os.remove(output)
    return data


def neutralize_atoms(mol):
    # https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def get_mol_ids(mols):
    return [mol.GetProp('_Name') for mol in mols]


def get_rmsd(child_mol, parent_mol):
    """
    Returns best fit rmsd between a common part of child and parent molecules taking into account symmetry of molecules
    :param child_mol: Mol
    :param parent_mol: Mol
    :return:
    """
    child_mol = neutralize_atoms(Chem.RemoveHs(child_mol))
    parent_mol = neutralize_atoms(Chem.RemoveHs(parent_mol))
    match_ids = child_mol.GetSubstructMatches(parent_mol, uniquify=False, useChirality=True)
    best_rms = float('inf')
    for ids in match_ids:
        diff = np.array(child_mol.GetConformer().GetPositions()[ids, ]) - np.array(
            parent_mol.GetConformer().GetPositions())
        rms = np.sqrt((diff ** 2).sum() / len(diff))
        if rms < best_rms:
            best_rms = rms
    return round(best_rms, 3)