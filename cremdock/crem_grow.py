import logging

import numpy as np
from crem.crem import grow_mol
from rdkit import Chem, rdBase
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from scipy.spatial import distance_matrix

from cremdock.auxiliary import calc_rtb
from cremdock.molecules import neutralize_atoms
from cremdock.user_protected_atoms import get_atom_idxs_for_canon


def get_protected_ids(mol, protein_xyz, dist_threshold):
    """
    Returns list of ids of heavy atoms ids which have ALL hydrogen atoms close to the protein
    :param mol: molecule
    :param protein_xyz: coordinates of heavy atoms of a protein
    :param dist_threshold: minimum distance to hydrogen atoms
    :return:
    """
    hids = np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1])
    xyz = mol.GetConformer().GetPositions()[hids]
    min_xyz = xyz.min(axis=0) - dist_threshold
    max_xyz = xyz.max(axis=0) + dist_threshold
    # select protein atoms which are within a box of min-max coordinates of ligand hydrogen atoms
    pids = (protein_xyz >= min_xyz).any(axis=1) & (protein_xyz <= max_xyz).any(axis=1)
    pxyz = protein_xyz[pids]
    m = distance_matrix(xyz, pxyz)  # get matrix (ligandH x protein)
    ids = set(hids[(m <= dist_threshold).any(axis=1)].tolist())  # ids of H atoms close to a protein

    output_ids = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() > 1:
            # all hydrogens of a heavy atom are close to protein
            if not (set(n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() == 1) - ids):
                output_ids.append(a.GetIdx())

    return output_ids


def __get_protein_heavy_atom_xyz(protein):
    """
    Returns coordinates of heavy atoms
    :param protein: protein file (pdb or pdbqt), explicit hydrogens are not necessary
    :return: 2d_array (n_atoms x 3)
    """
    with open(protein) as f:
        pdb_block = f.read()
    return get_protein_heavy_atoms_xyz_from_string(pdb_block)


def get_protein_heavy_atoms_xyz_from_string(pdb_block):
    protein = Chem.MolFromPDBBlock('\n'.join([line[:66] for line in pdb_block.split('\n')]), sanitize=False)
    if protein is None:
        raise ValueError("Protein structure is incorrect. Please check protein pdbqt file.")
    xyz = protein.GetConformer().GetPositions()
    xyz = xyz[[a.GetAtomicNum() > 1 for a in protein.GetAtoms()], ]
    return xyz


def grow_mol_crem(mol, protein_xyz, max_mw, max_rtb, max_logp, max_tpsa, h_dist_threshold=2, ncpu=1, **kwargs):
    mol_0 = neutralize_atoms(Chem.RemoveHs(mol))  # add neutralize_atoms to calc correct logp and tpsa
    mw = max_mw - Chem.Descriptors.MolWt(mol_0)
    if mw <= 0:
        return []
    rtb = max_rtb - calc_rtb(mol_0) - 1  # it is necessary to take into account the formation of bonds during the growth of the molecule
    if rtb == -1:
        rtb = 0
    logp = max_logp - MolLogP(mol_0) + 0.5
    tpsa = max_tpsa - CalcTPSA(mol_0)

    mol = Chem.AddHs(mol, addCoords=True)
    _protected_user_ids = set()
    if mol.HasProp('protected_user_canon_ids'):
        _protected_user_ids = set(
            get_atom_idxs_for_canon(mol, list(map(int, mol.GetProp('protected_user_canon_ids').split(',')))))
    _protected_alg_ids = set(get_protected_ids(mol, protein_xyz, h_dist_threshold))
    protected_ids = _protected_alg_ids | _protected_user_ids

    # remove explicit hydrogen and charges and redefine protected atom ids
    for i in protected_ids:
        mol.GetAtomWithIdx(i).SetIntProp('__tmp', 1)
    mol = neutralize_atoms(Chem.RemoveHs(mol))
    protected_ids = []
    for a in mol.GetAtoms():
        if a.HasProp('__tmp') and a.GetIntProp('__tmp'):
            protected_ids.append(a.GetIdx())
            a.ClearProp('__tmp')

    blocker = rdBase.BlockLogs()  # suppress CReM warnings, https://github.com/rdkit/rdkit/issues/2683
    try:
        res = list(grow_mol(mol, protected_ids=protected_ids, return_rxn=False, return_mol=True, ncores=ncpu,
                            symmetry_fixes=True, mw=(1, mw), rtb=(0, rtb), logp=(-100, logp), tpsa=(0, tpsa), **kwargs))

    except Exception as e:
        logging.error(f'grow error, {mol.GetProp("_Name")} {Chem.MolToSmiles(mol)}, {e}',
                      stack_info=True, exc_info=True)
        res = []

    res = tuple(m for smi, m in res)

    return res


def grow_mols_crem(mols, protein_xyz, max_mw, max_rtb, max_logp, max_tpsa, h_dist_threshold=2, ncpu=1, **kwargs):
    """

    :param mols: list of molecules
    :param protein_xyz: 2D array of heavy atoms coordinates
    :param max_mw:
    :param max_rtb:
    :param max_logp:
    :param max_tpsa:
    :param h_dist_threshold: maximum distance from H atoms to the protein to mark them as protected from further grow
    :param ncpu: number of cpu
    :param kwargs: arguments passed to crem function grow_mol
    :return: dict of parent mols and lists of corresponding generated mols
    """
    res = dict()
    for mol in mols:
        tmp = grow_mol_crem(mol, protein_xyz, max_mw=max_mw, max_rtb=max_rtb, max_logp=max_logp, max_tpsa=max_tpsa,
                            h_dist_threshold=h_dist_threshold, ncpu=ncpu, **kwargs)
        if tmp:
            res[mol] = tmp
    return res