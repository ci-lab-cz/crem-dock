import numpy as np
from rdkit import Chem


def assign_protected_ids(mol_dict, prop_name='protected_crem'):
    """
    assign boolean atom property 'protected_crem' in child mols according to a parent mol
    :param mol_dict: {parent_mol: [child_mol1, child_mol2, ...], ...}
    :param prop_name:
    :return:
    """
    for parent_mol, mols in mol_dict.items():
        new_mols = list()
        for m in mols:
            if parent_mol.HasProp('protected_user_canon_ids'):
                parent_protected_user_ids = get_atom_idxs_for_canon(parent_mol, list(
                    map(int, parent_mol.GetProp('protected_user_canon_ids').split(','))))
                child_protected_user_ids = set(get_child_protected_atom_ids(m, parent_protected_user_ids))
                for a in m.GetAtoms():
                    a.SetBoolProp(prop_name, True if a.GetIdx() in child_protected_user_ids else False)
            new_mols.append(m)
        mol_dict[parent_mol] = new_mols
    return mol_dict


def set_isotope_to_parent_protected_atoms(mol_dict, prop_name='protected_crem', isotope=13):
    """

    :param mol_dict: {parent_mol: [child_mol1, child_mol2, ...], ...}
    :param prop_name:
    :param isotope:
    :return:
    """
    for parent_mol, mols in mol_dict.items():
        if parent_mol.HasProp('protected_user_canon_ids'):
            for m in mols:
                for atom in m.GetAtoms():
                    if atom.GetBoolProp(prop_name):
                        atom.SetIsotope(isotope)
    return mol_dict


def assign_protected_ids_from_isotope(mol_dict, prop_name='protected_crem', isotope=13):
    """
    remove isotope label and assign boolean atom property 'protected_crem' True for labeled atoms, otherwise False
    :param mol_dict: {parent_mol: [child_mol1, child_mol2, ...], ...}
    :param prop_name:
    :param isotope:
    :return:
    """
    for parent_mol, mols in mol_dict.items():
        if parent_mol.HasProp('protected_user_canon_ids'):
            for m in mols:
                for atom in m.GetAtoms():
                    if atom.GetIsotope() == isotope:
                        atom.SetBoolProp(prop_name, True)
                        atom.SetIsotope(0)
                    else:
                        atom.SetBoolProp(prop_name, False)
    return mol_dict


def get_atom_idxs_for_canon(mol, canon_idxs):
    """
    get the rdkit current indices for the canonical indices of the molecule
    :param mol:
    :param canon_idxs: list[int]
    :return: sorted list of integers
    """
    canon_ranks = np.array(Chem.CanonicalRankAtoms(mol))
    return sorted(np.where(np.isin(canon_ranks, canon_idxs))[0].tolist())


def get_canon_for_atom_idx(mol, idx):
    """
    get the canonical numeration of the current molecule indices
    :param mol:
    :param idx: list[int]
    :return: sorted list of integers
    """
    canon_ranks = np.array(Chem.CanonicalRankAtoms(mol))
    return sorted(canon_ranks[idx].tolist())


def get_child_protected_atom_ids(mol, protected_parent_ids):
    """

    :param mol:
    :param protected_parent_ids: list[int]
    :type  protected_parent_ids: list[int]
    :return: sorted list of integers
    """
    # After RDKit reaction procedure there is a field <react_atom_idx> with initial parent atom idx in product mol
    protected_product_ids = []
    for a in mol.GetAtoms():
        if a.HasProp('react_atom_idx') and int(a.GetProp('react_atom_idx')) in protected_parent_ids:
            protected_product_ids.append(a.GetIdx())
    return sorted(protected_product_ids)


def get_protected_canon_ids(mol, prop_name='protected_crem'):
    """
    get canonical ids of labeled atoms for hydrogenized molecule
    :param mol:
    :param prop_name:
    :return:
    """
    output = list()
    mol = Chem.AddHs(mol)
    for i, atom in zip(Chem.CanonicalRankAtoms(mol), mol.GetAtoms()):
        if atom.HasProp(prop_name) and atom.GetBoolProp(prop_name):
            output.append(i)
    return sorted(output)
