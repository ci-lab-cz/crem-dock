#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import argparse
import os
import sys

from openbabel import openbabel as ob

from meeko import MoleculePreparation
from meeko import obutils


def main(molecule_string, build_macrocycle=True, add_water=False,
         merge_hydrogen=True, add_hydrogen=False, pH_value=None, verbose=False, mol_format='SDF'):

    mol = obutils.load_molecule_from_string(molecule_string, molecule_format=mol_format)

    if pH_value is not None:
        mol.CorrectForPH(float(pH_value))

    if add_hydrogen:
        mol.AddHydrogens()
        charge_model = ob.OBChargeModel.FindType("Gasteiger")
        charge_model.ComputeCharges(mol)

    preparator = MoleculePreparation(merge_hydrogens=merge_hydrogen, macrocycle=build_macrocycle,
                                     hydrate=add_water, amide_rigid=True)
                                     #additional parametrs
                                     #rigidify_bonds_smarts=[], rigidify_bonds_indices=[])
    preparator.prepare(mol)
    if verbose:
        preparator.show_setup()

    return preparator.write_pdbqt_string()


if __name__ == '__main__':
    main()