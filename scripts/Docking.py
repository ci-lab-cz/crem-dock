import os
import glob
import re
from os import system
from rdkit import Chem


# class VinaDock:
#
#     def __init__(self, vina_scripts_path, vina_path, ...):
#         pass
#
#     def prepare_ligands(self, dname, ncpu):
#         pass
#
#     def dock_ligands(self, dname, target_pdbqt, target_setup, ncpu):
#         pass


def get_mol_scores(dname, const_suffix="_dock.pdbqt"):
    """
    Extract docking scores from pdbqt files matching the supplied suffix from the given directory
    :param dname: directory with pdbqt files
    :param const_suffix: constant suffix to select pdbqt files
    :return: dict {mol_id: score}, where mol_id is a variable part of pdbqt file names
    """
    d = {}
    for fname in glob.glob(os.path.join(dname, '*' + const_suffix)):
        with open(fname) as f:
            score = float(f.read().split()[5])
        d[re.sub('^.*/(.*)' + const_suffix, '\\1', fname)] = score
    return d


def prepare_target(i_fname, o_fname, pythonADT, script_file, param=''):
    
    '''
    Preprocess target file for docking to Vina
    :param i_fname: path for target file (pdb)
    :param o_fname: output path for target file (pdbqt)
    :param param: param. Example '-A bonds_hydrogens -e True'
    :param pythonADT: python path for AutoDock . Example bin/mgltools_x86_64Linux2_1.5.6/bin/python'
    :param script_file: script path for AutoDock. Example 'bin/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py',
    :return: None
    '''

    system(' '.join([pythonADT, script_file, '-r', i_fname, '-o', o_fname, param]))


def prepare_ligand(i_fname, o_fname, pythonADT, script_file, param=''):
    '''
    Preprocess ligand file for docking to Vina
    :param i_fname: path for ligand file (pdb)
    :param o_fname: output path for ligand file (pdbqt)
    :param param: param. Example for default setting use (), another use  '-A bonds_hydrogens'
    :param pythonADT: python path for AutoDock . Example 'bin/mgltools_x86_64Linux2_1.5.6/bin/python'
    :param script_file: script path for AutoDock. Example 'bin/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py',
    :return: None
    '''

    system(' '.join([pythonADT, script_file, '-l', '"{}"'.format(i_fname), '-o',  '"{}"'.format(o_fname), param]))



def run_docking(ligand_in_fname, ligand_out_fname, target_fname, config, script_file, param=''):
    '''
    Run vina docking
    :param target_fname: target file
    :param ligand: ligand file
    :param out: output file
    :param config: config file contains coordinations of gridbox
    :param param: param. Default: None. example: '--energy_range 0 --cpu 2'. More about param in bin/AutodockVina/vina --help
    :param script_file: Vina path. Default: bin/AutodockVina/vina

    :return: None
    '''

    system(' '.join([script_file,
                     '--receptor', '"{}"'.format(target_fname), '--ligand', '"{}"'.format(ligand_in_fname),
                     '--out', '"{}"'.format(ligand_out_fname), '--config', '"{}"'.format(config), param]))


if __name__ == '__main__':
    python = 'bin/mgltools_x86_64Linux2_1.5.6/bin/python'
    script = 'bin/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/{}'
    prepare_target(i_fname='1t4u.pdb', o_fname='1t4u.pdbqt', param='-A bonds_hydrogens -e True',
                   pythonADT=python, script_file=script.format('prepare_receptor4.py'))
    prepare_ligand(i_fname='ref.pdb', o_fname='ref_lig.pdbqt', param='',
                   pythonADT=python, script_file=script.format('prepare_ligand4.py'))
    # run_docking(t_fname='1t4u.pdb', l_fname='1t4u_lig.pdbqt', o_fname='ttt', config='1t4u.log', script_file='bin/vina')
