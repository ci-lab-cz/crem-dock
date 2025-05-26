# CReM-dock: design of chemically reasonable compounds guided by molecular docking

This tool automates generation of molecules using [CReM](https://github.com/DrrDom/crem) starting from a set of fragments by their growing within a protein binding site guided by molecular docking ([EasyDock](https://github.com/ci-lab-cz/easydock)).

![cremdock](./pics/crem-dock-600.gif)

## Features
- fully automated workflow
- different selection strategies (greedy, Pareto, k-means clustering)
- different objective functions (docking score, docking score augmented with QED or the fraction of sp3-carbon atoms, docking score divided on the number of heavy atoms, etc)
- user-defined thresholds for maximum allowed physicochemical parameters (MW, logP, RTB, TPSA)
- using protein-ligand interaction fingerprints (ProLIF) and/or RMSD to a parent molecule to control the pose of a constructed ligand
- indirect control over synthetic feasibility of generated structures (based on CReM parameters)
- support docking programs integrated in EasyDock, including ligand preparation steps and distributed calculations
- output is SQLite database from which a user can retrieve all necessary information 
- continue interrupted/unfinished generation by rerun of the same command

## Installation

Install dependencies and the software:

1. Install a conda environment with [easydock](https://github.com/ci-lab-cz/easydock) (minimal version 1.0)
2. On top install further dependencies and crem-dock
```
pip install prolif crem

pip install cremdock
# or the latest version from the repository
pip install git+https://github.com/DrrDom/crem-dock.git
```

## Generation pipeline

1. Docking of starting fragments.
2. (optional) Selection of molecules satisfying user-defined PLIF or RMSD to a parent molecule
3. Selection of the best candidates for the next iteration (greedy, Pareto, clustering) using the selected objective function (docking score, augmented docking score, etc)
4. Growing the selected molecules with CReM
5. Filtration of generated molecules by user-defined physicochemical parameters
6. Docking of passed molecules
7. Go to step 2  

Generation is stopped if no more molecules can be grown to satisfy the defined physicochemical parameters.

#### Notes

All molecules undergo automatic preparation for docking which should include the protonation step to get proper results. This is implemented in EasyDock and a user can choose which approach to use. At this time there are two options: Chemaxon (required a license) or pkasolver model (open-source). Installation of pkasolver and possible issues are described in [EasyDock repository](https://github.com/ci-lab-cz/easydock). It is still possible to omit protonation, but it is not advisable. 

## Usage

### CReM fragment databases

Precompiled CReM databases created from ChEMBL22 molecules are available at http://qsar4u.com/pages/crem.php.  
Details about creation of custom databases is at the [CReM repository](https://github.com/DrrDom/crem).

### Modes

All examples below are only illustrative and settings were chosen to quickly demonstrate particular functionality. Real use cases will require more carefully chosen parameters. All files required to reproduce these examples are available in `example` directory.    

1. **Hit generation**. Generation of molecules which starts from a set of fragment supplied as SMILES or 2D SDF. In these case the supplied fragments are docked, pass PLIF/RMSD check (optional) and best candidates are used for growing on the next iteration.


- simplest example (disregarding protonation and protein-ligand interaction fingerprints)
```bash
cremdock -i example/input.smi -o example/mode1_1.db -d chembl22_sa2_hac12.db --nclust 2 --max_replacements 2 --program vina --config example/vina_config.yml -c 2
```
`input.smi` - set of starting fragments  
`mode1_1.db` - output DB  
`chembl22_sa2_hac12.db` - CReM DB, which should be preliminary downloaded  
`--n_clust` - the number of clusters for the default selection strategy (clustering)  
`--max_replacements` - the number of new molecules derived from the selected ones on the previous iteration, 2 was selected for the reason of speed, usually we use 2000    
`--program` - docking program from the list of programs available in EasyDock  
`--config` - yml-file with dockign setup, more details at the [EasyDock repository](https://github.com/ci-lab-cz/easydock)  
`-c` - the number of molecules docked in parallel (the number of cores used for docking of individual molecules are provided in `--config`)


- protonation

It is recommended to protonate compounds before docking. This can be achieved by using the argument `--protonation` and specify the protonation model. These options come from `easydock`.

```bash
cremdock -i example/input.smi -o example/mode1_2.db -d chembl22_sa2_hac12.db --nclust 2 --max_replacements 2 --program vina --config example/vina_config.yml -c 2 --protonation molgpka
```


- identification of protein-ligand interaction fingerprints (PLIF) for a reference ligand

PLIF can be identified invoking `cremdock_plif` script feeded with a protein and a reference ligand (both should have all hydrogens explicit). The output is a text file with detected interactions. Full names of desired contacts should be used for the subsequent `cremdock` call.  
In this case the contacts will be `leu83.ahbdonor` and `leu83.ahbacceptor` which encode interaction with a hidge region.
```bash
cremdock_plif -p example/2BTR_H.pdb -l example/2BTR_lig.sdf -o example/2BTR_lig.plif 
```
Running this script is not necessary, a user may select contacts manually, however, names of contacts should follow the same conventions/format (otherwise no matches will be found).


- run `cremdock` with PLIF constraints and enabled protonation  
```bash
cremdock -i example/input.smi -o example/mode1_3.db -d chembl22_sa2_hac12.db --nclust 2 --max_replacements 2 --program vina --config example/vina_config.yml -c 2 --plif leu83.ahbdonor leu83.ahbacceptor --plif_cutoff 1 --plif_protein example/2BTR_H.pdb --protonation pkasolver 
```
There are three additional arguments:  
`--plif` - takes a list of contacts  
`--plif_cutoff` - the minimum fraction of satisfied contact (1 means 100%, thus molecules should have all of them to have a chance to be selected for the next iteration)  
`--plif_protein` - a protein the same as used for docking, but having all hydrogens explicit  


- biasing generated molecules to more Csp<sup>3</sup>-rich structures

Custom sampling functions can be implemented to bias selection of fragments for decoration of a parent molecule. We integrated a function which selects fragments proportionally to their fractiomn of sp<sup>3</sup> carbon atoms. However, the most useful approach would be to use starting fragments with a high fraction of sp<sup>3</sup> atoms. 
```bash
cremdock -i example/input.smi -o example/mode1_4.db -d chembl22_sa2_hac12.db --nclust 2 --max_replacements 2 --program vina --config example/vina_config.yml -c 2 --plif leu83.ahbdonor leu83.ahbacceptor --plif_cutoff 1 --plif_protein example/2BTR_H.pdb --protonation pkasolver --sample_func sample_csp3
```

- using augmented scoring function

There are several implemented objective functions. One of them is a geometric mean of a docking score (linearly scaled to the range [0;1]) and QED. To use a spceific objective functions one should set argument `--ranking` with a corresponding number. The full list is available in the help message (`-h`).

```bash
cremdock -i example/input.smi -o example/mode1_5.db -d chembl22_sa2_hac12.db --nclust 2 --max_replacements 2 --program vina --config example/vina_config.yml -c 2 --plif leu83.ahbdonor leu83.ahbacceptor --plif_cutoff 1 --plif_protein example/2BTR_H.pdb --protonation pkasolver --ranking 2
```


- restriction on physico-chemical properties of generated compounds

It is important to set reasonable thresholds for some physicochemical properties to avoid exploration of undesirable chemical space and save time. There are four such parameters avaiable with corresponding default threshold values (MW 450, logP 4, RTB 5, TPSA 120) which can be changed by command line arguments 

```bash
cremdock -i example/input.smi -o example/mode1_6.db -d chembl22_sa2_hac12.db --nclust 2 --max_replacements 2 --program vina --config example/vina_config.yml -c 2 --plif leu83.ahbdonor leu83.ahbacceptor --plif_cutoff 1 --plif_protein example/2BTR_H.pdb --protonation pkasolver --mw 400 --rtb 6 --logp 3 --tpsa 100
```


- change selection strategy

There are three major selection stratigies implemented: greedy (1), clustering (2) and Pareto (4). To enable a strategy it is necessary to pass its number to `--search` argument. There may be necessary to adjust other parameters accordingly.

```bash
cremdock -i example/input.smi -o example/mode1_7.db -d chembl22_sa2_hac12.db --search 1 --max_replacements 2 --program vina --config example/vina_config.yml -c 2 --plif leu83.ahbdonor leu83.ahbacceptor --plif_cutoff 1 --plif_protein example/2BTR_H.pdb --protonation pkasolver --mw 400 --rtb 6 --logp 3 --tpsa 100
```


2. **Hit expansion**. Expansion of a co-crystallized ligand by supplying 3D ligand structure (SDF). In this case an input molecule is directly going to the growing stage.

The difference from the first mode is to use SDF file with 3D structure of a starting fragment (it should be properly protonated, because it will not pass through the protonation step).  

Usually for such studies it is required to specify PLIF and RMSD (`--rmsd`) value to select for further iterations compounds which preserve both contacts and position of a parent molecule. In the example below we set to preserve at least one contact out of two and do not restrict RMSD, because with given settings we enumerate only few molecules each iteration and the chances to generate compounds satisfying PLIF and RMSD is very low. 
```bash
cremdock -i example/input.sdf -o example/mode2_1.db -d chembl22_sa2_hac12.db --nclust 2 --max_replacements 25 --program vina --config example/vina_config.yml -c 2 --plif leu83.ahbdonor leu83.ahbacceptor --plif_cutoff 0.5 --plif_protein example/2BTR_H.pdb --protonation pkasolver
```

To additionally restricts generation and to point growing to a desired direction it is possible to provide a field named `protected_user_ids` in an inpit SDF file. The field should contain a comma-separated list of ids of heavy atoms (starting from 1), which should be protected from expansion. This prevents growing of molecules at the specified points on all subsequent interations. It makes sens to combine this option with the restriction by RMSD value. Thus, only molecules alighed with the parent structures will pass to the next iterations and growing direction will remain relevant. Otherwise, if a molecule will flip, the specified growing direction can become not relevant.

3. Continuation of an interrupted run

If `cremdock` running was interrupted it can be re-run using the same command and the calcultions will be automatically continued. To continue calculations it will be even enough to run `cremdock` with only a single argument `--output` pointing out on the existing database. All necessary settings will be read from the database (almost all other input arguments are ignored if an output database exists).


### Retrieval data from the database

To retrieve data from an output database one can use funtionality from EasyDock. More examples are in EasyDock repository.  

```bash
get_sdf_from_dock_db -i mode1_1.db -o mode1_1.sdf --fields docking_score
```

Data can also be retrieved in SMILES format as well as any additional columns (we usually use tab as a separator).
```bash
sqlite3 -header -separator " " mode1_1.db "select smi, id, docking_score from mols"
```


## License
BSD-3

## Citation
CReM-dock: de novo design of synthetically feasible compounds guided by molecular docking  
Guzel Minibaeva, Pavel Polishchuk  
https://doi.org/10.26434/chemrxiv-2024-fpzqb-v2
