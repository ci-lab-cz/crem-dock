## Dependency
* **Python (>=3.9)**
* **RDKit**  
* **AutoDock Vina**
* **Meeko**
* **Scipy**
* **Scikit-learn**
* **CReM (>= 0.2.7)**
* **ProLIF**

## Installation
```
conda create -n vina
conda activate vina
```

Install dependencies from conda
```
conda install -c conda-forge python=3.9 numpy rdkit dask distributed scipy scikit-learn
```

Install dependencies from pip and github
```
pip install easydock crem meeko
```

Install vina from sources (https://autodock-vina.readthedocs.io/en/latest/installation.html) or from compiled binaries. In the latter case you need to install python bindings to your anaconda environment as showing below.
```
# download zip archive with compiled repository and unpack
conda activate vina
cd AutoDock-Vina/build/python
conda install -c conda-forge boost-cpp swig
rm -rf build dist *.egg-info (to clean previous installation)
python setup.py build install 
```

Install ProLIF with corrected SMARTS patterns
```
conda install -c conda-forge prolif
pip install prolif
```

## Usage

Input SDF should not contain hydrogens or hydrogens should be listed after heavy atoms, otherwise protected ids can be broken and will not be correctly transferred to child molecules