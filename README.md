## Dependency
* **Python (>=3.7)**
* **RDKit**  
* **AutoDock Vina** 
* **Openbabel (>=3)**
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
conda install -c conda-forge python=3.9 rdkit openbabel cython dask distributed scipy scikit-learn
```

Install dependencies from pip and github
```
pip install vina
pip install crem
pip install git+https://github.com/forlilab/Meeko@7b1a60d9451eabaeb16b08a4a497cf8e695acc63
```

Install ProLIF with corrected SMARTS patterns
```
conda install -c conda-forge prolif
pip install git+https://github.com/DrrDom/ProLIF.git@smarts_patterns
```
