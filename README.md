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
pip install vina easydock crem meeko
```

Install ProLIF with corrected SMARTS patterns
```
conda install -c conda-forge prolif=1.1.0
pip install prolif==1.1.0
```

## Usage

Input SDF should not contain hydrogens or hydrogens should be listed after heavy atoms, otherwise protected ids can be broken and will not be correctly transferred to child molecules