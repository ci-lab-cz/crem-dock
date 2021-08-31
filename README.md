## Dependency
* **Python (>=3.5)**
* **RDKit**  
* **AutoDock Vina** 
* **Openbabel (>=3)**
* **Meeko**
* **Scipy**
* **CReM**

## Installation
```
conda create -n vina python=3.7
conda activate vina
```

**RDKit**
```
conda install -c conda-forge rdkit
```


**AutoDock Vina**
```
conda install -c conda-forge -c ccsb-scripps vina 
``` 

or installation using pip:

```
pip install vina
```    


**Openbabel**
```
conda install -c conda-forge openbabel
```

**Meeko**
```
pip install git+https://github.com/ccsb-scripps/Meeko
```

**Scipy**
```
conda install scipy
```

**CReM**
```
pip install crem
```