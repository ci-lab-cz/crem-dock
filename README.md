## Dependency
* **Python (>=3.5)**
* **RDKit**  
* **AutoDock Vina** 
* **Openbabel (>=3)**
* **Meeko**
* **Scipy**
* **Scikit-learn**
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
due to recent changes in Meeko the latest version may be not working, in this case the following installation can be used
```
pip install git+https://github.com/forlilab/Meeko@7b1a60d9451eabaeb16b08a4a497cf8e695acc63
```

**Scipy and Scikit-learn**
```
conda install scipy scikit-learn
```

**CReM**
```
pip install crem
```