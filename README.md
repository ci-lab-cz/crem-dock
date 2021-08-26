## Dependency
* **Python (>=3.5)**
* **RDKit**  
* **AutoDock Vina** 
* **Openbabel (>=3)**
* **Meeko (dev version)**

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

``` pip install vina```    


**Openbabel**
```
conda install -c conda-forge openbabel
```

**Meeko (from source)**
```
git clone https://github.com/ccsb-scripps/Meeko
cd Meeko
python setup.py build install
```