# Predicting MiRNA-disease Associations by Multiple Meta-paths Fusion Graph Embedding Model（M2GMDA）


### Implemented evironment
Python>=3.6


###Required libraries
`numpy,numba,openpyxl,xlrd，torch,itertools,sys,os,importlib`

We recommended that you could install Anaconda to meet these requirements


### How to run M2GMDA? 
####Data
All datas or mid results are orgnized in `DATA` fold, which contains miRNA-disease associations,disease semantic similarity, miRNA functional similarity, encode result of disease and  miRNA.

####The starting point for running M2GMDA is:

(1)**M2GMDA.py**：gereating meta-paths from the dataset of miRNA-disease associations,disease semantic similarity, miRNA functional similarity. all the result is saved in the folds named `"5.mid result"` and `"6.meta path"`, which need to be created by yourselves.


(2)**training.py**: training the model of M2GMDA, which will referece `auto_encoder.py, M_D_coder_by_AE.py,MLP.py,SelfAttention.py. `
And it outputs the parameter of M2GMDA.

####other relateive files:
**auto_encoder.py**: an auto_encoder model

**M_D_coder_by_AE.py**: encode the M and D nodes by auto_encoder,  obtain the inputs of M2GMDA

**MLP.py**: a MLP model in M2GMDA
**SelfAttention.py**:SelfAttention model in M2GMDA
