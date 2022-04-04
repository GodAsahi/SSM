# coding:utf-8

import sys
import rdkit
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
import scaffoldgraph as sg
from copy import deepcopy
from rdkit.Chem import Draw
from utils import extract_scaffolds
from extract_scaffold_from_NRmol import extract_scaffold_from_NRmol

scaffolds=[]
err_mols=[]

with open('','r') as f_r:
    smiles=f_r.read().split('\n')
    for s in smiles:
        if extract_scaffolds(s) !='':
            scaffolds.append(extract_scaffolds(s))
        elif extract_scaffolds(s) =='':
            try:
                scaffolds.append(extract_scaffold_from_NRmol(s))
            except:
                print(s)
                err_mols.append(s)
        else:
            print('Error extracting scaffold')
            pass
        pass
with open('','w') as f_w:
    for i in scaffolds:
        f_w.write(i+'\n')
        pass
    pass
print('Success')
print(err_mols)