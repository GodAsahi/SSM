from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem import rdBase
from rdkit.Chem import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetUSRScore, GetUSRCAT
from rdkit.Chem import DataStructs
from rdkit.Chem import Draw

p_smiles=''
p_mol = Chem.MolFromSmiles(p_smiles)
#Draw.MolToFile(p_mol, 'm.png', size=(500, 500))
AllChem.EmbedMolecule(p_mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
p_usrcats = GetUSRCAT(p_mol)


with open('','r') as f_r:
    smiles=f_r.read().split('\n')
#print(smiles)
with open('','w') as f_w:
    for s in smiles:
        try:
            mol = Chem.MolFromSmiles(s)
            AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
            usrcats = GetUSRCAT(mol)
            u_score = GetUSRScore(usrcats, p_usrcats)
            #print(s)
            f_w.write(s+'\t'+str(u_score)+'\n')
        except:
            continue

print('Done')