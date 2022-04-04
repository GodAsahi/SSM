from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
import scaffoldgraph as sg
from copy import deepcopy
from rdkit.Chem import Draw

def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

s=''
s2=''

mol = Chem.MolFromSmiles(s2)
mol=mol_with_atom_index(mol)
Draw.ShowMol(mol, size=(300,300), kekulize=False)

mol=RWMol(mol)
mol.BeginBatchEdit()
remove_atoms = set()
remove_bonds = set()
idx_atoms=set()
idx_bonds=set()
for i in mol.GetAtoms():
    print(i)
for i in mol.GetBonds():
    #print(i.GetBondType())
    if i.GetBondType()==Chem.rdchem.BondType.DOUBLE or i.GetBondType()==Chem.BondType.TRIPLE:
        idx_atoms.add(i.GetBeginAtomIdx())
        idx_atoms.add(i.GetEndAtomIdx())
        idx_bonds.add(i.GetIdx())
        pass
    else:
        begin_atom=i.GetBeginAtomIdx()
        end_atom=i.GetEndAtomIdx()
        remove_atoms.add(i.GetBeginAtomIdx())
        remove_atoms.add(i.GetEndAtomIdx())
        remove_bonds.add((begin_atom,end_atom))
        #remove_bonds.add(i.GetIdx())
        pass

# print(remove_bonds)

for i in remove_bonds:
    mol.RemoveBond(*i)
    pass
print(remove_atoms)
remove=deepcopy(remove_atoms)
for i in remove:
    if i in idx_atoms:
        remove_atoms.remove(i)
print(remove_atoms)
for i in remove_atoms:
    mol.RemoveAtom(i)
mol.CommitBatchEdit()
print(1)
mol.GetMol()
try:
    Chem.SanitizeMol(mol)
except:
    mol = None
mol = Chem.MolToSmiles(mol)
print(mol)
mol = Chem.MolFromSmiles(mol)
Draw.ShowMol(mol, size=(300,300), kekulize=False)
