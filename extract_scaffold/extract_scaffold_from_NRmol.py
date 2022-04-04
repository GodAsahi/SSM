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

def extract_scaffold_from_NRmol(smiles):
    c=True
    while c:
        smiles_old=smiles
        #print('smiles:',smiles)
        mol = Chem.MolFromSmiles(smiles)
        # mol = mol_with_atom_index(mol)
        # Draw.ShowMol(mol, size=(300, 300), kekulize=False)

        mol = RWMol(mol)
        mol.BeginBatchEdit()

        idx_atom_BTb_list = set()
        idx_atom_Sb_list = set()
        idx_atom_in_middle_list = set()
        idx_will_remove_atoms = set()

        for i in mol.GetBonds():
            if i.GetBondType() == Chem.rdchem.BondType.DOUBLE or i.GetBondType() == Chem.BondType.TRIPLE:
                idx_atom_BTb_list.add(i.GetBeginAtomIdx())
                idx_atom_BTb_list.add(i.GetEndAtomIdx())
                pass
            else:
                idx_atom_Sb_list.add(i.GetBeginAtomIdx())
                idx_atom_Sb_list.add(i.GetEndAtomIdx())
                pass
            pass

        idx_atom_BTb_list=list(idx_atom_BTb_list)
        idx_atom_BTb_list.sort()
        for atom in idx_atom_Sb_list:
            if idx_atom_BTb_list[0] <= atom <= idx_atom_BTb_list[-1]:
                idx_atom_in_middle_list.add(atom)
                pass
            pass

        for i in idx_atom_BTb_list:
            if i not in idx_atom_in_middle_list:
                idx_atom_in_middle_list.add(i)
                pass
            pass
        pass

        for idx in idx_atom_Sb_list:
            if idx not in idx_atom_in_middle_list:
                idx_will_remove_atoms.add(idx)
                pass
            pass

        for idx in idx_atom_in_middle_list:
            if len(mol.GetAtomWithIdx(idx).GetNeighbors()) < 2 and idx not in idx_atom_BTb_list:
                idx_will_remove_atoms.add(idx)
                pass
            pass

        for atom in idx_will_remove_atoms:
            mol.RemoveAtom(atom)

        print('idx_atom_BTb_list{}'.format(idx_atom_BTb_list))
        print('idx_atom_Sb_list{}'.format(idx_atom_Sb_list))
        print('idx_atom_in_middle_list{}'.format(idx_atom_in_middle_list))
        print('idx_will_remove_atoms{}'.format(idx_will_remove_atoms))

        mol.CommitBatchEdit()
        # mol = Chem.MolToSmiles(mol)
        # print(mol)
        # Draw.ShowMol(mol, size=(300, 300), kekulize=False)

        try:
            #Chem.SanitizeMol(mol)
            mol.GetMol()
            # Chem.AddHs(mol)
            smiles = Chem.MolToSmiles(mol)
            smiles_now=smiles
            #print(smiles)
            # Draw.ShowMol(mol, size=(300, 300), kekulize=False)

        except:
            mol = None
            print('Error')
            pass

        if smiles_old == smiles_now:
            c=False
        else:
            c=True
        #print(c)
    return smiles
