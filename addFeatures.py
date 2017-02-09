from __future__ import print_function
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import MolSurf
from rdkit.Chem.AtomPairs import Utils
import time
start_time = time.time()


"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

sssr_train = []
tpsa_train = []
hdonors_train = []
hacceptors_train = []
rotatablebonds_train = []
pibonds_train = []
i = 0
for mol in df_train.smiles:
    i = i+1
    print(i)
    m = Chem.MolFromSmiles(mol)
    sssr_train.append(Chem.rdmolops.GetSSSR(m))
    tpsa_train.append(MolSurf.TPSA(m))
    hdonors_train.append(Lipinski.NumHDonors(m))
    hacceptors_train.append(Lipinski.NumHAcceptors(m))
    rotatablebonds_train.append(Lipinski.NumRotatableBonds(m))
    pi_bonds = 0
    for atom in m.GetAtoms():
        pi_bonds = pi_bonds + Utils.NumPiElectrons(atom)
    pibonds_train.append(pi_bonds)

df_train['sssr'] = sssr_train
df_train['tpsa'] = tpsa_train
df_train['hdonors'] = hdonors_train
df_train['hacceptors'] = hacceptors_train
df_train['rotatablebonds'] = rotatablebonds_train
df_train['pibonds'] = pibonds_train

sssr_test = []
tpsa_test = []
hdonors_test = []
hacceptors_test = []
rotatablebonds_test = []
pibonds_test = []
i = 0
for mol in df_test.smiles:
    i = i+1
    print(i)
    m = Chem.MolFromSmiles(mol)
    sssr_test.append(Chem.rdmolops.GetSSSR(m))
    tpsa_test.append(MolSurf.TPSA(m))
    hdonors_test.append(Lipinski.NumHDonors(m))
    hacceptors_test.append(Lipinski.NumHAcceptors(m))
    rotatablebonds_test.append(Lipinski.NumRotatableBonds(m))
    pi_bonds = 0
    for atom in m.GetAtoms():
        pi_bonds = pi_bonds + Utils.NumPiElectrons(atom)
    pibonds_test.append(pi_bonds)



df_test['sssr'] = sssr_test
df_test['tpsa'] = tpsa_test
df_test['hdonors'] = hdonors_test
df_test['hacceptors'] = hacceptors_test
df_test['rotatablebonds'] = rotatablebonds_test
df_test['pibonds'] = pibonds_test

print(df_train.head())
df_train.to_csv('newTrain.csv', sep=',')
df_test.to_csv('newTest.csv', sep=',')
print("--- %s seconds ---" % (time.time() - start_time))
