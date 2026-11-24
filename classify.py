import sys
from rdkit import Chem
m = Chem.MolFromSmiles(sys.argv[1])
if m is None:
    print("BAD 0 0 0"); sys.exit()
organic = {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
def real_radical(a):
    if a.GetSymbol() not in organic:
        return False
    return all(n.GetSymbol() in organic for n in a.GetNeighbors())
q = Chem.GetFormalCharge(m)
u = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms() if real_radical(a))
r3 = sum(1 for r in m.GetRingInfo().AtomRings() if len(r) == 3)
la = sum(1 for a in m.GetAtoms() if a.GetSymbol() not in organic)
print(q, u, r3, la)
