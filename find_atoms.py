import sys
from rdkit import Chem
mol = Chem.MolFromSmiles(sys.argv[1])
organic = {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
rings3 = [set(r) for r in mol.GetRingInfo().AtomRings() if len(r) == 3]
def real_radical(a):
    return (a.GetNumRadicalElectrons() > 0 and a.GetSymbol() in organic
            and all(n.GetSymbol() in organic for n in a.GetNeighbors()))
def activated_carbonyl(a):
    if a.GetSymbol() != "C":
        return False
    for b in a.GetBonds():
        o = b.GetOtherAtom(a)
        if o.GetSymbol() == "O" and b.GetBondType() == Chem.BondType.DOUBLE and o.GetDegree() > 1:
            return True
    return False
found = []
for r in mol.GetAtoms():
    if not (r.GetFormalCharge() != 0 or real_radical(r) or activated_carbonyl(r)):
        continue
    iR = r.GetIdx()
    for c1 in r.GetNeighbors():
        iC = c1.GetIdx()
        for ring in rings3:
            if iC in ring and iR not in ring:
                a, b = sorted(ring - {iC})
                t = (iR + 1, iC + 1, a + 1, b + 1)
                if t not in found:
                    found.append(t)
for f in found:
    print(*f)
if not found:
    print("no R-C1(cyclopropane) pattern found", file=sys.stderr)
