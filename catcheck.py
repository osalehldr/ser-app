import sys
from rdkit import Chem
m = Chem.MolFromSmiles(sys.argv[1])
if m is None:
    print("RDKit could not read that SMILES"); sys.exit(1)
organic = {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
frags = Chem.GetMolFrags(m)
q = Chem.GetFormalCharge(m)
rad = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()
          if a.GetSymbol() in organic and all(n.GetSymbol() in organic for n in a.GetNeighbors()))
rings3 = [set(r) for r in m.GetRingInfo().AtomRings() if len(r) == 3]
pos = [a for a in m.GetAtoms() if a.GetFormalCharge() > 0]
print("charge: %+d | unpaired e-: %d | fragments: %d | cyclopropanes: %d" % (q, rad, len(frags), len(rings3)))
ok = True
if len(frags) > 1:
    print("WARNING: %d separate fragments ('.' in SMILES) -- something extra was copied" % len(frags)); ok = False
if q != 1:
    print("WARNING: net charge is %+d, expected +1 for a carbocation (fine only if intended)" % q); ok = False
if rad:
    print("WARNING: %d unpaired electron(s) -- a cation carbon is probably missing an H" % rad); ok = False
if not pos:
    print("WARNING: no positively charged atom found"); ok = False
for a in pos:
    i, s = a.GetIdx() + 1, a.GetSymbol()
    if s != "C":
        print("note: + charge is on %s%d (onium ion, not a carbocation)" % (s, i)); continue
    nH, val = a.GetTotalNumHs(), a.GetTotalValence()
    where = "not next to a cyclopropane"
    if any(a.GetIdx() in r for r in rings3):
        where = "INSIDE a cyclopropane (ring analysis only, no A/B analysis)"
    elif any(n.GetIdx() in r for n in a.GetNeighbors() for r in rings3):
        where = "next to a cyclopropane (A/B analysis will run)"
    print("cation C%d: %d H, %d bond(s) to heavy atoms -- %s" % (i, nH, a.GetDegree(), where))
    if val != 3:
        print("WARNING: C%d has %d attachments (bonds + H), a carbocation needs 3 -- check its H count" % (i, val)); ok = False
if ok:
    print("OK: clean carbocation SMILES")
