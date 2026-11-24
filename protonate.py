import sys
from rdkit import Chem
smi, prefix = sys.argv[1], sys.argv[2]
mol = Chem.MolFromSmiles(smi)
seen = set()
for ring in mol.GetRingInfo().AtomRings():
    if len(ring) != 3:
        continue
    for i in range(3):
        a, b = ring[i], ring[(i + 1) % 3]
        for h_at, c_at in ((a, b), (b, a)):
            nh = {x: mol.GetAtomWithIdx(x).GetTotalNumHs() for x in (a, b)}
            rw = Chem.RWMol(mol)
            rw.RemoveBond(a, b)
            for x in (a, b):
                at = rw.GetAtomWithIdx(x)
                at.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                at.SetNoImplicit(True)
                at.SetNumExplicitHs(nh[x] + (1 if x == h_at else 0))
            rw.GetAtomWithIdx(c_at).SetFormalCharge(1)
            m = rw.GetMol()
            Chem.SanitizeMol(m)
            key = Chem.MolToSmiles(m)
            if key in seen:
                continue
            seen.add(key)
            print("%s_H%d_C%d %s" % (prefix, h_at + 1, c_at + 1,
                                      Chem.MolToSmiles(m, canonical=False)))
