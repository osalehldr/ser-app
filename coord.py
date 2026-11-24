import sys, os
import numpy as np
from rdkit import Chem
name, smiles = sys.argv[1], sys.argv[2]
organic = {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
def read(p):
    return open(p).read() if os.path.exists(p) else ""
L = read("xtbopt.xyz").splitlines()
mol = Chem.MolFromSmiles(smiles)
if not L or mol is None:
    sys.exit()
n = int(L[0].split()[0])
sym = [l.split()[0].capitalize() for l in L[2:2 + n]]
X = np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + n]])
metals = [i for i in range(n) if sym[i] not in organic]
if not metals:
    sys.exit()
pt = Chem.GetPeriodicTable()
rcov = [pt.GetRcovalent(pt.GetAtomicNumber(s)) for s in sym]
dist = lambda i, j: float(np.linalg.norm(X[i] - X[j]))
def nb(i):
    return [j for j in range(n) if j != i and dist(i, j) < 1.3 * (rcov[i] + rcov[j])]
lab = lambda i: "%s%d" % (sym[i], i + 1)
nheavy = mol.GetNumAtoms()
rw = Chem.RWMol(mol)
for b in list(mol.GetBonds()):
    if b.GetBeginAtom().GetSymbol() not in organic or b.GetEndAtom().GetSymbol() not in organic:
        rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
org = rw.GetMol()
def real_radical(a):
    return (a.GetNumRadicalElectrons() > 0 and a.GetSymbol() in organic
            and all(x.GetSymbol() in organic for x in a.GetNeighbors()))
def activated_carbonyl(a):
    if a.GetSymbol() != "C":
        return False
    for b in a.GetBonds():
        o = b.GetOtherAtom(a)
        if o.GetSymbol() == "O" and b.GetBondType() == Chem.BondType.DOUBLE and o.GetDegree() > 1:
            return True
    return False
centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and
           (a.GetFormalCharge() != 0 or real_radical(a) or activated_carbonyl(a))]
abset = []
for line in read(name + "_atoms.txt").splitlines():
    if line.strip():
        R, C1, A, B = map(int, line.split())
        abset.append((C1 - 1, A - 1, B - 1))
def shield(t, axis, skip):
    occ = 0.0
    for k in range(n):
        if k == t or k in skip:
            continue
        v = X[k] - X[t]; d = np.linalg.norm(v)
        if d > 4.5 or np.dot(v, axis) / d < 0.5:
            continue
        occ += (2.0 if sym[k] != "H" else 1.0) / d ** 2
    return occ
P = print
P("-" * 64); P(" METAL COORDINATION / CHELATION / FACIAL BIAS"); P("-" * 64)
summ = []
for M in metals:
    lig = nb(M)
    P("\n%s: coordination number %d" % (lab(M), len(lig)))
    for j in lig:
        P("  %-6s %.2f A" % (lab(j), dist(M, j)))
    donors = [j for j in lig if j < nheavy and sym[j] in ("O", "N", "S", "P") and
              any(x.GetSymbol() in organic and x.GetSymbol() != "H"
                  for x in mol.GetAtomWithIdx(j).GetNeighbors())]
    if len(donors) >= 2:
        for x in range(len(donors)):
            for y in range(x + 1, len(donors)):
                path = Chem.GetShortestPath(org, donors[x], donors[y])
                if path:
                    P("  CHELATE: %s + %s bind %s -> %d-membered chelate ring"
                      % (lab(donors[x]), lab(donors[y]), lab(M), len(path) + 1))
                    summ.append("chelate %s/%s %d-ring" % (lab(donors[x]), lab(donors[y]), len(path) + 1))
    elif len(donors) == 1:
        P("  monodentate: binds %s only (no chelate)" % lab(donors[0]))
        summ.append("monodentate %s" % lab(donors[0]))
for t in centers:
    nt = nb(t)
    if len(nt) < 2:
        continue
    Bv = np.array([(X[j] - X[t]) / dist(t, j) for j in nt])
    ax = np.linalg.svd(Bv)[2][-1]
    Mm = min(metals, key=lambda m: dist(m, t))
    sp, sm = shield(t, ax, nt), shield(t, -ax, nt)
    side = lambda s: "syn to %s" % lab(Mm) if np.sign(np.dot(X[Mm] - X[t], ax)) == s else "anti to %s" % lab(Mm)
    P("\nFACES of %s (shielding: higher = more blocked)" % lab(t))
    P("  face 1 (%s): %.3f" % (side(1), sp))
    P("  face 2 (%s): %.3f" % (side(-1), sm))
    lo, hi = (sp, sm) if sp < sm else (sm, sp)
    open_face = side(1) if sp < sm else side(-1)
    if hi > 1.3 * lo + 0.02:
        P("  -> nucleophile favored from the %s face (%.1fx less shielded)" % (open_face, hi / max(lo, 1e-3)))
        summ.append("%s attack %s" % (lab(t), open_face))
    else:
        P("  -> no strong facial bias")
        summ.append("%s no facial bias" % lab(t))
for C1, A, B in abset:
    sa = shield(A, (X[A] - X[C1]) / dist(A, C1), nb(A))
    sb = shield(B, (X[B] - X[C1]) / dist(B, C1), nb(B))
    P("\nBACKSIDE ACCESS for ring opening (lower = easier Nu approach)")
    P("  at A %s (opposite C%d-%s): %.3f" % (lab(A), C1 + 1, lab(A), sa))
    P("  at B %s (opposite C%d-%s): %.3f" % (lab(B), C1 + 1, lab(B), sb))
    better = lab(A) if sa < sb else lab(B)
    if max(sa, sb) > 1.3 * min(sa, sb) + 0.02:
        P("  -> backside of %s is more open (steric bias)" % better)
        summ.append("backside favors %s" % better)
    else:
        P("  -> no strong backside bias")
P("\nCOORD| " + (" | ".join(summ) if summ else "no data"))
