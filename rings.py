import sys, os, re
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
rings = [r for r in mol.GetRingInfo().AtomRings() if len(r) == 3]
if not rings:
    sys.exit()
n = int(L[0].split()[0])
sym = [l.split()[0].capitalize() for l in L[2:2 + n]]
X = np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + n]])
pt = Chem.GetPeriodicTable()
rcov = [pt.GetRcovalent(pt.GetAtomicNumber(s)) for s in sym]
def nb(i):
    return [j for j in range(n) if j != i and np.linalg.norm(X[j] - X[i]) < 1.3 * (rcov[i] + rcov[j])]
lab = lambda i: "%s%d" % (sym[i], i + 1)
wbo = {}
for l in read("wbo").splitlines():
    p = l.split()
    if len(p) == 3:
        a, b = int(p[0]) - 1, int(p[1]) - 1
        wbo[(a, b)] = wbo[(b, a)] = float(p[2])
fk = {}
FL = read(name + "_fukui.out").splitlines()
for k, l in enumerate(FL):
    if "f(+)" in l and "f(-)" in l:
        for m in FL[k + 1:]:
            t = m.split()
            if len(t) < 4 or not re.match(r"\d+", t[0]):
                break
            fk[int(re.match(r"\d+", t[0]).group()) - 1] = tuple(float(v) for v in t[1:4])
        break
def real_radical(a):
    return (a.GetNumRadicalElectrons() > 0 and a.GetSymbol() in organic
            and all(x.GetSymbol() in organic for x in a.GetNeighbors()))
charge = Chem.GetFormalCharge(mol)
radical = any(real_radical(a) for a in mol.GetAtoms())
centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"
           and (a.GetFormalCharge() != 0 or real_radical(a))]
col = 2 if radical else (1 if charge < 0 else 0)
fname = ["f(+)", "f(-)", "f(0)"][col]
D = Chem.GetDistanceMatrix(mol)
axes = {}
for c in centers:
    nc = nb(c)
    if len(nc) >= 2:
        B = np.array([(X[j] - X[c]) / np.linalg.norm(X[j] - X[c]) for j in nc])
        axes[c] = np.linalg.svd(B)[2][-1]
P = print
P("-" * 64); P(" CYCLOPROPANE ANALYSIS (every 3-ring, any distance from the center)"); P("-" * 64)
P("reference: unstrained cyclopropane C-C ~1.51 A, WBO ~0.97")
for k, ring in enumerate(rings):
    a, b, c = ring
    P("\nRing %d: %s" % (k + 1, "-".join(lab(i) for i in ring)))
    if centers:
        path = min(int(D[x][y]) for x in centers for y in ring)
        dmin = min(np.linalg.norm(X[x] - X[y]) for x in centers for y in ring)
        P("  distance to reactive center: %d bond(s) through bonds, %.2f A through space" % (path, dmin))
    else:
        path = -1
        P("  no charged/radical center in the SMILES")
    rows = []
    for i, j in ((a, b), (b, c), (a, c)):
        r = np.linalg.norm(X[i] - X[j])
        v = (X[j] - X[i]) / r
        devs = [np.degrees(np.arccos(np.clip(abs(np.dot(ax, v)), 0, 1))) for ax in axes.values()]
        rows.append((wbo.get((i, j), 0.0), -r, i, j, r, min(devs) if devs else None))
    rows.sort()
    P("  %-12s %7s %6s %12s" % ("bond", "r(A)", "WBO", "p-axis dev"))
    for w, _, i, j, r, dv in rows:
        P("  %-12s %7.3f %6.3f %12s" % (lab(i) + "-" + lab(j), r, w, "-" if dv is None else "%.1f deg" % dv))
    w1, _, i1, j1, r1, _ = rows[0]
    w2, _, i2, j2, r2, _ = rows[1]
    opened = [x for x in rows if x[4] > 1.80]
    if opened:
        call, i1, j1 = "OPENED", opened[0][2], opened[0][3]
        P("  ** ring OPENED during optimization: %s-%s is now %.2f A **" % (lab(i1), lab(j1), opened[0][4]))
    elif abs(w1 - w2) < 0.03 and abs(r1 - r2) < 0.01:
        call = "tie"
        P("  tie: %s-%s and %s-%s are indistinguishable" % (lab(i1), lab(j1), lab(i2), lab(j2)))
    elif w1 < 0.90 or r1 > 1.56:
        call = "activated"
        P("  activated: %s-%s is the weakest bond" % (lab(i1), lab(j1)))
    else:
        call = "unperturbed"
        P("  unperturbed: no ring bond significantly weakened")
    site = "-"
    if fk and call in ("OPENED", "activated"):
        fi, fj = fk.get(i1, (0, 0, 0))[col], fk.get(j1, (0, 0, 0))[col]
        site = lab(i1) if fi >= fj else lab(j1)
        P("  if %s-%s breaks: %s favors %s (%.3f vs %.3f)" % (lab(i1), lab(j1), fname, site, max(fi, fj), min(fi, fj)))
    P("RING| atoms %d %d %d | weak %d %d | weak2 %d %d | call %s | path %d | site %s"
      % (a + 1, b + 1, c + 1, i1 + 1, j1 + 1, i2 + 1, j2 + 1, call, path, site))
