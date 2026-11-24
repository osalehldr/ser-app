import sys, re, os
import numpy as np
from rdkit import Chem
name, smiles = sys.argv[1], sys.argv[2]
pt = Chem.GetPeriodicTable()
organic = {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
def read(p):
    return open(p).read() if os.path.exists(p) else ""
L = read("xtbopt.xyz").splitlines()
if not L:
    print("GENERAL| no optimized structure"); sys.exit()
n = int(L[0].split()[0])
sym = [l.split()[0].capitalize() for l in L[2:2 + n]]
X = np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + n]])
rcov = [pt.GetRcovalent(pt.GetAtomicNumber(s)) for s in sym]
def nb(i):
    return [j for j in range(n) if j != i and np.linalg.norm(X[j] - X[i]) < 1.3 * (rcov[i] + rcov[j])]
lab = lambda i: "%s%d" % (sym[i], i + 1)
q = [float(v) for v in read("charges").split()]
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
mol = Chem.MolFromSmiles(smiles)
charge = Chem.GetFormalCharge(mol) if mol else 0
def real_radical(a):
    return (a.GetNumRadicalElectrons() > 0 and a.GetSymbol() in organic
            and all(x.GetSymbol() in organic for x in a.GetNeighbors()))
centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"
           and (a.GetFormalCharge() != 0 or real_radical(a))] if mol else []
radical = any(real_radical(a) for a in mol.GetAtoms()) if mol else False
P = print
P("-" * 64); P(" GENERAL REACTIVITY (works on any molecule)"); P("-" * 64)
heavy = [i for i in range(n) if sym[i] != "H"]
summ = []
if fk:
    if radical:
        col, what = 2, "f(0): radical sites (radical coupling / atom abstraction)"
    elif charge < 0:
        col, what = 1, "f(-): nucleophilic sites (where an ELECTROPHILE attacks)"
    else:
        col, what = 0, "f(+): electrophilic sites (where a NUCLEOPHILE attacks)"
    acyl_c = set()
    if mol is not None:
        for a_ in mol.GetAtoms():
            if a_.GetSymbol() == 'C' and any(b_.GetBondType() == Chem.BondType.DOUBLE and
                    b_.GetOtherAtom(a_).GetSymbol() == 'O' for b_ in a_.GetBonds()) and \
                    sum(1 for x_ in a_.GetNeighbors() if x_.GetSymbol() in ('O', 'N', 'S')) >= 2:
                acyl_c.add(a_.GetIdx())
    sites = [i for i in heavy if sym[i] == 'C' and i not in acyl_c] or heavy
    rank = sorted(sites, key=lambda i: -fk.get(i, (0, 0, 0))[col])
    P(what)
    for i in rank[:4]:
        P("  %-6s %7.3f   charge %+.3f" % (lab(i), fk.get(i, (0, 0, 0))[col], q[i] if i < len(q) else 0))
    top = rank[0]
    summ.append("top site %s (%.3f)" % (lab(top), fk.get(top, (0, 0, 0))[col]))
    if centers and top not in centers:
        P("  note: top site is not the drawn center (%s) -- delocalized or rearranged"
          % ", ".join(lab(c) for c in centers))
else:
    P("Fukui data not available.")
def kind(x):
    s = sym[x]
    if radical:
        return {"H": "beta-H transfer / disproportionation",
                "C": "beta-scission (C-C cleavage)"}.get(s, "beta-elimination of %s radical" % s)
    return {"H": "1,2-H shift or E1 proton loss",
            "C": "1,2-alkyl shift (Wagner-Meerwein)"}.get(s, "1,2-shift or beta-elimination of %s" % s)
for R in centers:
    nR = nb(R)
    P("\nCENTER %s (%s)  contacts: %s" % (lab(R), "radical" if radical else
      ("cation" if charge > 0 else "anion"), ", ".join(lab(j) for j in nR)))
    if charge < 0 and not radical:
        P("  anionic center acts as the nucleophile -- beta-bond analysis not applicable"); continue
    if len(nR) >= 4:
        P("  ** 4+ close contacts: possibly bridged (nonclassical) or rearranged -- check structure **")
    if len(nR) < 2:
        continue
    B = np.array([(X[j] - X[R]) / np.linalg.norm(X[j] - X[R]) for j in nR])
    ax = np.linalg.svd(B)[2][-1]
    pyr = np.mean([abs(np.degrees(np.arccos(np.clip(np.dot(ax, b), -1, 1))) - 90) for b in B])
    P("  pyramidalization %.1f deg (0 = planar)%s" % (pyr, "" if pyr < 12 else "  -- p-axis poorly defined"))
    rows = []
    for c1 in nR:
        if sym[c1] == "H":
            continue
        for x in nb(c1):
            if x == R:
                continue
            v = (X[x] - X[c1]) / np.linalg.norm(X[x] - X[c1])
            dev = np.degrees(np.arccos(np.clip(abs(np.dot(ax, v)), 0, 1)))
            rows.append((dev, -wbo.get((c1, x), 0), c1, x))
    rows.sort()
    if not rows:
        P("  no beta bonds found"); continue
    P("  BETA BONDS (best aligned with the p orbital first; 0 deg = ideal overlap)")
    P("  %-12s %8s %7s %6s  %s" % ("bond", "dev(deg)", "r(A)", "WBO", "possible process"))
    for dev, mw, c1, x in rows[:6]:
        P("  %-12s %8.1f %7.3f %6.3f  %s" % (lab(c1) + "-" + lab(x), dev,
          np.linalg.norm(X[x] - X[c1]), -mw, kind(x)))
    dev, mw, c1, x = rows[0]
    summ.append("top beta bond %s-%s (%.0f deg, WBO %.2f) -> %s" % (lab(c1), lab(x), dev, -mw, kind(x)))
P("\nGENERAL| " + (" | ".join(summ) if summ else "no data"))
