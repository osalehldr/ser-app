#!/usr/bin/env python3
"""
rearr.py -- rearrangement / scission analysis for carbocations and radicals.

Usage (run inside a job folder, after xtb optimization + Fukui):
    python3 rearr.py NAME "SMILES" XTB_PATH CHARGE UHF
Optional environment variables:
    SOLV         implicit solvent for xtb (ALPB), e.g. acetonitrile  (default: gas phase)
    REARR_NSCAN  number of pathways to scan (default 6)
    REARR_STEPS  points per relaxed scan   (default 12)

What it does
  1. Compares the optimized structure with the SMILES bonding: reports bonds that are
     already partially formed / stretched (bridged, nonclassical, or rearranged ions).
  2. Enumerates every plausible elementary step around each cationic/radical carbon,
     at ANY distance from the center:
       - bond formation R-T to atoms two bonds away (1,2-shifts AND sigma/pi participation,
         e.g. cyclopropane bonds that are not attached to the center)
       - bond formation to atoms further away but close in space (transannular,
         neighbouring-group participation), and to nearby H atoms (1,n-H shifts / HAT)
       - beta-bond cleavage (radical beta-scission; strained-ring opening for cations)
     Each candidate is ranked by orbital alignment with the center's p orbital,
     through-space proximity and bond weakening (Wiberg bond order).
  3. For the best candidates runs a relaxed GFN2-xTB scan of that coordinate
     (imposed activation, cf. IACTA: Lavigne, Gomes, Pollice, Aspuru-Guzik,
     Chem. Sci. 2022, 13, 13857), then relaxes the end point without constraints.
     The quantum-mechanical scan decides WHICH bond actually breaks, and gives an
     approximate barrier (scan maximum) and reaction energy.
  4. Predicts where the charge/radical ends up and the most likely nucleophile site
     (Fukui f(+) for cations / f(0) for radicals, plus steric accessibility).
"""
import sys, os, re, subprocess
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

KCAL = 627.509
name, smiles, XTB = sys.argv[1], sys.argv[2], sys.argv[3]
CHG = int(sys.argv[4]) if len(sys.argv) > 4 else 0
UHF = int(sys.argv[5]) if len(sys.argv) > 5 else 0
SOLV = os.environ.get("SOLV", "").strip()
NSCAN = int(os.environ.get("REARR_NSCAN", "6"))
NSTEP = int(os.environ.get("REARR_STEPS", "12"))
FACILE = float(os.environ.get("REARR_FACILE", "20"))
# REARR_EXCLUDE="5-6,5-7": bonds (1-based atom numbers) not to use -- set by the cascade when a step would
# only lead back to a structure it has already seen (the two forms of one bridged, non-classical ion)
EXCL = set(frozenset(int(x) - 1 for x in p.split("-")) for p in os.environ.get("REARR_EXCLUDE", "").split(",")
           if re.match(r"^\d+-\d+$", p.strip()))
TIE = float(os.environ.get("REARR_TIE", "2"))
UPHILL = float(os.environ.get("REARR_UPHILL", "15"))   # cascade may try uphill steps up to this (kcal/mol)
uphill = None   # kcal/mol: max barrier counted as a real rearrangement
GXTB = os.environ.get("GXTB", "").strip()          # optional g-xTB-enabled xtb binary for energy refinement
if GXTB and not os.path.exists(GXTB):
    GXTB = ""
ORGANIC = {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
PT = Chem.GetPeriodicTable()
P = print


# ---------------------------------------------------------------- helpers
def read(p):
    return open(p).read() if os.path.exists(p) else ""


def read_xyz(p):
    L = read(p).splitlines()
    if not L:
        return None, None, None
    n = int(L[0].split()[0])
    m = re.search(r"energy:\s*(-?\d+\.\d+)", L[1])
    sym = [l.split()[0].capitalize() for l in L[2:2 + n]]
    X = np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + n]])
    return sym, X, (float(m.group(1)) if m else None)


def write_xyz(p, sym, X, comment=""):
    with open(p, "w") as f:
        f.write("%d\n%s\n" % (len(sym), comment))
        for s, x in zip(sym, X):
            f.write("%-2s %14.8f %14.8f %14.8f\n" % (s, x[0], x[1], x[2]))


def scan_frames(p):
    L = read(p).splitlines()
    out, k = [], 0
    while k < len(L) and L[k].strip():
        n = int(L[k].split()[0])
        m = re.search(r"(-?\d+\.\d+)", L[k + 1])
        X = np.array([[float(v) for v in l.split()[1:4]] for l in L[k + 2:k + 2 + n]])
        out.append((float(m.group(1)), X))
        k += n + 2
    return out


def run_xtb(args, cwd, out):
    cmd = [XTB] + args + ["--chrg", str(CHG), "--uhf", str(UHF), "--gfn", "2"]
    if SOLV:
        cmd += ["--alpb", SOLV]
    with open(os.path.join(cwd, out), "w") as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)


def single_point(X, tag, method):
    """method 'gxtb' (g-xTB, gas) or 'gfn2gas' (GFN2-xTB, gas); returns energy in Eh or None"""
    wd = os.path.join("rearr", "sp", tag + "_" + method)
    os.makedirs(wd, exist_ok=True)
    write_xyz(os.path.join(wd, "s.xyz"), sym, X)
    if method == "gxtb":
        cmd = [GXTB, "s.xyz", "--gxtb", "--chrg", str(CHG)] + (["--uhf", str(UHF)] if UHF > 0 else [])
    else:
        cmd = [XTB, "s.xyz", "--chrg", str(CHG), "--uhf", str(UHF), "--gfn", "2"]
    with open(os.path.join(wd, "sp.out"), "w") as f:
        subprocess.run(cmd, cwd=wd, stdout=f, stderr=subprocess.STDOUT)
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", read(os.path.join(wd, "sp.out")))
    return float(m.group(1)) if m else None


def refined(X, E_gfn2, tag):
    """g-xTB energy (+ GFN2 ALPB solvation correction when a solvent is used)"""
    if not GXTB:
        return None
    eg = single_point(X, tag, "gxtb")
    if eg is None:
        return None
    if SOLV:
        e2 = single_point(X, tag, "gfn2gas")
        if e2 is None or E_gfn2 is None:
            return None
        return eg + (E_gfn2 - e2)
    return eg


def fukui(path):
    fk = {}
    L = read(path).splitlines()
    for k, l in enumerate(L):
        if "f(+)" in l and "f(-)" in l:
            for m in L[k + 1:]:
                t = m.split()
                if len(t) < 4 or not re.match(r"\d+", t[0]):
                    break
                fk[int(re.match(r"\d+", t[0]).group()) - 1] = tuple(float(v) for v in t[1:4])
            break
    return fk


def rcov(s):
    return PT.GetRcovalent(PT.GetAtomicNumber(s))


def dist(X, i, j):
    return float(np.linalg.norm(X[i] - X[j]))


def ratio(X, i, j):
    return dist(X, i, j) / (rcov(sym[i]) + rcov(sym[j]))


def state(r):
    # bond state from distance / (sum of covalent radii)
    if r < 1.05:
        return 3, "full"
    if r < 1.10:
        return 2, "stretched"
    if r < 1.32:
        return 1, "partial"
    return 0, "none"


def angle(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def lab(i):
    return "%s%d" % (sym[i], i + 1)


# ---------------------------------------------------------------- input
mol = Chem.MolFromSmiles(smiles)
sym, X0, E0 = read_xyz("xtbopt.xyz")
if mol is None or sym is None:
    sys.exit()
n, nh = len(sym), mol.GetNumAtoms()
if [a.GetSymbol() for a in mol.GetAtoms()] != sym[:nh]:
    P("REARRANGEMENT ANALYSIS skipped: atom order in xtbopt.xyz does not match the SMILES.")
    sys.exit()


def real_radical(a):
    return (a.GetNumRadicalElectrons() > 0 and a.GetSymbol() in ORGANIC
            and all(x.GetSymbol() in ORGANIC for x in a.GetNeighbors()))


centers = [a.GetIdx() for a in mol.GetAtoms() if (a.GetSymbol() == "C" and a.GetFormalCharge() > 0)
           or (a.GetSymbol() in ("C", "N", "O", "S") and real_radical(a))]      # incl. alkoxy / aminyl radicals
# onium ions (e.g. a bromonium [Br+] bridging two carbons): both carbons carry the positive character
ONIUM = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() > 0
         and a.GetSymbol() in ("Cl", "Br", "I", "S", "Se", "O", "N") and a.GetDegree() >= 2
         and sum(1 for x in a.GetNeighbors() if x.GetSymbol() == "C") >= 2]
for o in ONIUM:
    for x in mol.GetAtomWithIdx(o).GetNeighbors():
        if x.GetSymbol() == "C" and x.GetIdx() not in centers:
            centers.append(x.GetIdx())
# carbons that a nucleophile does not trap: ester / acid / amide carbonyl carbons
ACYL = set()
for a in mol.GetAtoms():
    if a.GetSymbol() == "C":
        dbl_o = any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() == "O" for b in a.GetBonds())
        het = sum(1 for x in a.GetNeighbors() if x.GetSymbol() in ("O", "N", "S"))
        if dbl_o and het >= 2:
            ACYL.add(a.GetIdx())
# heteroatoms of those groups (their participation is reported, but not used as the prediction)
ACYL_HET = set(x.GetIdx() for i in ACYL for x in mol.GetAtomWithIdx(i).GetNeighbors() if x.GetSymbol() in ("O", "N", "S"))
UNSPEC = [i for i, t in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False) if t == "?"]
# anions (deprotonated species from the scenario analysis): the negatively charged C/N/O/S atom
if not centers and CHG < 0 and UHF == 0:
    centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() < 0 and a.GetSymbol() in ("C", "N", "O", "S")]
# radical ions made from a neutral SMILES: the centers come from where the hole / extra electron sits
_cf = name + "_centers.txt"
if os.path.exists(_cf) and UHF > 0:
    _m = re.search(r"CENTERS\|\s*([\d ]+)", open(_cf).read())
    if _m:
        centers = [int(t) - 1 for t in _m.group(1).split() if 0 < int(t) <= nh] or centers
if not centers:
    sys.exit()                      # neutral closed shell: nothing to rearrange
radical = UHF > 0 or any(real_radical(a) for a in mol.GetAtoms())
kind = "radical" if radical else ("anion" if CHG < 0 else "cation")
fcol, fname = (2, "f(0)") if radical else ((1, "f(-)") if kind == "anion" else (0, "f(+)"))
KIND_TXT = kind if not radical or CHG == 0 else ("radical cation" if CHG > 0 else "radical anion")
NU_TXT = "site where an electrophile bonds" if kind == "anion" else "nucleophile/trapping site"
NU_TAG = "E+ at" if kind == "anion" else "Nu at"

# reference bonding: SMILES heavy-atom bonds + each H to its nearest heavy atom
D = Chem.GetDistanceMatrix(mol)
# H atoms: same order as RDKit AddHs (how the structure was built), so each H
# keeps its SMILES parent even if it has moved during the optimization
molH = Chem.AddHs(mol)
parent = {}
if molH.GetNumAtoms() == n and all(molH.GetAtomWithIdx(h).GetSymbol() == "H" for h in range(nh, n)):
    for h in range(nh, n):
        parent[h] = molH.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
else:
    for h in range(nh, n):
        parent[h] = min(range(nh), key=lambda j: dist(X0, h, j))
ref = set(frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds())
ref |= set(frozenset((h, p)) for h, p in parent.items())


def topo(i, j):
    a, b = parent.get(i, i), parent.get(j, j)
    return int(D[a][b]) + (i >= nh) + (j >= nh)


wbo = {}
for l in read("wbo").splitlines():
    p_ = l.split()
    if len(p_) == 3:
        a, b = int(p_[0]) - 1, int(p_[1]) - 1
        wbo[(a, b)] = wbo[(b, a)] = float(p_[2])
FK0 = fukui(name + "_fukui.out")
ring_info = mol.GetRingInfo()


def small_ring(i, j):
    if i >= nh or j >= nh:
        return 0
    b = mol.GetBondBetweenAtoms(i, j)
    if b is None:
        return 0
    for s in (3, 4):
        if ring_info.IsBondInRingOfSize(b.GetIdx(), s):
            return s
    return 0


def p_axis(X, c, nbrs):
    if len(nbrs) < 2:
        return None
    B = np.array([(X[j] - X[c]) / dist(X, c, j) for j in nbrs])
    return np.linalg.svd(B)[2][-1]


def ref_nbrs(c):
    return [j for j in range(n) if frozenset((c, j)) in ref]


def shield(X, t, axis, skip):
    occ = 0.0
    for k in range(n):
        if k == t or k in skip:
            continue
        v = X[k] - X[t]
        d = np.linalg.norm(v)
        if d > 4.5 or np.dot(v, axis) / d < 0.5:
            continue
        occ += (2.0 if sym[k] != "H" else 1.0) / d ** 2
    return occ


def bond_states(X):
    """all pairs that are bonded in the reference OR at least partially bonded in X"""
    st = {}
    for i in range(n):
        for j in range(i + 1, n):
            if sym[i] == "H" and sym[j] == "H":
                continue
            key = frozenset((i, j))
            r = ratio(X, i, j)
            if key in ref or r < 1.32:
                st[key] = state(r)
    return st


def changes(Xa, Xb):
    """bond-state changes going from geometry Xa to Xb"""
    sa, sb = bond_states(Xa), bond_states(Xb)
    out = []
    for key in set(sa) | set(sb):
        i, j = sorted(key)
        ra = state(ratio(Xa, i, j))
        rb = state(ratio(Xb, i, j))
        if ra[0] != rb[0] and abs(ra[0] - rb[0]) >= 2:
            out.append((i, j, ra[1], rb[1], dist(Xa, i, j), dist(Xb, i, j)))
    return sorted(out)


def nu_sites(X, FK):
    """carbon sites ranked by Fukui; carbonyl carbons excluded (acyl attack is not cation trapping)"""
    carbonyl = set()
    for i in range(n):
        if sym[i] == "C":
            for j in range(n):
                if sym[j] == "O" and dist(X, i, j) < 1.26:
                    carbonyl.add(i)
    cs = [i for i in range(n) if sym[i] == "C" and i not in carbonyl and i not in ACYL and i in FK]
    cs.sort(key=lambda i: -FK[i][fcol])
    rows = []
    for i in cs[:3]:
        nb = [j for j in range(n) if j != i and ratio(X, i, j) < 1.10]
        weak = [j for j in range(n) if j != i and 1.10 <= ratio(X, i, j) < 1.32]
        if len(nb) == 3:
            ax = p_axis(X, i, nb)
            sh = min(shield(X, i, ax, nb), shield(X, i, -ax, nb))
            how = "open face"
        elif weak:
            far = max(weak, key=lambda j: dist(X, i, j))
            ax = (X[i] - X[far]) / dist(X, i, far)
            sh = shield(X, i, ax, nb + weak)
            how = "backside of %s" % lab(far)
        elif nb:
            far = max(nb, key=lambda j: dist(X, i, j))
            ax = (X[i] - X[far]) / dist(X, i, far)
            sh = shield(X, i, ax, nb)
            how = "backside of %s" % lab(far)
        else:
            sh, how = 0.0, "-"
        rows.append((i, FK[i][fcol], sh, how))
    if not rows:
        return rows, None, ""
    best, why = rows[0][0], "highest %s" % fname
    if len(rows) > 1 and rows[0][1] - rows[1][1] < 0.02 and rows[1][2] < rows[0][2]:
        best, why = rows[1][0], "%s similar (<0.02), less crowded" % fname
    return rows, best, why


# ---------------------------------------------------------------- report header
P("-" * 70)
P(" REARRANGEMENT / SCISSION ANALYSIS  (%s; GFN2-xTB, %s)" % (KIND_TXT, ("ALPB " + SOLV) if SOLV else "gas phase"))
P("-" * 70)
P("energies: GFN2-xTB%s" % ("; refined with g-xTB single points (gas phase g-xTB%s)" % (" + GFN2 ALPB solvation" if SOLV else "")
                            if GXTB else " only (set GXTB to a g-xTB-enabled xtb for better energies)"))
if UNSPEC:
    P("note: %d stereocenter(s) not specified in the SMILES (%s): RDKit chose one arrangement at random."
      % (len(UNSPEC), ", ".join(lab(i) for i in UNSPEC)))
    P("      For cages, the face of attack (exo/endo) changes the answer -- add @/@@ to the SMILES.")
if ONIUM:
    P("onium ion: %s bridges %s; both carbons are treated as cationic centers"
      % (", ".join(lab(o) for o in ONIUM),
         " and ".join(lab(x.GetIdx()) for o in ONIUM for x in mol.GetAtomWithIdx(o).GetNeighbors() if x.GetSymbol() == "C")))
if not SOLV and CHG != 0:
    P("note: ion computed in the gas phase. Ions often bridge/rearrange differently in solution;")
    P("      consider rerunning with the reaction solvent (run \"SMILES\" name SOLVENT).")

# ---- 0. undo artefacts: the ester (or amide) captured a proton or bonded to the cation during the optimization
fixes = []
for h in range(nh, n):
    if ratio(X0, h, parent[h]) >= 1.32 and any(ratio(X0, h, o) < 1.32 for o in ACYL_HET):
        fixes.append((h, parent[h], 1.09))
for o in ACYL_HET:
    for c in range(nh):
        if c != o and frozenset((o, c)) not in ref and sym[c] == "C" and ratio(X0, o, c) < 1.32:
            fixes.append((o, c, None))
if fixes:
    _, Xi, _ = read_xyz(name + ".xyz")
    if Xi is not None and len(Xi) == n:
        fd = os.path.join("rearr", "startfix")
        os.makedirs(fd, exist_ok=True)
        write_xyz(os.path.join(fd, "start.xyz"), sym, Xi)
        with open(os.path.join(fd, "fix.inp"), "w") as f:
            f.write("$constrain\n   force constant=1.0\n")
            for a_, b_, v in fixes:
                f.write("   distance: %d, %d, %.3f\n" % (a_ + 1, b_ + 1, v if v else max(3.0, dist(Xi, a_, b_))))
            f.write("$end\n")
        run_xtb(["start.xyz", "--opt", "--input", "fix.inp"], fd, "opt.out")
        _, Xf, Ef = read_xyz(os.path.join(fd, "xtbopt.xyz"))
        if Xf is not None:
            P("\nnote: during the free optimization the ester/amide group took part (%s)."
              % ", ".join("%s-%s" % (lab(a_), lab(b_)) for a_, b_, v in fixes))
            P("      That is a typical artefact of ions without solvent; those distances were held at their")
            P("      starting values and the analysis below starts from that constrained structure.")
            X0, E0 = Xf, Ef
            run_xtb(["xtbopt.xyz", "--vfukui"], fd, "fukui.out")
            FK0 = fukui(os.path.join(fd, "fukui.out")) or FK0
            wbo = {}
            for l in read(os.path.join(fd, "wbo")).splitlines():
                p_ = l.split()
                if len(p_) == 3:
                    a_, b_ = int(p_[0]) - 1, int(p_[1]) - 1
                    wbo[(a_, b_)] = wbo[(b_, a_)] = float(p_[2])

# ---- 1. is the optimized structure already bridged / rearranged?
obs, act = [], []          # obs: real bonding changes; act: bonds only stretched (activated)
for key in ref:
    i, j = sorted(key)
    s_ = state(ratio(X0, i, j))
    if s_[0] <= 1:
        obs.append(("broken/partial", i, j, dist(X0, i, j), s_[1]))
    elif s_[0] == 2:
        act.append(("stretched", i, j, dist(X0, i, j), s_[1]))
for i in range(n):
    for j in range(i + 1, n):
        if frozenset((i, j)) in ref or (sym[i] == "H" and sym[j] == "H"):
            continue
        s_ = state(ratio(X0, i, j))
        if s_[0] >= 1:
            obs.append(("new contact", i, j, dist(X0, i, j), s_[1]))
# an H that has (partly) left its carbon belongs to the nearest other heavy atom: the transfer is complete
# (GFN2 sometimes stops H-atom / hydride transfers half-way, e.g. alkoxy-radical 1,5-HAT)
for what, i, j, d, s_ in list(obs):
    if what == "broken/partial" and (sym[i] == "H") != (sym[j] == "H"):
        h, p_ = (i, j) if sym[i] == "H" else (j, i)
        others = [k for k in range(nh) if k != p_]
        if others:
            y = min(others, key=lambda k: dist(X0, h, k))
            if ratio(X0, h, y) < 1.6 and not any({h, y} == {a_, b_} for _, a_, b_, _, _ in obs):
                obs.append(("new contact", y, h, dist(X0, y, h), "partial"))
if obs:
    P("\n1. OPTIMIZED STRUCTURE vs SMILES: bonding changed during the optimization")
    for what, i, j, d, s_ in sorted(obs + act, key=lambda x: x[0]):
        P("   %-15s %-12s %6.3f A  (%s)" % (what, lab(i) + "-" + lab(j), d, s_))
    P("   -> the ion/radical is bridged (nonclassical) or rearranged on its own: no barrier at this level")
elif act:
    P("\n1. OPTIMIZED STRUCTURE vs SMILES: no bonds broken, but these bonds are activated (stretched):")
    for what, i, j, d, s_ in act:
        P("   %-15s %-12s %6.3f A" % (what, lab(i) + "-" + lab(j), d))
    P("   -> hyperconjugative weakening; the scans below test whether they actually break")
else:
    P("\n1. OPTIMIZED STRUCTURE vs SMILES: no bond changes (classical structure)")

# ---- 2. enumerate candidates
cands = []
# extra conformers (CREST ensemble, conformers.py): bond-forming steps that need a folded chain
# (1,5-HAT, ring closure) are looked for in every conformer; scans start from the best one
CONFS = [X0]
if os.path.exists("conformers_extra.xyz"):
    _L = read("conformers_extra.xyz").split("\n")
    _k = 0
    while _k < len(_L) and _L[_k].strip():
        _m = int(_L[_k].split()[0])
        if _m == n:
            CONFS.append(np.array([[float(v) for v in l.split()[1:4]] for l in _L[_k + 2:_k + 2 + _m]]))
        _k += _m + 2
if len(CONFS) > 1:
    P("   (%d extra low-energy conformers from the conformer search are also searched for steps)" % (len(CONFS) - 1))
for _ci, XC in enumerate(CONFS):
 for R in centers:
     nbR = ref_nbrs(R)
     ax = p_axis(XC, R, [j for j in nbR if j not in ONIUM])
     alpha = [j for j in nbR if j < nh]
     for T in (range(n) if kind != "anion" else []):
         if T == R or frozenset((R, T)) in ref:
             continue
         d = dist(XC, R, T)
         if T in ACYL or T in ACYL_HET:
             continue
         if T < nh:
             dt = int(D[R][T])
             if not ((dt == 2 and d <= 3.2) or (dt >= 3 and d <= 3.0)):
                 continue
             ws = [wbo.get((T, j), 1.0) for j in ref_nbrs(T) if j < nh]
             weak = 1 + 3 * max(0.0, 1.0 - min(ws)) if ws else 1.0
             ring = 1.5 if any(len(r) == 3 and T in r for r in ring_info.AtomRings()) else 1.0
         else:
             pT = parent[T]
             dp = int(D[R][pT])
             if kind == "radical" and dp == 1:
                 continue                      # 1,2-H shifts in radicals: not competitive
             if not ((dp == 1 and d <= 2.4) or (dp >= 2 and d <= 2.7)):
                 continue
             weak = 1 + 3 * max(0.0, 0.97 - wbo.get((pT, T), 0.97))
             ring = 1.0
         orient = abs(np.dot(ax, (XC[T] - XC[R]) / d)) if ax is not None else 0.5
         score = (0.25 + orient) * np.exp(-(d - 2.0) / 0.6) * weak * ring
         cands.append(dict(kind="form", R=R, a=R, b=T, d0=d, orient=orient, score=score, X=XC, conf=_ci))
     for A in alpha:
         for Xb in ref_nbrs(A):
             if Xb == R or Xb >= nh:
                 continue
             sr = small_ring(A, Xb)
             if kind in ("cation", "anion") and not sr:
                 continue
             al = abs(np.dot(ax, (XC[Xb] - XC[A]) / dist(XC, A, Xb))) if ax is not None else 0.5
             weak = 1 + 3 * max(0.0, 1.0 - wbo.get((A, Xb), 1.0))
             score = (0.25 + al ** 2) * weak * (1.5 if sr else 1.0)
             cands.append(dict(kind="break", R=R, a=A, b=Xb, d0=dist(XC, A, Xb), orient=al, score=score, X=XC, conf=_ci,
                               frag=bool(sr) and kind == "cation"))
# ---- cyclizations onto C=C / C#C, found by structure (Baldwin / Beckwith), whatever the current distance
BALDWIN_OK = {("exo", "tet"): range(3, 8), ("exo", "trig"): range(3, 8), ("endo", "trig"): range(6, 8),
              ("exo", "dig"): range(5, 8), ("endo", "dig"): range(3, 8)}


def _beckwith(n, mode, geom, subst):
    """radical cyclization guideline (Beckwith): short verdict"""
    if mode == "exo" and n == 5:
        v = "preferred (fastest common radical cyclization)"
    elif mode == "exo" and n == 6:
        v = "slower than 5-exo; competitive when no 5-exo is possible"
    elif mode == "endo" and n == 6:
        v = "usually minor next to 5-exo (gains when the 5-exo carbon is substituted)"
    elif n in (3, 4):
        v = "fast but reversible (the strained ring reopens)"
    elif mode == "endo" and n == 5:
        v = "disfavored"
    else:
        v = "slow"
    if subst and mode == "exo":
        v += "; attacked carbon is substituted, which slows this mode"
    return v


CYCINFO = {}
if kind in ("radical", "cation"):
    for R in centers:
        for b in mol.GetBonds():
            bt = b.GetBondType()
            if bt not in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) or b.GetIsAromatic():
                continue
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if R in (i, j) or mol.GetAtomWithIdx(i).GetSymbol() != "C" or mol.GetAtomWithIdx(j).GetSymbol() != "C":
                continue
            for T, U in ((i, j), (j, i)):
                if frozenset((R, T)) in ref:
                    continue
                path = list(Chem.GetShortestPath(mol, R, T))
                nring = len(path)
                if not 3 <= nring <= 7:
                    continue
                mode = "endo" if U in path else "exo"
                geom = "dig" if bt == Chem.BondType.TRIPLE else "trig"
                bald = "favored" if nring in BALDWIN_OK.get((mode, geom), ()) else "disfavored"
                subst = sum(1 for x in mol.GetAtomWithIdx(T).GetNeighbors() if x.GetSymbol() != "H" and x.GetIdx() != U) > 1
                beck = _beckwith(nring, mode, geom, subst) if kind == "radical" else ""
                rank = {("exo", 5): 1, ("exo", 6): 2, ("endo", 6): 3, ("exo", 3): 4, ("exo", 4): 5}.get((mode, nring), 6)
                if bald == "disfavored":
                    rank += 4
                name_ = "%d-%s-%s" % (nring, mode, geom)
                CYCINFO[frozenset((R, T))] = dict(R=R, T=T, name=name_, baldwin=bald, beckwith=beck, rank=rank,
                                                  label="[%s: Baldwin %s%s]" % (name_, bald,
                                                                                  "; Beckwith " + beck.split(";")[0]
                                                                                  if beck else ""))
                cands.append(dict(kind="form", R=R, a=R, b=T, d0=dist(X0, R, T), orient=0.0, score=0.0, X=X0, conf=0,
                                  cyc=True))
# complete an already-stretched bond (bridged start)
for what, i, j, d, s_ in obs + act:
    if what in ("broken/partial", "stretched") and s_ in ("stretched", "partial") and i < nh and j < nh \
            and frozenset((i, j)) not in EXCL:
        cands.append(dict(kind="break", R=centers[0], a=i, b=j, d0=d, orient=1.0, score=99.0))
# atoms of an explicit Lewis acid (B, Al, or the Si of a silyloxonium): its bonds are not reaction steps
LA_AT = set(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("B", "Al") or (a.GetSymbol() == "Si" and any(
    x.GetSymbol() == "O" and x.GetFormalCharge() > 0 for x in a.GetNeighbors())))
LA_AT |= set(x.GetIdx() for i in list(LA_AT) for x in mol.GetAtomWithIdx(i).GetNeighbors() if x.GetSymbol() in ("F", "Cl"))
cands = [c for c in cands if c["a"] not in LA_AT and c["b"] not in LA_AT]
seen, todo = set(), []
for c in sorted(cands, key=lambda c: -c["score"]):
    key = (c["kind"], frozenset((c["a"], c["b"])))
    if key in seen or frozenset((c["a"], c["b"])) in EXCL:
        continue
    seen.add(key)
    todo.append(c)
# fragmentation of a strained ring next to a cation (cyclopropyl-/cyclobutylcarbinyl, Grob-type) is always
# scanned: in a bridged (non-classical) geometry its orientation score can look poor although it is the real path
_frag = [c for c in todo[NSCAN:] if c.get("frag")][:3]
# the best cyclizations by Baldwin / Beckwith are always scanned, even if the chain is far away now
_cyc = sorted([c for c in todo[NSCAN:] if frozenset((c["a"], c["b"])) in CYCINFO and c["kind"] == "form"],
              key=lambda c: CYCINFO[frozenset((c["a"], c["b"]))]["rank"])[:4]
todo = todo[:NSCAN] + _frag + _cyc

P("\n2. CANDIDATE STEPS (ranked by orbital alignment x proximity x bond weakening)")
P("   %-6s %-14s %7s %9s %7s" % ("type", "coordinate", "d (A)", "p-align", "score"))
for c in todo:
    P("   %-6s %-14s %7.3f %9.2f %7.2f%s" % (c["kind"], lab(c["a"]) + "-" + lab(c["b"]), c["d0"], c["orient"], c["score"],
                                           ("   " + CYCINFO[frozenset((c["a"], c["b"]))]["label"])
                                           if c["kind"] == "form" and frozenset((c["a"], c["b"])) in CYCINFO else ""))
P("   p-align = |cos| between the center's p orbital and the bond/approach vector (1 = ideal overlap)")

# ---------------------------------------------------------------- 3. scans
RANK = {"none": 0, "partial": 1, "stretched": 2, "full": 3}


def formed_broken(chg):
    return ([(a_, b_) for a_, b_, sa, sb, da, db in chg if RANK[sb] > RANK[sa]],
            [(a_, b_) for a_, b_, sa, sb, da, db in chg if RANK[sb] < RANK[sa]])


def is_sigma(p_, q_):
    if p_ >= nh or q_ >= nh:
        return True
    bd = mol.GetBondBetweenAtoms(p_, q_)
    return bd is not None and bd.GetBondType() == Chem.BondType.SINGLE and not bd.GetIsAromatic()


def scan_relax(Xs, i, j, d1, wd):
    """relaxed scan of distance i-j from its value in Xs to d1, then free relaxation of the end point"""
    os.makedirs(wd, exist_ok=True)
    d0 = dist(Xs, i, j)
    write_xyz(os.path.join(wd, "start.xyz"), sym, Xs)
    with open(os.path.join(wd, "scan.inp"), "w") as f:
        f.write("$constrain\n   force constant=1.0\n   distance: %d, %d, %.4f\n"
                "$scan\n   1: %.4f, %.4f, %d\n$end\n" % (i + 1, j + 1, d0, d0, d1, NSTEP))
    run_xtb(["start.xyz", "--opt", "--input", "scan.inp"], wd, "scan.out")
    F = scan_frames(os.path.join(wd, "xtbscan.log"))
    if len(F) < 3:
        return None
    E = np.array([e for e, _ in F])
    imax = int(np.argmax(E))
    resp = []
    for key in ref:
        p_, q_ = sorted(key)
        if frozenset((p_, q_)) != frozenset((i, j)) and is_sigma(p_, q_):
            resp.append((dist(F[-1][1], p_, q_) - dist(F[0][1], p_, q_), p_, q_))
    resp.sort(reverse=True)
    k = imax + 1 + int(np.argmin(E[imax + 1:])) if imax < len(E) - 1 else len(E) - 1
    rd = os.path.join(wd, "relax")
    os.makedirs(rd, exist_ok=True)
    write_xyz(os.path.join(rd, "guess.xyz"), sym, F[k][1])
    run_xtb(["guess.xyz", "--opt"], rd, "opt.out")
    _, Xp, Ep = read_xyz(os.path.join(rd, "xtbopt.xyz"))
    return dict(F=F, E=E, imax=imax, Xp=Xp, Ep=Ep, rd=rd,
                stretch=[(dd, p_, q_) for dd, p_, q_ in resp[:3] if dd > 0.10])


def break_target(i, j, X):
    return {3: 2.50, 4: 2.70}.get(small_ring(i, j), dist(X, i, j) + 1.4)


os.makedirs("rearr", exist_ok=True)
ER0 = refined(X0, E0, "start") if GXTB else None
results = []
for c in todo:
    i, j = c["a"], c["b"]
    tag = "%s_%d_%d" % (c["kind"], i + 1, j + 1)
    wd = os.path.join("rearr", tag)
    XS = c.get("X", X0)
    d0 = dist(XS, i, j)
    d1 = 1.05 * (rcov(sym[i]) + rcov(sym[j])) if c["kind"] == "form" else break_target(i, j, XS)
    res = dict(c, tag=tag, d1=d1, note="")
    if c["kind"] == "form" and d0 <= d1 + 0.03:
        res["note"] = "already bonded at the start"
        results.append(res)
        continue
    s1 = scan_relax(XS, i, j, d1, wd)
    if s1 is None or s1["Xp"] is None:
        res["note"] = "scan or relaxation failed"
        results.append(res)
        continue
    Eref = E0 if E0 is not None else s1["E"][0]
    res["profile"] = (s1["E"] - Eref) * KCAL
    res["barrier"] = float(res["profile"].max())
    res["stretch"] = s1["stretch"]
    Xmax, Emax = s1["F"][s1["imax"]][1], s1["E"][s1["imax"]]
    Xp, Ep, rdir = s1["Xp"], s1["Ep"], s1["rd"]
    chg = changes(X0, Xp)
    fo, br = formed_broken(chg)
    # bridged intermediate (T now bonded to both R and its old neighbour): complete the 1,2-shift
    if c["kind"] == "form" and j < nh and fo and not br:
        As = [A for A in ref_nbrs(j) if A < nh and frozenset((A, c["R"])) in ref]
        if As and ratio(Xp, As[0], j) < 1.10:
            A = As[0]
            res["inter"] = ((Ep - Eref) * KCAL, A, j)
            res["Xinter"], res["Einter"] = Xp, Ep
            s2 = scan_relax(Xp, A, j, break_target(A, j, Xp), os.path.join(wd, "complete"))
            if s2 is not None and s2["Xp"] is not None:
                chg2 = changes(X0, s2["Xp"])
                fo2, br2 = formed_broken(chg2)
                if br2:
                    if s2["E"][s2["imax"]] > Emax:
                        Xmax, Emax = s2["F"][s2["imax"]][1], s2["E"][s2["imax"]]
                    res["barrier"] = float((Emax - Eref) * KCAL)
                    Xp, Ep, rdir, chg = s2["Xp"], s2["Ep"], s2["rd"], chg2
                    res["two_step"] = True
    res["dE"] = (Ep - Eref) * KCAL
    res["chg"] = chg
    res["Xp"] = Xp
    if GXTB and ER0 is not None:
        erm = refined(Xmax, Emax, tag + "_max")
        erp = refined(Xp, Ep, tag + "_prod")
        if None not in (erm, erp):
            res["barrier_g"] = (erm - ER0) * KCAL
            res["dE_g"] = (erp - ER0) * KCAL
        if "inter" in res:
            eri = refined(res["Xinter"], res["Einter"], tag + "_inter")
            if eri is not None:
                res["inter_g"] = (eri - ER0) * KCAL
    if chg:
        write_xyz(os.path.join(wd, "product.xyz"), sym, Xp, "relaxed product of %s" % tag)
        run_xtb(["xtbopt.xyz", "--vfukui"], rdir, "fukui.out")
        res["FK"] = fukui(os.path.join(rdir, "fukui.out"))
    results.append(res)

P("\n3. RELAXED xTB SCANS (imposed activation; energies relative to the optimized start)")
for r in results:
    co = lab(r["a"]) + "-" + lab(r["b"])
    if r["note"]:
        P("\n   %s %s: %s" % (r["kind"], co, r["note"]))
        continue
    P("\n   %s %s  (%.2f -> %.2f A)" % (r["kind"], co, r["d0"], r["d1"]))
    P("     barrier along scan : %5.1f kcal/mol%s" % (r["barrier"], "   (still rising at the end: no product basin)"
                                                       if int(np.argmax(r["profile"])) == len(r["profile"]) - 1 else ""))
    if "inter" in r:
        ei, A_, T_ = r["inter"]
        P("     bridged intermediate: %s bonded to both %s and %s at %+.1f kcal/mol%s"
          % (lab(T_), lab(r["a"]), lab(A_), r.get("inter_g", ei), " (g-xTB)" if "inter_g" in r else ""))
        P("     %s" % ("second scan breaks %s-%s: 1,2-shift completed (two steps)" % (lab(A_), lab(T_))
                       if r.get("two_step") else "the shift does not complete: the bridged species is the product"))
    if "barrier_g" in r:
        P("     g-xTB refined      : barrier %5.1f, reaction energy %+5.1f kcal/mol" % (r["barrier_g"], r["dE_g"]))
    if r.get("stretch"):
        P("     bonds that stretch : " + ", ".join("%s-%s +%.2f A" % (lab(p_), lab(q_), dd) for dd, p_, q_ in r["stretch"]))
    if "dE" in r:
        if r["chg"]:
            P("     relaxed product    : %+5.1f kcal/mol; bond changes:" % r["dE"])
            for a_, b_, sa, sb, da, db in r["chg"]:
                P("         %-12s %-9s -> %-9s (%.2f -> %.2f A)" % (lab(a_) + "-" + lab(b_), sa, sb, da, db))
        else:
            P("     relaxed product    : returns to the starting structure (no separate minimum)")

def new_ring_size(Xg, bond):
    """size of the smallest ring closed by `bond` in geometry Xg (0 if no ring)"""
    a0, b0 = bond
    adj = {i: [j for j in range(n) if j != i and ratio(Xg, i, j) < 1.32 and {i, j} != {a0, b0}] for i in range(n)}
    seen, frontier, depth = {a0}, [a0], 1
    while frontier and depth < 9:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v == b0:
                    return depth + 1
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier, depth = nxt, depth + 1
    return 0


def classify(form, brk, Xg):
    if form and brk:
        m = set(form) & set(brk)
        if m:
            M = m.pop()
            new_p = form[0] if form[1] == M else form[1]
            old_p = brk[0] if brk[1] == M else brk[1]
            if sym[M] == "H":
                if sym[new_p] in ("O", "N", "S") and kind != "radical":
                    return "intramolecular proton transfer to %s (E1-type loss of H+, alkene forms)" % lab(new_p)
                return ("1,%d-hydride shift" % (topo(new_p, old_p) + 1)) if kind == "cation" else "1,%d-H-atom transfer" % (topo(new_p, old_p) + 1)
            rs = new_ring_size(Xg, form)
            ring_txt = (", new %d-membered ring" % rs) if rs else ""
            shift = "%s shifts from %s to %s%s" % (lab(M), lab(old_p), lab(new_p), ring_txt)
            if small_ring(*brk) == 3:
                return "cyclopropane C-C bond breaks: " + shift
            if frozenset((old_p, new_p)) in ref:
                arom = M < nh and mol.GetAtomWithIdx(M).GetIsAromatic()
                if kind == "radical":
                    label = "1,2-aryl shift (neophyl-type)" if arom else "1,2-shift (radical)"
                else:
                    label = "1,2-aryl shift (via phenonium/bridged ion)" if arom else "1,2-shift (Wagner-Meerwein)"
                return label + ": " + shift
            return "sigma-bond participation: " + shift
        if small_ring(*brk) == 3:
            return "cyclopropane C-C bond participation"
        return "sigma-bond participation"
    if brk and (brk[0] in ONIUM or brk[1] in ONIUM):
        c_ = brk[1] if brk[0] in ONIUM else brk[0]
        return "%s-onium opens: positive charge on %s" % (sym[brk[0] if brk[0] in ONIUM else brk[1]], lab(c_))
    if brk:
        if small_ring(*brk):
            return "strained-ring opening (%d-ring)" % small_ring(*brk)
        return "beta-scission" if kind == "radical" else ("anionic fragmentation" if kind == "anion" else "fragmentation")
    if form and kind == "cation" and any(sym[x] in ("O", "N", "S") for x in form):
        het = form[0] if sym[form[0]] in ("O", "N", "S") else form[1]
        rs = new_ring_size(Xg, form)
        return "intramolecular trapping by %s%s" % (lab(het), (" (%d-membered ring)" % rs) if rs else "")
    return "bond formation (participation / cyclization)"


def carbocycle3(b):
    """True if bond b lies in a three-membered ring made only of carbons"""
    if b[0] >= nh or b[1] >= nh:
        return False
    return any(len(r) == 3 and b[0] in r and b[1] in r and all(sym[k] == "C" for k in r)
               for r in ring_info.AtomRings())


def pick_pair(formed, broken):
    """choose the formed/broken pair that describes one elementary step"""
    pairs = [(f, b) for f in formed for b in broken if set(f) & set(b)]
    for test in (lambda f, b: sym[(set(f) & set(b)).pop()] == "H",
                 lambda f, b: carbocycle3(b),
                 lambda f, b: sym[b[0]] == "C" and sym[b[1]] == "C"):
        for f, b in pairs:
            if test(f, b):
                return f, b
    if pairs:
        return pairs[0]
    cc = [b for b in broken if carbocycle3(b)]
    return (formed[0] if formed else None), (cc[0] if cc else (broken[0] if broken else None))


def hetero_participation(chg):
    """bond formation to an ester/amide heteroatom (neighbouring-group participation or proton transfer)"""
    return any(RANK[sb] > RANK[sa] and (a_ in ACYL_HET or b_ in ACYL_HET) for a_, b_, sa, sb, da, db in chg)


# ---------------------------------------------------------------- 4. prediction
_cy_rows = [(CYCINFO[frozenset((r["a"], r["b"]))], r) for r in results
            if r.get("kind") == "form" and frozenset((r["a"], r["b"])) in CYCINFO and "barrier" in r]
if _cy_rows:
    P("\n   CYCLIZATIONS TESTED (ring size / exo-endo / trig-dig; Baldwin and, for radicals, Beckwith):")
    for ci_, r in sorted(_cy_rows, key=lambda x: x[0]["rank"]):
        P("     %-8s %-14s barrier %5.1f   dE %+6.1f   Baldwin %-10s %s" % (
            lab(ci_["R"]) + "-" + lab(ci_["T"]), ci_["name"], r.get("barrier_g", r["barrier"]), r.get("dE_g", r["dE"]),
            ci_["baldwin"], ("Beckwith: " + ci_["beckwith"]) if ci_["beckwith"] else ""))
P("\n4. PREDICTION")
pred, notes = None, []
if obs:
    formed = [(i, j) for w, i, j, d, s_ in obs if w == "new contact" and frozenset((i, j)) not in EXCL]
    broken = [(i, j) for w, i, j, d, s_ in sorted(obs + act, key=lambda x: -x[3] / (rcov(sym[x[1]]) + rcov(sym[x[2]])))
              if w in ("broken/partial", "stretched") and frozenset((i, j)) not in EXCL]
    if formed or broken:
        f_, b_ = pick_pair(formed, broken)
        pred = dict(form=f_, brk=b_, dEa=0.0, dE=0.0, src="optimized structure (happens without a barrier at this level)",
                    X=X0, FK=FK0, all_f=formed, all_b=broken)
valid = [r for r in results if r.get("chg") and not hetero_participation(r["chg"])]
ngp = [r for r in results if r.get("chg") and hetero_participation(r["chg"])]
if ngp:
    notes.append("   (not used: %s -- the ester/amide group takes part there, which these reactions do not show)"
                 % ", ".join(r["tag"] for r in ngp))
bar = lambda r: r.get("barrier_g", r["barrier"])
rxe = lambda r: r.get("dE_g", r["dE"])
try:                                             # every scanned pathway with its product, for the DFT check list
    import json as _jpw
    _pw = []
    for _r in results:
        if _r.get("chg") and _r.get("Xp") is not None and "barrier" in _r:
            _f = [[int(a_), int(b_)] for a_, b_, sa, sb, *_ in _r["chg"] if sa == "none" and sb in ("full", "stretched")]
            _b = [[int(a_), int(b_)] for a_, b_, sa, sb, *_ in _r["chg"] if sa == "full" and sb in ("none", "partial")]
            if _f or _b:
                _pw.append(dict(tag=_r["tag"], barrier=float(bar(_r)), dE=float(rxe(_r)), form=_f, brk=_b,
                                xyz=os.path.join("rearr", _r["tag"], "product.xyz")))
    os.makedirs("rearr", exist_ok=True)
    _jpw.dump(_pw, open(os.path.join("rearr", "pathways.json"), "w"))
except Exception:
    pass
if valid:
    b0 = min(bar(r) for r in valid)
    comp = [r for r in valid if bar(r) <= b0 + TIE]
    best = min(comp, key=rxe)        # among kinetically competitive paths, the more stable product wins
    form = [(a_, b_) for a_, b_, sa, sb, da, db in best["chg"] if RANK[sb] > RANK[sa]]
    brk = [(a_, b_) for a_, b_, sa, sb, da, db in best["chg"] if RANK[sb] < RANK[sa]]
    if best["kind"] == "form" and not form:
        form = [(best["a"], best["b"])]
    if best["kind"] == "break" and not brk:
        brk = [(best["a"], best["b"])]
    if not brk and form:
        cand = [(dd, p_, q_) for dd, p_, q_ in best.get("stretch", []) if set((p_, q_)) & set(form[0])]
        if cand:
            brk = [(cand[0][1], cand[0][2])]
    fp, bp = pick_pair(form, brk)
    scan_pred = dict(form=fp, brk=bp, dEa=bar(best), dE=rxe(best), X=best["Xp"], FK=best.get("FK", {}),
                     all_f=form, all_b=brk,
                     src="relaxed scan %s%s" % (best["tag"], " (g-xTB energies)" if "barrier_g" in best else ""))
    facile = scan_pred["dEa"] <= FACILE and scan_pred["dE"] <= 5.0
    if not facile and scan_pred["dEa"] <= FACILE and scan_pred["dE"] <= UPHILL:
        uphill = scan_pred
    if pred is None and facile:
        pred = scan_pred
        if len(comp) > 1:
            notes.append("   competitive pathways (barriers within %.0f kcal/mol):" % TIE)
            for r in sorted(comp, key=bar):
                notes.append("       %-16s barrier %5.1f   reaction energy %+6.1f kcal/mol%s"
                             % (r["tag"], bar(r), rxe(r), "   <- chosen (more stable product)" if r is best else ""))
    elif pred is not None:
        notes.append("   (scans: best activated pathway %s, barrier %.1f, reaction energy %+.1f kcal/mol -- slower than the"
                     " barrierless change found in the optimization)" % (best["tag"], bar(best), rxe(best)))
    else:
        notes.append("   best scanned pathway %s: barrier %.1f, reaction energy %+.1f kcal/mol -> not facile "
                     "(limits: barrier <= %.0f, reaction energy <= +5)" % (best["tag"], bar(best), rxe(best), FACILE))
if pred is None:
    try:
        import json as _json
        os.makedirs("rearr", exist_ok=True)
        _up = None
        if uphill is not None and (uphill["form"] or uphill["brk"]):
            write_xyz(os.path.join("rearr", "predicted.xyz"), sym, uphill["X"], "uphill step (tried by the cascade)")
            _up = dict(form=[list(map(int, p_)) for p_ in uphill.get("all_f", [uphill["form"]] if uphill["form"] else [])],
                       brk=[list(map(int, p_)) for p_ in uphill.get("all_b", [uphill["brk"]] if uphill["brk"] else [])],
                       type=classify(uphill["form"], uphill["brk"], uphill["X"]), dEa=float(max(0.0, uphill["dEa"])),
                       dE=float(uphill["dE"]), src=uphill["src"], kind=kind)
        _json.dump(dict(step=False, kind=kind, uphill=_up), open(os.path.join("rearr", "predicted.json"), "w"))
    except Exception:
        pass
    for l in notes:
        P(l)
    P("   No facile rearrangement or scission found: the ion/radical is predicted to react at its original center.")
    rows, nb_, why = nu_sites(X0, FK0)
    if nb_ is not None:
        for i, fv, sh, how in rows:
            P("       %-6s %s %.3f   crowding %.3f (%s)" % (lab(i), fname, fv, sh, how))
        P("   predicted %s: %s  (%s)" % (NU_TXT, lab(nb_), why))
        P("PRED| no rearrangement | %s %s" % (NU_TAG, lab(nb_)))
        P("REARR| form 0 0 | break 0 0 | nu %d" % (nb_ + 1))
    sys.exit()
for l in notes:
    P(l)

f_, b_ = pred["form"], pred["brk"]
P("   source              : %s" % pred["src"])
if b_:
    P("   bond that breaks    : %s" % (lab(b_[0]) + "-" + lab(b_[1])))
if f_:
    P("   bond that forms     : %s" % (lab(f_[0]) + "-" + lab(f_[1])))
ptype = classify(f_, b_, pred["X"])
_ci_ = CYCINFO.get(frozenset(f_)) if f_ else None
if _ci_ is not None:
    ptype = "%s %scyclization" % (_ci_["name"], "radical " if kind == "radical" else "")
    P("   ring closure rules  : Baldwin %s%s" % (_ci_["baldwin"], ("; Beckwith: " + _ci_["beckwith"]) if _ci_["beckwith"] else ""))
    if _ci_["baldwin"] == "disfavored" or _ci_["rank"] > 3:
        P("   NOTE: the xtb scans favour a ring closure that the rules rate as less likely -- check with DFT.")
try:
    import json as _json
    os.makedirs("rearr", exist_ok=True)
    write_xyz(os.path.join("rearr", "predicted.xyz"), sym, pred["X"], "predicted product: %s" % ptype)
    _json.dump(dict(step=True, form=[list(map(int, p_)) for p_ in pred.get("all_f", [f_] if f_ else [])],
                    brk=[list(map(int, p_)) for p_ in pred.get("all_b", [b_] if b_ else [])],
                    type=ptype, dEa=float(max(0.0, pred["dEa"])), dE=float(pred["dE"]), src=pred["src"],
                    kind=kind), open(os.path.join("rearr", "predicted.json"), "w"))
except Exception:
    pass
P("   type                : %s" % ptype)
extra = ["%s-%s forms" % (lab(a_), lab(b_2)) for a_, b_2 in pred.get("all_f", []) if (a_, b_2) != f_] + \
        ["%s-%s breaks" % (lab(a_), lab(b_2)) for a_, b_2 in pred.get("all_b", []) if (a_, b_2) != b_]
if extra:
    P("   also                : " + ", ".join(extra))
if pred["src"].startswith("optimized"):
    P("   barrier             : none - it happened during the optimization itself")
else:
    btxt = "~0 (downhill along this coordinate)" if pred["dEa"] < 0.5 else "%.1f kcal/mol" % pred["dEa"]
    P("   barrier (scan max)  : %s    reaction energy: %+.1f kcal/mol" % (btxt, pred["dE"]))
if "proton transfer" in ptype:
    P("   the cation is quenched by losing a proton (alkene + protonated acceptor); no carbocation left to trap")
rows, nb_, why = nu_sites(pred["X"], pred["FK"])
if "proton transfer" in ptype:
    rows, nb_ = [], None
if rows:
    P("   charge/radical after the step (%s, carbons only, carbonyl C excluded):" % fname)
    for i, fv, sh, how in rows:
        P("       %-6s %s %.3f   crowding %.3f (%s)" % (lab(i), fname, fv, sh, how))
    P("   predicted %s: %s  (%s)" % (NU_TXT, lab(nb_), why))
if nb_ is not None and nb_ < nh:
    rb = [(bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()) for bd in mol.GetAtomWithIdx(nb_).GetBonds()
          if carbocycle3((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()))]
    if rb:
        P("   cyclopropane bonds at the nucleophile site %s (start -> after the step):" % lab(nb_))
        for i, j in rb:
            brk_ = b_ is not None and {i, j} == set(b_)
            P("       %-12s %.3f -> %.3f A   WBO %.2f   %s" % (lab(i) + "-" + lab(j), dist(X0, i, j), dist(pred["X"], i, j),
                                                         wbo.get((i, j), 0.0), "<- BREAKS" if brk_ else "stays"))
fl = (lab(f_[0]) + "-" + lab(f_[1])) if f_ else "-"
bl = (lab(b_[0]) + "-" + lab(b_[1])) if b_ else "-"
P("\n   Scan barriers are upper-bound estimates along one coordinate; confirm with DFT (TS search).")
P("PRED| %s breaks, %s forms, %s %s | %s | dEa %.1f | dE %+.1f" % (bl, fl, NU_TAG, lab(nb_) if nb_ is not None else "-",
                                                                  ptype.split(":")[0].split(" (")[0], max(0.0, pred["dEa"]), pred["dE"]))
fa = (f_[0] + 1, f_[1] + 1) if f_ else (0, 0)
ba = (b_[0] + 1, b_[1] + 1) if b_ else (0, 0)
P("REARR| form %d %d | break %d %d | nu %d" % (fa + ba + ((nb_ + 1) if nb_ is not None else 0,)))
