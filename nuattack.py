#!/usr/bin/env python3
"""
nuattack.py -- explicit nucleophile attack on a carbocation / onium ion (GFN2-xTB relaxed scans).

Usage (inside a job folder, after xtb optimization):
    python3 nuattack.py NAME "SMILES" XTB_PATH CHARGE UHF
Environment: SOLV (ALPB solvent), NU (preset name or SMILES, see nucleophiles.py; default water), NUA_STEPS (default 12)

For a cation it places one nucleophile molecule next to:
  * every cyclopropane carbon near the cationic center, on the backside of EACH of its ring bonds
    (attack with inversion: the bond on the other side is the one that can break), and
  * each cationic carbon (direct capture, from its less crowded face),
then pulls the nucleophile onto that carbon in a relaxed scan and lets the end point relax freely.
The quantum chemistry decides which bond breaks, whether a new ring closes (e.g. a bromonium being
opened by the electrons of the breaking cyclopropane bond) and how high the barrier is.
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
NU = os.environ.get("NU", "none").strip() or "none"
NSTEP = int(os.environ.get("NUA_STEPS", "12"))
TIE = float(os.environ.get("NUA_TIE", "3"))   # kcal/mol: closer barriers count as competitive
LOWBAR = float(os.environ.get("NUA_LOWBAR", "6"))   # both barriers below this: practically barrierless, a tie
PT = Chem.GetPeriodicTable()
P = print


def read(p):
    return open(p).read() if os.path.exists(p) else ""


def read_xyz(p):
    L = read(p).splitlines()
    if not L:
        return None, None, None
    k = int(L[0].split()[0])
    m = re.search(r"energy:\s*(-?\d+\.\d+)", L[1])
    return ([l.split()[0].capitalize() for l in L[2:2 + k]],
            np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + k]]), float(m.group(1)) if m else None)


def write_xyz(p, S, X, c=""):
    with open(p, "w") as f:
        f.write("%d\n%s\n" % (len(S), c))
        for s_, x in zip(S, X):
            f.write("%-2s %14.8f %14.8f %14.8f\n" % (s_, x[0], x[1], x[2]))


def frames(p):
    L = read(p).splitlines()
    out, k = [], 0
    while k < len(L) and L[k].strip():
        m_ = int(L[k].split()[0])
        e = float(re.search(r"(-?\d+\.\d+)", L[k + 1]).group(1))
        out.append((e, np.array([[float(v) for v in l.split()[1:4]] for l in L[k + 2:k + 2 + m_]])))
        k += m_ + 2
    return out


def xtb(args, cwd, out):
    cmd = [XTB] + args + ["--chrg", str(CHG + QNU), "--uhf", str(UHF), "--gfn", "2"] + (["--alpb", SOLV] if SOLV else [])
    with open(os.path.join(cwd, out), "w") as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)


mol = Chem.MolFromSmiles(smiles)
RADICAL = UHF > 0 and CHG == 0                   # neutral radical: radical partners instead of nucleophiles
if mol is None or (CHG <= 0 and not RADICAL):
    sys.exit()                                   # cations, radical cations, onium ions and neutral radicals
start = "rearr/startfix/xtbopt.xyz" if os.path.exists("rearr/startfix/xtbopt.xyz") else "xtbopt.xyz"
sym, X0, E0 = read_xyz(start)
if sym is None:
    sys.exit()
n, nh = len(sym), mol.GetNumAtoms()
if [a.GetSymbol() for a in mol.GetAtoms()] != sym[:nh]:
    sys.exit()
rc = lambda s_: PT.GetRcovalent(PT.GetAtomicNumber(s_))
centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and a.GetFormalCharge() > 0]
onium = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() > 0 and a.GetSymbol() != "C"
         and sum(1 for x in a.GetNeighbors() if x.GetSymbol() == "C") >= 2]
for o in onium:
    centers += [x.GetIdx() for x in mol.GetAtomWithIdx(o).GetNeighbors() if x.GetSymbol() == "C" and x.GetIdx() not in centers]
if RADICAL:
    centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0 and a.GetSymbol() in ("C", "N", "O", "S")]
# radical cations made from a neutral SMILES: centers = where the hole sits (scenarios.py centers)
_cf = name + "_centers.txt"
if os.path.exists(_cf) and UHF > 0:
    _m = re.search(r"CENTERS\|\s*([\d ]+)", open(_cf).read())
    if _m:
        centers = [int(t) - 1 for t in _m.group(1).split() if 0 < int(t) <= nh] or centers
if not centers:
    sys.exit()
molH = Chem.AddHs(mol)
parent = {h: molH.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx() for h in range(nh, n)} if molH.GetNumAtoms() == n else {}
D = Chem.GetDistanceMatrix(mol)
rings3 = [tuple(r) for r in mol.GetRingInfo().AtomRings()
          if len(r) == 3 and all(sym[k] == "C" for k in r) and min(D[c][k] for c in centers for k in r) <= 3]
ref = set(frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds())
ref |= set(frozenset((h, p)) for h, p in parent.items())


def lab(i):
    if i < n:
        return "%s%d" % (sym[i], i + 1)
    return "%s(Nu)" % NUSYM[i - n]


# ---- nucleophile (atoms appended after the substrate; attacking atom first)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nucleophiles
try:
    _nu = nucleophiles.resolve(NU, XTB)
except ValueError as e_:
    print("NUCLEOPHILE ATTACK skipped: %s" % e_)
    print("NUAT| none")
    sys.exit()


def _fk(path):
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


FKS = _fk(name + "_fukui.out")
FCOL, FNAME = (2, "f(0)") if UHF > 0 else (0, "f(+)")
if _nu["kind"] in ("none", "oxidant"):
    what = "radical" if RADICAL else "cation"
    P("-" * 70)
    P(" %s  (%s, GFN2-xTB, %s)" % ("REACTIVE SITES OF THE RADICAL" if RADICAL else "ELECTROPHILIC SITES",
                                   "no explicit partner" if _nu["kind"] == "none" else "oxidant: see the cascade",
                                   ("ALPB " + SOLV) if SOLV else "gas phase"))
    P("-" * 70)
    P("Where any %s would most likely end up: carbons ranked by %s, with how crowded each site is" %
      ("radical trap" if RADICAL else "nucleophile", FNAME))
    acyl = set(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() == "O" for b in a.GetBonds())
        and sum(1 for x in a.GetNeighbors() if x.GetSymbol() in ("O", "N", "S")) >= 2)
    rows = []
    for i in range(nh):
        if sym[i] not in ("C", "N", "O", "S") or i in acyl or i not in FKS:
            continue
        near = sum(1 for k in range(n) if k != i and 1.7 < np.linalg.norm(X0[k] - X0[i]) < 3.2 and sym[k] != "H")
        rows.append((FKS[i][FCOL], near, i))
    rows.sort(key=lambda r: (-r[0], r[1]))
    P("   site    %s   heavy atoms within 3.2 A (crowding)" % FNAME)
    for f_, near, i in rows[:6]:
        P("   %-6s  %.3f   %d" % (lab(i), f_, near))
    best = None
    if rows:
        top = [r for r in rows if rows[0][0] - r[0] < 0.02]
        best = min(top, key=lambda r: r[1])
        P("most likely site: %s  (%s)" % (lab(best[2]), "highest %s" % FNAME if best is rows[0] else
                                           "%s about as high, less crowded" % FNAME))
    if _nu["kind"] == "oxidant":
        P("oxidant chosen: the radical-polar crossover (radical -> cation) is followed in the CASCADE section.")
    P("NUAT| none%s" % ((" | site %d" % (best[2] + 1)) if best else ""))
    sys.exit()
if RADICAL and _nu["kind"] != "radical":
    print("NUCLEOPHILE ATTACK skipped: %s is a nucleophile; a neutral radical reacts with radical partners "
          "(tin hydride, thiol, iodine, acrylate, oxidant) -- choose one of those." % _nu["name"])
    print("NUAT| none")
    sys.exit()
if not RADICAL and _nu["kind"] == "radical":
    print("NUCLEOPHILE ATTACK skipped: %s is a radical partner; a cation reacts with nucleophiles." % _nu["name"])
    print("NUAT| none")
    sys.exit()
NUC = nucleophiles.geometry(_nu)
NU, NUSYM, QNU = NUC["name"], NUC["sym"], NUC["charge"]
if RADICAL:                                       # delocalized radicals: also the other high-f(0) carbons
    extra = sorted((i for i in range(nh) if sym[i] == "C" and i not in centers and i in FKS),
                   key=lambda i: -FKS[i][2])[:2]
    top0 = max([FKS[c][2] for c in centers if c in FKS] or [0.0])
    centers += [i for i in extra if FKS[i][2] >= 0.5 * top0]


def place_nu(T, u):
    """attacking atom 3.1-4.2 A from carbon T along unit vector u, the rest of the nucleophile pointing away"""
    return nucleophiles.place(NUC, X0[T], u, X0[:n], skip={T})


def state(i, j, X):
    r = np.linalg.norm(X[i] - X[j]) / (rc(S[i]) + rc(S[j]))
    return 3 if r < 1.05 else 2 if r < 1.10 else 1 if r < 1.32 else 0


S = sym + NUSYM
NREF = ref | {frozenset((n + i, n + j)) for i, j in NUC["bonds"]}


def changes(Xa, Xb):
    out = []
    N = len(S)
    for i in range(N):
        for j in range(i + 1, N):
            if S[i] == "H" and S[j] == "H":
                continue
            sa, sb = state(i, j, Xa), state(i, j, Xb)
            if abs(sa - sb) >= 2 or (frozenset((i, j)) in NREF and sa >= 2 and sb == 0):
                out.append((i, j, sa, sb))
    return out


def crowding(T, u):
    """how blocked the approach direction u to carbon T is (higher = more crowded)"""
    occ = 0.0
    for k in range(n):
        if k == T:
            continue
        v = X0[k] - X0[T]
        d = np.linalg.norm(v)
        if d < 4.5 and np.dot(v, u) / d > 0.5:
            occ += (2.0 if sym[k] != "H" else 1.0) / d ** 2
    return occ


# activated carbonyl carbons (C+ with an OH: protonated or Lewis-acid-activated C=O): capture there is a
# reversible 1,2-addition, reported but not used as the prediction when a ring can open
CARBONYL = set(c for c in centers if any(x.GetSymbol() == "O" and x.GetTotalNumHs() > 0
                                         for x in mol.GetAtomWithIdx(c).GetNeighbors()))
# ---- trajectories
traj = []
for r in (rings3 if not RADICAL else []):
    for T in r:
        for Xb in r:
            if Xb != T:
                u = (X0[T] - X0[Xb]) / np.linalg.norm(X0[T] - X0[Xb])
                traj.append(("ring", T, Xb, u))
for c in centers:
    # still bridged by an onium atom (Br+, I+, Cl+, Se+, protonated epoxide O): anti opening, i.e. from the
    # backside of the C-onium bond (SN2-like); an ion that has already opened falls through to the face choice below
    _on = [o for o in onium if np.linalg.norm(X0[c] - X0[o]) < 2.4]
    if _on:
        traj.append(("center", c, None, (X0[c] - X0[_on[0]]) / np.linalg.norm(X0[c] - X0[_on[0]])))
        continue
    nb = [j for j in range(n) if j != c and frozenset((c, j)) in ref and j not in onium]
    if len(nb) >= 2:
        B = np.array([(X0[j] - X0[c]) / np.linalg.norm(X0[j] - X0[c]) for j in nb])
        ax = np.linalg.svd(B)[2][-1]
        crowd = lambda v: sum(1.0 / np.linalg.norm(X0[k] - X0[c]) ** 2 for k in range(n)
                              if k != c and np.dot(X0[k] - X0[c], v) / np.linalg.norm(X0[k] - X0[c]) > 0.5
                              and np.linalg.norm(X0[k] - X0[c]) < 4.5)
        traj.append(("center", c, None, ax if crowd(ax) <= crowd(-ax) else -ax))

P("-" * 70)
P(" %s  (%s, GFN2-xTB, %s)" % ("RADICAL PARTNER SCANS" if RADICAL else "NUCLEOPHILE ATTACK SCANS", NU,
                                 ("ALPB " + SOLV) if SOLV else "gas phase"))
P("-" * 70)
P("Nucleophile: %s (%s). It is pulled onto each candidate carbon; for cyclopropane carbons it comes" % (NU, NUC["desc"]))
P("from the backside of one ring bond at a time, so each ring bond gets its own test.")
if QNU:
    P("Charged nucleophile: the cation and %s form an ion pair in the implicit solvent; barriers are rougher" % NU)
    P("than for neutral nucleophiles (no counter-ion/solvent shell), treat them as qualitative.")
if NUC["approach"] == "pi":
    P("Carbon (pi) nucleophile: it approaches face-on, as in a Friedel-Crafts-type attack.")
if not traj:
    P("no cyclopropane near the cationic center and no capture direction found.")
    sys.exit()
os.makedirs("nuattack", exist_ok=True)
res = []
for kind, T, Xb, u in traj:
    nuX = place_nu(T, u)
    tag = "%s_C%d%s" % (kind, T + 1, "_anti%d" % (Xb + 1) if Xb is not None else "")
    if nuX is None:
        res.append(dict(tag=tag, kind=kind, T=T, Xb=Xb, note="no room for the nucleophile"))
        continue
    wd = os.path.join("nuattack", tag)
    os.makedirs(wd, exist_ok=True)
    Xs = np.vstack([X0, nuX])
    write_xyz(os.path.join(wd, "start.xyz"), S, Xs)
    d0 = float(np.linalg.norm(nuX[0] - X0[T]))
    with open(os.path.join(wd, "scan.inp"), "w") as f:
        f.write("$constrain\n   force constant=1.0\n   distance: %d, %d, %.3f\n$scan\n   1: %.3f, %.2f, %d\n$end\n"
                % (n + 1, T + 1, d0, d0, NUC["bond_to_C"], NSTEP))
    xtb(["start.xyz", "--opt", "--input", "scan.inp"], wd, "scan.out")
    F = frames(os.path.join(wd, "xtbscan.log"))
    if len(F) < 3:
        res.append(dict(tag=tag, kind=kind, T=T, Xb=Xb, note="scan failed"))
        continue
    E = np.array([(e - F[0][0]) * KCAL for e, _ in F])
    imax = int(np.argmax(E))
    k = imax + 1 + int(np.argmin(E[imax + 1:])) if imax < len(E) - 1 else len(E) - 1
    rd = os.path.join(wd, "relax")
    os.makedirs(rd, exist_ok=True)
    write_xyz(os.path.join(rd, "guess.xyz"), S, F[k][1])
    xtb(["guess.xyz", "--opt"], rd, "opt.out")
    _, Xp, Ep = read_xyz(os.path.join(rd, "xtbopt.xyz"))
    if Xp is None:
        res.append(dict(tag=tag, kind=kind, T=T, Xb=Xb, note="relaxation failed"))
        continue
    chg = changes(F[0][1], Xp)
    bonded = state(n, T, Xp) >= 2
    res.append(dict(tag=tag, kind=kind, T=T, Xb=Xb, note="", barrier=float(E[imax]), dE=(Ep - F[0][0]) * KCAL,
                    chg=chg, bonded=bonded, Xp=Xp, crowd=crowding(T, u)))
    write_xyz(os.path.join(wd, "product.xyz"), S, Xp, "relaxed product of %s" % tag)

P("\n%-26s %8s %8s %7s   %s" % ("attack", "barrier", "dE", "crowd", "what happens (relaxed end point)"))
for r in res:
    where = "C%d %s" % (r["T"] + 1, ("(backside of %s-%s)" % (lab(r["T"]), lab(r["Xb"]))) if r["Xb"] is not None
                        else ("(at the radical center)" if RADICAL else "(direct capture)"))
    if r["note"]:
        P("%-26s %8s %8s %7s   %s" % (where, "-", "-", "-", r["note"]))
        continue
    br = ["%s-%s breaks" % (lab(i), lab(j)) for i, j, a, b in r["chg"] if b < a and max(i, j) < n]
    fo = ["%s-%s forms" % (lab(i), lab(j)) for i, j, a, b in r["chg"] if b > a and max(i, j) < n]
    txt = ", ".join(br + fo) or ("no change in the skeleton" if r["bonded"] else "nucleophile does not stay bonded")
    if not r["bonded"]:
        txt = "nucleophile does not stay bonded; " + txt if (br or fo) else txt
    P("%-26s %8.1f %+8.1f %7.2f   %s" % (where, r["barrier"], r["dE"], r["crowd"], txt))

ok = [r for r in res if not r["note"] and r["bonded"]]
P("")
if not ok:
    P("No attack ends with the nucleophile bonded.")
    P("NUAT| none")
    sys.exit()
in_ring = lambda i, j: any(i in rr and j in rr for rr in rings3)
for r in ok:
    r["ring_brk"] = [(i, j) for i, j, a, b in r["chg"] if b < a and max(i, j) < n and in_ring(i, j)]
    r["frm"] = [(i, j) for i, j, a, b in r["chg"] if b > a and max(i, j) < n]
opening = [r for r in ok if r["ring_brk"]]
capture = [r for r in ok if r["kind"] == "center" and not (r["T"] in CARBONYL and opening)]
if opening and any(r["kind"] == "center" and r["T"] in CARBONYL for r in ok):
    P("note: attack at the activated carbonyl carbon (1,2-addition) is listed but not used as the prediction:")
    P("      it is usually reversible; the cyclopropane ring opening below is the productive pathway.")
# which ring bond breaks, bond by bond
if opening:
    P("CYCLOPROPANE BONDS (which one breaks when the nucleophile attacks the ring):")
    for rr in rings3:
        for i, j in ((rr[0], rr[1]), (rr[1], rr[2]), (rr[0], rr[2])):
            hits = [r for r in opening if set((i, j)) in [set(b) for b in r["ring_brk"]]]
            if hits:
                b = min(hits, key=lambda r: r["barrier"])
                P("   %-10s BREAKS in %d attack(s); lowest barrier %.1f kcal/mol (%s at C%d)"
                  % ("%s-%s" % (lab(i), lab(j)), len(hits), b["barrier"], NU, b["T"] + 1))
            else:
                P("   %-10s never breaks" % ("%s-%s" % (lab(i), lab(j))))
FACILE = float(os.environ.get("NUA_FACILE", "20"))
best_open = min([r for r in opening if r["barrier"] <= FACILE], key=lambda r: r["barrier"], default=None)
best_cap = min(capture, key=lambda r: r["barrier"], default=None)
P("")
P("RESULT")
if best_open:
    P("  ring opening by %s at C%d: %s breaks, %s forms -- barrier %.1f, reaction energy %+.1f kcal/mol"
      % (NU, best_open["T"] + 1, ", ".join("%s-%s" % (lab(i), lab(j)) for i, j in best_open["ring_brk"]),
         ", ".join("%s-%s" % (lab(i), lab(j)) for i, j in best_open["frm"]) or "no new bond",
         best_open["barrier"], best_open["dE"]))
if best_cap:
    P("  %s at C%d (no ring opening): barrier %.1f, reaction energy %+.1f kcal/mol"
      % ("reaction" if RADICAL else "direct capture", best_cap["T"] + 1, best_cap["barrier"], best_cap["dE"]))
_both_low = bool(best_open and best_cap and best_open["barrier"] <= LOWBAR and best_cap["barrier"] <= LOWBAR)
if best_open and best_cap:
    gap = best_open["barrier"] - best_cap["barrier"]
    if _both_low and abs(gap) > TIE:
        P("  -> CLOSE CALL: both are practically barrierless at this level (%.1f and %.1f kcal/mol; the difference"
          % (best_open["barrier"], best_cap["barrier"]))
        P("     is within the method's error). The ring-opening (rearranged) product is reported as the main")
        P("     prediction and direct capture as the competitor; check with DFT (run_ts).")
    elif abs(gap) <= TIE:
        P("  -> competitive (within %.0f kcal/mol, the accuracy of this method): which site the nucleophile picks"
          % TIE)
        P("     is not resolved at this level; the ring-opening result above says WHICH ring bond breaks.")
    elif gap < 0:
        P("  -> ring opening is clearly preferred")
    else:
        P("  -> direct capture is clearly preferred; ring opening is the minor/alternative pathway")
_rank = list(Chem.CanonicalRankAtoms(mol, breakTies=False))


def _bkey(bonds):
    """ring bonds up to molecular symmetry (equivalent bonds are not a 'close call')"""
    return frozenset(frozenset((_rank[i] if i < len(_rank) else -1, _rank[j] if j < len(_rank) else -1)) for i, j in bonds)


if best_open:
    alt = [r for r in opening if r["barrier"] <= FACILE and _bkey(r["ring_brk"]) != _bkey(best_open["ring_brk"])]
    alt = min(alt, key=lambda r: r["barrier"], default=None)
    if alt and alt["barrier"] - best_open["barrier"] <= TIE:
        P("  -> CLOSE CALL: a different ring bond (%s, attack at C%d) opens only %.1f kcal/mol higher -- both are"
          % (", ".join("%s-%s" % (lab(i), lab(j)) for i, j in alt["ring_brk"]), alt["T"] + 1, alt["barrier"] - best_open["barrier"]))
        P("     within the accuracy of this method; decide with DFT (or compare different nucleophiles).")
P("  (barriers along one coordinate, upper-bound estimates; confirm with DFT)")
main = best_open if (best_open and (not best_cap or best_open["barrier"] - best_cap["barrier"] <= TIE or _both_low)) \
    else best_cap
if main is None:
    main = min(ok, key=lambda r: r["barrier"])
rb = main["ring_brk"][0] if main.get("ring_brk") else (0, -1)
fb = main["frm"][0] if main.get("frm") else (0, -1)
extra = (" | capture %d %.1f" % (best_cap["T"] + 1, best_cap["barrier"])) if (best_cap and main is not best_cap) else ""
# steric warning: bulky nucleophile + direct capture at a crowded (tertiary / quaternary) cationic carbon.
# GFN2 and g-xTB both underestimate this steric clash (Tenaglia 1996: tert-butanol gives the rearranged
# C4 product, the tool predicts C8 capture as for methanol), so such captures are flagged as uncertain.
def _bulky(nuc):
    m_ = nuc.get("mol")
    a_ = nuc.get("attack")
    if m_ is None or isinstance(a_, tuple) or nuc.get("approach") == "pi":
        return False
    at_ = m_.GetAtomWithIdx(a_)
    return any(x.GetSymbol() == "C" and sum(1 for y in x.GetNeighbors() if y.GetSymbol() == "C") >= 2
               for x in at_.GetNeighbors())


STERIC = ""
if best_cap and main is best_cap and best_cap["barrier"] <= FACILE and not RADICAL and _bulky(NUC):
    _T = best_cap["T"]
    if _T < nh and sum(1 for x in mol.GetAtomWithIdx(_T).GetNeighbors() if x.GetSymbol() != "H") >= 3:
        STERIC = " | steric warning"
        P("")
        P("  STERIC WARNING: %s is a bulky nucleophile and %s is a crowded (tertiary/quaternary) carbon." % (NU, lab(_T)))
        P("  This method underestimates steric hindrance for such approaches (e.g. Tenaglia 1996: tert-butanol")
        P("  attacks the rearranged C4 position, while the scans favour C8 capture as for methanol).")
        P("  Treat the direct capture at %s as uncertain%s" % (lab(_T), (
            "; the best alternative is ring opening at C%d (%s breaks), barrier %.1f kcal/mol."
            % (best_open["T"] + 1, ", ".join("%s-%s" % (lab(i), lab(j)) for i, j in best_open["ring_brk"]),
               best_open["barrier"])) if best_open else "."))
        P("  Settle it with a DFT transition-state calculation using the real nucleophile.")
close = ""
if best_open and main is best_open:
    alt = [r for r in opening if r["barrier"] <= FACILE and _bkey(r["ring_brk"]) != _bkey(best_open["ring_brk"])]
    alt = min(alt, key=lambda r: r["barrier"], default=None)
    if alt and alt["barrier"] - best_open["barrier"] <= TIE:
        i_, j_ = alt["ring_brk"][0]
        close = " | close call: %d-%d +%.1f" % (i_ + 1, j_ + 1, alt["barrier"] - best_open["barrier"])
P("NUAT| site %d | break %d %d | form %d %d | dEa %.1f | dE %+.1f%s%s%s | nu %s"
  % (main["T"] + 1, rb[0] + 1, rb[1] + 1, fb[0] + 1, fb[1] + 1, main["barrier"], main["dE"], extra, close, STERIC, NU))
# picture of the predicted product (from the relaxed end point of the chosen attack)
try:
    import prodpic
    _q = CHG + (Chem.GetFormalCharge(NUC["mol"]) if NUC.get("mol") is not None else 0)
    _pp = prodpic.product(os.path.join("nuattack", main["tag"], "product.xyz"), _q, main["T"])
    if _pp:
        _png = prodpic.draw(_pp["smiles"], name + "_nuproduct.png",
                            "predicted product%s" % (" (after loss of H+)" if _pp["lost_h"] else ""))
        P("  predicted product: %s%s" % (_pp["smiles"], "   (after loss of H+)" if _pp["lost_h"] else ""))
        P("NUPROD| %s | %s | %s | %s" % (_pp["smiles"], "lostH" if _pp["lost_h"] else "-", _png or "-",
                                         _pp.get("smiles_onium") or _pp["smiles"]))
except Exception:
    pass
