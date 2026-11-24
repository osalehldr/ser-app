#!/usr/bin/env python3
"""
scenarios.py -- the three-scenario analysis of a NEUTRAL molecule.

A neutral input is analysed as three species, each in its own sub-folder:
  neutral/          the molecule as drawn
  cation/           an electrophile / acid / Lewis acid (modelled as H+) adds: every plausible site, ranked
  radical_cation/   one electron removed (SET oxidation)
Cations, radical cations and neutral radicals given as input are single runs (not handled here).
Anions are not analysed by this tool.

Sub-commands (cluster: called by scenario.sh / pipeline.sh; web app: imported):
  sites cation SMILES                    list the candidate protonated species (no calculation)
  screen cation NAME SMILES XTB          optimize every candidate (in the cation folder), rank,
                                         write NAME_<kind>_ranking.txt + the list of sites to analyse
  finish cation NAME                     after the full analyses: add site info / formation to their reports
  centers NAME XTB CHG UHF               radical cation: where the hole sits (-> NAME_centers.txt)
  formation NAME XTB                     radical cation: oxidation potential (needs SCEN, SOLV)
  stamp NAME                             write the species line into NAME_report.txt (needs SCEN, CHRG, UHF)
  rank JOBDIR NAME                       cation vs radical cation: which forms more easily (neutral report)

Formation energies are xtb (GFN2) energies converted to familiar scales with linear fits to
experimental reference data (see CAL below and calibration.txt):
  cation          proton affinity (gas phase, kcal/mol)      NIST / Hunter & Lias 1998
  radical cation  oxidation potential, V vs SCE (MeCN)       experimental E1/2 compilations
Expect roughly +/-5 kcal/mol (PA) and +/-0.3 V: good for ranking, not exact values.
"""
import os, re, sys, glob, json, shutil, subprocess
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

KCAL = 627.509
SINGLE, DOUBLE, TRIPLE = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE
UNSPEC = Chem.ChiralType.CHI_UNSPECIFIED
PREFIX = {"cation": "cat", "anion": "an"}
RUNTAG = {"neutral": "neutral", "radical_cation": "radcat", "radical_anion": "radan"}
SCALE_SOLV = {"PA": "", "pKa": "dmso", "Eox": "acetonitrile", "Ered": "acetonitrile"}

# ---- calibration: experimental = a * xtb_quantity + b   (fitted on reference sets, see calibration.txt)
#   PA  : xtb quantity = E(B) - E(BH+), gas phase, kcal/mol           -> proton affinity, kcal/mol
#   pKa : xtb quantity = E(A-) - E(HA), ALPB(DMSO), kcal/mol          -> pKa in DMSO
#   Eox : xtb quantity = E(M+.) - E(M), ALPB(acetonitrile), kcal/mol  -> E1/2(ox), V vs SCE
#   Ered: xtb quantity = E(M-.) - E(M), ALPB(acetonitrile), kcal/mol  -> E1/2(red), V vs SCE
CAL = {"PA": (1.0, 0.0), "pKa": (1.0, 0.0), "Eox": (1.0 / 23.06, 0.0), "Ered": (1.0 / 23.06, 0.0)}
CAL_STATS = {}
_cf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
if os.path.exists(_cf):
    try:
        _j = json.load(open(_cf))
        for k_ in CAL:
            if k_ in _j:
                CAL[k_] = tuple(_j[k_]["coef"])
                CAL_STATS[k_] = _j[k_]
    except Exception:
        pass

WINDOW = float(os.environ.get("SITE_WINDOW", "12"))     # kcal/mol above the best site still analysed in full
NMAX = int(os.environ.get("SITE_MAX", "4"))              # at most this many sites analysed in full


# ============================================================== xtb helpers
def solvent():
    return os.environ.get("SOLV", "") or ""


def xtb_run(xtb, xyz, folder, chg, uhf, solv, opt=True, extra=()):
    """runs xtb in folder (created); returns (energy_Eh or None, path of final xyz)"""
    os.makedirs(folder, exist_ok=True)
    src = os.path.join(folder, "in.xyz")
    if os.path.abspath(xyz) != os.path.abspath(src):
        shutil.copy(xyz, src)
    cmd = [xtb, "in.xyz"] + (["--opt"] if opt else []) + ["--chrg", str(chg), "--uhf", str(uhf), "--gfn", "2"]
    cmd += (["--alpb", solv] if solv else []) + list(extra)
    env = os.environ.copy()
    env.setdefault("OMP_STACKSIZE", "2G")
    with open(os.path.join(folder, "xtb.out"), "w") as f:
        subprocess.run(cmd, cwd=folder, stdout=f, stderr=subprocess.STDOUT, env=env)
    out = open(os.path.join(folder, "xtb.out")).read()
    m = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", out)
    e = float(m[-1]) if m else None
    fin = os.path.join(folder, "xtbopt.xyz") if opt else src
    if opt and not os.path.exists(fin):
        return None, None
    return e, fin


def single_point(xtb, xyz, folder, chg, uhf, solv):
    return xtb_run(xtb, xyz, folder, chg, uhf, solv, opt=False)[0]


def read_xyz(p):
    L = open(p).read().split("\n")
    n = int(L[0].split()[0])
    sym = [l.split()[0] for l in L[2:2 + n]]
    X = np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + n]])
    return sym, X


def read_charges(folder):
    p = os.path.join(folder, "charges")
    return [float(x) for x in open(p).read().split()] if os.path.exists(p) else None


# ============================================================== building 3D structures
def build_xyz(smiles, path, seed=7):
    """heavy atoms in SMILES order, then H in RDKit AddHs order (the order every script assumes)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    molH = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    ok = AllChem.EmbedMolecule(molH, p) == 0
    if not ok:
        p.useRandomCoords = True
        ok = AllChem.EmbedMolecule(molH, p) == 0
    if ok:
        try:
            if AllChem.MMFFHasAllMoleculeParams(molH):
                AllChem.MMFFOptimizeMolecule(molH, maxIters=1000)
        except Exception:
            pass
        X = molH.GetConformer().GetPositions()
        syms = [a.GetSymbol() for a in molH.GetAtoms()]
    else:
        # cage ions: build the neutral C-H analogue, then remove that H again (same as pyldr's fallback)
        ch = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() != 0 and a.GetSymbol() == "C"]
        if not ch:
            return False
        rw = Chem.RWMol(mol)
        for i in ch:
            a = rw.GetAtomWithIdx(i)
            h = a.GetTotalNumHs() + (1 if a.GetFormalCharge() > 0 else -1)
            a.SetFormalCharge(0)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(max(0, h))
        neu = rw.GetMol()
        try:
            Chem.SanitizeMol(neu)
        except Exception:
            return False
        nH = Chem.AddHs(neu)
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(nH, p) != 0:
            return False
        try:
            AllChem.MMFFOptimizeMolecule(nH, maxIters=1000)
        except Exception:
            pass
        rwh = Chem.RWMol(nH)
        drop = []
        for i in ch:
            if mol.GetAtomWithIdx(i).GetFormalCharge() > 0:
                hs = [x.GetIdx() for x in rwh.GetAtomWithIdx(i).GetNeighbors() if x.GetSymbol() == "H"]
                if hs:
                    drop.append(hs[-1])
        for h in sorted(drop, reverse=True):
            rwh.RemoveAtom(h)
        m2 = rwh.GetMol()
        X = m2.GetConformer().GetPositions()
        syms = [a.GetSymbol() for a in m2.GetAtoms()]
        if len(syms) != molH.GetNumAtoms():
            return False
    with open(path, "w") as f:
        f.write("%d\n%s\n" % (len(syms), smiles))
        for s, x in zip(syms, X):
            f.write("%-2s %14.8f %14.8f %14.8f\n" % (s, x[0], x[1], x[2]))
    return True


# ============================================================== candidate species
def _addH(a, d):
    a.SetNumExplicitHs(max(0, a.GetTotalNumHs() + d))
    a.SetNoImplicit(True)


def _frozen(mol):
    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        a.SetNumExplicitHs(a.GetTotalNumHs())
        a.SetNoImplicit(True)
    Chem.Kekulize(m, clearAromaticFlags=True)
    return m


def _finish(m):
    m = m.GetMol() if isinstance(m, Chem.RWMol) else Chem.Mol(m)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    return m


def _stereo_variants(m, atoms):
    """a newly created stereocentre (the atom that received the H) -> both configurations"""
    out = [m]
    for a in atoms:
        new = []
        for x in out:
            cc = dict(Chem.FindMolChiralCenters(x, includeUnassigned=True, useLegacyImplementation=False))
            if cc.get(a) == "?":
                for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                    y = Chem.Mol(x)
                    y.GetAtomWithIdx(a).SetChiralTag(tag)
                    Chem.AssignStereochemistry(y, cleanIt=True, force=True)
                    new.append(y)
            else:
                new.append(x)
        out = new
    return out


def _smiles_keep_order(m):
    """non-canonical SMILES; returns (smiles, order) where order[k] = original index of atom k"""
    s = Chem.MolToSmiles(m, canonical=False)
    try:
        order = [int(t) for t in re.findall(r"\d+", m.GetProp("_smilesAtomOutputOrder"))]
    except Exception:
        order = list(range(m.GetNumAtoms()))
    return s, order


def _lab(mol, i):
    return "%s%d" % (mol.GetAtomWithIdx(i).GetSymbol(), i + 1)


def _unspecified(mol):
    return any(t == "?" for _, t in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))


def cation_sites(smiles):
    """every plausible place an electrophile (H+) adds; each -> one cation (or onium ion)"""
    mol = Chem.MolFromSmiles(smiles)
    base = _frozen(mol)
    out, seen = [], set()
    exact = _unspecified(mol)          # stereo missing: symmetry-equivalent-looking sites may differ in 3D

    def add(m, desc, hat, cen):
        m = _finish(m)
        if m is None:
            return
        for v in _stereo_variants(m, [hat] if hat is not None else []):
            key = Chem.MolToSmiles(v) + ("|" + desc if exact else "")
            if key in seen:
                continue
            seen.add(key)
            s, order = _smiles_keep_order(v)
            out.append(dict(smiles=s, order=order, desc=desc, center=cen, key=key))

    r3 = [r for r in mol.GetRingInfo().AtomRings() if len(r) == 3]
    for bd in base.GetBonds():
        i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        si, sj = base.GetAtomWithIdx(i).GetSymbol(), base.GetAtomWithIdx(j).GetSymbol()
        bt = bd.GetBondType()
        if bt in (DOUBLE, TRIPLE) and si == "C" and sj == "C":
            arom = mol.GetBondWithIdx(bd.GetIdx()).GetIsAromatic()
            what = "arene" if arom else ("alkyne" if bt == TRIPLE else "alkene")
            for h, c in ((i, j), (j, i)):
                m = Chem.RWMol(base)
                m.GetBondBetweenAtoms(h, c).SetBondType(SINGLE if bt == DOUBLE else DOUBLE)
                a = m.GetAtomWithIdx(h)
                _addH(a, 1)
                m.GetAtomWithIdx(c).SetFormalCharge(1)
                for x in (h, c):
                    m.GetAtomWithIdx(x).SetChiralTag(UNSPEC)
                add(m, "%s: H+ on %s -> cation at %s" % (what, _lab(mol, h), _lab(mol, c)), h, c)
        if bt == DOUBLE and {si, sj} == {"C", "O"}:
            o, c = (i, j) if si == "O" else (j, i)
            m = Chem.RWMol(base)
            m.GetBondBetweenAtoms(o, c).SetBondType(SINGLE)
            a = m.GetAtomWithIdx(o)
            _addH(a, 1)
            m.GetAtomWithIdx(c).SetFormalCharge(1)
            add(m, "carbonyl: H+ on %s -> cation at %s (Lewis-acid model)" % (_lab(mol, o), _lab(mol, c)), None, c)
    for at in base.GetAtoms():
        k, s = at.GetIdx(), at.GetSymbol()
        if at.GetFormalCharge() != 0:
            continue
        single = all(b.GetBondType() == SINGLE for b in at.GetBonds())
        if s in ("O", "S") and single and at.GetDegree() >= 1:
            ring = [r for r in r3 if k in r]
            amide_like = any(x.GetSymbol() == "C" and any(b.GetBondType() == DOUBLE and
                             b.GetOtherAtomIdx(x.GetIdx()) != k for b in x.GetBonds()) for x in at.GetNeighbors())
            kind = "epoxide" if ring else ("ester/enol O" if amide_like else "ether/alcohol")
            m = Chem.RWMol(base)
            a = m.GetAtomWithIdx(k)
            a.SetFormalCharge(1)
            _addH(a, 1)
            add(m, "%s: H+ on %s (onium ion)" % (kind, _lab(mol, k)), None, k)
            if ring:
                for c in [x.GetIdx() for x in at.GetNeighbors() if x.GetSymbol() == "C"]:
                    m = Chem.RWMol(base)
                    m.RemoveBond(k, c)
                    a = m.GetAtomWithIdx(k)
                    _addH(a, 1)
                    m.GetAtomWithIdx(c).SetFormalCharge(1)
                    m.GetAtomWithIdx(c).SetChiralTag(UNSPEC)
                    add(m, "epoxide: H+ on %s, %s-%s opens -> cation at %s" % (_lab(mol, k), _lab(mol, c), _lab(mol, k),
                                                                             _lab(mol, c)), None, c)
        if s == "N":
            amide = any(x.GetSymbol() == "C" and any(b.GetBondType() == DOUBLE and b.GetOtherAtom(x).GetSymbol() in ("O", "S")
                        for b in x.GetBonds()) for x in at.GetNeighbors())
            arom = mol.GetAtomWithIdx(k).GetIsAromatic()
            if amide or (arom and at.GetTotalNumHs() > 0):
                continue
            m = Chem.RWMol(base)
            a = m.GetAtomWithIdx(k)
            a.SetFormalCharge(1)
            _addH(a, 1)
            add(m, "nitrogen: H+ on %s" % _lab(mol, k), None, k)
    for r in r3:
        if not all(mol.GetAtomWithIdx(x).GetSymbol() == "C" for x in r):
            continue
        for q in range(3):
            i, j = r[q], r[(q + 1) % 3]
            for h, c in ((i, j), (j, i)):
                m = Chem.RWMol(base)
                m.RemoveBond(i, j)
                a = m.GetAtomWithIdx(h)
                _addH(a, 1)
                m.GetAtomWithIdx(c).SetFormalCharge(1)
                for x in (h, c):
                    m.GetAtomWithIdx(x).SetChiralTag(UNSPEC)
                add(m, "cyclopropane: H+ on %s, %s-%s opens -> cation at %s" % (_lab(mol, h), _lab(mol, i), _lab(mol, j),
                                                                             _lab(mol, c)), h, c)
    return out


def anion_sites(smiles):
    """every distinct H a base could remove; each -> one anion"""
    mol = Chem.MolFromSmiles(smiles)
    base = _frozen(mol)
    out, seen = [], set()
    exact = _unspecified(mol)
    for at in base.GetAtoms():
        k, s = at.GetIdx(), at.GetSymbol()
        if at.GetTotalNumHs() == 0 or s not in ("C", "N", "O", "S", "P", "Si") or at.GetFormalCharge() != 0:
            continue
        m = Chem.RWMol(base)
        a = m.GetAtomWithIdx(k)
        _addH(a, -1)
        a.SetFormalCharge(-1)
        a.SetChiralTag(UNSPEC)
        m = _finish(m)
        if m is None:
            continue
        key = Chem.MolToSmiles(m) + ("|%d" % k if exact else "")
        if key in seen:
            continue
        seen.add(key)
        orig = mol.GetAtomWithIdx(k)
        if s == "C":
            hyb = str(orig.GetHybridization()).replace("SP", "sp").replace("2", "2").lower()
            nxt = [x for x in orig.GetNeighbors()]
            alpha = any(any(b.GetBondType() == DOUBLE and b.GetOtherAtom(x).GetSymbol() in ("O", "N", "S")
                            for b in x.GetBonds()) for x in nxt if x.GetSymbol() == "C")
            what = "C-H (%s%s)" % (hyb, ", next to C=O/C=N" if alpha else "")
        else:
            what = "%s-H" % s
        st, order = _smiles_keep_order(m)
        out.append(dict(smiles=st, order=order, desc="%s: H removed from %s" % (what, _lab(mol, k)), center=k, key=key))
    return out


# ============================================================== bonding check after optimization
def _rcov(s):
    return Chem.GetPeriodicTable().GetRcovalent(Chem.GetPeriodicTable().GetAtomicNumber(s))


def bonding_changes(smiles, xyz):
    mol = Chem.MolFromSmiles(smiles)
    sym, X = read_xyz(xyz)
    nh = mol.GetNumAtoms()
    ch = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if np.linalg.norm(X[i] - X[j]) > 1.35 * (_rcov(sym[i]) + _rcov(sym[j])):
            ch.append("%s-%s broke" % (_lab(mol, i), _lab(mol, j)))
    bonded = set(frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds())
    for i in range(nh):
        for j in range(i + 1, nh):
            if frozenset((i, j)) not in bonded and np.linalg.norm(X[i] - X[j]) < 1.15 * (_rcov(sym[i]) + _rcov(sym[j])):
                ch.append("%s-%s formed" % (_lab(mol, i), _lab(mol, j)))
    molH = Chem.AddHs(mol)
    if molH.GetNumAtoms() == len(sym):
        for h in range(nh, len(sym)):
            p = molH.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
            near = min(range(nh), key=lambda j: np.linalg.norm(X[h] - X[j]))
            if near != p and np.linalg.norm(X[h] - X[near]) < 1.3:
                ch.append("H moved %s->%s" % (_lab(mol, p), _lab(mol, near)))
    return ch


# ============================================================== calibrated scales and tiers
def cal(kind, x):
    a, b = CAL[kind]
    return a * x + b


TIERS = {
    # (upper/lower bounds of each tier, description); difficulty 0 (easy) .. 4 (very hard)
    "PA": ([215, 205, 195, 185], ["a weak acid (e.g. acetic acid) is enough",
                                  "a moderate acid (carboxylic acid, TFA)",
                                  "a strong acid (TfOH, H2SO4) or a Lewis acid",
                                  "a very strong acid or Lewis acid; the cation is short-lived",
                                  "superacid territory; the cation is only a transient intermediate"]),
    "pKa": ([12, 20, 28, 36], ["a weak base (amine, carbonate)",
                               "an alkoxide or amidine base (NaOMe, DBU)",
                               "a strong base (t-BuOK, NaH, NaHMDS)",
                               "a very strong base (LDA, alkyllithium)",
                               "an alkyllithium/superbase; essentially not deprotonated"]),
    "Eox": ([0.5, 1.2, 1.8, 2.3], ["a mild oxidant (air, Fe(III))",
                                   "a common oxidant (Cu(II), Ru(bpy)3(3+), DDQ)",
                                   "a strong oxidant / photo-oxidant (CAN, Ir/Ru photocatalysts)",
                                   "a very strong photo-oxidant (acridinium, ~ +2 V)",
                                   "beyond common reagents (anodic oxidation only)"]),
    "Ered": ([-1.0, -1.6, -2.1, -2.7], ["a mild reductant (Zn, Fe(II))",
                                        "a moderate reductant (SmI2 alone, Ru(bpy)3(+), ~ -1.3 to -1.6 V)",
                                        "a strong reductant (SmI2/HMPA, Ir(ppy)3*, ~ -1.7 to -2.1 V)",
                                        "a very strong reductant (Li/Na arenide, ~ -2.5 V)",
                                        "dissolving-metal territory (Li/NH3, ~ -3 V)"]),
}


def difficulty(kind, v):
    """continuous difficulty 0..5 (0 = trivial) and the tier description"""
    bounds, txt = TIERS[kind]
    sgn = -1 if kind in ("PA", "Ered") else 1      # PA / Ered: larger value = easier
    x = sgn * v
    b = [sgn * t for t in bounds]
    if x <= b[0]:
        return max(0.0, 1.0 - (b[0] - x) / 10.0) * 0.999, txt[0]
    for k in range(1, 4):
        if x <= b[k]:
            return k + (x - b[k - 1]) / (b[k] - b[k - 1]), txt[k]
    return 4.0 + min(1.0, (x - b[3]) / (abs(b[3] - b[2]) or 1)), txt[4]


def fmt_value(kind, v):
    return {"PA": "proton affinity %.0f kcal/mol" % v, "pKa": "pKa(DMSO) %.0f" % v,
            "Eox": "E(ox) %+.2f V vs SCE" % v, "Ered": "E(red) %+.2f V vs SCE" % v}[kind]


def cal_note(kind):
    st = CAL_STATS.get(kind)
    if not st:
        return "uncalibrated"
    unit = {"PA": "kcal/mol", "pKa": "pKa units", "Eox": "V", "Ered": "V"}[kind]
    return "calibrated on %d reference molecules, typical error +/-%.2g %s" % (st.get("n", 0), st.get("mae", 0), unit)


# ============================================================== screening (cation / anion)
def _ref_neutral(smiles, xtb, solv, folder):
    """the neutral molecule optimized in the run solvent (reference for formation energies)"""
    os.makedirs(folder, exist_ok=True)
    x0 = os.path.join(folder, "start.xyz")
    if not build_xyz(smiles, x0):
        return None, None
    e, fin = xtb_run(xtb, x0, os.path.join(folder, "opt"), 0, 0, solv)
    return e, fin


def screen(kind, name, smiles, xtb, log=print):
    solv = solvent()
    sites = cation_sites(smiles) if kind == "cation" else anion_sites(smiles)
    chg = 1 if kind == "cation" else -1
    pre = PREFIX[kind]
    mol = Chem.MolFromSmiles(smiles)
    if not sites:
        open("%s_%s_ranking.txt" % (name, kind), "w").write("No %s site found for %s.\n" % (
            "protonation" if kind == "cation" else "deprotonation", smiles))
        open("%s_%s_selected.txt" % (name, kind), "w").close()
        return []
    log("%s: %d candidate sites" % (kind, len(sites)))
    os.makedirs("screen", exist_ok=True)
    eN, xN = _ref_neutral(smiles, xtb, solv, "screen/neutral")
    sc = SCALE_SOLV["PA" if kind == "cation" else "pKa"]
    eN_s = single_point(xtb, xN, "screen/neutral/sp_scale", 0, 0, sc) if xN else None
    res = []
    for k, s in enumerate(sites):
        d = "screen/%02d" % (k + 1)
        os.makedirs(d, exist_ok=True)
        log("  %s site %d/%d: %s" % (kind, k + 1, len(sites), s["desc"]))
        if not build_xyz(s["smiles"], os.path.join(d, "start.xyz")):
            s.update(E=None, note="3D build failed")
            res.append(s)
            continue
        e, fin = xtb_run(xtb, os.path.join(d, "start.xyz"), os.path.join(d, "opt"), chg, 0, solv)
        s.update(E=e, xyz=fin, note="")
        if fin:
            ch = bonding_changes(s["smiles"], fin)
            s["note"] = ("changed during optimization: " + ", ".join(ch[:4])) if ch else ""
            es = single_point(xtb, fin, os.path.join(d, "sp_scale"), chg, 0, sc)
            if es is not None and eN_s is not None:
                if kind == "cation":
                    s["scale"] = cal("PA", (eN_s - es) * KCAL)
                    if s["desc"].split(":")[0] in ("alkene", "arene", "alkyne", "cyclopropane"):
                        s["scale"] += CAL_STATS.get("PA", {}).get("carbon_offset", 0.0)
                else:
                    s["scale"] = cal("pKa", (es - eN_s) * KCAL)
        res.append(s)
    ok = [s for s in res if s.get("E") is not None]
    ok.sort(key=lambda s: s["E"])
    if not ok:
        open("%s_%s_ranking.txt" % (name, kind), "w").write("All %d candidate optimizations failed.\n" % len(res))
        open("%s_%s_selected.txt" % (name, kind), "w").close()
        return []
    e0 = ok[0]["E"]
    used = {}
    for r, s in enumerate(ok):
        s["rel"] = (s["E"] - e0) * KCAL
        lb = _lab(mol, s["center"])
        used[lb] = used.get(lb, 0) + 1
        s["folder"] = "%s%02d_%s" % (pre, r + 1, lb)
        s["run"] = "%s_%s%02d" % (name, pre, r + 1)
        s["full"] = (r < NMAX and s["rel"] <= WINDOW)
        os.makedirs(s["folder"], exist_ok=True)
        shutil.copy(s["xyz"], os.path.join(s["folder"], "screen.xyz"))
        open(os.path.join(s["folder"], "%s_site.txt" % s["run"]), "w").write(
            "site       : %s\nSMILES     : %s\nrank       : %d of %d (%.1f kcal/mol above the best site)\n%s\n%s\n" % (
                s["desc"], s["smiles"], r + 1, len(ok), s["rel"],
                ("" if s.get("scale") is None else fmt_value("PA" if kind == "cation" else "pKa", s["scale"])),
                "full analysis: yes" if s["full"] else "full analysis: no (screened only: higher in energy)"))
    lines = ["=" * 100,
             " %s SCENARIO -- %s   (%s; GFN2-xTB%s)" % (kind.upper(), name, "electrophile modelled as H+" if kind == "cation"
                                                     else "a base removes one proton", ", ALPB " + solv if solv else ", gas phase"),
             " SMILES: %s" % smiles,
             "=" * 100,
             " All species are isomers, so their energies compare directly: rank 1 is the most stable %s." % kind,
             " %-16s %9s  %-22s %s" % ("site folder", "rel kcal", "PA (kcal/mol)" if kind == "cation" else "pKa (DMSO)",
                                       "what happens"), ]
    for s in ok:
        sc_ = "-" if s.get("scale") is None else "%.0f" % s["scale"]
        lines.append(" %-16s %9.1f  %-22s %s%s" % (s["folder"], s["rel"], sc_, s["desc"], "" if s["full"] else "   [screened only]"))
        if s["note"]:
            lines.append(" %-16s %9s  %-22s   (%s)" % ("", "", "", s["note"]))
        if s["order"] != list(range(len(s["order"]))):
            lines.append(" %-16s %9s  %-22s   (atom numbers in this site's own files differ from the neutral numbering)" % ("", "", ""))
    for s in res:
        if s.get("E") is None:
            lines.append(" %-16s %9s  %-22s %s   [%s]" % ("-", "-", "-", s["desc"], s.get("note") or "optimization failed"))
    lines.append(" Full analysis (rearrangement, nucleophile attack, picture) for sites within %.0f kcal/mol of the best "
                 "(at most %d); change with SITE_WINDOW / SITE_MAX." % (WINDOW, NMAX))
    lines.append(" Atom numbers = the neutral molecule's numbering (SMILES order).")
    best = ok[0]
    if best.get("scale") is not None:
        k_ = "PA" if kind == "cation" else "pKa"
        dval, txt = difficulty(k_, best["scale"])
        lines.append("")
        lines.append(" FORMATION (most stable site, %s): %s -> needs %s" % (best["folder"], fmt_value(k_, best["scale"]), txt))
        lines.append(" (%s)" % cal_note(k_))
        open("%s_%s_formation.txt" % (name, kind), "w").write(
            "FORMATION| %s | %s | %.3f | %.2f | %s | %s\n" % (kind, k_, best["scale"], dval, best["folder"], txt))
    open("%s_%s_ranking.txt" % (name, kind), "w").write("\n".join(lines) + "\n")
    with open("%s_%s_selected.txt" % (name, kind), "w") as f:
        for s in ok:
            if s["full"]:
                f.write("%s %s %s\n" % (s["folder"], s["run"], s["smiles"]))
    json.dump([{k: v for k, v in s.items() if k != "order" or True} for s in ok],
              open("screen/sites.json", "w"), indent=1, default=str)
    return ok


def relabel(text, order):
    """site numbering -> neutral numbering (only heavy-atom labels; H labels unchanged)"""
    if not order or order == list(range(len(order))):
        return text

    def f(m):
        k = int(m.group(2))
        return "%s%d" % (m.group(1), order[k - 1] + 1) if 1 <= k <= len(order) else m.group(0)
    return re.sub(r"\b([A-Z][a-z]?)(\d+)\b", f, text)


def finish(kind, name):
    """after the full analyses: site info into each analysed report, predictions into the ranking file"""
    rk = "%s_%s_ranking.txt" % (name, kind)
    if not os.path.exists("screen/sites.json") or not os.path.exists(rk):
        return
    sites = json.load(open("screen/sites.json"))
    add = ["", " RESULTS OF THE FULL ANALYSES (neutral numbering; details in each site folder's report):"]
    for r, s in enumerate(sites):
        rep = os.path.join(s["folder"], "%s_report.txt" % s["run"])
        if not os.path.exists(rep):
            continue
        txt = open(rep).read()
        if "SCENARIO|" not in txt:
            hdr = ("SCENARIO| %s | site %d of %d | %.1f kcal/mol above the best site | %s\n" % (
                kind, r + 1, len(sites), s["rel"], s["desc"]))
            if s.get("scale") is not None:
                k_ = "PA" if kind == "cation" else "pKa"
                hdr += "FORMATION| %s for this site: %s -> needs %s\n" % (k_, fmt_value(k_, s["scale"]), difficulty(k_, s["scale"])[1])
            order = s.get("order") or []
            if order and order != list(range(len(order))):
                pairs = ["%d->%d" % (order[k] + 1, k + 1) for k in range(len(order)) if order[k] != k]
                hdr += ("NUMBERING| this site's files number the atoms differently from the neutral molecule "
                        "(neutral -> here): %s\n" % ", ".join(pairs))
            open(rep, "w").write(hdr + txt)
        m = re.search(r"PRED\|\s*(.+)", txt)
        n_ = re.search(r"NUAT\|\s*(.+)", txt)
        order = s.get("order") or []
        add.append(" %-16s rearrangement: %s" % (s["folder"], relabel(m.group(1).strip(), order) if m else "-"))
        if n_ and kind == "cation":
            add.append(" %-16s nucleophile attack (atom numbers as in the site report): %s" % ("", n_.group(1).strip()))
    t = open(rk).read()
    if "RESULTS OF THE FULL ANALYSES" not in t:
        open(rk, "w").write(t + "\n".join(add) + "\n")


# ============================================================== radical ions
def centers(name, xtb, chg, uhf):
    """where the hole (radical cation) or the extra electron (radical anion) sits:
       charge difference ion - neutral at the ion geometry (both in the run solvent)"""
    if uhf == 0 or chg == 0 or not os.path.exists("xtbopt.xyz"):
        return []
    solv = solvent()
    single_point(xtb, "xtbopt.xyz", "spin/ion", chg, uhf, solv)
    single_point(xtb, "xtbopt.xyz", "spin/neutral", 0, 0, solv)
    qi, qn = read_charges("spin/ion"), read_charges("spin/neutral")
    if not qi or not qn or len(qi) != len(qn):
        return []
    sym, X = read_xyz("xtbopt.xyz")
    smiles = os.environ.get("SMILES_IN", "")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    nh = mol.GetNumAtoms() if mol is not None else sum(1 for s in sym if s != "H")
    w = np.array(qi) - np.array(qn)
    if chg < 0:
        w = -w
    heavy = np.zeros(nh)
    for i in range(len(sym)):
        if i < nh:
            heavy[i] += w[i]
        else:
            heavy[min(range(nh), key=lambda j: np.linalg.norm(X[i] - X[j]))] += w[i]
    order = np.argsort(-heavy)
    top = heavy[order[0]]
    cands = [int(i) for i in order if heavy[i] >= max(0.15, 0.45 * top) and sym[i] in ("C", "N", "O", "S")]
    carb = [i for i in cands if sym[i] == "C"]
    cen = (carb or cands or [int(order[0])])[:3]
    what = "hole (radical cation)" if chg > 0 else "extra electron (radical anion)"
    lines = ["SPIN| where the %s sits (charge difference ion - neutral at the ion geometry, H folded onto heavy atoms):" % what]
    for i in order[:6]:
        lines.append("SPIN|   %s%-4d %6.3f" % (sym[i], i + 1, heavy[i]))
    lines.append("CENTERS| " + " ".join(str(i + 1) for i in cen))
    open("%s_centers.txt" % name, "w").write("\n".join(lines) + "\n")
    return cen


def formation(name, xtb):
    """radical ions: E(ox) / E(red) from the optimized ion and the neutral (re-optimized here)"""
    scen = os.environ.get("SCEN", "")
    if scen not in ("radical_cation", "radical_anion") or not os.path.exists("xtbopt.xyz"):
        return None
    smiles = os.environ.get("SMILES_IN", "")
    solv = solvent()
    chg = 1 if scen == "radical_cation" else -1
    eN, xN = _ref_neutral(smiles, xtb, solv, "ref_neutral")
    if xN is None:
        return None
    sc = SCALE_SOLV["Eox"]
    e_ion = single_point(xtb, "xtbopt.xyz", "ref_neutral/sp_ion", chg, 1, sc)
    e_neu = single_point(xtb, xN, "ref_neutral/sp_neutral", 0, 0, sc)
    if e_ion is None or e_neu is None:
        return None
    k_ = "Eox" if chg > 0 else "Ered"
    v = cal(k_, (e_ion - e_neu) * KCAL)
    dval, txt = difficulty(k_, v)
    open("%s_%s_formation.txt" % (name, scen), "w").write(
        "FORMATION| %s | %s | %.3f | %.2f | - | %s\n" % (scen, k_, v, dval, txt))
    return v, dval, txt


def stamp(name):
    """write the species into the report (the report itself is generated from the SMILES)"""
    rep = "%s_report.txt" % name
    if not os.path.exists(rep):
        return
    scen = os.environ.get("SCEN", "")
    q, u = os.environ.get("CHRG", ""), os.environ.get("UHF", "")
    txt = open(rep).read()
    if "SPECIES|" in txt:
        return
    head = {"neutral": "neutral molecule as drawn (scenario 1 of 5)",
            "radical_cation": "RADICAL CATION: one electron removed (SET oxidation)",
            "radical_anion": "RADICAL ANION: one electron added (SET reduction)",
            "cation": "CATION: electrophile (H+) added", "anion": "ANION: one proton removed"}.get(scen, scen)
    if q != "" and u != "":
        txt = re.sub(r"charge \S+, unpaired electrons \d+", "charge %+d, unpaired electrons %s" % (int(q), u), txt)
    extra = ""
    if scen in ("radical_cation", "radical_anion"):
        cf = "%s_centers.txt" % name
        if os.path.exists(cf):
            extra += "\n" + open(cf).read().rstrip() + "\n"
        ff = "%s_%s_formation.txt" % (name, scen)
        if os.path.exists(ff):
            p = ff and open(ff).read().split("|")
            k_, v = p[2].strip(), float(p[3])
            extra += "FORMATION| %s -> needs %s\n           (%s)\n" % (fmt_value(k_, v), p[6].strip(), cal_note(k_))
    hdr = "SPECIES| %s%s\n" % (head, "" if q == "" else " -- charge %+d, unpaired electrons %s" % (int(q), u))
    open(rep, "w").write(hdr + txt.rstrip() + "\n" + extra)


# ============================================================== ranking (neutral report)
def rank(jobdir, name):
    rows = []
    labels = {"cation": "acid / electrophile -> cation", "radical_cation": "one-electron oxidation -> radical cation"}
    for scen in ("cation", "radical_cation"):
        for f in glob.glob(os.path.join(jobdir, scen, "*_formation.txt")):
            p = [x.strip() for x in open(f).read().split("|")]
            if len(p) >= 7:
                rows.append(dict(scen=scen, kind=p[2], val=float(p[3]), d=float(p[4]), site=p[5], txt=p[6]))
    L = ["", "=" * 100, " CATION OR RADICAL CATION: WHICH FORMS MORE EASILY?  (%s)" % name, "=" * 100]
    if not rows:
        L.append(" No scenario results found.")
    else:
        rows.sort(key=lambda r: r["d"])
        L.append(" Easiest first. Each species needs a different reagent, so the order compares how strong")
        L.append(" that reagent must be (difficulty 0 = trivial ... 5 = practically impossible):")
        for k, r in enumerate(rows):
            L.append(" %d. %-42s %-30s  difficulty %.1f" % (k + 1, labels[r["scen"]], fmt_value(r["kind"], r["val"]), r["d"]))
            L.append("    needs %s%s" % (r["txt"], ("   [site %s]" % r["site"]) if r["site"] not in ("-", "") else ""))
        L.append("")
        L.append(" In one line: " + " > ".join(labels[r["scen"]].split(" ->")[0] for r in rows) + "   (easiest first)")
        L.append(" Which one actually forms depends on the reagent: an acid / Lewis acid / electrophile (e.g. NBS) makes")
        L.append(" the cation, an oxidant or photo-oxidant (SET) makes the radical cation.")
        L.append(" Scales: xtb energies converted with linear fits to experimental data:")
        for k_ in ("PA", "Eox"):
            if any(r["kind"] == k_ for r in rows):
                L.append("   %-5s %s" % (k_, cal_note(k_)))
        L.append(" Good for ranking and order-of-magnitude reagent choice, not for exact values.")
    txt = "\n".join(L) + "\n"
    nd = os.path.join(jobdir, "neutral")
    os.makedirs(nd, exist_ok=True)
    open(os.path.join(nd, "%s_ranking.txt" % name), "w").write(txt.lstrip("\n"))
    rep = os.path.join(nd, "%s_neutral_report.txt" % name)
    if os.path.exists(rep):
        t = open(rep).read()
        for hd in ("\n WHICH REACTIVE SPECIES", "\n CATION OR RADICAL CATION"):
            t = t.split("\n" + "=" * 100 + hd)[0]
        t = t.rstrip() + "\n"
        open(rep, "w").write(t + txt)
    return rows


# ============================================================== CLI
if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd = a[0]
    if cmd in ("sites", "screen", "finish") and a[1] != "cation":
        sys.exit("only the cation scenario is supported")
    if cmd == "sites":
        f = cation_sites if a[1] == "cation" else anion_sites
        for s in f(a[2]):
            print("%-60s %s" % (s["desc"], s["smiles"]))
    elif cmd == "screen":
        screen(a[1], a[2], a[3], a[4])
    elif cmd == "usestart":
        try:
            s1, _ = read_xyz(a[1])
            s2, _ = read_xyz(a[2])
            if s1 == s2:
                shutil.copy(a[2], a[1])
                print("start geometry: screened structure %s" % a[2])
        except Exception as e:
            print("start geometry not reused (%s)" % e)
    elif cmd == "finish":
        finish(a[1], a[2])
    elif cmd == "centers":
        centers(a[1], a[2], int(a[3]), int(a[4]))
    elif cmd == "formation":
        formation(a[1], a[2])
    elif cmd == "stamp":
        stamp(a[1])
    elif cmd == "rank":
        for r in rank(a[1], a[2]):
            print(r["scen"], fmt_value(r["kind"], r["val"]), "difficulty %.1f" % r["d"])
    else:
        sys.exit(__doc__)
