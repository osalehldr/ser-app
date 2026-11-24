#!/usr/bin/env python3
"""
cascade.py NAME SMILES XTB CHG UHF     (run in the species' folder after rearr.py and nuattack.py)

Follows a multi-step sequence: the product of the predicted step (rearr.py) becomes the next species,
which is optimized and analysed again, until no facile step is left (or CASCADE_MAX steps).
  * radicals + oxidant (NU=oxidant or oxidant+NUCLEOPHILE): radical-polar crossover -- a tertiary,
    secondary, benzylic, allylic or alpha-heteroatom carbon radical is oxidized to the cation at once;
    a primary carbon radical only when no further radical step is facile; O/N radicals never
    (the printed E(ox) is a rough xtb estimate, for orientation only)
  * a cation trapped by its own OH/OR/NR (onium ion formed) ends the sequence (ether / amine after loss of H+)
  * at the end, the chosen nucleophile (cations) or radical partner (radicals) is run on the final species
Environment: SOLV, NU, CASCADE (off to skip), CASCADE_MAX (4), GXTB (passed to rearr)
Writes cascade/stepK/ folders; prints the summary (pipeline: NAME_cascade.txt, appended to the report).
"""
import os, re, sys, json, shutil, subprocess
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nucleophiles
import scenarios

KCAL = 627.509
VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2, "S": 2, "B": 3, "Si": 4, "P": 3, "F": 1, "Cl": 1, "Br": 1, "I": 1}


def read_xyz(p):
    L = open(p).read().split("\n")
    n = int(L[0].split()[0])
    sym = [l.split()[0] for l in L[2:2 + n]]
    X = np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + n]])
    m = re.search(r"energy:\s*(-?\d+\.\d+)", L[1])
    return sym, X, (float(m.group(1)) if m else None)


def write_xyz(p, sym, X, comment=""):
    with open(p, "w") as f:
        f.write("%d\n%s\n" % (len(sym), comment))
        for s, x in zip(sym, X):
            f.write("%-2s %14.8f %14.8f %14.8f\n" % (s, x[0], x[1], x[2]))


def product_species(smiles, Xp, form, brk, chg, uhf):
    """new SMILES + coordinates (reordered to the new SMILES' atom order) after the bond changes, or (None, why)"""
    mol = Chem.MolFromSmiles(smiles)
    molH = Chem.AddHs(mol)
    if molH.GetNumAtoms() != len(Xp):
        return None, "atom count mismatch"
    rw = Chem.RWMol(molH)
    Chem.Kekulize(rw, clearAromaticFlags=True)
    touched = set()
    for i, j in brk:
        if rw.GetBondBetweenAtoms(i, j) is not None:
            rw.RemoveBond(i, j)
        touched |= {i, j}
    for i, j in form:
        if rw.GetBondBetweenAtoms(i, j) is None:
            rw.AddBond(i, j, Chem.BondType.SINGLE)
        touched |= {i, j}
    for a in list(rw.GetAtoms()):
        if a.GetSymbol() == "H" and a.GetDegree() == 0:
            k = a.GetIdx()
            y = min((j for j in range(rw.GetNumAtoms()) if j != k and rw.GetAtomWithIdx(j).GetSymbol() != "H"),
                    key=lambda j: np.linalg.norm(Xp[k] - Xp[j]))
            rw.AddBond(k, y, Chem.BondType.SINGLE)
            touched |= {k, y}
    touched |= {a.GetIdx() for a in rw.GetAtoms() if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() > 0}
    # the pi system around the changed atoms is re-derived from scratch
    zone = set(touched)
    for i in list(touched):
        for b in rw.GetAtomWithIdx(i).GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE:
                zone.add(b.GetOtherAtomIdx(i))
    grown = True
    while grown:                                   # whole conjugated stretches attached to the zone
        grown = False
        for i in list(zone):
            for b in rw.GetAtomWithIdx(i).GetBonds():
                j = b.GetOtherAtomIdx(i)
                if b.GetBondType() != Chem.BondType.SINGLE and j not in zone:
                    zone.add(j)
                    grown = True
    for b in rw.GetBonds():
        if b.GetBeginAtomIdx() in zone and b.GetEndAtomIdx() in zone:
            b.SetBondType(Chem.BondType.SINGLE)
    for i in zone:
        a = rw.GetAtomWithIdx(i)
        a.SetFormalCharge(0)
        a.SetNumRadicalElectrons(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    deficit = {}
    for i in zone:
        a = rw.GetAtomWithIdx(i)
        v = VALENCE.get(a.GetSymbol())
        if v is None:
            return None, "element %s not handled" % a.GetSymbol()
        deficit[i] = v - sum(int(b.GetBondTypeAsDouble()) for b in a.GetBonds())
    # pi bonds: pair up neighbouring atoms that both miss valence, shortest bonds first
    pairs = sorted(((np.linalg.norm(Xp[b.GetBeginAtomIdx()] - Xp[b.GetEndAtomIdx()]), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                    for b in rw.GetBonds() if b.GetBeginAtomIdx() in zone and b.GetEndAtomIdx() in zone))
    changed = True
    while changed:
        changed = False
        for d, i, j in pairs:
            if deficit[i] >= 1 and deficit[j] >= 1:
                b = rw.GetBondBetweenAtoms(i, j)
                if b.GetBondType() == Chem.BondType.SINGLE and d < 1.45:
                    b.SetBondType(Chem.BondType.DOUBLE)
                elif b.GetBondType() == Chem.BondType.DOUBLE and d < 1.25:
                    b.SetBondType(Chem.BondType.TRIPLE)
                else:
                    continue
                deficit[i] -= 1
                deficit[j] -= 1
                changed = True
    q_left, u_left = chg, uhf
    for i, dv in sorted(deficit.items(), key=lambda t: t[1]):
        a = rw.GetAtomWithIdx(i)
        if dv < 0:                                 # over-valent (onium ion): positive charge
            a.SetFormalCharge(-dv)
            q_left -= -dv
    for i, dv in deficit.items():
        if dv <= 0:
            continue
        a = rw.GetAtomWithIdx(i)
        if dv == 1 and u_left > 0:
            a.SetNumRadicalElectrons(1)
            u_left -= 1
        elif dv == 1 and q_left > 0 and a.GetSymbol() == "C":
            a.SetFormalCharge(1)
            q_left -= 1
        elif dv == 1 and q_left < 0:
            a.SetFormalCharge(-1)
            q_left += 1
        else:
            return None, "valence of %s%d could not be assigned" % (a.GetSymbol(), i + 1)
    if q_left != 0 or u_left != 0:
        return None, "charge/spin bookkeeping failed"
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception as e:
        return None, "sanitize failed (%s)" % e
    conf = Chem.Conformer(m.GetNumAtoms())
    for k, x in enumerate(Xp):
        conf.SetAtomPosition(k, x.tolist())
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m)
    nh = sum(1 for a in m.GetAtoms() if a.GetSymbol() != "H")
    parent = {}
    for a in m.GetAtoms():
        if a.GetSymbol() == "H":
            nb = [x.GetIdx() for x in a.GetNeighbors()]
            if len(nb) != 1:
                return None, "H with %d bonds" % len(nb)
            parent[a.GetIdx()] = nb[0]
    try:                                           # also the H that only defines a C=C stereo (vinyl radical E/Z:
        _ps = Chem.RemoveHsParameters()            # meaningless, it inverts fast) -- keeps the atom order check exact
        _ps.removeDefiningBondStereo = True
        heavy = Chem.RemoveHs(m, _ps)
    except Exception:
        heavy = Chem.RemoveHs(m)
    smi = Chem.MolToSmiles(heavy, canonical=False)
    try:
        order = [int(t) for t in re.findall(r"\d+", heavy.GetProp("_smilesAtomOutputOrder"))]
    except Exception:
        order = list(range(heavy.GetNumAtoms()))
    # heavy atoms of the product in new order -> indices in the full list (heavy atoms keep their indices)
    newX, newS = [], []
    for k in order:
        newX.append(Xp[k])
        newS.append(m.GetAtomWithIdx(k).GetSymbol())
    for k in order:
        for h in sorted(h for h, p in parent.items() if p == k):
            newX.append(Xp[h])
            newS.append("H")
    chk = Chem.AddHs(Chem.MolFromSmiles(smi))
    if [a.GetSymbol() for a in chk.GetAtoms()] != newS:
        return None, "atom order check failed"
    fullmap = list(order) + [h for k in order for h in sorted(h for h, p in parent.items() if p == k)]
    return dict(smiles=smi, sym=newS, X=np.array(newX), order=order, map=fullmap), ""


def run(cmd, cwd, out, env=None, timeout=None):
    with open(os.path.join(cwd, out), "w") as f:
        try:
            subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


def analyse(sp, d, name, xtb, env):
    """optimize + Fukui + rearr in folder d for species sp; returns the predicted.json dict"""
    os.makedirs(d, exist_ok=True)
    write_xyz(os.path.join(d, name + ".xyz"), sp["sym"], sp["X"])
    solv = env.get("SOLV", "")
    base = ["--chrg", str(sp["chg"]), "--uhf", str(sp["uhf"]), "--gfn", "2"] + (["--alpb", solv] if solv else [])
    run([xtb, name + ".xyz", "--opt"] + base, d, name + "_xtb.out", env)
    if not os.path.exists(os.path.join(d, "xtbopt.xyz")):
        return None
    run([xtb, "xtbopt.xyz", "--vfukui"] + base, d, name + "_fukui.out", env)
    run([sys.executable, os.path.join(HERE, "rearr.py"), name, sp["smiles"], xtb, str(sp["chg"]), str(sp["uhf"])],
        d, name + "_rearr.txt", env)
    p = os.path.join(d, "rearr", "predicted.json")
    return json.load(open(p)) if os.path.exists(p) else dict(step=False)


def ox_potential(d, sp, xtb, env):
    """radical -> cation: calibrated E(ox), V vs SCE (MeCN); cation optimized from the radical geometry"""
    w = os.path.join(d, "ox")
    os.makedirs(w, exist_ok=True)
    shutil.copy(os.path.join(d, "xtbopt.xyz"), os.path.join(w, "r.xyz"))
    run([xtb, "r.xyz", "--chrg", "0", "--uhf", str(sp["uhf"]), "--gfn", "2", "--alpb", "acetonitrile"], w, "rad.out", env)
    run([xtb, "r.xyz", "--opt", "--chrg", "1", "--uhf", "0", "--gfn", "2", "--alpb", "acetonitrile"], w, "cat.out", env)
    vert = False
    s0, X0, _ = read_xyz(os.path.join(w, "r.xyz"))
    if not os.path.exists(os.path.join(w, "xtbopt.xyz")) or _bonds(s0, X0) != _bonds(*read_xyz(os.path.join(w, "xtbopt.xyz"))[:2]):
        vert = True                              # the cation reacts on optimization: use the cation at the radical geometry
        run([xtb, "r.xyz", "--chrg", "1", "--uhf", "0", "--gfn", "2", "--alpb", "acetonitrile"], w, "cat.out", env)
    E = []
    for f in ("rad.out", "cat.out"):
        m = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", open(os.path.join(w, f)).read())
        E.append(float(m[-1]) if m else None)
    if None in E:
        return None, vert
    return scenarios.cal("Eox", (E[1] - E[0]) * KCAL), vert


RC = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "Si": 1.11, "Cl": 1.02, "Br": 1.20, "I": 1.39}


def _bonds(sym, X):
    return set((i, j) for i in range(len(sym)) for j in range(i + 1, len(sym))
               if np.linalg.norm(X[i] - X[j]) < 1.25 * (RC.get(sym[i], 1.0) + RC.get(sym[j], 1.0)))


def radical_class(smiles):
    """type of the carbon radical: decides whether an oxidant converts it to the cation at once"""
    m = Chem.MolFromSmiles(smiles)
    for a in m.GetAtoms():
        if a.GetNumRadicalElectrons() > 0 and a.GetSymbol() == "C":
            nb = [x for x in a.GetNeighbors()]
            if any(x.GetSymbol() in ("O", "N", "S") for x in nb):
                return "alpha-heteroatom", True
            if any(x.GetIsAromatic() for x in nb):
                return "benzylic", True
            if any(b.GetBondType() == Chem.BondType.DOUBLE for x in nb for b in x.GetBonds() if b.GetOtherAtom(x).GetIdx() != a.GetIdx()):
                return "allylic", True
            heavy = sum(1 for x in nb if x.GetSymbol() != "H")
            return {3: "tertiary", 2: "secondary"}.get(heavy, "primary"), heavy >= 2
    return "non-carbon", False


def to_cation(smiles):
    m = Chem.RWMol(Chem.MolFromSmiles(smiles))
    rad = [a for a in m.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if len(rad) != 1:
        return None
    a = rad[0]
    a.SetNumRadicalElectrons(0)
    a.SetFormalCharge(a.GetFormalCharge() + 1)
    m = m.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return Chem.MolToSmiles(m, canonical=False)


KT = 0.593                     # kcal/mol at 298 K
E1_ALPHA = 0.3                 # Bell-Evans-Polanyi: fraction of the alkene energy difference felt in the barrier


def _energy(path):
    t = open(path).read() if os.path.exists(path) else ""
    m = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)", t)
    return float(m[-1]) if m else None


def e1_options(sp, xtb, env, k):
    """loss of H+ from each C-H next to a cationic carbon -> alkenes, optimized and compared"""
    mol = Chem.MolFromSmiles(sp["smiles"])
    if mol is None:
        return []
    nh = mol.GetNumAtoms()
    p_opt = os.path.join(sp["dir"], "xtbopt.xyz") if sp.get("dir") else ""
    sym, X = (read_xyz(p_opt)[:2]) if p_opt and os.path.exists(p_opt) else (sp["sym"], sp["X"])
    X = np.array(X)
    solv = env.get("SOLV", "")
    opts = {}
    for c in [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and a.GetFormalCharge() > 0]:
        nb = [x.GetIdx() for x in mol.GetAtomWithIdx(c).GetNeighbors()]
        ax = None
        if len(nb) >= 3:
            ax = np.cross(X[nb[1]] - X[nb[0]], X[nb[2]] - X[nb[0]])
            ax = ax / (np.linalg.norm(ax) + 1e-9)
        for n in nb:
            an = mol.GetAtomWithIdx(n)
            if an.GetSymbol() != "C" or an.GetTotalNumHs() == 0:
                continue
            hs = [h for h in range(nh, len(sym)) if sym[h] == "H" and np.linalg.norm(X[h] - X[n]) < 1.25]
            if not hs:
                continue
            h = max(hs, key=lambda h_: abs(np.dot(ax, (X[h_] - X[n]) / np.linalg.norm(X[h_] - X[n]))) if ax is not None else 0)
            rw = Chem.RWMol(mol)
            Chem.Kekulize(rw, clearAromaticFlags=True)
            ac, an2 = rw.GetAtomWithIdx(c), rw.GetAtomWithIdx(n)
            ac.SetFormalCharge(0)
            an2.SetNumExplicitHs(max(0, an2.GetTotalNumHs() - 1))
            an2.SetNoImplicit(True)
            ac.SetNoImplicit(True)
            ac.SetNumExplicitHs(ac.GetTotalNumHs())
            b = rw.GetBondBetweenAtoms(c, n)
            b.SetBondType(Chem.BondType.DOUBLE if b.GetBondType() == Chem.BondType.SINGLE else Chem.BondType.TRIPLE)
            try:
                m2 = rw.GetMol()
                Chem.SanitizeMol(m2)
                smi = Chem.MolToSmiles(m2, canonical=False)
                key = Chem.MolToSmiles(m2, isomericSmiles=False)
            except Exception:
                continue
            if key in opts:                       # equivalent H's (e.g. the two methyls of C(CH3)2+)
                opts[key]["nH"] += len(hs)
                continue
            keep = [i for i in range(len(sym)) if i != h]
            d = os.path.join("cascade", "e1_%d_C%d_C%d" % (k, c + 1, n + 1))
            os.makedirs(d, exist_ok=True)
            write_xyz(os.path.join(d, "alkene.xyz"), [sym[i] for i in keep], X[keep])
            run([xtb, "alkene.xyz", "--opt", "--chrg", "0", "--uhf", "0", "--gfn", "2"] + (["--alpb", solv] if solv else []),
                d, "xtb.out", env)
            E = _energy(os.path.join(d, "xtb.out"))
            if E is None:
                continue
            opts[key] = dict(c=c, n=n, h=h, nH=len(hs), smiles=smi, E=E, dir=d, keep=keep,
                             sym=[sym[i] for i in keep], X=read_xyz(os.path.join(d, "xtbopt.xyz"))[1]
                             if os.path.exists(os.path.join(d, "xtbopt.xyz")) else X[keep])
    out = list(opts.values())
    if not out:
        return []
    e0 = min(o["E"] for o in out)
    for o in out:
        o["rel"] = (o["E"] - e0) * 627.509
        o["w"] = o["nH"] * np.exp(-E1_ALPHA * o["rel"] / KT)
    tot = sum(o["w"] for o in out)
    for o in out:
        o["pct"] = 100.0 * o["w"] / tot
    return sorted(out, key=lambda o: -o["w"])


def _parents(sym, X):
    """H atom -> its heavy atom (nearest, within 1.3 A)"""
    X = np.array(X)
    heavy = [i for i, s_ in enumerate(sym) if s_ != "H"]
    par = {}
    for i, s_ in enumerate(sym):
        if s_ == "H" and heavy:
            j = min(heavy, key=lambda j_: np.linalg.norm(X[i] - X[j_]))
            if np.linalg.norm(X[i] - X[j]) < 1.3:
                par[i] = j
    return par


def overall(start_smiles, s_sym, s_X, f_smiles, f_sym, f_X, amap, name):
    """net change start -> final in the start structure's atom numbers; also writes NAME_overall.png"""
    m0, m1 = Chem.MolFromSmiles(start_smiles), Chem.MolFromSmiles(f_smiles)
    if m0 is None or m1 is None:
        return None
    for m_ in (m0, m1):
        Chem.Kekulize(m_, clearAromaticFlags=True)
    lab = lambda i: "%s%d" % (m0.GetAtomWithIdx(i).GetSymbol(), i + 1)
    b0 = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())): b.GetBondTypeAsDouble() for b in m0.GetBonds()}
    b1 = {}
    for b in m1.GetBonds():
        i, j = amap[b.GetBeginAtomIdx()], amap[b.GetEndAtomIdx()]
        b1[frozenset((i, j))] = b.GetBondTypeAsDouble()
    sym_ = lambda o: {1.0: "-", 2.0: "=", 3.0: "#", 1.5: "-"}.get(o, "-")
    broken = sorted(tuple(sorted(k)) for k in b0 if k not in b1)
    formed = sorted(tuple(sorted(k)) for k in b1 if k not in b0)
    order = sorted(tuple(sorted(k)) for k in b1 if k in b0 and b1[k] != b0[k])
    p0, p1 = _parents(s_sym, s_X), _parents(f_sym, f_X)
    p1o = {amap[h]: amap[j] for h, j in p1.items() if h < len(amap) and j < len(amap)}
    lost = sorted(set(p0) - set(p1o))
    moved = sorted((h, p0[h], p1o[h]) for h in p0 if h in p1o and p1o[h] != p0[h])
    gained = sorted(h for h in p1o if h not in p0)
    txt_b = ", ".join("%s-%s" % (lab(i), lab(j)) for i, j in broken) or "none"
    txt_f = ", ".join("%s%s%s" % (lab(i), sym_(b1[frozenset((i, j))]), lab(j)) for i, j in formed) or "none"
    txt_o = ", ".join("%s-%s %s -> %s" % (lab(i), lab(j), {1.0: "single", 2.0: "double", 3.0: "triple"}.get(b0[frozenset((i, j))], "?"),
                                          {1.0: "single", 2.0: "double", 3.0: "triple"}.get(b1[frozenset((i, j))], "?"))
                      for i, j in order) or "none"
    # net H count per heavy atom (which individual H atom moves does not matter chemically)
    n0 = {a.GetIdx(): a.GetTotalNumHs() for a in m0.GetAtoms()}       # from the SMILES: exact, and not affected by
    n1 = {}                                                            # changes already made in the start optimization
    for a in m1.GetAtoms():
        if a.GetIdx() < len(amap):
            n1[amap[a.GetIdx()]] = n1.get(amap[a.GetIdx()], 0) + a.GetTotalNumHs()
    net = sorted((a_, n1.get(a_, 0) - n0.get(a_, 0)) for a_ in set(n0) | set(n1) if n1.get(a_, 0) != n0.get(a_, 0))
    hl = ["%s %+d H" % (lab(a_), d_) for a_, d_ in net]
    dH = sum(n1.values()) - sum(n0.values())
    if dH < 0:
        hl.append("%d H+ lost overall" % -dH)
    elif dH > 0:
        hl.append("%d H gained overall" % dH)
    lost = [a_ for a_, d_ in net if d_ < 0]
    res = dict(broken=txt_b, formed=txt_f, order=txt_o, h="; ".join(hl) or "none", product=f_smiles, pic="")
    try:
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        d0 = Chem.MolFromSmiles(start_smiles)
        rdDepictor.Compute2DCoords(d0)
        hb, hc, ha, ac = [], {}, [], {}
        for i, j in broken:
            bd = d0.GetBondBetweenAtoms(i, j)
            if bd is not None:
                hb.append(bd.GetIdx()); hc[bd.GetIdx()] = (0.30, 0.75, 0.35)
        for i, j in order:
            bd = d0.GetBondBetweenAtoms(i, j)
            if bd is not None:
                hb.append(bd.GetIdx()); hc[bd.GetIdx()] = (1.0, 0.62, 0.15)
        for i, j in formed:
            for a_ in (i, j):
                if a_ not in ac:
                    ha.append(a_); ac[a_] = (1.0, 0.62, 0.15)
        for a_ in lost:
            if a_ not in ac:
                ha.append(a_)
            ac[a_] = (0.65, 0.40, 0.85)
        dv = rdMolDraw2D.MolDraw2DCairo(1000, 460, 500, 460)
        op = dv.drawOptions()
        op.addAtomIndices = True
        op.legendFontSize = 18
        d1 = Chem.MolFromSmiles(f_smiles)
        dv.DrawMolecules([d0, d1], highlightAtoms=[ha, []], highlightBonds=[hb, []],
                         highlightAtomColors=[ac, {}], highlightBondColors=[hc, {}],
                         legends=["start: green = breaks, orange = new bond, purple = loses H", "final"])
        dv.FinishDrawing()
        fn = name + "_overall.png"
        open(fn, "wb").write(dv.GetDrawingText())
        res["pic"] = fn
    except Exception:
        pass
    return res


def main():
    name, smiles, xtb, chg, uhf = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    _cm = os.environ.get("CASCADE", "auto").lower()
    if _cm in ("off", "0", "no"):
        return
    if _cm == "auto" and not (uhf > 0 and chg == 0):    # by default only neutral radicals (radical cascades)
        return
    maxs = int(os.environ.get("CASCADE_MAX", "4"))
    oxe = float(os.environ.get("OXIDANT_E", "0.5"))
    env = os.environ.copy()
    try:
        partner = nucleophiles.resolve(os.environ.get("NU", "none"), xtb)
    except ValueError:
        partner = nucleophiles.resolve("none")
    if chg < 0 or not os.path.exists("rearr/predicted.json") or not os.path.exists("xtbopt.xyz"):
        return
    pj = json.load(open("rearr/predicted.json"))
    sym, X, E = read_xyz("xtbopt.xyz")
    sp = dict(smiles=smiles, sym=sym, X=X, chg=chg, uhf=uhf, dir=".", name=name)
    steps, final_note = [], ""
    canon = lambda s_: (lambda m_: Chem.MolToSmiles(m_, isomericSmiles=False) if m_ is not None else s_)(Chem.MolFromSmiles(s_))
    visited = [canon(smiles)]
    amap = list(range(len(sym)))                 # current atom -> atom number in the start structure
    s_sym, s_X = list(sym), np.array(X)
    excl, merged = set(), 0
    pending = None                               # an uphill step on trial: kept only if the next step is downhill
    oxidant = partner["kind"] == "oxidant"
    if not pj.get("step") and not pj.get("uphill") and not (oxidant and uhf > 0 and chg == 0):
        return                                   # nothing to follow: the one-step report says it all
    shutil.rmtree("cascade", ignore_errors=True)
    os.makedirs("cascade")
    k = 0
    while True:
        k += 1
        if sum(1 for s_ in steps if s_["k"] is not None and not s_.get("free")) >= maxs or k > maxs + 6:
            final_note = "stopped after %d steps (CASCADE_MAX)" % maxs
            break
        radical = sp["uhf"] > 0 and sp["chg"] == 0
        rmol = Chem.MolFromSmiles(sp["smiles"])
        c_radical = radical and any(a.GetNumRadicalElectrons() > 0 and a.GetSymbol() == "C" for a in rmol.GetAtoms())
        if c_radical and oxidant:                  # only carbon radicals cross over (not alkoxy / aminyl radicals)
            eox, vert = ox_potential(sp["dir"], sp, xtb, env)
            rcls, easy = radical_class(sp["smiles"])
            if easy or not pj.get("step"):
                cs = to_cation(sp["smiles"])
                if cs is None:
                    final_note = "radical-polar crossover: the cation SMILES could not be made"
                    break
                etxt = ("E(ox) ~%+.1f V%s" % (eox, ", vertical" if vert else "")) if eox is not None else "E(ox) n/a"
                steps.append(dict(k=k, what="radical-polar crossover: %s radical oxidized to the cation (%s)"
                                  % (rcls, etxt if easy else etxt + "; no radical step left"),
                                  dEa=None, dE=None, smiles=cs, charge="cation"))
                nd = os.path.join("cascade", "step%d" % k)
                sp = dict(smiles=cs, sym=sp["sym"], X=read_xyz(os.path.join(sp["dir"], "xtbopt.xyz"))[1], chg=1, uhf=0,
                          dir=nd, name="%s_s%d" % (name, k))
                pj = analyse(sp, nd, sp["name"], xtb, env)
                if pj is None:
                    final_note = "the cation could not be optimized"
                    break
                continue
        if not pj.get("step"):
            up = pj.get("uphill")
            if up and pending is None:
                pending = dict(sp=sp, amap=list(amap), visited=list(visited), n=len(steps), dE=up["dE"], k=k,
                               cum=up["dE"], count=0)
                pj = dict(up, step=True, uphill_step=True)
            else:
                final_note = "no further facile step: %s is the final species" % ("radical" if radical else "cation")
                break
        if "proton transfer" in pj.get("type", ""):
            steps.append(dict(k=k, what=pj["type"], dEa=pj["dEa"], dE=pj["dE"], smiles="-", charge="-"))
            final_note = "the cation is quenched by proton loss (alkene)"
            break
        Xp = read_xyz(os.path.join(sp["dir"], "rearr", "predicted.xyz"))[1]
        prod, why = product_species(sp["smiles"], Xp, [tuple(p) for p in pj["form"]], [tuple(p) for p in pj["brk"]],
                                    sp["chg"], sp["uhf"])
        tries = 0
        while prod is not None and canon(prod["smiles"]) in visited and tries < 3:
            tries += 1
            for p_ in list(pj.get("form", [])) + list(pj.get("brk", [])):
                excl.add("%d-%d" % (min(p_) + 1, max(p_) + 1))
            steps.append(dict(k=None, what="%s would lead back to an earlier structure: both are forms of one bridged "
                                           "(non-classical) cation, merged; next pathway tried" % pj.get("type", "the step"),
                              dEa=None, dE=None, smiles="-", charge="-"))
            alt = dict(sp, dir=os.path.join("cascade", "step%d_alt%d" % (k, tries)), name="%s_s%da%d" % (name, k, tries))
            pj = analyse(alt, alt["dir"], alt["name"], xtb, dict(env, REARR_EXCLUDE=",".join(sorted(excl))))
            sp = dict(alt, X=read_xyz(os.path.join(alt["dir"], "xtbopt.xyz"))[1]) \
                if os.path.exists(os.path.join(alt["dir"], "xtbopt.xyz")) else alt
            if pj is None or not pj.get("step") or "proton transfer" in pj.get("type", ""):
                prod, why = None, "no other pathway"
                break
            Xp = read_xyz(os.path.join(sp["dir"], "rearr", "predicted.xyz"))[1]
            prod, why = product_species(sp["smiles"], Xp, [tuple(p) for p in pj["form"]], [tuple(p) for p in pj["brk"]],
                                        sp["chg"], sp["uhf"])
        if tries and (prod is None or canon(prod["smiles"]) in visited):
            final_note = "no further step (only interconversion of the bridged forms): %s is the final species" % (
                "radical" if radical else "cation")
            k -= 1
            break
        if prod is None:
            steps.append(dict(k=k, what=pj["type"], dEa=pj["dEa"], dE=pj["dE"], smiles="?", charge="?"))
            final_note = "product structure of this step could not be assigned (%s); sequence stops here" % why
            break
        pm = Chem.MolFromSmiles(prod["smiles"])
        onium = [a for a in pm.GetAtoms() if a.GetFormalCharge() > 0 and a.GetSymbol() in ("O", "N", "S")]
        if pending is not None and not pj.get("uphill_step"):
            pending["cum"] += pj["dE"]
            pending["count"] += 1
            if pending["cum"] <= 0.0:                    # uphill step pays off: keep it
                steps[pending["n"]]["what"] += " (uphill, made possible by the downhill steps after it)"
                pending = None
            elif pending["cum"] > 15.0 or pending["count"] > 3:
                break
        free = pj["dEa"] <= 1.0 and abs(pj["dE"]) <= 1.0 and not pj.get("uphill_step") and \
            str(pj.get("src", "")).startswith("relaxed scan")
        steps.append(dict(k=k, what=pj["type"] + (" (switch between bridged forms of the same cation; not counted as a "
                                                  "step)" if free else ""), dEa=pj["dEa"], dE=pj["dE"],
                          smiles=prod["smiles"], charge="radical" if sp["uhf"] else "cation", free=free))
        if onium and sp["chg"] > 0 and not any(a.GetFormalCharge() > 0 and a.GetSymbol() in ("O", "N", "S")
                                               for a in Chem.MolFromSmiles(sp["smiles"]).GetAtoms()):
            final_note = "trapped intramolecularly by %s: onium ion -> %s after loss of H+" % (
                onium[0].GetSymbol(), "ether" if onium[0].GetSymbol() == "O" else "amine/sulfide")
            amap = [amap[i] for i in prod["map"]]
            sp = dict(smiles=prod["smiles"], sym=prod["sym"], X=prod["X"], chg=sp["chg"], uhf=sp["uhf"], dir=None,
                      name=None)
            break
        visited.append(canon(prod["smiles"]))
        amap = [amap[i] for i in prod["map"]]
        excl = set()
        nd = os.path.join("cascade", "step%d" % (k + 1))
        sp = dict(smiles=prod["smiles"], sym=prod["sym"], X=prod["X"], chg=sp["chg"], uhf=sp["uhf"], dir=nd,
                  name="%s_s%d" % (name, k + 1))
        pj = analyse(sp, nd, sp["name"], xtb, env)
        if pj is None:
            final_note = "the next species could not be optimized"
            break
    if pending is not None:                       # the uphill step led nowhere downhill: undo it
        sp, amap, visited, k = pending["sp"], pending["amap"], pending["visited"], pending["k"]
        del steps[pending["n"]:]
        final_note = ("the only pathway found is uphill (%+.1f kcal/mol) and is not followed by a downhill step: "
                      "no rearrangement" % pending["dE"])
        pending = None
    # E1: a cation with nothing left to do and no nucleophile chosen loses H+ to give an alkene
    e1 = []
    if sp["chg"] > 0 and sp["uhf"] == 0 and partner["kind"] == "none" and not final_note.startswith(("stopped", "trapped")) \
            and any(a.GetSymbol() == "C" and a.GetFormalCharge() > 0 for a in Chem.MolFromSmiles(sp["smiles"]).GetAtoms()):
        e1 = e1_options(sp, xtb, env, k)
        for o in e1:
            o["lab"] = "C%d=C%d" % (amap[o["c"]] + 1, amap[o["n"]] + 1)
        if e1:
            b_ = e1[0]
            close = len(e1) > 1 and e1[1]["pct"] >= e1[0]["pct"] / 3.0
            steps.append(dict(k=k + 1 if steps and steps[-1]["k"] == k else k,
                              what="loss of H+ (E1) from C%d next to C%d+: alkene C%d=C%d%s" % (
                                  amap[b_["n"]] + 1, amap[b_["c"]] + 1, amap[b_["c"]] + 1, amap[b_["n"]] + 1,
                                  " (close call)" if close else ""),
                              dEa=None, dE=None, smiles=b_["smiles"], charge="neutral"))
            final_note = "the cation is quenched by loss of H+ (E1): neutral alkene"
            amap = [amap[i] for i in b_["keep"]]
            sp = dict(smiles=b_["smiles"], sym=b_["sym"], X=b_["X"], chg=0, uhf=0, dir=None, name=None)
    # final species: chosen nucleophile / partner
    nuat = ""
    fin = partner.get("then") if oxidant else partner
    if sp.get("dir") and sp["dir"] != "." and fin and fin["kind"] != "none":
        e2 = dict(env, NU=fin["name"])
        run([sys.executable, os.path.join(HERE, "nuattack.py"), sp["name"], sp["smiles"], xtb, str(sp["chg"]), str(sp["uhf"])],
            sp["dir"], sp["name"] + "_nuattack.txt", e2)
        t = open(os.path.join(sp["dir"], sp["name"] + "_nuattack.txt")).read()
        m = re.search(r"NUAT\|\s*(.+)", t)
        r_ = re.search(r"\n(RESULT\n(?:  .*\n)+)", t)
        nuat = (r_.group(1).rstrip() if r_ else "") + ("\n" if r_ else "") + ("NUAT| " + m.group(1) if m else "")
        mp_ = re.search(r"NUPROD\|\s*(\S+) \| (\S+) \| (\S+)(.*)", t)
        if mp_:
            _pic = os.path.join(sp["dir"], mp_.group(3)) if mp_.group(3) != "-" else "-"
            nuat += "\nNUPROD| %s | %s | %s%s" % (mp_.group(1), mp_.group(2), _pic, mp_.group(4))
    # report
    print("-" * 70)
    print(" CASCADE: MULTI-STEP SEQUENCE  (each step's product analysed again; GFN2-xTB%s)" %
          (", ALPB " + os.environ["SOLV"] if os.environ.get("SOLV") else ", gas phase"))
    print("-" * 70)
    print(" start: %s  (%s)" % (smiles, "radical" if uhf and not chg else "radical cation" if uhf else "cation"))
    ov = None
    if any(s["k"] is not None for s in steps):
        try:
            ov = overall(smiles, s_sym, s_X, sp["smiles"], sp["sym"], sp["X"], amap, name)
        except Exception:
            ov = None
    if ov:
        print("")
        print(" OVERALL TRANSFORMATION  (start -> final, atom numbers of the start structure)")
        print("   bonds broken   : %s" % ov["broken"])
        print("   bonds formed   : %s" % ov["formed"])
        print("   bond order     : %s" % ov["order"])
        print("   hydrogens      : %s" % ov["h"])
        print("   final product  : %s" % ov["product"])
        if ov["pic"]:
            print("   picture        : %s  (green = breaks, orange = new bond, purple = loses H)" % ov["pic"])
        print("OVERALL| broken %s | formed %s | order %s | H %s | %s | pic %s" % (
            ov["broken"], ov["formed"], ov["order"], ov["h"], ov["product"], ov["pic"] or "-"))
        print("")
        print(" the steps:")
    print(" step  what happens")
    for s in steps:
        if s["k"] is None:
            print("      note: %s" % s["what"])
            print("CNOTE| %s" % s["what"])
            continue
        print(" %-4d %s%s" % (s["k"], s["what"], "" if s["dEa"] is None else
                               "   [barrier %.1f, dE %+.1f kcal/mol]" % (s["dEa"], s["dE"])))
        if s["smiles"] not in ("-", "?"):
            print("      -> %s" % s["smiles"])
        print("CASCADE| %d | %s | %s | %s | %s" % (s["k"], s["what"], "-" if s["dEa"] is None else "%.1f" % s["dEa"],
                                                  "-" if s["dE"] is None else "%+.1f" % s["dE"], s["smiles"]))
    if e1:
        print("      E1 options (alkenes; rel. E = alkene stability, H = hydrogens that can leave, share = statistics x")
        print("      stability; the H lost is often decided kinetically, e.g. methyl H giving isopropenyl groups):")
        for o in e1:
            print("        %-10s rel. E %+5.1f kcal/mol   H %d   share %3.0f%%   %s" % (
                o["lab"], o["rel"], o["nH"], o["pct"], o["smiles"]))
            print("E1| %s | %.1f | %d | %.0f | %s" % (o["lab"], o["rel"], o["nH"], o["pct"], o["smiles"]))
    print(" final: %s" % final_note)
    print("FINAL| %s | %s" % (sp["smiles"], final_note))
    if nuat:
        print(" final species + %s (folder %s):" % (fin["name"], sp["dir"]))
        print("  " + nuat.replace("\n", "\n  "))
    print(" (barriers: kcal/mol, each relative to its own species; after a crossover the energy zero changes;")
    print("  every step is a screening result -- confirm the key steps with DFT)")


if __name__ == "__main__":
    main()
