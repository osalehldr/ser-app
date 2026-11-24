#!/usr/bin/env python3
"""
stereo.py -- when a SMILES leaves stereocenters unspecified, every combination is tried; the ones that
cannot exist in 3D (e.g. a trans-bromonium on a cage) are dropped and the rest are analysed separately.

  python3 stereo.py list SMILES          -> one line per buildable isomer:  label<TAB>SMILES
                                            (nothing printed if the SMILES has no unspecified centers)
  python3 stereo.py summary JOBDIR NAME  -> JOBDIR/NAME_stereo.txt: the isomers compared side by side

Labels: exo / endo for substituents on a norbornane-type two-carbon bridge (exo = towards the one-atom
bridge), otherwise the R/S labels of the enumerated atoms (e.g. iso1_C4R_C5S).
Environment: STEREO=off disables the enumeration; STEREO_MAX (8) = most isomers analysed.
"""
import os, re, sys, glob, itertools
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
KCAL = 627.509
MAXC = int(os.environ.get("STEREO_MAX", "8"))


def unspecified(mol):
    return [i for i, t in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
            if t == "?"]


def _embed(mol, seeds=(7, 11, 23)):
    mh = Chem.AddHs(mol)
    for s in seeds:
        p = AllChem.ETKDGv3()
        p.randomSeed = s
        p.maxIterations = 50
        if AllChem.EmbedMolecule(mh, p) == 0:
            return mh
    # cations in cages: neutral C-H analogue (as the structure builder does)
    rw = Chem.RWMol(mol)
    ch = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "C" and a.GetFormalCharge() != 0]
    if not ch:
        return None
    for i in ch:
        a = rw.GetAtomWithIdx(i)
        h = a.GetTotalNumHs() + (1 if a.GetFormalCharge() > 0 else -1)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(max(0, h))
    try:
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
    except Exception:
        return None
    mh = Chem.AddHs(m2)
    for s in seeds:
        p = AllChem.ETKDGv3()
        p.randomSeed = s
        p.maxIterations = 50
        if AllChem.EmbedMolecule(mh, p) == 0:
            return mh
    return None


def _norbornanes(mol):
    """(M, bridge1, bridge2): one-atom bridge M and the two two-atom bridges of every bicyclo[2.2.1] unit"""
    ri = mol.GetRingInfo()
    r5 = [set(r) for r in ri.AtomRings() if len(r) == 5]
    out = []
    for a, b in itertools.combinations(r5, 2):
        sh = a & b
        if len(sh) != 3:
            continue
        mid = [x for x in sh if sum(1 for y in sh if y != x and mol.GetBondBetweenAtoms(x, y)) == 2]
        if len(mid) != 1:
            continue
        out.append((mid[0], sorted(a - sh), sorted(b - sh)))
    return out


def label_isomer(mol, X, centers):
    """exo/endo for centers on a norbornane two-atom bridge, else R/S"""
    nbs = _norbornanes(mol)
    parts, all_exoendo = [], True
    for c in centers:
        lab = None
        for M, b1, b2 in nbs:
            for br, other in ((b1, b2), (b2, b1)):
                if c in br:
                    ring = set(br) | set(other) | {M}
                    subs = [x for x in mol.GetAtomWithIdx(c).GetNeighbors() if x.GetIdx() not in ring
                            and x.GetIdx() not in br and x.GetSymbol() != "H"]
                    if not subs:
                        continue
                    s = max(subs, key=lambda x: (x.GetFormalCharge() > 0, x.GetAtomicNum())).GetIdx()
                    dM = np.linalg.norm(X[s] - X[M])
                    dO = min(np.linalg.norm(X[s] - X[o]) for o in other)
                    lab = "exo" if dM < dO else "endo"
                    break
            if lab:
                break
        if lab is None:
            all_exoendo = False
            from rdkit.Chem import rdCIPLabeler
            m2 = Chem.Mol(mol)
            try:
                rdCIPLabeler.AssignCIPLabels(m2)
                lab = m2.GetAtomWithIdx(c).GetProp("_CIPCode") if m2.GetAtomWithIdx(c).HasProp("_CIPCode") else "?"
            except Exception:
                lab = "?"
        parts.append((c, lab))
    labs = [l for _, l in parts]
    if all_exoendo and len(set(labs)) == 1:
        return labs[0]
    if all_exoendo:
        return "_".join("C%d%s" % (c + 1, l) for c, l in parts)
    return "_".join("%s%d%s" % (mol.GetAtomWithIdx(c).GetSymbol(), c + 1, l) for c, l in parts)


def _mirror(m):
    x = Chem.Mol(m)
    for at in x.GetAtoms():
        t = at.GetChiralTag()
        if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            at.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            at.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(x)


def _from3d(mh, m_ref):
    """all stereocenters from the 3D structure; None if the constitution changed"""
    try:
        x = Chem.Mol(mh)
        Chem.AssignStereochemistryFrom3D(x)
        x = Chem.RemoveHs(x)
        if Chem.MolToSmiles(x, isomericSmiles=False) != Chem.MolToSmiles(m_ref, isomericSmiles=False):
            return None
        return x
    except Exception:
        return None


def isomers(smiles):
    """-> (list of dict(label, smiles), number of impossible combinations, number of unspecified centers)
    few unspecified centers: every combination is tried (impossible ones dropped);
    many (e.g. a whole cage drawn without stereo): the 3D structure is built many times with random
    arrangements and the distinct isomers that come out are kept (impossible ones never come out)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], 0, 0
    cen = unspecified(mol)
    if not cen or os.environ.get("STEREO", "on").lower() == "off":
        return [], 0, len(cen)
    nspec = sum(1 for i, t in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
                if t != "?")
    found, bad = {}, 0
    if len(cen) <= 4:
        for tags in itertools.product((Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW),
                                      repeat=len(cen)):
            m = Chem.Mol(mol)
            for i, t in zip(cen, tags):
                m.GetAtomWithIdx(i).SetChiralTag(t)
            Chem.AssignStereochemistry(m, cleanIt=True, force=True)
            mh = _embed(m)
            if mh is None:
                bad += 1
                continue
            full = _from3d(mh, mol) or m
            key = Chem.MolToSmiles(full)
            if key not in found:
                found[key] = (full, mh.GetConformer().GetPositions())
    else:
        rejected = set()
        mh0 = Chem.AddHs(mol)
        for seed in range(60):
            p = AllChem.ETKDGv3()
            p.randomSeed = 1000 + seed
            p.maxIterations = 30
            mh = Chem.Mol(mh0)
            if AllChem.EmbedMolecule(mh, p) != 0:
                continue
            full = _from3d(mh, mol)
            if full is None:
                continue
            key = Chem.MolToSmiles(full)
            if key in found or key in rejected:
                continue
            mh2 = _embed(full)                        # a distorted random structure can fake an isomer:
            if mh2 is None:                           # keep it only if it can be built with its own stereo
                rejected.add(key)
                continue
            found[key] = (full, mh2.GetConformer().GetPositions())
    # drop isomers with a trans-fused three-membered ring on a 4- to 7-membered ring (e.g. a trans-bromonium on
    # cyclohexane): they can be built in 3D but are far too strained to form
    def _trans3(m_, X_):
        X_ = np.array(X_)
        rings = [set(r) for r in m_.GetRingInfo().AtomRings()]
        for r3 in [r for r in rings if len(r) == 3]:
            a3 = sorted(r3)
            nrm = np.cross(X_[a3[1]] - X_[a3[0]], X_[a3[2]] - X_[a3[0]])
            for R in [r for r in rings if 4 <= len(r) <= 7 and len(r & r3) == 2]:
                bb, cc = sorted(r3 & R)
                nb = [x.GetIdx() for x in m_.GetAtomWithIdx(bb).GetNeighbors() if x.GetIdx() in R and x.GetIdx() not in r3]
                nc = [x.GetIdx() for x in m_.GetAtomWithIdx(cc).GetNeighbors() if x.GetIdx() in R and x.GetIdx() not in r3]
                if nb and nc and np.dot(nrm, X_[nb[0]] - X_[bb]) * np.dot(nrm, X_[nc[0]] - X_[cc]) < 0:
                    return True
        return False
    for key_ in [k_ for k_, (f_, X_) in found.items() if _trans3(f_, X_)]:
        del found[key_]
        if bad >= 0:
            bad += 1
    # with no stereo given at all, mirror images are the same chemistry: keep one of each pair
    if nspec == 0:
        kept = {}
        for k, v in found.items():
            if _mirror(v[0]) not in kept:
                kept[k] = v
        found = kept
    # label by the centers that carry a real substituent (or an onium bridge); bare cage CH atoms are implied
    ring_atoms = set(a for r in mol.GetRingInfo().AtomRings() for a in r if len(r) > 3)
    lab_c = [c for c in cen if any(x.GetSymbol() != "H" and (x.GetIdx() not in ring_atoms or x.GetFormalCharge() != 0
                                                             or x.GetSymbol() not in ("C",))
                                   for x in mol.GetAtomWithIdx(c).GetNeighbors())] or cen
    out = []
    for k, (full, X) in found.items():
        lab = label_isomer(full, X, lab_c)
        out.append(dict(label=lab, smiles=Chem.MolToSmiles(full, canonical=False)))
    used = {}
    for k, d in enumerate(out):
        base = d["label"] if re.match(r"^[A-Za-z0-9_]+$", d["label"]) else "iso%d" % (k + 1)
        if base not in ("exo", "endo") and not base.startswith("iso"):
            base = "iso%d_%s" % (k + 1, base)
        used[base] = used.get(base, 0) + 1
        d["label"] = base if used[base] == 1 else "%s%d" % (base, used[base])
    out.sort(key=lambda d: (d["label"] != "exo", d["label"]))
    if len(cen) > 4:
        bad = -1                                     # sampled: impossible ones are not counted
    return out[:MAXC], bad, len(cen)


# ---------------------------------------------------------------- summary of the analysed isomers
def _read(p):
    return open(p).read() if os.path.exists(p) else ""


def _energy(xyz):
    t = _read(xyz).split("\n")
    m = re.search(r"energy:\s*(-?\d+\.\d+)", t[1]) if len(t) > 1 else None
    return float(m.group(1)) if m else None


def summary(jobdir, name, kind="stereo", la_name="", write=True):
    rows = []
    for d in sorted(glob.glob(os.path.join(jobdir, "*")), key=lambda p: (os.path.basename(p) != "exo", os.path.basename(p))):
        if not os.path.isdir(d) or os.path.basename(d) in ("logs",):
            continue
        lab = os.path.basename(d)
        rep = glob.glob(os.path.join(d, "*_report.txt"))
        neu = os.path.join(d, "neutral")
        if os.path.isdir(neu):                       # neutral input: compare the neutral species + cation site 1
            rep = glob.glob(os.path.join(neu, "*_report.txt"))
            e = _energy(os.path.join(neu, "xtbopt.xyz"))
            cat = sorted(glob.glob(os.path.join(d, "cation", "cat01_*", "*_report.txt")))
            t = _read(cat[0]) if cat else ""
        else:
            e = _energy(os.path.join(d, "xtbopt.xyz"))
            t = _read(rep[0]) if rep else ""
        if not rep and not t:
            continue
        smi = re.search(r"SMILES\s*:\s*(\S+)", _read(rep[0]) if rep else t)
        pred = re.search(r"PRED\|\s*(.+)", t)
        nuat = re.search(r"NUAT\|\s*(.+)", t)
        fin = re.search(r"FINAL\|\s*(.+)", t)
        rows.append(dict(lab=lab, e=e, smiles=smi.group(1) if smi else "?", pred=pred.group(1).strip() if pred else "-",
                         nuat=nuat.group(1).strip() if nuat else "-", final=fin.group(1).strip() if fin else ""))
    if not rows:
        return ""
    es = [r["e"] for r in rows if r["e"] is not None]
    e0 = min(es) if es else None
    if kind == "lewis":
        L = ["=" * 100, (" %s ADDUCTS OF %s (Lewis acid bound to each Lewis-basic site)" % (la_name, name)) if la_name in
             ("BF3", "AlCl3", "TMS+") else (" %s ONIUM IONS OF %s (electrophile added across each C=C, both faces)" %
                                            (la_name, name)), "=" * 100]
    else:
        L = ["=" * 100, " STEREOISOMERS OF %s (stereochemistry was not specified in the SMILES)" % name, "=" * 100]
    imp = _read(os.path.join(jobdir, ".stereo_note")).strip()
    if imp:
        L.append(" " + imp)
    L.append(" %-12s %10s   %s" % ("site" if kind == "lewis" else "isomer", "rel. E", "result"))
    for r in rows:
        L.append(" %-12s %10s   %s" % (r["lab"], "-" if r["e"] is None or e0 is None else "%+.1f" % ((r["e"] - e0) * KCAL),
                                       r["pred"]))
        if r["nuat"] != "-":
            L.append(" %-12s %10s   nucleophile/partner: %s" % ("", "", r["nuat"]))
        if r["final"]:
            L.append(" %-12s %10s   cascade end: %s" % ("", "", r["final"]))
        L.append(" %-12s %10s   %s" % ("", "", r["smiles"]))
    preds = set(re.sub(r"\| dEa.*", "", r["pred"]) for r in rows)
    nus = set(re.sub(r"\| dEa.*", "", r["nuat"]) for r in rows)
    L.append("")
    if kind == "lewis":
        L.append(" rel. E = most stable adduct / onium ion (lowest = preferred; all have the same atoms).")
        L.append(" The preferred site is the likeliest pathway, but a less favoured site can still react if its step is faster.")
    elif len(preds) == 1 and len(nus) == 1:
        L.append(" The isomers AGREE: the predicted outcome does not depend on the unspecified stereochemistry.")
    else:
        L.append(" The isomers DIFFER: the outcome depends on the stereochemistry -- specify it if you know it.")
    if any("steric warning" in r["nuat"] for r in rows):
        L.append(" Note: a STERIC WARNING is raised for at least one isomer (bulky nucleophile at a crowded carbon).")
    if kind != "lewis":
        L.append(" rel. E = relative stability of the isomers (kcal/mol). It does NOT say which isomer forms: that depends on")
        L.append(" how it is made (e.g. NBS adds from the exo face). Energies of neutral inputs: the neutral molecule.")
    txt = "\n".join(L) + "\n"
    if write:
        open(os.path.join(jobdir, name + ("_lewis.txt" if kind == "lewis" else "_stereo.txt")), "w").write(txt)
    return txt


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "list":
        iso, bad, n = isomers(a[1])
        if iso:
            for d in iso:
                print("%s\t%s" % (d["label"], d["smiles"]))
        if len(a) > 2:                                # list SMILES IMPOSSIBLE_FILE: note for the summary
            open(a[2], "w").write(("%d unspecified stereocenter(s): %d buildable isomer(s) analysed" % (n, len(iso)) +
                                   (", %d combination(s) impossible in 3D and dropped" % bad if bad >= 0 else
                                    " (found by building the structure many times; impossible arrangements never form)"))
                                  if iso else "")
    elif a[0] == "summary":
        print(summary(a[1], a[2]))
    else:
        sys.exit(__doc__)
