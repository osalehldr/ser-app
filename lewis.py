#!/usr/bin/env python3
"""
lewis.py -- explicit Lewis acid activation (instead of modelling the acid as H+).

The Lewis acid is bound to every Lewis-basic site (O of ethers, epoxides, C=O, OH; N of amines, imines,
nitriles, pyridines) and each adduct is analysed like any other species. Adducts of BF3 and AlCl3 are
zwitterions (O+ -> B-/Al-, net charge unchanged); TMS+ (TMSOTf, TMSI) gives a silyloxonium ion (+1).

  python3 lewis.py sites SMILES LA        -> one line per site:  label<TAB>adduct SMILES
  python3 lewis.py summary JOBDIR NAME LA -> JOBDIR/NAME_lewis.txt comparing the sites
LA: bf3 (BF3, BF3.OEt2), alcl3 (AlCl3), tms (TMS+: TMSOTf, TMSI, TMSCl).  LA_MAX (4) = most sites analysed.
"""
import os, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LAS = {"bf3": dict(name="BF3", frag="F[B-](F)F", acc=1, dq=0),
       "alcl3": dict(name="AlCl3", frag="Cl[Al-](Cl)Cl", acc=1, dq=0),
       "tms": dict(name="TMS+", frag="C[Si](C)C", acc=1, dq=1)}
# electrophiles: added across each C=C as a three-membered onium ion (bromonium, iodonium, chloronium,
# seleniranium); both faces come out as stereoisomers (exo / endo) and each is analysed
ELECTROPHILES = {"br": dict(name="Br+", atom="Br"), "i": dict(name="I+", atom="I"), "cl": dict(name="Cl+", atom="Cl"),
                 "phse": dict(name="PhSe+", atom="Se", phenyl=True)}
ALIASES = {"nbs": "br", "br2": "br", "br+": "br", "br": "br", "nis": "i", "i2": "i", "i+": "i", "iodine": "i",
           "ncs": "cl", "cl2": "cl", "cl+": "cl", "phsecl": "phse", "phse+": "phse", "phse": "phse",
           "bf3": "bf3", "bf3.oet2": "bf3", "bf3oet2": "bf3", "bf3-oet2": "bf3", "alcl3": "alcl3",
           "tms": "tms", "tms+": "tms", "tmsotf": "tms", "tmsi": "tms", "tmscl": "tms", "h+": "h", "h": "h",
           "proton": "h", "none": "h", "": "h"}
MAXS = int(os.environ.get("LA_MAX", "4"))


def norm(la):
    k = ALIASES.get((la or "").strip().lower())
    if k is None:
        raise ValueError("unknown Lewis acid '%s' (choose: BF3, AlCl3, TMS+ or H+)" % la)
    return k


def _site_ok(a):
    if a.GetFormalCharge() != 0:
        return False
    if a.GetSymbol() == "O":
        return True
    if a.GetSymbol() != "N":
        return False
    if a.GetIsAromatic():                         # pyridine-type only (no H, two ring bonds)
        return a.GetTotalNumHs() == 0 and a.GetDegree() == 2
    for nb in a.GetNeighbors():                   # no amide / sulfonamide N, no nitro N
        if nb.GetSymbol() in ("C", "S") and any(b.GetBondType() == Chem.BondType.DOUBLE and
                                                b.GetOtherAtom(nb).GetSymbol() == "O" for b in nb.GetBonds()):
            return False
    return not any(nb.GetSymbol() == "O" for nb in a.GetNeighbors())


def onium_sites(mol, k):
    """electrophile E+ across each non-aromatic C=C -> three-membered onium ion"""
    E = ELECTROPHILES[k]
    out, seen = [], set()
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE or b.GetIsAromatic():
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetSymbol() != "C" or mol.GetAtomWithIdx(j).GetSymbol() != "C":
            continue
        rw = Chem.RWMol(mol)
        Chem.Kekulize(rw, clearAromaticFlags=False)
        rw.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.SINGLE)
        e = rw.AddAtom(Chem.Atom(E["atom"]))
        rw.GetAtomWithIdx(e).SetFormalCharge(1)
        rw.AddBond(i, e, Chem.BondType.SINGLE)
        rw.AddBond(j, e, Chem.BondType.SINGLE)
        if E.get("phenyl"):
            ph = Chem.MolFromSmiles("c1ccccc1")
            base = rw.GetNumAtoms()
            for a in ph.GetAtoms():
                rw.AddAtom(Chem.Atom(6))
            for bb in ph.GetBonds():
                rw.AddBond(base + bb.GetBeginAtomIdx(), base + bb.GetEndAtomIdx(), bb.GetBondType())
            for a in range(6):
                rw.GetAtomWithIdx(base + a).SetIsAromatic(True)
            for bb in range(6):
                pass
            rw.AddBond(e, base, Chem.BondType.SINGLE)
        try:
            m2 = rw.GetMol()
            Chem.SanitizeMol(m2)
        except Exception:
            continue
        key = Chem.MolToSmiles(m2)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(label="%s_C%dC%d" % (E["name"].replace("+", ""), i + 1, j + 1),
                        smiles=Chem.MolToSmiles(m2, canonical=False), site=i))
    return out[:MAXS]


def sites(smiles, la):
    """-> list of dict(label, smiles): the adduct at each distinct Lewis-basic site (Lewis acids), or the onium
    ion at each C=C (electrophiles)"""
    k = norm(la)
    if k == "h":
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    if k in ELECTROPHILES:
        return onium_sites(mol, k)
    L = LAS[k]
    frag = Chem.MolFromSmiles(L["frag"], sanitize=False)
    out, seen = [], set()
    for a in mol.GetAtoms():
        if not _site_ok(a):
            continue
        rw = Chem.RWMol(Chem.CombineMols(mol, frag))
        acc = mol.GetNumAtoms() + L["acc"]
        rw.AddBond(a.GetIdx(), acc, Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(a.GetIdx()).SetFormalCharge(1)
        try:
            m2 = rw.GetMol()
            Chem.SanitizeMol(m2)
        except Exception:
            continue
        key = Chem.MolToSmiles(m2)
        if key in seen:                           # symmetry-equivalent sites
            continue
        seen.add(key)
        out.append(dict(label="%s_%s%d" % (L["name"].replace("+", "").replace("3", "3"), a.GetSymbol(), a.GetIdx() + 1),
                        smiles=Chem.MolToSmiles(m2, canonical=False), site=a.GetIdx()))
    return out[:MAXS]


def label(la):
    k = norm(la)
    return (ELECTROPHILES.get(k) or LAS.get(k) or dict(name=la))["name"]


def summary(jobdir, name, la, write=False):
    import stereo
    return stereo.summary(jobdir, name, kind="lewis", la_name=label(la), write=write)


def annotate(jobdir, la):
    """one line in each adduct's report: how it compares with the most stable adduct / onium ion"""
    import re, glob
    rows = []
    for d in sorted(glob.glob(os.path.join(jobdir, "*"))):
        rep = glob.glob(os.path.join(d, "*_report.txt"))
        xyz = os.path.join(d, "xtbopt.xyz")
        if not rep or not os.path.exists(xyz):
            continue
        m = re.search(r"energy:\s*(-?\d+\.\d+)", open(xyz).read().split("\n")[1])
        if m:
            rows.append((os.path.basename(d), float(m.group(1)), rep[0]))
    if len(rows) < 2:
        return
    best = min(rows, key=lambda r: r[1])
    what = "adducts" if norm(la) in LAS else "onium ions (sites and faces)"
    for lab, e, rep in rows:
        rel = (e - best[1]) * 627.509
        line = ("comparison      : %s, %d %s of %s compared: %s\n" % (
            "the most stable one" if lab == best[0] else "%+.1f kcal/mol less stable than %s" % (rel, best[0]),
            len(rows), what, label(la), ", ".join(r[0] for r in rows)))
        t = open(rep).read()
        t = re.sub(r"(?m)^comparison      : .*\n", "", t)
        t = re.sub(r"(?m)^(solvent\s*:.*\n)", lambda m_: m_.group(1) + line, t, count=1)
        open(rep, "w").write(t)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "sites":
        try:
            for d in sites(a[1], a[2]):
                print("%s\t%s" % (d["label"], d["smiles"]))
        except ValueError as e:
            sys.exit(str(e))
    elif len(a) >= 4 and a[0] == "summary":
        print(summary(a[1], a[2], a[3]))
    elif len(a) >= 3 and a[0] == "annotate":
        annotate(a[1], a[2])
    else:
        sys.exit(__doc__)
