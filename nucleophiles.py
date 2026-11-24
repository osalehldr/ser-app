#!/usr/bin/env python3
"""
nucleophiles.py -- nucleophiles (for cations) and radical partners (for neutral radicals) used by the
attack scans (nuattack.py) and the cascade (cascade.py).

Choose one by name or give any SMILES:
  run "SMILES" name solvent methanol            nucleophile for a cation / radical cation
  run "[CH2]C1CC1" name benzene "tin hydride"    partner for a neutral radical
  run "SMILES" name solvent "CC(C)(C)[OH:1]"     custom (atom map :1 marks the attacking atom;
                                                  for a hydride donor mark the H: "[H:1][Si](CC)(CC)CC")
"none" (default): no explicit partner -- the report gives the most electrophilic (cation) or most
reactive (radical) site, i.e. where any nucleophile / trap would most likely end up.
"oxidant" (radicals): radical-polar crossover -- the radical is oxidized to the cation and the cascade
continues as a cation ("oxidant+methanol" = then attacked by methanol).

  python3 nucleophiles.py list            show the presets
  python3 nucleophiles.py check SPEC      prints "label|kind", or "BAD <reason>"
"""
import os, re, sys, shutil, subprocess, tempfile
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog("rdApp.*")

# name: (SMILES, attacking atom, description, kind)
#   attacking atom: index of an atom of the SMILES, or ("H", i) = an H on atom i (hydride / H-atom donors)
#   kind: "nucleophile" (cations), "radical" (partners for neutral radicals), "none", "oxidant"
PRESETS = {
    "none": ("", None, "no explicit partner: most electrophilic / most reactive site", "none"),
    "water": ("O", 0, "O attacks", "nucleophile"),
    "methanol": ("CO", 1, "O attacks", "nucleophile"),
    "isopropanol": ("CC(C)O", 3, "O attacks (secondary alcohol)", "nucleophile"),
    "tert-butanol": ("CC(C)(C)O", 4, "O attacks (bulky alcohol)", "nucleophile"),
    "methoxide": ("C[O-]", 1, "alkoxide, O- attacks", "nucleophile"),
    "isopropoxide": ("CC(C)[O-]", 3, "alkoxide, O- attacks", "nucleophile"),
    "tert-butoxide": ("CC(C)(C)[O-]", 4, "bulky alkoxide, O- attacks", "nucleophile"),
    "hydride (NaBH4)": ("[BH4-]", ("H", 0), "BH4- transfers H-", "nucleophile"),
    "hydride (LiAlH4)": ("[AlH4-]", ("H", 0), "AlH4- transfers H-", "nucleophile"),
    # more nucleophiles (by name on the command line; not in the app menu)
    "acetic acid": ("CC(=O)O", 2, "carbonyl O attacks", "nucleophile"),
    "acetate": ("CC(=O)[O-]", 3, "anion, O attacks", "nucleophile"),
    "bromide": ("[Br-]", 0, "anion", "nucleophile"),
    "chloride": ("[Cl-]", 0, "anion", "nucleophile"),
    "iodide": ("[I-]", 0, "anion", "nucleophile"),
    "acetonitrile": ("CC#N", 2, "N attacks (Ritter-type)", "nucleophile"),
    "anisole": ("COc1ccccc1", 5, "para carbon attacks (Friedel-Crafts type)", "nucleophile"),
    "trimethoxybenzene": ("COc1cc(OC)cc(OC)c1", 3, "ring CH attacks (Friedel-Crafts type)", "nucleophile"),
    "triethylsilane": ("[SiH](CC)(CC)CC", ("H", 0), "Et3SiH transfers H- (ionic hydrogenation)", "nucleophile"),
    # partners for neutral radicals
    "tin hydride": ("C[SnH](C)C", ("H", 1), "H-atom donor (Bu3SnH, modelled as Me3SnH)", "radical"),
    "thiol": ("CS", ("H", 1), "H-atom donor (thiol, modelled as MeSH)", "radical"),
    "iodine": ("II", 0, "I2: iodine-atom transfer", "radical"),
    "acrylate": ("C=CC(=O)OC", 0, "electron-poor alkene (methyl acrylate): Giese addition", "radical"),
    "oxidant": ("", None, "radical-polar crossover: the radical is oxidized to the cation", "oxidant"),
}
APP_NUCLEOPHILES = ["none", "water", "methanol", "isopropanol", "tert-butanol", "methoxide", "isopropoxide",
                    "tert-butoxide", "hydride (NaBH4)", "hydride (LiAlH4)"]
APP_PARTNERS = ["none", "tin hydride", "thiol", "iodine", "acrylate", "oxidant"]
ALIASES = {"h2o": "water", "meoh": "methanol", "ipa": "isopropanol", "iproh": "isopropanol", "2-propanol": "isopropanol",
           "tbuoh": "tert-butanol", "t-buoh": "tert-butanol", "t-butanol": "tert-butanol", "meo-": "methoxide",
           "naome": "methoxide", "ipro-": "isopropoxide", "tbuo-": "tert-butoxide", "t-buo-": "tert-butoxide",
           "tbuok": "tert-butoxide", "nabh4": "hydride (NaBH4)", "bh4-": "hydride (NaBH4)", "hydride": "hydride (NaBH4)",
           "h-": "hydride (NaBH4)", "borohydride": "hydride (NaBH4)", "lialh4": "hydride (LiAlH4)", "lah": "hydride (LiAlH4)",
           "alh4-": "hydride (LiAlH4)", "acoh": "acetic acid", "aceticacid": "acetic acid", "aco-": "acetate",
           "oac": "acetate", "br": "bromide", "br-": "bromide", "cl": "chloride", "cl-": "chloride", "i-": "iodide",
           "mecn": "acetonitrile", "tmb": "trimethoxybenzene", "1,3,5-trimethoxybenzene": "trimethoxybenzene",
           "phome": "anisole", "et3sih": "triethylsilane", "bu3snh": "tin hydride", "tin_hydride": "tin hydride",
           "tinhydride": "tin hydride", "rsh": "thiol", "i2": "iodine", "methyl acrylate": "acrylate",
           "giese": "acrylate", "crossover": "oxidant", "no": "none", "no nucleophile": "none", "no partner": "none"}


def _parse(smi):
    """SMILES -> mol; explicit [H] atoms are kept (a hydride donor's transferring H can be the attacking atom)"""
    p = Chem.SmilesParserParams()
    p.removeHs = False
    return Chem.MolFromSmiles(smi, p)


def _rcov(s):
    return Chem.GetPeriodicTable().GetRcovalent(Chem.GetPeriodicTable().GetAtomicNumber(s))


def _fukui_minus(mol, xtb):
    """xtb f(-) per heavy atom of the neutral/charged nucleophile (None if xtb is not available)"""
    if not xtb or not (os.path.exists(xtb) or shutil.which(xtb)):
        return None
    mh = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mh, randomSeed=7) != 0:
        return None
    d = tempfile.mkdtemp(prefix="nu_")
    try:
        X = mh.GetConformer().GetPositions()
        with open(os.path.join(d, "n.xyz"), "w") as f:
            f.write("%d\n\n" % mh.GetNumAtoms())
            for a, x in zip(mh.GetAtoms(), X):
                f.write("%s %f %f %f\n" % (a.GetSymbol(), x[0], x[1], x[2]))
        out = subprocess.run([xtb, "n.xyz", "--vfukui", "--chrg", str(Chem.GetFormalCharge(mol))], cwd=d,
                             capture_output=True, text=True).stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)
    L = out.splitlines()
    fk = {}
    for k, l in enumerate(L):
        if "f(+)" in l and "f(-)" in l:
            for m in L[k + 1:]:
                t = m.split()
                if len(t) < 4 or not re.match(r"\d+", t[0]):
                    break
                fk[int(re.match(r"\d+", t[0]).group()) - 1] = float(t[2])
            break
    return fk or None


def split_spec(spec):
    """'oxidant+methanol' -> ('oxidant', 'methanol'); anything else -> (spec, None)"""
    s = (spec or "none").strip()
    if "+" in s and s.split("+", 1)[0].strip().lower() in ("oxidant", "ox", "crossover"):
        return "oxidant", s.split("+", 1)[1].strip() or "none"
    return s, None


def resolve(spec, xtb=None):
    """-> dict(name, smiles, mol, attack, charge, desc, kind[, then]) or raises ValueError"""
    s, then = split_spec(spec)
    k0 = s.lower().replace("_", " ")
    key = ALIASES.get(k0, ALIASES.get(s.lower(), k0))
    key = {p.lower(): p for p in PRESETS}.get(key.lower(), key)
    if key in PRESETS:
        smi, at, desc, kind = PRESETS[key]
        mol = _parse(smi) if smi else None
        out = dict(name=key, smiles=smi, mol=mol, attack=at, charge=Chem.GetFormalCharge(mol) if mol else 0,
                   desc=desc, kind=kind)
        if kind == "oxidant":
            out["then"] = resolve(then or "none", xtb)
            if out["then"]["kind"] not in ("nucleophile", "none"):
                raise ValueError("after 'oxidant+' give a nucleophile for the cation (e.g. oxidant+methanol)")
            if then:
                out["name"] = "oxidant+" + out["then"]["name"]
        return out
    mol = _parse(s)
    if mol is None:
        raise ValueError("'%s' is neither a preset (%s) nor a valid SMILES" % (s, ", ".join(PRESETS)))
    if len(Chem.GetMolFrags(mol)) > 1:
        raise ValueError("the nucleophile SMILES must be one molecule or ion (no '.')")
    if mol.GetNumHeavyAtoms() > 30:
        raise ValueError("nucleophile too large (max 30 heavy atoms)")
    q = Chem.GetFormalCharge(mol)
    if q > 0:
        raise ValueError("a nucleophile cannot be positively charged")
    mapped = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 1]
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    if not mapped:                     # explicit H atoms only matter when marked: drop the others
        mol = Chem.RemoveHs(mol)
    if mapped:
        at, how = mapped[0], "attacking atom marked with :1"
    else:
        neg = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() < 0]
        if neg:
            at, how = neg[0], "the negatively charged atom attacks"
        else:
            fk = _fukui_minus(mol, xtb)
            heavy = [a.GetIdx() for a in mol.GetAtoms()]
            if fk:
                at, how = max(heavy, key=lambda i: fk.get(i, 0.0)), "attacking atom = largest Fukui f(-)"
            else:
                het = [i for i in heavy if mol.GetAtomWithIdx(i).GetSymbol() in ("N", "S", "O", "P")]
                at, how = (het or heavy)[0], "attacking atom = first heteroatom (xtb not available for Fukui)"
    clean = Chem.MolToSmiles(mol)
    return dict(name="custom " + clean, smiles=Chem.MolToSmiles(mol, canonical=False), mol=mol, attack=at, charge=q,
                desc="%s%d attacks (%s)" % (mol.GetAtomWithIdx(at).GetSymbol(), at + 1, how),
                kind="radical" if Descriptors.NumRadicalElectrons(mol) else "nucleophile")


def geometry(nu):
    """3D structure, attacking atom first: adds sym, X (local coordinates), bonds, approach ('pi' or 'lp')"""
    mol = nu["mol"]
    mh = Chem.AddHs(mol)
    if mh.GetNumAtoms() == 1:
        X = np.zeros((1, 3))
    else:
        if AllChem.EmbedMolecule(mh, randomSeed=11) != 0:
            AllChem.EmbedMolecule(mh, randomSeed=11, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mh, maxIters=500)
        except Exception:
            pass
        X = mh.GetConformer().GetPositions()
    a = nu["attack"]
    if isinstance(a, tuple):                     # an H on atom a[1] is transferred (hydride / H-atom donors)
        a = [x.GetIdx() for x in mh.GetAtomWithIdx(a[1]).GetNeighbors() if x.GetSymbol() == "H"][0]
    order = [a] + [i for i in range(mh.GetNumAtoms()) if i != a]
    new = {o: k for k, o in enumerate(order)}
    nu["sym"] = [mh.GetAtomWithIdx(i).GetSymbol() for i in order]
    nu["X"] = X[order] - X[a]
    nu["bonds"] = [(new[b.GetBeginAtomIdx()], new[b.GetEndAtomIdx()]) for b in mh.GetBonds()]
    at = mh.GetAtomWithIdx(a)
    pi = at.GetSymbol() == "C" and (at.GetIsAromatic() or at.GetHybridization() == Chem.HybridizationType.SP2)
    nu["approach"] = "pi" if pi else "lp"
    nbr = [new[x.GetIdx()] for x in at.GetNeighbors()]
    if len(nu["X"]) == 1:
        away = None
    elif pi and len(nbr) >= 2:
        away = np.cross(nu["X"][nbr[0]], nu["X"][nbr[1]])        # normal of the pi plane at the attacking carbon
    elif nbr:
        # substituent bisector: the lone pairs point the other way (towards the carbon being attacked)
        away = sum(nu["X"][j] / np.linalg.norm(nu["X"][j]) for j in nbr)
    else:
        away = nu["X"][1:].mean(axis=0)
    nu["away"] = None if away is None or np.linalg.norm(away) < 1e-6 else away / np.linalg.norm(away)
    nu["bond_to_C"] = 1.05 * (_rcov("C") + _rcov(nu["sym"][0]))
    return nu


def _rot(a, b):
    """rotation matrix taking unit vector a onto unit vector b"""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def place(nu, XT, u, X0, skip, dists=(3.1, 3.4, 3.8, 4.2)):
    """nucleophile coordinates with the attacking atom at distance d from XT along u, body pointing away;
    tries several spins about u; None if it does not fit (closest contact < 1.9 A)"""
    Y = nu["X"]
    R0 = _rot(nu["away"], u) if nu["away"] is not None else np.eye(3)
    Y0 = Y @ R0.T
    best = None
    for d in dists:
        for k in range(6):
            ang = np.radians(60 * k)
            K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            Rk = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            P = (Y0 @ Rk.T) + XT + d * u
            others = [X0[j] for j in range(len(X0)) if j not in skip]
            clash = min(np.linalg.norm(p - q) for p in P for q in others) if others else 9.0
            if best is None or clash > best[0]:
                best = (clash, P)
        if best[0] > 1.9:
            return best[1]
    return None


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "list":
        for k, (sm, i, d, kd) in PRESETS.items():
            print("%-18s %-12s %-20s %s" % (k, kd, sm or "-", d))
        sys.exit()
    if a[0] == "check":
        try:
            nu = resolve(a[1] if len(a) > 1 else "none", a[2] if len(a) > 2 else os.environ.get("XTB"))
            print("%s|%s" % (nu["name"], nu["kind"]))
        except ValueError as e:
            print("BAD", e)
