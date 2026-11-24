"""
pythonldr.py -- general molecular structure analyzer.
Input: a SMILES string. Output: everything RDKit can tell you about it.

Usage (Windows Command Prompt):
    py C:\\chem\\scripts\\pythonldr.py "SMILES"
    py C:\\chem\\scripts\\pythonldr.py "SMILES" --xyz out.xyz
"""

import sys
import argparse

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
    from rdkit import RDLogger
except ImportError:
    sys.exit("RDKit missing.  Fix:  pip install rdkit")

RDLogger.DisableLog("rdApp.*")

HYB = {Chem.HybridizationType.SP: "sp", Chem.HybridizationType.SP2: "sp2",
       Chem.HybridizationType.SP3: "sp3", Chem.HybridizationType.SP3D: "sp3d",
       Chem.HybridizationType.SP3D2: "sp3d2",
       Chem.HybridizationType.S: "s", Chem.HybridizationType.UNSPECIFIED: "-"}

BT = {Chem.BondType.SINGLE: "-", Chem.BondType.DOUBLE: "=",
      Chem.BondType.TRIPLE: "#", Chem.BondType.AROMATIC: "~"}


def line(c="-", n=72):
    print(c * n)


def head(t):
    print()
    line("=")
    print(t)
    line("=")


def classify(a):
    """Return a reactive-center label for an atom, or ''."""
    s, q, r = a.GetSymbol(), a.GetFormalCharge(), a.GetNumRadicalElectrons()
    deg, nH = a.GetDegree(), a.GetTotalNumHs()
    tags = []
    if r == 1:
        tags.append("RADICAL")
    elif r == 2:
        tags.append("CARBENE/DIRADICAL" if s == "C" else "DIRADICAL")
    elif r > 2:
        tags.append("%d UNPAIRED e-" % r)
    if q > 0:
        if s == "C" and deg + nH == 3 and r == 0:
            tags.append("CARBOCATION")
        else:
            tags.append("CATION %+d" % q)
    elif q < 0:
        if s == "C" and r == 0:
            tags.append("CARBANION")
        else:
            tags.append("ANION %+d" % q)
    if s == "C" and q == 0 and r == 0 and deg == 2 and nH == 0 \
            and all(b.GetBondType() == Chem.BondType.SINGLE for b in a.GetBonds()):
        tags.append("possible CARBENE (written as [C]?)")
    return ", ".join(tags)


def analyze(smi, xyz_out=None, png_out=None):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        sys.exit("SMILES failed to parse. Check brackets and ring-closure digits.")

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    try:
        from rdkit.Chem import rdCIPLabeler
        rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        pass

    # ---------------------------------------------------------- overview
    head("MOLECULE")
    print("SMILES in   :", smi)
    print("canonical   :", Chem.MolToSmiles(mol))
    print("formula     :", rdMolDescriptors.CalcMolFormula(mol))
    print("MW          : %.3f" % Descriptors.MolWt(mol))
    print("heavy atoms :", mol.GetNumAtoms())
    print("total atoms :", Chem.AddHs(mol).GetNumAtoms())
    print("net charge  : %+d" % Chem.GetFormalCharge(mol))
    unp = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    print("unpaired e- :", unp, " -> multiplicity", unp + 1,
          "(charge/mult for QM:  %d %d)" % (Chem.GetFormalCharge(mol), unp + 1))
    print("rings       :", rdMolDescriptors.CalcNumRings(mol))
    print("rotatable   :", rdMolDescriptors.CalcNumRotatableBonds(mol))

    # ----------------------------------------------------------- atoms
    head("ATOMS  (1-based indices -- use these in Gaussian/ORCA)")
    print("%4s %4s %4s %4s %4s %6s %4s %5s  %-18s %s"
          % ("idx", "sym", "chg", "rad", "nH", "hyb", "arom", "ring",
             "neighbours", "note"))
    line()
    for a in mol.GetAtoms():
        i = a.GetIdx() + 1
        nb = sorted(n.GetIdx() + 1 for n in a.GetNeighbors())
        cip = a.GetPropsAsDict().get("_CIPCode", "")
        note = classify(a)
        if cip:
            note = ("(%s) " % cip) + note
        print("%4d %4s %+4d %4d %4d %6s %4s %5s  %-18s %s"
              % (i, a.GetSymbol(), a.GetFormalCharge(),
                 a.GetNumRadicalElectrons(), a.GetTotalNumHs(),
                 HYB.get(a.GetHybridization(), "?"),
                 "yes" if a.GetIsAromatic() else "-",
                 "yes" if a.IsInRing() else "-", str(nb), note))

    # --------------------------------------------------- reactive centers
    head("REACTIVE CENTERS")
    found = False
    for a in mol.GetAtoms():
        lab = classify(a)
        if lab:
            found = True
            nb = sorted(n.GetIdx() + 1 for n in a.GetNeighbors())
            print("  atom %d (%s): %s   neighbours %s"
                  % (a.GetIdx() + 1, a.GetSymbol(), lab, nb))
    if not found:
        print("  none -- closed-shell neutral molecule")

    # ----------------------------------------------------------- bonds
    head("MULTIPLE BONDS")
    any_mb = False
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.SINGLE:
            continue
        any_mb = True
        i, j = b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1
        kind = {Chem.BondType.DOUBLE: "double", Chem.BondType.TRIPLE: "triple",
                Chem.BondType.AROMATIC: "aromatic"}.get(b.GetBondType(), "other")
        st = str(b.GetStereo()).replace("STEREO", "")
        cip = b.GetPropsAsDict().get("_CIPCode", "")
        extra = ""
        if cip:
            extra = "  CIP: " + cip
        elif st not in ("NONE", ""):
            extra = "  stereo: " + st
        print("  C%d%sC%d   %-8s %s%s"
              % (i, BT.get(b.GetBondType(), "?"), j, kind,
                 "in ring" if b.IsInRing() else "acyclic", extra))
    if not any_mb:
        print("  none")

    # ----------------------------------------------------------- rings
    head("RINGS")
    ri = mol.GetRingInfo()
    rings = [tuple(x + 1 for x in r) for r in ri.AtomRings()]
    if not rings:
        print("  acyclic")
    for r in sorted(rings, key=len):
        size = len(r)
        arom = all(mol.GetAtomWithIdx(i - 1).GetIsAromatic() for i in r)
        tag = []
        if arom:
            tag.append("aromatic")
        if size == 3:
            tag.append("STRAINED (cyclopropane-type, ~27 kcal/mol)")
        elif size == 4:
            tag.append("STRAINED (cyclobutane-type, ~26 kcal/mol)")
        print("  %d-membered %s   %s" % (size, str(r), ", ".join(tag)))
    try:
        print("  spiro atoms      :", rdMolDescriptors.CalcNumSpiroAtoms(mol))
        print("  bridgehead atoms :", rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    except Exception:
        pass

    # ------------------------------------------------------ stereochem
    head("STEREOCHEMISTRY")
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                        useLegacyImplementation=False)
    if centers:
        for idx, lab in centers:
            print("  atom %d : %s" % (idx + 1, lab if lab != "?" else "UNSPECIFIED"))
    else:
        print("  no tetrahedral stereocenters")
    try:
        si = Chem.FindPotentialStereo(mol)
        und = [e for e in si if str(e.specified) == "Unspecified"]
        if und:
            print("  %d stereo element(s) UNSPECIFIED -- the SMILES does not"
                  % len(und))
            print("  define them. Fix before any 3D work.")
    except Exception:
        pass

    # ------------------------------------------- strained-ring reactivity
    strained = [r for r in rings if len(r) in (3, 4)]
    reactive = [a.GetIdx() + 1 for a in mol.GetAtoms()
                if a.GetNumRadicalElectrons() or a.GetFormalCharge()]
    if strained and reactive:
        head("STRAINED RING NEXT TO A REACTIVE CENTER")
        for ring in strained:
            for ri_ in reactive:
                ra = mol.GetAtomWithIdx(ri_ - 1)
                rn = set(n.GetIdx() + 1 for n in ra.GetNeighbors())
                if ri_ in ring:
                    print("  atom %d is INSIDE the %d-ring %s"
                          % (ri_, len(ring), str(ring)))
                    continue
                for c1 in sorted(rn & set(ring)):
                    others = sorted(x for x in ring if x != c1)
                    print("  reactive atom %d -- attached to %d-ring %s via atom %d"
                          % (ri_, len(ring), str(ring), c1))
                    print("    candidate scission bonds:")
                    for t in others:
                        if mol.GetBondBetweenAtoms(c1 - 1, t - 1):
                            print("      C%d-C%d   ->  fragment lands on atom %d"
                                  % (c1, t, t))
                    if len(others) == 2:
                        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
                        if ranks[others[0] - 1] == ranks[others[1] - 1]:
                            print("    *** atoms %d and %d are EQUIVALENT by"
                                  % (others[0], others[1]))
                            print("        symmetry -- the two bonds cannot be")
                            print("        distinguished. Strictly 1:1. ***")

    # ----------------------------------------------------------- output
    if png_out:
        pic = Chem.Mol(mol)
        for a in pic.GetAtoms():
            a.SetProp("atomNote", str(a.GetIdx() + 1))
        AllChem.Compute2DCoords(pic)
        Draw.MolToFile(pic, png_out, size=(900, 900))
        print()
        print("wrote", png_out, " (numbered structure -- check it)")

    if xyz_out:
        molH = Chem.AddHs(mol)
        ref = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        p = AllChem.ETKDGv3()
        p.randomSeed = 0xF00D
        p.useSmallRingTorsions = True
        p.enforceChirality = True
        p.useRandomCoords = True
        p.maxIterations = 200
        p.numThreads = 0
        cids = list(AllChem.EmbedMultipleConfs(molH, numConfs=30, params=p))
        if not cids:
            print("direct embedding failed -- building from neutral CH analog")
            rw = Chem.RWMol(mol)
            cat = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "C" and (a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() > 0)]
            for i in cat:
                a = rw.GetAtomWithIdx(i)
                a.SetFormalCharge(0); a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
                a.SetNoImplicit(True)
            Chem.SanitizeMol(rw)
            neutH = Chem.AddHs(rw)
            cids = list(AllChem.EmbedMultipleConfs(neutH, numConfs=30, params=p))
            if cids:
                if AllChem.MMFFHasAllMoleculeParams(neutH): AllChem.MMFFOptimizeMoleculeConfs(neutH, maxIters=2000)
                rwH = Chem.RWMol(neutH)
                drop = []
                for i in cat:
                    hs = [n.GetIdx() for n in rwH.GetAtomWithIdx(i).GetNeighbors() if n.GetSymbol() == "H"]
                    drop.append(hs[0])
                    a = rwH.GetAtomWithIdx(i)
                    a.SetFormalCharge(mol.GetAtomWithIdx(i).GetFormalCharge()); a.SetNumRadicalElectrons(mol.GetAtomWithIdx(i).GetNumRadicalElectrons())
                    a.SetNumExplicitHs(0)
                    a.SetNoImplicit(True)
                for h in sorted(drop, reverse=True):
                    rwH.RemoveAtom(h)
                molH = rwH.GetMol()
                Chem.SanitizeMol(molH)
                print("built cation geometry from neutral analog (%d conformers)" % len(cids))
        if not cids:
            print("3D embedding FAILED with stereochemistry enforced")
            return
        if AllChem.MMFFHasAllMoleculeParams(molH):
            res = AllChem.MMFFOptimizeMoleculeConfs(molH, maxIters=5000)
            ff = "MMFF"
        else:
            res = AllChem.UFFOptimizeMoleculeConfs(molH, maxIters=5000) if AllChem.UFFHasAllMoleculeParams(molH) else [(0, 0.0)] * len(cids)
            ff = "UFF"
        good = []
        for i, cid in enumerate(cids):
            m = Chem.Mol(molH, False, cid)
            Chem.AssignStereochemistryFrom3D(m)
            got = dict(Chem.FindMolChiralCenters(m, includeUnassigned=True))
            if all(got.get(k) == v for k, v in ref.items() if v != "?"):
                good.append(i)
        print("%d of %d conformers kept correct stereochemistry" % (len(good), len(cids)))
        if not good:
            print("No conformer matches the SMILES stereochemistry -- nothing written")
            return
        best = min(good, key=lambda i: res[i][1])
        Chem.MolToXYZFile(molH, xyz_out, confId=cids[best])
        print("wrote %s  (%d atoms, %s %.2f kcal/mol)"
              % (xyz_out, molH.GetNumAtoms(), ff, res[best][1]))
        print("heavy-atom indices 1-%d are unchanged; H appended after"
              % mol.GetNumAtoms())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="General SMILES analyzer.")
    ap.add_argument("smiles")
    ap.add_argument("--xyz", default=None, help="also write a 3D geometry")
    ap.add_argument("--png", default=None, help="also write a numbered picture")
    a = ap.parse_args()
    analyze(a.smiles, a.xyz, a.png)
