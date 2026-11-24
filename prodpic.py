#!/usr/bin/env python3
"""prodpic.py -- product structure from a relaxed end point (xyz), and a picture of it.

product(xyz, charge, keep): bonds and charges assigned from the geometry (RDKit DetermineBonds), the
fragment containing atom `keep` (0-based) kept; an O/N/S onium formed from a neutral nucleophile loses
its H+ (as in work-up). Stereochemistry is read from the 3D structure.
-> dict(smiles, smiles_onium, lost_h) or None
"""
import os
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def product(xyz, charge, keep):
    from rdkit.Chem import rdDetermineBonds
    pm = None
    try:
        m = Chem.MolFromXYZFile(xyz)
        rdDetermineBonds.DetermineBonds(m, charge=int(charge))
        frag = next((f for f in Chem.GetMolFrags(m) if keep in f), None)
        if frag is not None:
            rw = Chem.RWMol(m)
            for i in sorted(set(range(m.GetNumAtoms())) - set(frag), reverse=True):
                rw.RemoveAtom(i)
            pm = rw.GetMol()
    except Exception:
        pm = None
    if pm is None:          # e.g. radical + Bu3SnH: product + tin radical (odd electrons): bonds of the product alone
        try:
            m = Chem.MolFromXYZFile(xyz)
            rdDetermineBonds.DetermineConnectivity(m)
            frag = next((f for f in Chem.GetMolFrags(m) if keep in f), None)
            rw = Chem.RWMol(m)
            for i in sorted(set(range(m.GetNumAtoms())) - set(frag), reverse=True):
                rw.RemoveAtom(i)
            for q in (0, 1, -1):
                try:
                    cand = Chem.Mol(rw.GetMol())
                    rdDetermineBonds.DetermineBondOrders(cand, charge=q)
                    Chem.SanitizeMol(cand)
                    pm = cand
                    break
                except Exception:
                    continue
        except Exception:
            pm = None
    if pm is None:
        return None
    try:
        Chem.SanitizeMol(pm)
        Chem.AssignStereochemistryFrom3D(pm)
        onium = Chem.MolToSmiles(Chem.RemoveHs(pm))
    except Exception:
        return None
    lost = False
    for a in pm.GetAtoms():                   # onium from a neutral nucleophile: lose H+
        if a.GetSymbol() in ("O", "N", "S") and a.GetFormalCharge() == 1:
            hs = [x.GetIdx() for x in a.GetNeighbors() if x.GetSymbol() == "H"]
            if hs:
                rw2 = Chem.RWMol(pm)
                rw2.GetAtomWithIdx(a.GetIdx()).SetFormalCharge(0)
                rw2.RemoveAtom(hs[0])
                try:
                    p2 = rw2.GetMol()
                    Chem.SanitizeMol(p2)
                    Chem.AssignStereochemistryFrom3D(p2)
                    pm, lost = p2, True
                except Exception:
                    pass
                break
    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(pm))
    except Exception:
        return None
    return dict(smiles=smi, smiles_onium=onium, lost_h=lost)


def draw(smiles, path, legend=""):
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        m = Chem.MolFromSmiles(smiles)
        d = rdMolDraw2D.MolDraw2DCairo(520, 380)
        d.drawOptions().legendFontSize = 17
        d.DrawMolecule(m, legend=legend)
        d.FinishDrawing()
        open(path, "wb").write(d.GetDrawingText())
        return path
    except Exception:
        return None
