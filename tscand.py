#!/usr/bin/env python3
"""
tscand.py NAME SMILES CHARGE UHF  (run in the molecule folder after rearr / cascade)

Lists the steps worth checking with DFT (autodE + ORCA, command run_ts), numbered, with the reason:
  close call       two pathways within 3 kcal/mol: xtb cannot decide between them
  borderline       barrier 15-25 kcal/mol: fast or slow at room temperature depends on the exact value
  uphill           the step is uphill (dE > 0) at xtb level
  not facile       nothing passed at xtb level, but a pathway is within reach (barrier <= 30): DFT may differ
  key step         the predicted step itself (optional check)
Output lines:  DFT| k | yes/optional | reason | REACTANT>>PRODUCT | what
"""
import os, re, sys, json
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cascade import product_species, read_xyz          # noqa: E402

TIE, LOW, HIGH, REACH = 3.0, 15.0, 25.0, 30.0


def main():
    name, smiles, chg, uhf = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    pw = json.load(open("rearr/pathways.json")) if os.path.exists("rearr/pathways.json") else []
    pj = json.load(open("rearr/predicted.json")) if os.path.exists("rearr/predicted.json") else {}
    rows, seen = [], set()

    def add(worth, reason, prod_smiles, what, react=smiles):
        rx = "%s>>%s" % (react, prod_smiles)
        if prod_smiles in ("", "-", "?") or rx in seen:
            return
        seen.add(rx)
        rows.append([worth, reason, rx, what])

    def prod_of(p):
        if not os.path.exists(p["xyz"]):
            return None
        Xp = read_xyz(p["xyz"])[1]
        pr, _ = product_species(smiles, Xp, [tuple(x) for x in p["form"]], [tuple(x) for x in p["brk"]], chg, uhf)
        return pr["smiles"] if pr else None

    m0 = Chem.MolFromSmiles(smiles)
    sy = lambda i: m0.GetAtomWithIdx(i).GetSymbol() if m0 is not None and i < m0.GetNumAtoms() else "H"
    lab = lambda bs: ", ".join("%s%d-%s%d" % (sy(a), a + 1, sy(b), b + 1) for a, b in bs)
    desc = lambda p: "%s%s%s (xtb barrier %.1f, dE %+.1f)" % (
        ("breaks " + lab(p["brk"])) if p["brk"] else "", "; " if p["brk"] and p["form"] else "",
        ("forms " + lab(p["form"])) if p["form"] else "", p["barrier"], p["dE"])
    pw = sorted(pw, key=lambda p: p["barrier"])
    if pj.get("step") and pw:
        main_ = next((p for p in pw if sorted(map(sorted, p["brk"])) == sorted(map(sorted, pj.get("brk", []))) and
                      sorted(map(sorted, p["form"])) == sorted(map(sorted, pj.get("form", [])))), None)
        if main_ is not None:
            close = [p for p in pw if p is not main_ and abs(p["barrier"] - main_["barrier"]) <= TIE]
            why = []
            if close:
                why.append("close call")
            if LOW <= main_["barrier"] <= HIGH:
                why.append("borderline barrier")
            if main_["dE"] > 0:
                why.append("uphill")
            ps = prod_of(main_)
            if ps:
                add("yes" if why else "optional", ", ".join(why) or "key step", ps, "predicted step: " + desc(main_))
            for p in close[:2]:
                ps = prod_of(p)
                if ps:
                    add("yes", "close call (competitor)", ps, "competing step: " + desc(p))
    elif pw:                                        # nothing facile at xtb level
        for p in [p for p in pw if p["barrier"] <= REACH and p["dE"] <= 10.0][:2]:   # not strongly uphill
            ps = prod_of(p)
            if ps:
                add("yes", "not facile at xtb level", ps, "best pathway found: " + desc(p))
    # cascade steps (reactant = species before the step)
    ca = open(name + "_cascade.txt").read() if os.path.exists(name + "_cascade.txt") else ""
    prev = smiles
    for l in ca.splitlines():
        if not l.startswith("CASCADE|"):
            continue
        f = [x.strip() for x in l.split("|")[1:]]
        if len(f) < 5:
            continue
        k, what, dEa, dE, ps = f[0], f[1], f[2], f[3], f[4]
        if ps in ("-", "?") or "crossover" in what or "E1" in what:
            prev = ps if ps not in ("-", "?") else prev
            continue
        why = []
        if "uphill" in what or (dE not in ("-", "") and float(dE) > 0):
            why.append("uphill")
        if dEa not in ("-", "") and LOW <= float(dEa) <= HIGH:
            why.append("borderline barrier")
        add("yes" if why else "optional", ", ".join(why) or "cascade step", ps,
            "cascade step %s: %s (xtb barrier %s, dE %s)" % (k, what, dEa, dE), react=prev)
        prev = ps
    nuat = open(name + "_nuattack.txt").read() if os.path.exists(name + "_nuattack.txt") else ""
    # nucleophile attack: a close call between pathways, or a steric warning -> worth DFT (cation + Nu -> product)
    mn = re.search(r"NUAT\|\s*site (\d+) \| break (\d+) (\d+) \| form (\d+) (\d+) \| dEa (\S+) \| dE (\S+)(.*)", nuat)
    mp = re.search(r"NUPROD\|\s*\S+ \| \S+ \| \S+ \| (\S+)", nuat)
    if mp is None:                                  # older line: usable only if no H+ was removed for the picture
        mp = re.search(r"NUPROD\|\s*(\S+) \| - \|", nuat)
    if mn and mp:
        rest = mn.group(8)
        why = []
        if "capture" in rest or "CLOSE CALL" in nuat or "close call" in rest or "-> competitive" in nuat:
            why.append("nucleophile close call")
        if "steric warning" in rest:
            why.append("steric warning")
        if why:
            nm_ = re.search(r"\| nu (.+?)\s*$", rest)
            try:
                import nucleophiles
                nsmi = nucleophiles.resolve(nm_.group(1).strip())["smiles"] if nm_ else None
                nmol = Chem.MolFromSmiles(nsmi) if nsmi else None
                if nmol is not None:
                    for a in nmol.GetAtoms():
                        a.SetAtomMapNum(0)
                    nsmi = Chem.MolToSmiles(nmol)
            except Exception:
                nsmi = None
            if nsmi:
                cap = re.search(r"capture (\d+) ([\d.]+)", rest)
                what = "nucleophile attack at C%s (xtb barrier %s)%s" % (
                    mn.group(1), mn.group(6), (" vs direct capture at C%s (%s)" % cap.groups()) if cap else "")
                add("yes", ", ".join(why), mp.group(1), what, react="%s.%s" % (smiles, nsmi))
    # report
    print("-" * 70)
    print(" DFT CHECK  (steps worth a DFT transition-state calculation: run_ts %s <number> on the cluster)" % name)
    print("-" * 70)
    if not rows:
        print(" nothing to list (no step found by the scans)")
    for k, (worth, reason, rx, what) in enumerate(rows, 1):
        print(" %d. [%s] %s -- %s" % (k, "DFT worth it" if worth == "yes" else "optional", reason, what))
        print("    %s" % rx)
        print("DFT| %d | %s | %s | %s | %s" % (k, worth, reason, rx, what))
    if "steric warning" in nuat:
        print(" Note: the nucleophile attack has a STERIC WARNING. To check it with DFT, write the reaction yourself:")
        print("   run_ts \"CATION.NUCLEOPHILE>>PRODUCT\" name solvent")
    print(" (each run_ts job: autodE finds the transition state with xtb, then ORCA DFT; hours per step)")


if __name__ == "__main__":
    main()
