import sys, re, os
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D
RDLogger.DisableLog("rdApp.*")
# colours: green = bond that breaks, red = tie, orange = atoms that form a new bond, blue = nucleophile site
GREEN, RED, ORANGE, BLUE = (0.3, 0.8, 0.4), (0.95, 0.3, 0.3), (1.0, 0.65, 0.2), (0.35, 0.55, 1.0)
TPL = os.path.expanduser("~/chem/template.smi")
name = sys.argv[1]
_rr = open(name + "_rearr.txt").read() if os.path.exists(name + "_rearr.txt") else ""
NU = "E+" if "E+ at" in _rr else "Nu"
if not os.path.exists(name + "_map.txt"):
    sys.exit()
m_ = re.search(r"SMILES in\s*:\s*(\S+)", open(name + "_map.txt").read())
if not m_:
    sys.exit()
mol = Chem.MolFromSmiles(m_.group(1))
if mol is None:
    sys.exit()
NH = mol.GetNumAtoms()


def read(p):
    return open(p).read() if os.path.exists(p) else ""


def align(m2):
    if not os.path.exists(TPL):
        return False
    try:
        tok = open(TPL).read().split()
        t = Chem.MolFromSmiles(tok[0])
        rdDepictor.Compute2DCoords(t)
        if len(tok) > 1 and tok[1].lower() == "flip":
            conf = t.GetConformer()
            for i in range(t.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, Point3D(-p.x, p.y, 0.0))
        mcs = rdFMCS.FindMCS([t, m2], atomCompare=rdFMCS.AtomCompare.CompareElements,
                             bondCompare=rdFMCS.BondCompare.CompareAny,
                             ringMatchesRingOnly=True, completeRingsOnly=True, timeout=10)
        q = Chem.MolFromSmarts(mcs.smartsString)
        mt, mm = t.GetSubstructMatch(q), m2.GetSubstructMatch(q)
        if len(mm) < 3:
            return False
        rdDepictor.GenerateDepictionMatching2DStructure(m2, t, list(zip(mt, mm)))
        return True
    except Exception:
        return False


def base():
    m2 = Chem.Mol(mol)
    for a in m2.GetAtoms():
        a.SetProp("atomNote", str(a.GetIdx() + 1))
    if not align(m2):
        AllChem.Compute2DCoords(m2)
    return m2


def bidx(m2, i, j):
    if i < 1 or j < 1 or max(i, j) > m2.GetNumAtoms():
        return None
    b = m2.GetBondBetweenAtoms(i - 1, j - 1)
    return b.GetIdx() if b else None


def save(m2, hb, col, ha, hac, legend, out):
    n = m2.GetNumAtoms()
    hb = [x for x in hb if x is not None]
    col = {k: v for k, v in col.items() if k is not None}
    ha = [x for x in ha if 0 <= x < n]
    hac = {k: v for k, v in hac.items() if 0 <= k < n}
    d = rdMolDraw2D.MolDraw2DCairo(900, 900)
    d.drawOptions().legendFontSize = 22
    rdMolDraw2D.PrepareAndDrawMolecule(d, m2, legend=legend, highlightAtoms=ha, highlightAtomColors=hac,
                                       highlightBonds=hb, highlightBondColors=col)
    d.FinishDrawing()
    open(out, "wb").write(d.GetDrawingText())
    print("wrote", out)


def lab(i):
    if 1 <= i <= NH:
        return "%s%d" % (mol.GetAtomWithIdx(i - 1).GetSymbol(), i)
    return "H%d" % i


for old in [f for f in os.listdir(".") if f.startswith(name + "_break") and f.endswith(".png")]:
    os.remove(old)

# ---- inputs from the analyses
rep = read(name + "_report.txt")
sets = []
for blk in re.split(r"\n--- ", rep)[1:]:
    m = re.match(r"R=(\d+)\s+C1=(\d+)\s+bond A = C\d+-C(\d+)\s+bond B = C\d+-C(\d+)", blk)
    if m:
        w = re.search(r"Bond that breaks\s*:\s*(\S+)", blk)
        sv = re.search(r"Nucleophile site\s*:\s*(\S+)", blk)
        sets.append((int(m.group(2)), int(m.group(3)), int(m.group(4)),
                     w.group(1) if w else "none", sv.group(1) if sv else "none"))
rl = re.findall(r"RING\|\s*atoms (\d+) (\d+) (\d+) \| weak (\d+) (\d+) \| weak2 (\d+) (\d+) \| call (\S+) \| path -?\d+ \| site (\S+)",
                read(name + "_rings.txt"))
rr = re.search(r"REARR\|\s*form (\d+) (\d+) \| break (\d+) (\d+) \| nu (\d+)", read(name + "_rearr.txt"))
f1 = f2 = b1 = b2 = nu = 0
if rr:
    f1, f2, b1, b2, nu = map(int, rr.groups())
na = re.search(r"NUAT\|\s*site (\d+) \| break (\d+) (\d+) \| form (\d+) (\d+)(.*)", read(name + "_nuattack.txt"))
ns = nb1 = nb2 = nf1 = nf2 = 0
cap = None
if na:
    ns, nb1, nb2, nf1, nf2 = map(int, na.groups()[:5])
    mc = re.search(r"capture (\d+) ([\d.]+)", na.group(6))
    cap = (int(mc.group(1)), float(mc.group(2))) if mc else None


# the analysis concluded "no rearrangement" (or a nucleophile captures directly): the picture must say the same,
# not show a hypothetical ring opening from the cyclopropane section
no_rearr = rr is not None and not (f1 or b1)
capture_only = bool(na) and not (nb1 and nb2)


def heavy_partner(x, other):
    """for an H atom (index > NH) return the heavy atom it pairs with in this bond, else x"""
    return other if x > NH else x


# ---- 0. nucleophile attack that opens a cyclopropane (highest priority)
if nb1 and nb2:
    m2 = base()
    hb, col, ha, hac = [], {}, [], {}
    bb = bidx(m2, nb1, nb2)
    if bb is not None:
        hb.append(bb); col[bb] = GREEN
    parts = ["Nu attacks %s: %s-%s breaks (green)" % (lab(ns), lab(nb1), lab(nb2))]
    if nf1 and nf2:
        for x in (nf1, nf2):
            if x <= NH:
                ha.append(x - 1); hac.setdefault(x - 1, ORANGE)
        parts.append("%s...%s forms (orange)" % (lab(nf1), lab(nf2)))
    ha.append(ns - 1); hac[ns - 1] = BLUE
    if cap:
        parts.append("direct capture at %s competitive" % lab(cap[0]))
    save(m2, hb, col, ha, hac, "%s   %s" % (name, "; ".join(parts)), name + "_break.png")
# ---- 0b. the chosen nucleophile captures directly: no bond breaks
elif capture_only:
    m2 = base()
    save(m2, [], {}, [ns - 1], {ns - 1: BLUE}, "%s   no bond breaks; %s captured at %s (blue)" % (name, NU, lab(ns)),
         name + "_break.png")
# ---- 1. scan-based rearrangement / scission
elif f1 or b1:
    m2 = base()
    hb, col, ha, hac, parts = [], {}, [], {}, []
    if b1:
        bb = bidx(m2, b1, b2)
        if bb is not None:
            hb.append(bb); col[bb] = GREEN
        else:                                   # bond to an H: mark the heavy atom instead
            for x in (b1, b2):
                if x <= NH:
                    ha.append(x - 1); hac[x - 1] = GREEN
        parts.append("%s-%s breaks (green)" % (lab(b1), lab(b2)))
    if f1:
        for x in (f1, f2):
            if x <= NH:
                ha.append(x - 1); hac.setdefault(x - 1, ORANGE)
        parts.append("%s...%s forms (orange)" % (lab(f1), lab(f2)))
    if nu:
        ha.append(nu - 1); hac[nu - 1] = BLUE
        parts.append(NU + " at %s (blue)" % lab(nu))
    save(m2, hb, col, ha, hac, "%s   %s" % (name, "; ".join(parts)), name + "_break.png")
# ---- 2. cyclopropane directly on the reactive center (A/B vote)
elif sets and not no_rearr:
    for k, (C1, A, B, win, site) in enumerate(sets):
        m2 = base()
        bA, bB = bidx(m2, C1, A), bidx(m2, C1, B)
        if win == "A":
            hb, col, txt = [bA], {bA: GREEN}, "breaks C%d-C%d" % (C1, A)
        elif win == "B":
            hb, col, txt = [bB], {bB: GREEN}, "breaks C%d-C%d" % (C1, B)
        else:
            hb, col, txt = [bA, bB], {bA: RED, bB: RED}, "tie: C%d-C%d / C%d-C%d" % (C1, A, C1, B)
        ha, hac = [], {}
        at = (A if site == "A" else B) - 1 if site in ("A", "B") else (nu - 1 if nu else None)
        if at is not None:
            ha, hac = [at], {at: BLUE}
            txt += "   " + NU + " at %s (blue)" % lab(at + 1)
        out = name + "_break.png" if len(sets) == 1 else "%s_break_%d.png" % (name, k + 1)
        save(m2, hb, col, ha, hac, "%s   %s" % (name, txt), out)
# ---- 3. remote cyclopropanes (ring analysis)
elif rl and not no_rearr:
    m2 = base()
    hb, col, ha, hac, parts = [], {}, [], {}, []
    for a, b, c, i1, j1, i2, j2, call, site in rl:
        w1, w2 = bidx(m2, int(i1), int(j1)), bidx(m2, int(i2), int(j2))
        if call == "tie":
            for x in (w1, w2):
                if x is not None:
                    hb.append(x); col[x] = RED
            parts.append("tie C%s-C%s / C%s-C%s" % (i1, j1, i2, j2))
        elif call in ("activated", "OPENED"):
            if w1 is not None:
                hb.append(w1); col[w1] = GREEN
            parts.append("%s C%s-C%s" % ("opened" if call == "OPENED" else "breaks", i1, j1))
            mm = re.search(r"(\d+)$", site)
            if mm:
                at = int(mm.group(1)) - 1
                ha.append(at); hac[at] = BLUE
                parts.append(NU + " at C%d (blue)" % (at + 1))
        else:
            parts.append("ring C%s-C%s-C%s unperturbed" % (a, b, c))
    save(m2, hb, col, ha, hac, "%s   %s" % (name, "; ".join(parts)), name + "_break.png")
# ---- 4. no rearrangement predicted: just show the trapping site
elif ns or nu:
    nu = ns or nu
    m2 = base()
    save(m2, [], {}, [nu - 1], {nu - 1: BLUE}, "%s   no rearrangement; " + NU + " at %s (blue)" % (name, lab(nu)),
         name + "_break.png")
