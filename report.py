import sys, os, re
name, smiles = sys.argv[1], sys.argv[2]
def read(p):
    return open(p).read() if os.path.exists(p) else ""
def grab(pat, txt, d="?"):
    m = re.search(pat, txt)
    return m.group(1).strip() if m else d
mp, xo, geo = read(name + "_map.txt"), read(name + "_xtb.out"), read(name + "_geo.txt")
charge = grab(r"net charge\s*:\s*(\S+)", mp)
unp = grab(r"unpaired e-\s*:\s*(\d+)", mp)
centers = list(dict.fromkeys(c.strip() for c in re.findall(r"atom \d+ \(\w+\): .*", mp)))
conv = "converged" if "GEOMETRY OPTIMIZATION CONVERGED" in xo else "NOT CONVERGED"
topo = "YES - check structure" if "Topologies differ" in xo else "no"
E = grab(r"TOTAL ENERGY\s+(-?\d+\.\d+)", xo)
q = [float(v) for v in read("charges").split()]
wbo = {}
for l in read("wbo").splitlines():
    p = l.split()
    if len(p) == 3:
        wbo[(int(p[0]), int(p[1]))] = wbo[(int(p[1]), int(p[0]))] = float(p[2])
fp = {}
L = read(name + "_fukui.out").splitlines()
for k, l in enumerate(L):
    if "f(+)" in l and "f(-)" in l:
        for m in L[k + 1:]:
            t = m.split()
            if len(t) < 4 or not re.match(r"\d+", t[0]):
                break
            fp[int(re.match(r"\d+", t[0]).group())] = float(t[1])
        break
try:
    from morfeus import BuriedVolume, read_xyz
    el, xyz = read_xyz("xtbopt.xyz")
    vbur = lambda i: 100 * BuriedVolume(el, xyz, i).fraction_buried_volume
except Exception:
    vbur = None
blocks, cur = {}, None
for l in geo.splitlines():
    t = l.split()
    if l.startswith("##########"):
        cur = tuple(map(int, l.split("atoms")[1].split("#")[0].split()))
        blocks[cur] = {"intact": False}
    elif cur is None:
        continue
    elif l.startswith("pyramidalization"): blocks[cur]["pyr"] = float(t[1])
    elif l.startswith("bond-length asym"): blocks[cur]["dl"] = float(t[2])
    elif l.startswith("dihedral adv"): blocks[cur]["dv"] = float(t[2])
    elif l.startswith("p-axis/bond adv"): blocks[cur]["da"] = float(t[2])
    elif "REAL DIFF" in l: blocks[cur]["call"] = "REAL"
    elif l.startswith("FLAT"): blocks[cur]["call"] = "FLAT"
    elif "(scission bond A)" in l: blocks[cur]["rA"] = float(t[1])
    elif "(scission bond B)" in l: blocks[cur]["rB"] = float(t[1])
    elif "(distal bond)" in l: blocks[cur]["rC"] = float(t[1])
    elif "INTACT" in l: blocks[cur]["intact"] = True
def vote(x, thr):
    return "A" if x > thr else ("B" if x < -thr else "tie")
def consensus(vs):
    a, b = vs.count("A"), vs.count("B")
    if a + b == 0: return "none", "no clear preference (all ties)"
    if a and not b: return "A", "A (%d of %d indicators, none against)" % (a, len(vs))
    if b and not a: return "B", "B (%d of %d indicators, none against)" % (b, len(vs))
    w = "A" if a > b else ("B" if b > a else "split")
    return w, "%s (mixed: %d A vs %d B)" % (w, a, b)
print("=" * 64); print(" REPORT: " + name); print("=" * 64)
print("SMILES          :", smiles)
print("Species         : charge %s, unpaired electrons %s" % (charge, unp))
for c in centers or ["none flagged"]: print("Reactive center :", c)
print("xtb             : %s | bonding changed: %s | E = %s Eh" % (conv, topo, E))
print("solvent         : %s" % (os.environ.get("SOLV") or "none (gas phase)"))
summ = []
if not blocks:
    print("\nNo cyclopropane directly on the reactive center (A/B analysis not applicable) -- see REARRANGEMENT / SCISSION below.")
    summ.append("SUMMARY| %s | no pattern | ring=- | topo=%s" % (name, topo.split()[0]))
for (R, C1, A, B), b in blocks.items():
    print("\n--- R=%d  C1=%d  bond A = C%d-C%d  bond B = C%d-C%d ---" % (R, C1, C1, A, C1, B))
    ring = "INTACT" if b["intact"] else "NOT confirmed intact"
    print("Ring after opt  : %s (A %.3f, B %.3f, distal %.3f A)" % (ring, b.get("rA", 0), b.get("rB", 0), b.get("rC", 0)))
    print("Center planarity: %.1f deg (0 = planar)" % b.get("pyr", -1))
    print("geoldr call     : %s" % b.get("call", "?"))
    print("\n  %-26s %10s %10s   %s" % ("BOND BREAKING", "A", "B", "vote"))
    bv = []
    v = vote(b.get("dl", 0), 0.005); bv.append(v)
    print("  %-26s %10.3f %10.3f   %s" % ("bond length (A)", b.get("rA", 0), b.get("rB", 0), v))
    v = vote(b.get("dv", 0), 2.0); bv.append(v)
    print("  %-26s %21s   %s" % ("dihedral advantage", "%+.1f deg" % b.get("dv", 0), v))
    v = vote(b.get("da", 0), 2.0); bv.append(v)
    print("  %-26s %21s   %s" % ("p-axis/bond advantage", "%+.1f deg" % b.get("da", 0), v))
    if wbo:
        wa, wb = wbo.get((C1, A), 0), wbo.get((C1, B), 0)
        v = vote(wb - wa, 0.02); bv.append(v)
        print("  %-26s %10.3f %10.3f   %s" % ("WBO (lower = weaker)", wa, wb, v))
    print("\n  %-26s %10s %10s   %s" % ("NUCLEOPHILE SITE", "C%d" % A, "C%d" % B, "vote"))
    sv = []
    if q:
        v = vote(q[A - 1] - q[B - 1], 0.02); sv.append(v)
        print("  %-26s %10.3f %10.3f   %s" % ("charge (more + wins)", q[A - 1], q[B - 1], v))
    if fp:
        v = vote(fp.get(A, 0) - fp.get(B, 0), 0.01); sv.append(v)
        print("  %-26s %10.3f %10.3f   %s" % ("f(+) (higher wins)", fp.get(A, 0), fp.get(B, 0), v))
    if vbur:
        va, vb = vbur(A), vbur(B)
        v = vote(vb - va, 2.0); sv.append(v)
        print("  %-26s %10.1f %10.1f   %s" % ("%Vbur (less crowded wins)", va, vb, v))
    bw, bt = consensus(bv)
    sw, st = consensus(sv) if sv else ("none", "no data")
    print("\nCONSENSUS")
    print("  Bond that breaks : " + bt)
    print("  Nucleophile site : " + st)
    if bw in "AB" and sw in "AB" and bw != sw:
        print("  ** CONFLICT: breaking favors %s but Nu-site favors %s -- good DFT candidate **" % (bw, sw))
    if not b["intact"]:
        print("  ** Ring not confirmed intact -- treat geometry votes with caution **")
    if b.get("call") == "FLAT":
        print("  ** geoldr: scaffold does not differentiate the bonds **")
    summ.append("SUMMARY| %s | R%d C1%d A%d B%d | break=%s | site=%s | ring=%s | topo=%s"
                % (name, R, C1, A, B, bw, sw, "ok" if b["intact"] else "CHECK", topo.split()[0]))
na = read(name + "_nuattack.txt")
if na.strip():
    print("\n" + na.rstrip())
rr = read(name + "_rearr.txt")
if rr.strip():
    print("\n" + rr.rstrip())
cd = read(name + "_coord.txt")
if cd.strip():
    print("\n" + cd.rstrip())
rg = read(name + "_rings.txt")
if rg.strip():
    print("\n" + rg.rstrip())
gen = read(name + "_general.txt")
if gen.strip():
    print("\n" + gen.rstrip())
print("\nxtb-level screen -- confirm finalists with DFT.\n")
for s in summ: print(s)
