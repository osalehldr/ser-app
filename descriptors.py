import sys, re, os
name = sys.argv[1]
q = [float(l) for l in open("charges") if l.strip()]
wbo = {}
for l in open("wbo"):
    p = l.split()
    if len(p) == 3:
        i, j, w = int(p[0]), int(p[1]), float(p[2])
        wbo[(i, j)] = wbo[(j, i)] = w
fp = {}
if os.path.exists(name + "_fukui.out"):
    L = open(name + "_fukui.out").read().splitlines()
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
    vb = lambda i: 100 * BuriedVolume(el, xyz, i).fraction_buried_volume
except Exception:
    vb = None
for line in open(name + "_atoms.txt"):
    if not line.strip():
        continue
    R, C1, A, B = map(int, line.split())
    print("atoms R=%d C1=%d A=%d B=%d" % (R, C1, A, B))
    print("  %-30s %10s %10s" % ("", "A (C%d)" % A, "B (C%d)" % B))
    print("  %-30s %10.3f %10.3f" % ("WBO C1-x (lower = weaker)", wbo.get((C1, A), 0), wbo.get((C1, B), 0)))
    print("  %-30s %10.3f %10.3f" % ("charge q (more + = cationic)", q[A - 1], q[B - 1]))
    if fp:
        print("  %-30s %10.3f %10.3f" % ("f(+) (higher = Nu attacks)", fp.get(A, 0), fp.get(B, 0)))
    if vb:
        print("  %-30s %10.1f %10.1f" % ("%Vbur (higher = crowded)", vb(A), vb(B)))
    print()
