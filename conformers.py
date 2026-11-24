#!/usr/bin/env python3
"""
conformers.py NAME CREST XTB CHG UHF      (run in the species' folder right after the xtb optimization)

CREST conformer search (Grimme group; built-in GFN2-xTB, same ALPB solvent as the run):
  * the lowest conformer becomes the start structure (xtbopt.xyz is replaced if CREST found a lower one)
  * up to CONF_KEEP-1 further conformers within CONF_WINDOW kcal/mol are written to conformers_extra.xyz;
    rearr.py also looks for bond-forming steps (1,5-HAT, ring closures, intramolecular trapping) in them,
    which a single conformer often misses in flexible chains.
Environment: SOLV, CONF_WINDOW (3.0), CONF_KEEP (4), CONF_TIMEOUT (seconds, 3600), OMP_NUM_THREADS
Writes NAME_conformers.txt (summary).
"""
import os, re, sys, shutil, subprocess
import numpy as np

KCAL = 627.509


def read_multi(p):
    out = []
    L = open(p).read().split("\n")
    k = 0
    while k < len(L) and L[k].strip():
        m = int(L[k].split()[0])
        e = re.search(r"(-?\d+\.\d+)", L[k + 1])
        sym = [l.split()[0] for l in L[k + 2:k + 2 + m]]
        X = np.array([[float(v) for v in l.split()[1:4]] for l in L[k + 2:k + 2 + m]])
        out.append((float(e.group(1)) if e else None, sym, X))
        k += m + 2
    return out


def write_xyz(f, sym, X, comment=""):
    f.write("%d\n%s\n" % (len(sym), comment))
    for s, x in zip(sym, X):
        f.write("%-2s %14.8f %14.8f %14.8f\n" % (s, x[0], x[1], x[2]))


RCOV = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07, "S": 1.05,
        "Cl": 1.02, "Br": 1.20, "I": 1.39, "Sn": 1.39, "Al": 1.21}


def bonds(sym, X):
    """connectivity from distances (incl. H): conformers must keep it -- no H transfer or ring closure allowed"""
    out = set()
    for i in range(len(sym)):
        for j in range(i + 1, len(sym)):
            if np.linalg.norm(X[i] - X[j]) < 1.25 * (RCOV.get(sym[i], 1.0) + RCOV.get(sym[j], 1.0)):
                out.add((i, j))
    return out


def energy(xyz):
    m = re.search(r"energy:\s*(-?\d+\.\d+)", open(xyz).read().split("\n")[1])
    return float(m.group(1)) if m else None


def main():
    name, crest, xtb, chg, uhf = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    solv = os.environ.get("SOLV", "")
    win = float(os.environ.get("CONF_WINDOW", "3.0"))
    keep = int(os.environ.get("CONF_KEEP", "4"))
    tmo = int(os.environ.get("CONF_TIMEOUT", "3600"))
    ncpu = os.environ.get("OMP_NUM_THREADS", "1") or "1"
    rep = open(name + "_conformers.txt", "w")
    P = lambda s="": rep.write(s + "\n")
    P("CONFORMER SEARCH (CREST, GFN2-xTB%s)" % (", ALPB " + solv if solv else ", gas phase"))
    if not (os.path.exists("xtbopt.xyz") and (os.path.exists(crest) or shutil.which(crest))):
        P("   skipped (CREST or the optimized structure not found)")
        return
    if os.path.exists("conformers_extra.xyz"):
        os.remove("conformers_extra.xyz")
    shutil.rmtree("crest", ignore_errors=True)
    os.makedirs("crest")
    shutil.copy("xtbopt.xyz", "crest/start.xyz")
    cmd = [crest, "start.xyz", "--gfn2", "--quick", "--chrg", str(chg), "--uhf", str(uhf), "--T", str(ncpu)]
    if solv:
        cmd += ["--alpb", solv]
    try:
        with open("crest/crest.out", "w") as f:
            subprocess.run(cmd, cwd="crest", stdout=f, stderr=subprocess.STDOUT, timeout=tmo)
    except subprocess.TimeoutExpired:
        P("   CREST did not finish within %d s -- the single optimized conformer is used" % tmo)
        return
    ens = "crest/crest_conformers.xyz"
    if not os.path.exists(ens):
        P("   CREST produced no ensemble (see crest/crest.out) -- the single optimized conformer is used")
        return
    confs = [c for c in read_multi(ens) if c[0] is not None]
    sym0 = open("xtbopt.xyz").read().split("\n")[2:]
    sym0 = [l.split()[0] for l in sym0 if l.strip()]
    confs = [c for c in confs if c[1] == sym0]
    _L0 = open("xtbopt.xyz").read().split("\n")
    X0 = np.array([[float(v) for v in l.split()[1:4]] for l in _L0[2:2 + len(sym0)]])
    ref = bonds(sym0, X0)
    nall = len(confs)
    confs = [c for c in confs if bonds(c[1], c[2]) == ref]
    if nall - len(confs):
        P("   %d conformer(s) dropped: the bonding changed during the search (e.g. an H transfer) -- the" % (nall - len(confs)))
        P("   reaction itself is analysed by the scans, not by the conformer search")
    if not confs:
        P("   no usable conformers (atom order changed) -- the single optimized conformer is used")
        return
    e0 = confs[0][0]
    low = [c for c in confs if (c[0] - e0) * KCAL <= win][:keep]
    P("   %d conformers found, %d within %.1f kcal/mol (kept: %d)" % (len(confs), sum(1 for c in confs if (c[0] - e0) * KCAL <= win),
                                                                    win, len(low)))
    # re-optimize the lowest with the run settings; use it if it is lower than the current start
    d = "crest/best"
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "in.xyz"), "w") as f:
        write_xyz(f, low[0][1], low[0][2])
    cmd = [xtb, "in.xyz", "--opt", "--chrg", str(chg), "--uhf", str(uhf), "--gfn", "2"] + (["--alpb", solv] if solv else [])
    with open(os.path.join(d, "xtb.out"), "w") as f:
        subprocess.run(cmd, cwd=d, stdout=f, stderr=subprocess.STDOUT)
    e_old = energy("xtbopt.xyz")
    nb = os.path.join(d, "xtbopt.xyz")
    ok_topo = False
    if os.path.exists(nb):
        _L = open(nb).read().split("\n")
        ok_topo = bonds(sym0, np.array([[float(v) for v in l.split()[1:4]] for l in _L[2:2 + len(sym0)]])) == ref
    if ok_topo and energy(nb) is not None and e_old is not None and energy(nb) < e_old - 0.1 / KCAL:
        for f_ in ("xtbopt.xyz", "wbo", "charges", "xtbopt.log", "xtbtopo.mol"):
            if os.path.exists(os.path.join(d, f_)):
                shutil.copy(os.path.join(d, f_), f_)
        P("   start structure replaced by the lowest CREST conformer: %.1f kcal/mol lower than the first optimization"
          % ((e_old - energy("xtbopt.xyz")) * KCAL))
    else:
        P("   the first optimized structure is already the lowest conformer")
    if len(low) > 1:
        with open("conformers_extra.xyz", "w") as f:
            for e, sym, X in low[1:]:
                write_xyz(f, sym, X, "energy: %.8f  rel %.2f kcal/mol" % (e, (e - e0) * KCAL))
        P("   further conformers searched for reaction steps: " +
          ", ".join("+%.1f" % ((c[0] - e0) * KCAL) for c in low[1:]) + " kcal/mol")


if __name__ == "__main__":
    main()
