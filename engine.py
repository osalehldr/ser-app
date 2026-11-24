#!/usr/bin/env python3
"""
engine.py -- runs the full reactivity analysis for one molecule on any computer (no Slurm needed).

Same steps as ~/chem/pipeline.sh:
  structure (pyldr) -> xtb optimization -> cyclopropane A/B analysis (geoldr) -> Fukui + descriptors
  -> general / ring / metal analyses -> rearrangement scans -> report -> bond picture

A NEUTRAL molecule is analysed as three species (one folder each): neutral, cation (every protonation
site; acids / Lewis acids / electrophiles modelled as H+) and radical_cation -- plus a ranking of which
forms more easily (scripts/scenarios.py).  Cation, radical cation or radical input = that one species.
Anions are not analysed.  The nucleophile for the attack scans is selectable (scripts/nucleophiles.py).

Command line ("local mode"):
    python3 engine.py "SMILES" name [solvent] [--outdir DIR] [--cpus N] [--only neutral,cation,radical_cation]
                      [--nu NUCLEOPHILE]   (preset name or SMILES; default water)
Environment:
    XTB    path to the xtb binary   (default: "xtb" on the PATH)
    GXTB   optional g-xTB-enabled xtb binary for energy refinement
"""
import os, re, sys, time, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
SOLVENTS = ["acetone", "acetonitrile", "aniline", "benzaldehyde", "benzene", "ch2cl2", "chcl3", "cs2",
            "dioxane", "dmf", "dmso", "ether", "ethanol", "ethylacetate", "furane", "hexadecane", "hexane",
            "methanol", "nitromethane", "octanol", "woctanol", "phenol", "toluene", "thf", "water"]
ALIASES = {"gas": "", "none": "", "vacuum": "", "dcm": "ch2cl2", "dichloromethane": "ch2cl2",
           "chloroform": "chcl3", "mecn": "acetonitrile", "acn": "acetonitrile", "meoh": "methanol",
           "etoh": "ethanol", "etoac": "ethylacetate", "h2o": "water"}
KCAL = 627.509


XTB_URL = "https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz"
GXTB_URL = "https://github.com/grimme-lab/g-xtb/raw/main/binaries/xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz"
CREST_URL = "https://github.com/crest-lab/crest/releases/download/v3.0.2/crest-gnu-12-ubuntu-latest.tar.xz"
CACHE = os.environ.get("XTB_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "reactivity-screen"))


def _download_binary(url, folder, log=print):
    """download + unpack an xtb release tarball (Linux); returns the path of bin/xtb or ''"""
    import tarfile, urllib.request
    os.makedirs(folder, exist_ok=True)
    for root, _, files in os.walk(folder):
        if "xtb" in files and root.endswith("bin"):
            return os.path.join(root, "xtb")
    tmp = os.path.join(folder, "download.tar.xz")
    try:
        log("downloading %s (first start only) ..." % url.rsplit("/", 1)[-1])
        urllib.request.urlretrieve(url, tmp)
        with tarfile.open(tmp, "r:xz") as t:
            t.extractall(folder)
    except Exception as e:
        log("download failed: %s" % e)
        return ""
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    for root, _, files in os.walk(folder):
        if "xtb" in files and root.endswith("bin"):
            p = os.path.join(root, "xtb")
            os.chmod(p, 0o755)
            return p
    return ""


def ensure_binaries(log=print, want_gxtb=True):
    """find xtb (XTB variable, PATH, or download it); optionally g-xTB (GXTB variable or download)"""
    xtb = os.environ.get("XTB") or shutil.which("xtb") or ""
    if not xtb or not os.path.exists(xtb):
        xtb = _download_binary(XTB_URL, os.path.join(CACHE, "xtb"), log) if sys.platform.startswith("linux") else ""
    gx = os.environ.get("GXTB", "")
    if want_gxtb and (not gx or not os.path.exists(gx)) and sys.platform.startswith("linux"):
        gx = _download_binary(GXTB_URL, os.path.join(CACHE, "gxtb"), log)
    if xtb:
        os.environ["XTB"] = xtb
    if gx and os.path.exists(gx):
        os.environ["GXTB"] = gx
    return xtb, (gx if gx and os.path.exists(gx) else "")


def ensure_crest(log=print):
    """CREST binary (conformer search): $CREST, crest on the PATH, or downloaded once into the cache"""
    c = os.environ.get("CREST") or shutil.which("crest")
    if c and os.path.exists(c):
        return c
    p = os.path.join(CACHE, "crest", "crest", "crest")
    if not os.path.exists(p):
        try:
            _download_binary(CREST_URL, os.path.join(CACHE, "crest"), log)
        except Exception as err:
            log("CREST download failed (%s) -- conformer search disabled" % err)
            return ""
    if os.path.exists(p):
        os.chmod(p, 0o755)
        return p
    found = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(CACHE, "crest")) for f in fs if f == "crest"]
    return found[0] if found else ""


def norm_solvent(s):
    """returns the xtb ALPB solvent name, '' for gas phase, or None if unknown"""
    s = (s or "").strip().lower()
    s = ALIASES.get(s, s)
    return s if (s == "" or s in SOLVENTS) else None


def clean_name(s):
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "").strip())[:40].strip("_")
    return s or "molecule"


def _limits():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except Exception:
        pass


class Runner:
    def __init__(self, folder, solvent="", xtb=None, gxtb=None, cpus=1, timeout=900, log=print):
        self.d = folder
        self.solvent = solvent
        self.xtb = xtb or os.environ.get("XTB") or shutil.which("xtb") or "xtb"
        self.gxtb = gxtb if gxtb is not None else os.environ.get("GXTB", "")
        self.deadline = time.time() + timeout
        self.log = log
        self.env = os.environ.copy()
        self.env.update(OMP_NUM_THREADS=str(cpus), MKL_NUM_THREADS=str(cpus), OMP_STACKSIZE="2G",
                        PYTHONUNBUFFERED="1", SOLV=solvent or "")
        if self.gxtb and os.path.exists(self.gxtb):
            self.env["GXTB"] = self.gxtb
        else:
            self.env.pop("GXTB", None)
            self.gxtb = ""

    def run(self, cmd, out=None, append=False):
        left = self.deadline - time.time()
        if left <= 0:
            raise TimeoutError("time limit reached")
        f = open(os.path.join(self.d, out), "a" if append else "w") if out else subprocess.DEVNULL
        try:
            p = subprocess.run(cmd, cwd=self.d, env=self.env, stdout=f, stderr=subprocess.STDOUT if out else
                               subprocess.DEVNULL, timeout=left,
                               preexec_fn=_limits if os.name == "posix" else None)
            return p.returncode
        finally:
            if out:
                f.close()

    def py(self, script, args, out=None, append=False):
        path = os.path.join(SCRIPTS, script)
        if not os.path.exists(path):
            return None                      # optional analysis not installed -> skipped
        return self.run([sys.executable, path] + list(args), out, append)

    def solv(self):
        return ["--alpb", self.solvent] if self.solvent else []


def read(p):
    return open(p).read() if os.path.exists(p) else ""


def xyz_energy(p):
    m = re.search(r"energy:\s*(-?\d+\.\d+)", read(p))
    return float(m.group(1)) if m else None


def run_molecule(smiles, name, base, solvent="", cpus=1, timeout=900, log=print, xtb=None, gxtb=None,
                 outdir=None, chg=None, uhf=None, scen="", start_xyz=None, nu="none"):
    """runs one species in base/<outdir or name>/ (files named name_*) ; returns a result dict.
    chg/uhf override the SMILES (radical ions of a neutral SMILES); scen = scenario label"""
    d = os.path.join(base, outdir or name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    R = Runner(d, solvent, xtb, gxtb, cpus, timeout, log)
    R.env["SMILES_IN"] = smiles
    R.env["NU"] = nu or "none"
    if scen:
        R.env["SCEN"] = scen
        if chg is not None:
            R.env["CHRG"], R.env["UHF"] = str(chg), str(uhf or 0)
    res = dict(name=name, folder=d, smiles=smiles, ok=False, error="", scen=scen)
    try:
        log("1/7 building the 3D structure")
        R.py("pythonldr.py", [smiles, "--xyz", name + ".xyz", "--png", name + ".png"], name + "_map.txt")
        xyz = os.path.join(d, name + ".xyz")
        if not (os.path.exists(xyz) and os.path.getsize(xyz) > 0):
            res["error"] = "the 3D structure could not be built (see %s_map.txt)" % name
            return res
        if start_xyz and os.path.exists(start_xyz):
            R.py("scenarios.py", ["usestart", name + ".xyz", start_xyz])
        mp = read(os.path.join(d, name + "_map.txt"))
        m = re.search(r"net charge\s*:\s*([+-]?\d+)", mp)
        q = int(m.group(1)) if m else 0
        cl = subprocess.run([sys.executable, os.path.join(SCRIPTS, "classify.py"), smiles],
                            capture_output=True, text=True).stdout.split()
        u = int(cl[1]) if len(cl) > 1 and cl[1].lstrip("-").isdigit() else 0
        if chg is not None:
            q, u = int(chg), int(uhf or 0)
        res["charge"], res["uhf"] = q, u

        log("2/7 xtb geometry optimization (%s)" % (solvent or "gas phase"))
        R.run([R.xtb, name + ".xyz", "--opt", "--chrg", str(q), "--uhf", str(u), "--gfn", "2"] + R.solv(),
              name + "_xtb.out")
        if not os.path.exists(os.path.join(d, "xtbopt.xyz")):
            res["error"] = "xtb optimization failed (see %s_xtb.out)" % name
            return res
        crest = R.env.get("CREST", "")
        if R.env.get("CONF", "off") != "off" and crest and os.path.exists(crest):
            log("2b/7 conformer search (CREST)")
            R.py("conformers.py", [name, crest, R.xtb, str(q), str(u)])
        if scen:
            R.py("scenarios.py", ["centers", name, R.xtb, str(q), str(u)])

        log("3/7 cyclopropane next to the reactive center (A/B analysis)")
        R.py("find_atoms.py", [smiles], name + "_atoms.txt")
        sets = [l.split() for l in read(os.path.join(d, name + "_atoms.txt")).splitlines()
                if l.strip() and l.split()[0].isdigit()]
        open(os.path.join(d, name + "_atoms.txt"), "w").write("".join(" ".join(s) + "\n" for s in sets))
        if sets:
            open(os.path.join(d, name + "_geo.txt"), "w").close()
            for s in sets:
                with open(os.path.join(d, name + "_geo.txt"), "a") as f:
                    f.write("########## atoms %s ##########\n" % " ".join(s))
                R.py("geomldr.py", ["xtbopt.xyz"] + s, name + "_geo.txt", append=True)
                with open(os.path.join(d, name + "_geo.txt"), "a") as f:
                    f.write("\n")

        log("4/7 Fukui functions and descriptors")
        R.run([R.xtb, "xtbopt.xyz", "--vfukui", "--chrg", str(q), "--uhf", str(u)] + R.solv(), name + "_fukui.out")
        if sets:
            R.py("descriptors.py", [name], name + "_desc.txt")
        R.py("general.py", [name, smiles], name + "_general.txt")
        R.py("rings.py", [name, smiles], name + "_rings.txt")
        R.py("coord.py", [name, smiles], name + "_coord.txt")

        log("5/7 rearrangement / scission scans%s" % (" + g-xTB refinement" if R.gxtb else ""))
        R.py("rearr.py", [name, smiles, R.xtb, str(q), str(u)], name + "_rearr.txt")

        log("5b/7 nucleophile / partner scans (%s)" % (nu or "none"))
        R.py("nuattack.py", [name, smiles, R.xtb, str(q), str(u)], name + "_nuattack.txt")
        if R.env.get("CASCADE", "auto") != "off":
            log("5c/7 following multi-step sequences (cascade)")
            R.py("cascade.py", [name, smiles, R.xtb, str(q), str(u)], name + "_cascade.txt")
        R.py("tscand.py", [name, smiles, str(q), str(u)], name + "_dft.txt")
        log("6/7 writing the report")
        R.py("report.py", [name, smiles], name + "_report.txt")
        if scen:
            R.py("scenarios.py", ["formation", name, R.xtb])
            R.py("scenarios.py", ["stamp", name])
        for extra in ("_conformers.txt", "_cascade.txt", "_dft.txt"):
            t = read(os.path.join(d, name + extra))
            if t.strip():
                with open(os.path.join(d, name + "_report.txt"), "a") as f:
                    f.write("\n" + t)
        log("7/7 drawing the bond picture")
        R.py("bondpic.py", [name])
    except TimeoutError:
        res["error"] = "time limit reached -- partial results kept"
    except subprocess.TimeoutExpired:
        res["error"] = "time limit reached -- partial results kept"
    rep = read(os.path.join(d, name + "_report.txt"))
    res["report"] = rep
    m = re.search(r"PRED\|\s*(.+)", rep)
    res["pred"] = m.group(1).strip() if m else ""
    m = re.search(r"NUAT\|\s*site (\d+) \| break (\d+) (\d+) \| form (\d+) (\d+) \| dEa (\S+) \| dE (\S+)(.*)", rep)
    res["nuat"] = None
    if m:
        site, b1, b2, f1, f2 = (int(m.group(k)) for k in range(1, 6))
        cap = re.search(r"capture (\d+) ([\d.]+)", m.group(8))
        nn = re.search(r"\| nu (.+?)\s*$", m.group(8))
        cc = re.search(r"close call: (\d+)-(\d+) \+([\d.]+)", m.group(8))
        mp = re.search(r"NUPROD\|\s*(\S+) \| (\S+) \| (\S+)", rep + "\n" + read(os.path.join(d, name + "_nuattack.txt")))
        res["nuprod"] = dict(smiles=mp.group(1), lost=mp.group(2) == "lostH",
                             pic=os.path.join(d, mp.group(3)) if mp.group(3) != "-" else "") if mp else None
        res["nuat"] = dict(site=site, brk=(b1, b2) if b1 and b2 else None, form=(f1, f2) if f1 and f2 else None,
                           dEa=m.group(6), dE=m.group(7), capture=(int(cap.group(1)), cap.group(2)) if cap else None,
                           nu=nn.group(1) if nn else "none", steric="steric warning" in m.group(8),
                           close=(int(cc.group(1)), int(cc.group(2)), cc.group(3)) if cc else None)
    res["energy"] = xyz_energy(os.path.join(d, "xtbopt.xyz"))
    ca = read(os.path.join(d, name + "_cascade.txt"))
    res["cascade"] = [[x.strip() for x in l.split("|")[1:]] for l in ca.splitlines() if l.startswith("CASCADE|")]
    m = re.search(r"FINAL\|\s*(.+)", ca)
    res["final"] = m.group(1).strip() if m else ""
    m = re.search(r"OVERALL\|\s*broken (.*?) \| formed (.*?) \| order (.*?) \| H (.*?) \| (\S+) \| pic (\S+)", ca)
    res["overall"] = dict(broken=m.group(1), formed=m.group(2), order=m.group(3), h=m.group(4), product=m.group(5)) if m else None
    res["overall_pic"] = os.path.join(d, m.group(6)) if m and m.group(6) != "-" and os.path.exists(os.path.join(d, m.group(6))) else ""
    res["cnotes"] = [l.split("|", 1)[1].strip() for l in ca.splitlines() if l.startswith("CNOTE|")]
    mpc = re.search(r"NUPROD\|\s*(\S+) \| (\S+) \| (\S+)", ca)        # product at the end of a radical sequence
    if mpc:
        res["nuprod"] = dict(smiles=mpc.group(1), lost=mpc.group(2) == "lostH",
                             pic=os.path.join(d, mpc.group(3)) if mpc.group(3) != "-" else "", seq=True)
    dft = read(os.path.join(d, name + "_dft.txt"))
    res["dft"] = [[x.strip() for x in l.split("|")[1:]] for l in dft.splitlines() if l.startswith("DFT|")]
    res["e1"] = [[x.strip() for x in l.split("|")[1:]] for l in ca.splitlines() if l.startswith("E1|")]
    pics = sorted(f for f in os.listdir(d) if f.startswith(name + "_break") and f.endswith(".png"))
    res["pictures"] = [os.path.join(d, f) for f in pics]
    res["ok"] = bool(rep) and not res["error"]
    return res


def classify(smiles):
    out = subprocess.run([sys.executable, os.path.join(SCRIPTS, "classify.py"), smiles],
                         capture_output=True, text=True).stdout.split()
    if not out or out[0] == "BAD":
        return None
    v = [int(x) for x in out[:4]] + [0] * (4 - len(out[:4]))
    return dict(charge=v[0], unpaired=v[1], cyclopropanes=v[2], metals=v[3])


SCENARIOS = ["neutral", "cation", "radical_cation"]
SCEN_TITLE = {"neutral": "neutral molecule", "cation": "cation (acid / electrophile adds)",
              "radical_cation": "radical cation (SET oxidation)"}


def _script(args, cwd, env_extra=None, timeout=None):
    env = os.environ.copy()
    env.update(env_extra or {})
    return subprocess.run([sys.executable, os.path.join(SCRIPTS, "scenarios.py")] + list(args), cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)


def run_scenarios(smiles, name, job, solvent="", want=None, cpus=1, timeout=1800, log=print, site_max=None,
                  nu="none"):
    """the five-scenario analysis of a neutral molecule (same layout as the cluster)"""
    want = [w for w in (want or SCENARIOS) if w in SCENARIOS]
    t_end = time.time() + timeout
    left = lambda: max(30, t_end - time.time())
    xtb = os.environ.get("XTB") or shutil.which("xtb") or "xtb"
    env = dict(SOLV=solvent or "", OMP_NUM_THREADS=str(cpus), OMP_STACKSIZE="2G")
    if site_max is not None:
        env["SITE_MAX"] = str(site_max)
    out = dict(mode="scenarios", folder=job, scenarios={}, ranking="", results=[])
    for k, sc in enumerate(want):
        log("scenario %d of %d: %s" % (k + 1, len(want), SCEN_TITLE[sc]))
        sub = lambda s_, sc=sc: log("   [%s] %s" % (sc, s_))
        if time.time() > t_end:
            out["scenarios"][sc] = dict(error="time limit reached before this scenario started")
            continue
        if sc == "neutral":
            r = run_molecule(smiles, name + "_neutral", job, solvent, cpus, left(), sub, outdir="neutral", scen="neutral",
                             nu=nu)
            out["scenarios"][sc] = dict(results=[r])
        elif sc == "radical_cation":
            r = run_molecule(smiles, name + "_radcat", job, solvent, cpus, left(), sub, outdir=sc,
                             chg=1, uhf=1, scen=sc, nu=nu)
            out["scenarios"][sc] = dict(results=[r])
        else:
            d = os.path.join(job, sc)
            os.makedirs(d, exist_ok=True)
            sub("screening every protonation site")
            try:
                _script(["screen", sc, name, smiles, xtb], d, env, timeout=left())
            except subprocess.TimeoutExpired:
                out["scenarios"][sc] = dict(error="time limit reached during the site screening")
                continue
            res = []
            sel = os.path.join(d, "%s_%s_selected.txt" % (name, sc))
            rows = [l.split() for l in read(sel).splitlines() if len(l.split()) == 3]
            for i, (folder, run, smi) in enumerate(rows):
                sub("site %d of %d (%s): full analysis" % (i + 1, len(rows), folder))
                res.append(run_molecule(smi, run, d, solvent, cpus, left(),
                                        lambda s_, sc=sc, f=folder: log("   [%s/%s] %s" % (sc, f, s_)),
                                        outdir=folder, scen=sc, start_xyz=os.path.join(d, folder, "screen.xyz"), nu=nu))
            _script(["finish", sc, name], d, env)
            for r in res:
                r["report"] = read(os.path.join(r["folder"], r["name"] + "_report.txt")) or r.get("report", "")
            out["scenarios"][sc] = dict(results=res, ranking=read(os.path.join(d, "%s_%s_ranking.txt" % (name, sc))))
        out["results"] += out["scenarios"][sc].get("results", [])
    _script(["rank", job, name], job, env)
    out["ranking"] = read(os.path.join(job, "neutral", name + "_ranking.txt"))
    for r in out["scenarios"].get("neutral", {}).get("results", []):
        r["report"] = read(os.path.join(r["folder"], r["name"] + "_report.txt")) or r.get("report", "")
    return out


def effective_smiles(smiles):
    """the SMILES actually analysed: if only one stereo arrangement is possible, that fully specified one"""
    try:
        sys.path.insert(0, SCRIPTS)
        import stereo
        iso = stereo.isomers(smiles)[0]
        return iso[0]["smiles"] if len(iso) == 1 else smiles
    except Exception:
        return smiles


def lewis_targets(smiles, la):
    """the adducts / onium ions that will be analysed (each face separately), as (label, smiles)"""
    sys.path.insert(0, SCRIPTS)
    import lewis, stereo
    out = []
    for d in lewis.sites(smiles, la):
        iso = stereo.isomers(d["smiles"])[0]
        if len(iso) > 1:
            out += [dict(label="%s_%s" % (d["label"], x["label"]), smiles=x["smiles"]) for x in iso]
        else:
            out.append(dict(label=d["label"], smiles=iso[0]["smiles"] if iso else d["smiles"]))
    return out


def run_lewis(smiles, name, job, solvent, cpus, timeout, log, nu, la, only=None):
    """explicit Lewis acid: the acid bound to every Lewis-basic site, each adduct analysed as one species"""
    sys.path.insert(0, SCRIPTS)
    import lewis
    st_ = lewis.sites(smiles, la)
    if not st_:
        return None
    os.makedirs(job, exist_ok=True)
    t_end = time.time() + timeout
    subs = []
    todo = lewis_targets(smiles, la)               # each adduct / onium ion, split into its faces (exo / endo)
    if only:
        sel = [d for d in todo if d["label"] in only]
        if not sel:
            log("note: the chosen sites/faces (%s) were not found; all are analysed" % ", ".join(only))
        todo = sel or todo
    word = "adduct" if lewis.norm(la) in lewis.LAS else "onium ion"
    for k, d in enumerate(todo):
        log("%s %d of %d: %s" % (word, k + 1, len(todo), d["label"]))
        lg = lambda s_, lab=d["label"]: log("  [%s] %s" % (lab, s_))
        r = run_molecule(d["smiles"], "%s_%s" % (name, d["label"]), job, solvent, cpus, max(60, t_end - time.time()), lg,
                         outdir=d["label"], nu=nu)
        subs.append(dict(mode="single", folder=r["folder"], results=[r], label=d["label"], smiles=d["smiles"]))
    summ = lewis.summary(job, name, la)
    try:
        lewis.annotate(job, la)                     # one comparison line in each adduct's report (for the download)
    except Exception:
        pass
    _k = lewis.norm(la)
    head = ("**%s adducts** (the Lewis acid bound to each Lewis-basic site; lowest rel. E = preferred site)" % lewis.label(la)
            if _k in lewis.LAS else "**%s onium ions** (the electrophile added across each C=C, both faces; then the "
            "nucleophile you chose)" % lewis.label(la))
    return dict(mode="stereo", heading=head, folder=job, isomers=subs, summary=summ,
                results=[r for s_ in subs for r in s_["results"]])


def run_job(smiles, name, base, solvent="", screen=True, cpus=1, timeout=900, log=print, scenarios=None, site_max=None,
            nu="none", isomers=None, la=None, la_sites=None):
    """like run.sh: a neutral, closed-shell molecule without metals -> three scenarios (neutral, cation,
    radical cation; scenarios=None: all; [] or screen=False: the neutral molecule only); cations, radical
    cations and radicals = one species; anions are not analysed"""
    name = clean_name(name)
    c = classify(smiles)
    if c is None:
        return dict(mode="error", error="RDKit could not read the SMILES")
    if c["charge"] < 0:
        return dict(mode="error", error="This is an anion (charge %d). Anions are not analysed by this tool." % c["charge"])
    job = os.path.join(base, name)
    os.makedirs(job, exist_ok=True)
    # stereochemistry not fully specified: every buildable isomer, one folder each (isomers=[]: SMILES as given)
    if isomers != []:
        sys.path.insert(0, SCRIPTS)
        import stereo
        iso, bad, nun = stereo.isomers(smiles)
        if isomers:
            iso = [d for d in iso if d["label"] in isomers]
        if len(iso) == 1 and not isomers:         # only one possible: use it as the molecule, no isomer level
            smiles, iso = iso[0]["smiles"], []
        if iso:
            open(os.path.join(job, ".stereo_note"), "w").write(
                "%d unspecified stereocenter(s): %d isomer(s) analysed%s" % (nun, len(iso), (
                    ", %d combination(s) impossible in 3D and dropped" % bad) if bad > 0 else ""))
            t_end = time.time() + timeout
            subs = []
            for k, d in enumerate(iso):
                log("stereoisomer %d of %d: %s" % (k + 1, len(iso), d["label"]))
                lg = lambda s_, lab=d["label"]: log("  [%s] %s" % (lab, s_))
                left = max(60, t_end - time.time())
                nm2 = "%s_%s" % (name, d["label"])
                sub = None
                if la and c["charge"] == 0 and c["unpaired"] == 0:
                    sub = run_lewis(d["smiles"], nm2, os.path.join(job, d["label"]), solvent, cpus, left, lg, nu, la,
                                    la_sites)
                if sub is not None:
                    pass
                elif c["charge"] == 0 and c["unpaired"] == 0 and c["metals"] == 0 and screen and scenarios != []:
                    sub = run_scenarios(d["smiles"], nm2, os.path.join(job, d["label"]), solvent, scenarios, cpus, left,
                                        lg, site_max, nu)
                else:
                    r = run_molecule(d["smiles"], nm2, job, solvent, cpus, left, lg, outdir=d["label"], nu=nu)
                    sub = dict(mode="single", folder=r["folder"], results=[r])
                sub.update(label=d["label"], smiles=d["smiles"])
                subs.append(sub)
            summ = stereo.summary(job, name, write=False)
            return dict(mode="stereo", folder=job, isomers=subs, summary=summ, classify=c,
                        results=[r for s_ in subs for r in s_.get("results", [])])
    if la and c["charge"] == 0 and c["unpaired"] == 0:
        jl = run_lewis(smiles, name, job, solvent, cpus, timeout, log, nu, la, la_sites)
        if jl is not None:
            jl["classify"] = c
            return jl
        log("no Lewis-basic site (O or N) found: analysed without the Lewis acid")
    if c["charge"] == 0 and c["unpaired"] == 0 and c["metals"] == 0 and screen and scenarios != []:
        out = run_scenarios(smiles, name, job, solvent, scenarios, cpus, timeout, log, site_max, nu)
        out["classify"] = c
        return out
    r = run_molecule(smiles, name, base, solvent, cpus, timeout, log, nu=nu)
    return dict(mode="single", folder=r["folder"], results=[r], classify=c)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:]]
    outdir, cpus = os.getcwd(), os.cpu_count() or 1
    if "--outdir" in a:
        k = a.index("--outdir"); outdir = a[k + 1]; del a[k:k + 2]
    if "--cpus" in a:
        k = a.index("--cpus"); cpus = int(a[k + 1]); del a[k:k + 2]
    nu = "none"
    if "--nu" in a:
        k = a.index("--nu"); nu = a[k + 1]; del a[k:k + 2]
    jout = None
    if "--json" in a:
        k = a.index("--json"); jout = a[k + 1]; del a[k:k + 2]
    single = "--single" in a
    if single:
        a.remove("--single")
    las = None
    if "--la-sites" in a:
        k = a.index("--la-sites"); las = [w for w in a[k + 1].split(",") if w]; del a[k:k + 2]
    la = None
    if "--la" in a:
        k = a.index("--la"); la = a[k + 1]; del a[k:k + 2]
        la = None if la.lower() in ("h", "h+", "none", "") else la
    isos = None
    if "--isomers" in a:
        k = a.index("--isomers"); v = a[k + 1]; del a[k:k + 2]
        isos = [] if v == "asgiven" else (None if v == "all" else [w for w in v.split(",") if w])
    only = None
    if "--only" in a:
        k = a.index("--only"); only = [w.strip() for w in a[k + 1].split(",") if w.strip()]; del a[k:k + 2]
    if len(a) < 2:
        sys.exit(__doc__)
    solv = norm_solvent(a[2] if len(a) > 2 else "")
    if solv is None:
        sys.exit("Unknown solvent '%s'. Choose from: %s (or leave it out for gas phase)" % (a[2], " ".join(SOLVENTS)))
    ensure_binaries(want_gxtb=os.environ.get("AUTO_GXTB", "1") == "1")
    sys.path.insert(0, SCRIPTS)
    import nucleophiles
    try:
        nucleophiles.resolve(nu, os.environ.get("XTB"))
    except ValueError as err:
        sys.exit("Nucleophile: %s" % err)
    job = run_job(a[0], a[1], outdir, solv, cpus=cpus, timeout=int(os.environ.get("JOB_TIMEOUT", "86400")),
                  scenarios=[] if single else only, nu=nu, isomers=isos, la=la, la_sites=las,
                  site_max=int(os.environ["SITE_MAX"]) if os.environ.get("SITE_MAX") else None)
    if jout:
        import json
        json.dump(job, open(jout, "w"), default=str)
    if job["mode"] == "error":
        sys.exit(job["error"])
    if job["mode"] == "scenarios":
        print(job["ranking"])
    if job["mode"] == "stereo":
        print(job["summary"])
    for r in job["results"]:
        print("%s: %s" % (r["name"], r.get("pred") or r.get("error") or "done"))
    print("Results: %s" % job["folder"])
