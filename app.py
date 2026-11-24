"""
Reactivity Predictor web app (Streamlit).   Run:  streamlit run app.py
Settings (environment variables, all optional):
    XTB          path to xtb               (default: xtb on the PATH)
    GXTB         g-xTB-enabled xtb binary  (enables the "refine with g-xTB" option)
    MAX_JOBS     calculations allowed at the same time      (default 1)
    MAX_HEAVY    largest molecule accepted, heavy atoms     (default 60)
    JOB_TIMEOUT  seconds per job                            (default 1800)
    SITE_MAX     protonation/deprotonation sites analysed in full per scenario (default 2)
    JOB_CPUS     CPU cores per calculation                  (default 2)
    AUTO_GXTB    download g-xTB automatically if missing    (default 1 = yes)
"""
import os, io, re, json, time, uuid, shutil, zipfile, threading, subprocess
import streamlit as st
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import engine
import sys as _sys
_sys.path.insert(0, engine.SCRIPTS)
import nucleophiles

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
MAX_JOBS = int(os.environ.get("MAX_JOBS", "1"))
MAX_HEAVY = int(os.environ.get("MAX_HEAVY", "60"))
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "1800"))
SITE_MAX = int(os.environ.get("SITE_MAX", "2"))
CASCADE_MAX = int(os.environ.get("CASCADE_MAX", "3"))
CPUS = int(os.environ.get("JOB_CPUS", "2"))
ORGANIC = {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}

st.set_page_config(page_title="Reactivity Predictor", layout="wide")


@st.cache_resource(show_spinner="Setting up xtb (only on the first start of the server) ...")
def binaries():
    return engine.ensure_binaries(log=lambda s: None, want_gxtb=os.environ.get("AUTO_GXTB", "1") == "1")


XTB, GXTB = binaries()
HAVE_GXTB = bool(GXTB)
if not XTB:
    st.error("xtb is not available on this server, so calculations cannot run. "
             "Set the XTB environment variable to the xtb program.")


@st.cache_resource
def job_registry():
    """running calculations (separate processes, so Stop can end them at once)"""
    return dict(lock=threading.Lock(), procs=[])


def _alive(entry):
    p, t0 = entry
    if p.poll() is not None:
        return False
    if time.time() - t0 > TIMEOUT + 300:          # stale: over the time limit -> end it
        _kill(p)
        return False
    return True


def _kill(p):
    import signal
    try:
        os.killpg(p.pid, signal.SIGTERM)
        time.sleep(1.0)
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def cleanup_old_runs(hours=24):
    if not os.path.isdir(RUNS):
        return
    now = time.time()
    for d in os.listdir(RUNS):
        p = os.path.join(RUNS, d)
        if os.path.isdir(p) and now - os.path.getmtime(p) > hours * 3600:
            shutil.rmtree(p, ignore_errors=True)


def check_structure(smiles):
    """same checks as radcheck/catcheck, returns (mol, facts, warnings)"""
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return None, {}, ["RDKit could not read this SMILES."]
    warn = []

    def real_radical(a):
        return (a.GetNumRadicalElectrons() > 0 and a.GetSymbol() in ORGANIC
                and all(n.GetSymbol() in ORGANIC for n in a.GetNeighbors()))
    q = Chem.GetFormalCharge(m)
    u = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms() if real_radical(a))
    frags = len(Chem.GetMolFrags(m))
    r3 = sum(1 for r in m.GetRingInfo().AtomRings() if len(r) == 3)
    metals = sorted({a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() not in ORGANIC})
    facts = {"charge": "%+d" % q, "unpaired electrons": u, "fragments": frags, "cyclopropanes": r3,
             "heavy atoms": m.GetNumAtoms(), "metal / Lewis acid": ", ".join(metals) or "none"}
    if frags > 1:
        warn.append("%d separate fragments ('.' in the SMILES): something extra was copied." % frags)
    if u > 1:
        warn.append("%d unpaired electrons: a bracket carbon is probably missing an H (e.g. [C] should be [CH2])." % u)
    for a in m.GetAtoms():
        if a.GetSymbol() == "C" and a.GetFormalCharge() > 0 and a.GetTotalValence() != 3:
            warn.append("C%d is a cation with %d attachments (needs 3): check its H count." % (a.GetIdx() + 1,
                                                                                          a.GetTotalValence()))
    unspec = [i + 1 for i, t in Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
              if t == "?"]
    if unspec:
        warn.append("Stereocenters without @/@@ (atoms %s): a random arrangement will be used. In cages the "
                    "face (exo/endo) changes the result, so specify them." % ", ".join(map(str, unspec)))
    if m.GetNumAtoms() > MAX_HEAVY:
        warn.append("This molecule has %d heavy atoms; the limit on this server is %d." % (m.GetNumAtoms(), MAX_HEAVY))
    return m, facts, warn


def depict(m, size=420):
    m2 = Chem.Mol(m)
    for a in m2.GetAtoms():
        a.SetProp("atomNote", str(a.GetIdx() + 1))
    AllChem.Compute2DCoords(m2)
    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m2)
    d.FinishDrawing()
    return d.GetDrawingText()


def zip_folder(folder):
    """download in the same tidy layout as the cluster: NAME_REPORT.txt / NAME_PICTURE.png / NAME_PRODUCT.png /
    NAME_OVERALL.png on top of each species folder, everything else in its details/ folder"""
    import glob as _g
    main = (("_report.txt", "_REPORT.txt"), ("_break.png", "_PICTURE.png"), ("_nuproduct.png", "_PRODUCT.png"),
            ("_overall.png", "_OVERALL.png"), ("_ranking.txt", "_RANKING.txt"))
    spec = set(r for r, _, fs in os.walk(folder) if any(f.endswith("_report.txt") for f in fs))
    base = os.path.dirname(folder)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            owner = root
            while owner not in spec and owner.startswith(folder) and owner != folder:
                owner = os.path.dirname(owner)
            for f in files:
                if f.startswith(".") or f.endswith(("_stereo.txt", "_lewis.txt")) or f in ("job.json", "progress.log"):
                    continue
                p = os.path.join(root, f)
                arc = None
                if root in spec or root == folder:
                    for old, new in main:
                        if f.endswith(old) and (old != "_break.png" or not _g.glob(os.path.join(root, "*_break_*.png"))
                                                or True):
                            arc = os.path.join(os.path.relpath(root, base), f[:-len(old)] + new)
                            break
                if arc is None:
                    if owner in spec:
                        arc = os.path.join(os.path.relpath(owner, base), "details", os.path.relpath(p, owner))
                    else:
                        arc = os.path.relpath(p, base)
                z.write(p, arc)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _isomers(smi):
    import stereo
    return stereo.isomers(smi)


@st.cache_data(show_spinner=False)
def _lewis_targets(smi, la):
    try:
        return engine.lewis_targets(engine.effective_smiles(smi), la)
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _check_nu(spec):
    try:
        nucleophiles.resolve(spec, XTB)
        return ""
    except ValueError as err_:
        return str(err_)


ENERGY_NOTE = ("Rough xtb screening values: use them to rank pathways against each other, not as real "
               "barriers. Typical error is several kcal/mol (often ±5 or more); scan barriers are upper bounds "
               "along one coordinate, with no entropy, explicit solvent or counter-ions, and differences under "
               "~3 kcal/mol are not meaningful. For a reliable number, check the step with DFT (run_ts).")


def energy_line(dEa, dE):
    st.markdown("xtb estimate (rough): barrier **~%s**, reaction energy **~%s** kcal/mol" % (dEa, dE),
                help=ENERGY_NOTE)
    st.caption("Rough screening values, for ranking pathways only (error often ±5 kcal/mol or more); "
               "check key steps with DFT (run_ts).")


def show_nuat(nu):
    """result of the explicit nucleophile-attack scans (cations)"""
    if not nu:
        return
    who = (nu.get("nu") or "water")
    who = who[0].upper() + who[1:]
    if nu["brk"]:
        txt = "**%s attacks C%d:** C%d–C%d breaks" % (who, nu["site"], nu["brk"][0], nu["brk"][1])
        if nu["form"]:
            txt += ", C%d–C%d forms (new ring)" % tuple(nu["form"])
        st.success(txt)
    else:
        st.success("**%s attacks C%d** (direct capture, no ring opening)" % (who, nu["site"]))
    energy_line(nu["dEa"], nu["dE"])
    if nu["capture"]:
        st.caption("Direct capture at C%s (barrier %s kcal/mol) is within the accuracy of the method, so the "
                   "site of attack is not fully resolved; the bond that breaks is." % tuple(nu["capture"]))
    if nu.get("steric"):
        st.warning("Steric warning: %s is a bulky nucleophile attacking a crowded carbon. This method underestimates "
                   "steric hindrance here (e.g. tert-butanol in Tenaglia 1996 attacks the rearranged position instead). "
                   "Treat the direct capture as uncertain and check the ring-opening alternative in the report."
                   % (nu.get("nu") or "the nucleophile"))
    if nu.get("close"):
        st.warning("Close call: the C%d–C%d bond opens only %s kcal/mol higher. Both ring openings are within the "
                   "accuracy of this method; decide with DFT or compare nucleophiles." % tuple(nu["close"]))


def show_prediction(pred, report=""):
    if not pred:
        st.info("No rearrangement prediction for this species (neutral closed-shell molecules are not scanned).")
        return
    parts = [p.strip() for p in pred.split("|")]
    st.success("**Prediction:** " + parts[0] + ((" — " + parts[1]) if len(parts) > 1 else ""))
    nums = {}
    for p in parts[2:]:
        k = p.split()
        if len(k) == 2:
            nums[k[0]] = k[1]
    if "it happened during the optimization itself" in report:
        st.caption("This change happened spontaneously during the geometry optimization (no barrier at this level).")
    elif parts[0].startswith("no rearrangement"):
        pass
    elif nums:
        c1, c2 = st.columns(2)
        energy_line(nums.get("dEa", "-"), nums.get("dE", "-"))


# ------------------------------------------------------------------ page
cleanup_old_runs()
st.title("Reactivity Predictor")
st.caption("Fast GFN2-xTB screen: which bond breaks, whether the cation/radical rearranges, and where a "
           "nucleophile attacks. Results are hypotheses to prioritise DFT, not final answers.")

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Paste a SMILES, usually the **neutral starting material**.\n"
        "2. Choose the **solvent**, the **activation** (H+, a Lewis acid such as BF3, or an electrophile such as "
        "Br+ from NBS) and the **nucleophile**.\n"
        "3. **Check structure**, then **Run analysis**.\n\n"
        "**What it does:** it builds the reactive intermediate from your conditions (protonated cation, Lewis acid "
        "adduct, bromonium/iodonium ion on each C=C, or radical cation by SET oxidation), tests whether it "
        "rearranges or opens a ring, and where your nucleophile attacks it.\n\n"
        "**When to enter the intermediate yourself:** radicals made other ways (Bu3SnH, homolysis), cations from "
        "a leaving group (solvolysis), or conditions not in the menus: paste the cation `[CH2+]...` or radical "
        "`[CH2]...` directly. Anions are not analysed.\n\n"
        "**Stereochemistry:** specifying it in the SMILES (`@`/`@@`, e.g. ChemDraw → Copy As → SMILES with "
        "wedges drawn) gives the most reliable result, but it is not required. If stereocenters are left "
        "undetermined, every possible isomer is built (impossible ones are dropped) and listed, e.g. **exo** / "
        "**endo**, so you can choose which to analyse.\n\n"
        "**Results:** where the nucleophile attacks and a picture of the **product**, a picture of the predicted "
        "outcome (**green** = bond that breaks, **orange** = new bond, **blue** = where the nucleophile ends up), "
        "and a **DFT check** list of steps worth confirming (run_ts on the cluster).\n\n"
        "**Energies** are rough xtb screening values: use them to rank pathways, not as real barriers (error often "
        "±5 kcal/mol or more).")
    st.caption("Limits on this server: %d heavy atoms, %d min per job, %d job(s) at a time."
               % (MAX_HEAVY, TIMEOUT // 60, MAX_JOBS))

col_in, col_pic = st.columns([3, 2])
with col_in:
    smiles = st.text_input("SMILES", key="smiles", placeholder="e.g. [CH2+]C1CC1")
    name = st.text_input("Name", key="name", value="mol1", help="Letters, numbers, - and _ only")
    solv_opts = ["gas phase"] + engine.SOLVENTS
    solvent = st.selectbox("Solvent (implicit ALPB model)", solv_opts, index=solv_opts.index("acetonitrile"),
                           help="Ions should be run in the reaction solvent: in the gas phase they can "
                                "rearrange or lose protons artificially.")
    # ---- species detected from the SMILES: decides which options are shown
    _m0 = Chem.MolFromSmiles(smiles.strip()) if smiles.strip() else None
    _q = Chem.GetFormalCharge(_m0) if _m0 is not None else 0
    _nrad = sum(a.GetNumRadicalElectrons() for a in _m0.GetAtoms() if a.GetSymbol() in ORGANIC) if _m0 is not None else 0
    if _m0 is None:
        species = "invalid" if smiles.strip() else ""
    elif _q < 0:
        species = "anion"
    elif _q > 0:
        species = "radical cation" if _nrad else "cation"
    else:
        species = "radical" if _nrad else "neutral"
    _radical = species == "radical"
    if species == "neutral":
        st.info("**Detected: neutral molecule** — analysed as up to three species (choose below).")
    elif species in ("cation", "radical cation", "radical"):
        st.info("**Detected: %s** (charge %s, %d unpaired electron%s) — analysed as this one species."
                % (species, ("%+d" % _q) if _q else "0", _nrad, "" if _nrad == 1 else "s"))
    elif species == "anion":
        st.error("**Detected: anion** (charge %d) — anions are not analysed by this tool." % _q)
    elif species == "invalid":
        st.error("The SMILES could not be read.")
    picked_iso = None
    if species in ("neutral", "cation", "radical cation", "radical"):
        _iso, _bad, _nun = _isomers(smiles.strip())
        if len(_iso) == 1:                          # only one possible arrangement (e.g. a cage): nothing to choose
            _iso = []
        if _iso:
            st.info("**Stereochemistry not fully specified** (%d center%s): %d buildable isomer%s%s. Each is analysed "
                    "separately and compared." % (_nun, "" if _nun == 1 else "s", len(_iso), "" if len(_iso) == 1 else "s",
                                                  (", %d impossible combination%s dropped" % (_bad, "" if _bad == 1 else "s"))
                                                  if _bad > 0 else ""))
            picked_iso = st.multiselect("Isomers to analyse (each adds run time)", [d_["label"] for d_ in _iso],
                                        default=[d_["label"] for d_ in _iso],
                                        help="exo = substituent towards the one-atom bridge of a norbornane-type cage. "
                                             "If you know the stereochemistry, put @/@@ in the SMILES instead.")
            with st.expander("isomer SMILES"):
                for d_ in _iso:
                    st.code("%s   %s" % (d_["label"], d_["smiles"]), language=None)
    scen_labels = {"neutral": "neutral", "cation": "cation", "radical_cation": "radical cation"}
    la_pick = None
    _la_off = species in ("cation", "radical cation", "radical", "anion", "invalid")
    _la = st.selectbox("Activation (acid or electrophile)",
                       ["H+ (Brønsted acid; Lewis acids modelled as H+)", "BF3 (BF3·OEt2), explicit",
                        "AlCl3, explicit", "TMS+ (TMSOTf, TMSI), explicit", "Br+ electrophile (NBS, Br2)",
                        "I+ electrophile (NIS, I2)", "Cl+ electrophile (NCS)", "PhSe+ electrophile (PhSeCl)"],
                       disabled=_la_off, key="activation",
                       help="For a neutral starting material. Lewis acids: bound to every O/N site. Electrophiles: "
                            "added across each C=C as the onium ion (bromonium, iodonium ...), both faces (exo/endo), "
                            "then attacked by the nucleophile you choose. A cation or radical SMILES is analysed as "
                            "it is, so this menu does not apply to it.")
    if _la_off and species in ("cation", "radical cation", "radical"):
        st.caption("Activation applies to neutral starting materials; this SMILES is already a %s and is analysed "
                   "as it is." % species)
    if species == "neutral":
        la_pick = {"BF3": "bf3", "AlC": "alcl3", "TMS": "tms", "Br+": "nbs", "I+ ": "nis", "Cl+": "ncs",
                   "PhS": "phsecl"}.get(_la[:3])
    la_sites = None
    if species == "neutral" and la_pick:
        _tg = _lewis_targets(smiles.strip(), la_pick)
        if _tg:
            la_sites = st.multiselect("Sites / faces to analyse (each adds run time)", [t_["label"] for t_ in _tg],
                                      default=[t_["label"] for t_ in _tg],
                                      help="Electrophiles: each C=C, from both faces (exo/endo). Lewis acids: each O/N "
                                           "site. Untick what you do not need, e.g. the endo face when NBS is known "
                                           "to add exo.")
        else:
            st.warning("No %s found in this molecule: it will be analysed without the activation." % (
                "C=C" if la_pick in ("nbs", "nis", "ncs", "phsecl") else "O/N site"))
    if species == "neutral" and la_pick:
        st.caption("Each adduct / onium ion is analysed as one species (both faces for electrophiles); the "
                   "species choice below does not apply. Pick a nucleophile to trap it.")
        picked = list(scen_labels)
    elif species == "neutral":
        picked = st.multiselect("Species to analyse (neutral molecule)", list(scen_labels),
                                default=list(scen_labels), format_func=lambda k: scen_labels[k],
                                help="Each species adds run time (a medium-sized molecule takes about 2-10 min per "
                                     "species).")
    else:
        picked = list(scen_labels)          # not used: charged / radical input is one species
    screen = bool(picked)
    _lab = lambda k: k if k not in nucleophiles.PRESETS else (
        "no nucleophile (most electrophilic site)" if k == "none" and not _radical else
        "no partner (most reactive radical site)" if k == "none" else "%s  (%s)" % (k, nucleophiles.PRESETS[k][2]))
    if _radical:
        nu_pick = st.selectbox("Radical partner (what the radical reacts with)", nucleophiles.APP_PARTNERS, index=0,
                               format_func=_lab,
                               help="A neutral radical reacts with radical traps, not nucleophiles. 'oxidant' = "
                                    "radical-polar crossover: the radical is oxidized to the cation, which then meets "
                                    "the nucleophile chosen below.")
        nu_spec = nu_pick
        if nu_pick == "oxidant":
            then = st.selectbox("Nucleophile for the cation after oxidation", nucleophiles.APP_NUCLEOPHILES, index=0,
                                format_func=lambda k: "no nucleophile (most electrophilic site)" if k == "none" else _lab(k))
            nu_spec = "oxidant+" + then
    else:
        nu_pick = st.selectbox("Nucleophile (attack scans on cations)", nucleophiles.APP_NUCLEOPHILES +
                               ["custom SMILES ..."], index=0, format_func=_lab,
                               help="'no nucleophile': the report gives the most electrophilic site (where any "
                                    "nucleophile would most likely end up). Otherwise the nucleophile is pulled onto each "
                                    "cyclopropane carbon (backside of each ring bond) and onto the cationic carbon. "
                                    "Alkoxides and hydrides form ion pairs: their barriers are rougher.")
        nu_spec = nu_pick
    if nu_pick == "custom SMILES ...":
        nu_spec = st.text_input("Nucleophile SMILES", key="nu_smiles", placeholder="e.g. CC(C)(C)[OH:1]",
                                help="One neutral molecule or anion. Mark the attacking atom with :1, otherwise the "
                                     "most nucleophilic atom (xtb Fukui f-) is used.").strip()
    nu_err = ""
    if nu_spec:
        nu_err = _check_nu(nu_spec)
        if nu_err:
            st.error("Nucleophile: " + nu_err)
    else:
        nu_err = "enter a nucleophile SMILES"
    do_cascade = False
    do_conf = st.checkbox("Conformer search first (CREST)", value=False,
                          help="Finds the lowest conformer and folded conformers for steps such as 1,5-H transfer or "
                               "ring closure. Recommended for flexible chains; adds a few minutes.")
    use_g = st.checkbox("Refine energies with g-xTB", value=HAVE_GXTB, disabled=not HAVE_GXTB,
                        help="Much better reaction energies than GFN2-xTB alone." if HAVE_GXTB
                        else "g-xTB is not installed on this server.")
    b1, b2 = st.columns(2)
    do_check = b1.button("Check structure", use_container_width=True)
    do_run = b2.button("Run analysis", type="primary", use_container_width=True)

mol, facts, warns = check_structure(smiles.strip()) if smiles.strip() else (None, {}, [])
if smiles.strip() and (do_check or do_run):
    with col_pic:
        if mol is not None:
            st.image(depict(mol), caption="input structure (atom numbers used in the report)")
            st.table({"property": list(facts), "value": [str(v) for v in facts.values()]})
        for w in warns:
            st.warning(w)

if do_run:
    if mol is None:
        st.error("Please enter a valid SMILES first.")
    elif Chem.GetFormalCharge(mol) < 0:
        st.error("This is an anion. Anions are not analysed by this tool.")
    elif nu_err:
        st.error("Please fix the nucleophile first (%s)." % nu_err)
    elif any("limit on this server" in w for w in warns):
        st.error("Molecule too large for this server.")
    else:
        reg = job_registry()
        with reg["lock"]:
            reg["procs"] = [e for e in reg["procs"] if _alive(e)]
            free = len(reg["procs"]) < MAX_JOBS
            proc = None
            if free:
                run_dir = os.path.join(RUNS, time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
                os.makedirs(run_dir)
                env = os.environ.copy()
                env.update(CASCADE="auto", CASCADE_MAX=str(CASCADE_MAX), CONF="off",
                           PYTHONUNBUFFERED="1", JOB_TIMEOUT=str(TIMEOUT), SITE_MAX=str(SITE_MAX), AUTO_GXTB="0")
                if XTB:
                    env["XTB"] = XTB
                if do_conf:
                    _crest = engine.ensure_crest(log=lambda s_: None)
                    if _crest:
                        env["CREST"], env["CONF"] = _crest, "on"
                    else:
                        st.warning("CREST could not be set up on this server; running without the conformer search.")
                if use_g and HAVE_GXTB:
                    env["GXTB"] = GXTB
                else:
                    env.pop("GXTB", None)
                cmd = [_sys.executable, os.path.join(HERE, "engine.py"), smiles.strip(), engine.clean_name(name),
                       "gas" if solvent == "gas phase" else solvent, "--outdir", run_dir, "--cpus", str(CPUS),
                       "--nu", nu_spec or "none", "--json", os.path.join(run_dir, "job.json")]
                cmd += ["--only", ",".join(picked)] if picked else ["--single"]
                if la_pick:
                    cmd += ["--la", la_pick]
                    if la_sites:
                        cmd += ["--la-sites", ",".join(la_sites)]
                if picked_iso is not None:
                    cmd += ["--isomers", ",".join(picked_iso) if picked_iso else "asgiven"]
                logf = open(os.path.join(run_dir, "progress.log"), "w")
                proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                reg["procs"].append((proc, time.time()))
        if not free:
            st.warning("The server is busy with another calculation. Please try again in a minute.")
        else:
            try:
                with st.status("Running the analysis ... (press Stop at the top right to cancel)", expanded=True) as status:
                    t0, shown = time.time(), 0
                    while proc.poll() is None:
                        lines = open(os.path.join(run_dir, "progress.log")).read().splitlines()
                        for l in lines[shown:]:
                            if l.strip() and not l.startswith(("Results:", "====")):
                                status.write(l)
                        shown = len(lines)
                        if time.time() - t0 > TIMEOUT + 60:
                            _kill(proc)
                            break
                        time.sleep(1.0)
                    jf = os.path.join(run_dir, "job.json")
                    if os.path.exists(jf):
                        job = json.load(open(jf))

                        def _fix(o):
                            if isinstance(o, dict):
                                nu_ = o.get("nuat")
                                if isinstance(nu_, dict):
                                    for k_ in ("brk", "form", "capture", "close"):
                                        if isinstance(nu_.get(k_), list):
                                            nu_[k_] = tuple(nu_[k_])
                                for v_ in o.values():
                                    _fix(v_)
                            elif isinstance(o, list):
                                for v_ in o:
                                    _fix(v_)
                        _fix(job)
                        status.update(label="Finished in %.0f s" % (time.time() - t0), state="complete")
                    else:
                        el = time.time() - t0
                        steps = [l.strip() for l in open(os.path.join(run_dir, "progress.log")).read().splitlines()
                                 if re.match(r"^\s*(\[[^\]]+\]\s*)?(\d+[a-c]?/7|stereoisomer|scenario|species)", l)]
                        why = ("it reached the time limit of %d min" % (TIMEOUT // 60) if el >= TIMEOUT
                               else "it stopped unexpectedly after %.0f min (most often the free server running out of "
                                    "memory; try without CREST or with fewer isomers/species, or use the cluster)" % (el / 60))
                        job = dict(mode="error", error="The calculation stopped without results: %s.\nLast step: %s"
                                   % (why, steps[-1] if steps else "(none)"))
                        status.update(label="Stopped", state="error")
                job["solvent"] = solvent
                st.session_state["job"] = job
            finally:
                if proc.poll() is None:            # Stop pressed / page closed: end the calculation now
                    _kill(proc)
                with reg["lock"]:
                    reg["procs"] = [e for e in reg["procs"] if e[0] is not proc and _alive(e)]

job = st.session_state.get("job")
if job:
    st.divider()
    if job["mode"] == "error":
        st.error(job["error"])
    else:
        st.subheader("Results (%s)" % job.get("solvent", ""))

        def show_result(r, key):
            if r.get("error"):
                st.warning(r["error"])
            if r.get("cascade"):                       # radical sequence (cyclizations etc.); cations: off
                st.markdown("**Radical sequence** (each step's radical analysed again)")
                ov = r.get("overall")
                if ov:
                    st.markdown("Overall: bonds broken **%s**; bonds formed **%s**; hydrogens **%s**." % (
                        ov["broken"], ov["formed"], ov["h"]))
                    if r.get("overall_pic"):
                        st.image(r["overall_pic"], caption="start vs final: green = breaks, orange = new bond, "
                                                           "purple = loses H")
                st.table({"step": [c_[0] for c_ in r["cascade"]], "what happens": [c_[1] for c_ in r["cascade"]],
                          "xtb barrier, rough (kcal/mol)": [c_[2] for c_ in r["cascade"]],
                          "species after the step": [c_[4] for c_ in r["cascade"]]})
                if r.get("final"):
                    st.caption("End: " + r["final"].split("|")[-1].strip() + ". Barriers are rough xtb values for "
                               "ranking only.")
            if r.get("nuat"):
                show_nuat(r["nuat"])
            if r.get("nuprod") and r["nuprod"].get("pic"):
                st.image(r["nuprod"]["pic"], width=420,
                         caption="Predicted product%s: %s  (stereo shown = the face attacked in the model; a planar "
                                 "cation attacked from both faces gives both)" % (
                             " (after loss of H+)" if r["nuprod"].get("lost") else "", r["nuprod"]["smiles"]))
                with st.expander("cation alone (no nucleophile)"):
                    show_prediction(r.get("pred", ""), r.get("report", ""))
            else:
                show_prediction(r.get("pred", ""), r.get("report", ""))
            cpic, crep = st.columns([2, 3])
            with cpic:
                for p in r.get("pictures", []):
                    st.image(p)
            with crep:
                st.text_area("report", r.get("report", "") or "(no report)", height=420, key="rep_" + key)
            if r.get("dft"):
                worth = [x for x in r["dft"] if x[1] == "yes"]
                with st.expander("DFT check: %d step%s worth a DFT transition-state calculation" % (
                        len(worth), "" if len(worth) == 1 else "s") if worth else "DFT check (optional)", expanded=bool(worth)):
                    st.table({"#": [x[0] for x in r["dft"]],
                              "DFT": ["worth it" if x[1] == "yes" else "optional" for x in r["dft"]],
                              "why": [x[2] for x in r["dft"]], "step": [x[4] if len(x) > 4 else "" for x in r["dft"]],
                              "reaction (copy for run_ts)": [x[3] for x in r["dft"]]})
                    st.caption('On the cluster: run the same SMILES with run, then "run_ts NAME <number>"; or submit a '
                               'reaction directly: run_ts "REACTANT>>PRODUCT" name solvent (autodE + ORCA, hours per step).')

        def render(jb, kp=""):
            if jb.get("mode") == "error":
                st.error(jb.get("error", ""))
            elif jb.get("mode") == "scenarios":
                if jb.get("ranking"):
                    st.markdown("**Which reactive species forms most easily?**")
                    st.code(jb["ranking"], language=None)
                names = [k for k in engine.SCENARIOS if k in jb["scenarios"]]
                tabs = st.tabs([engine.SCEN_TITLE[k] for k in names])
                for tab, k in zip(tabs, names):
                    with tab:
                        sc = jb["scenarios"][k]
                        if sc.get("error"):
                            st.warning(sc["error"])
                        if sc.get("ranking"):
                            st.code(sc["ranking"], language=None)
                        for i, r in enumerate(sc.get("results", [])):
                            if k == "cation":
                                with st.expander(os.path.basename(r["folder"]), expanded=(i == 0)):
                                    show_result(r, "%s%s_%d" % (kp, k, i))
                            else:
                                show_result(r, kp + k)
            else:
                for r in jb.get("results", []):
                    with st.expander(r["name"], expanded=True):
                        show_result(r, kp + r["name"])

        def render_any(jb, kp=""):
            if jb.get("mode") == "stereo":
                st.markdown(jb.get("heading") or "**Stereoisomers** (the stereochemistry was not fully specified, so every "
                                                 "buildable isomer was analysed)")
                st.code(jb.get("summary", ""), language=None)
                itabs = st.tabs([s_["label"] for s_ in jb["isomers"]])
                for tab, sub in zip(itabs, jb["isomers"]):
                    with tab:
                        st.caption(sub.get("smiles", ""))
                        render_any(sub, kp + sub["label"] + "_")
            else:
                render(jb, kp)

        render_any(job)
        st.download_button("Download all files (.zip)", data=zip_folder(job["folder"]),
                           file_name=os.path.basename(job["folder"]) + ".zip", mime="application/zip")
