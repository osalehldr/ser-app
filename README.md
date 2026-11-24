---
title: Reactivity Predictor
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Reactivity Predictor

Web app (Streamlit) and command-line "local mode" for the GFN2-xTB reactivity workflow:
structure -> xtb optimization -> cyclopropane A/B analysis -> Fukui / descriptors ->
rearrangement & scission scans (imposed activation) -> report + bond picture.
No Slurm or HPC needed: one molecule takes about 10-60 s on one CPU core.

A neutral SMILES is analysed as three species (neutral, cation, radical cation) with a ranking
of which forms more easily; a cation, radical cation or radical SMILES as that one species
(anions are not analysed). The nucleophile for the attack scans is selectable.

Folder contents: `app.py` (web page), `engine.py` (runs the analysis), `scripts/` (the analysis
scripts), `requirements.txt`, `packages.txt`, `Dockerfile`.

## Put it online for free (Streamlit Community Cloud)

1. Make a free GitHub account and create a **new repository** (Public), e.g. `reactivity-screen`.
2. On the repository page: **Add file -> Upload files**, and drag in everything from this folder:
   `app.py`, `engine.py`, `requirements.txt`, `packages.txt`, `README.md`, `Dockerfile`
   and the whole `scripts` folder. Click **Commit changes**.
3. Go to share.streamlit.io, sign in with GitHub, click **Create app**, choose your repository,
   branch `main`, main file `app.py`. Under **Advanced settings** pick Python 3.12. Click **Deploy**.
4. The first start takes a few minutes (it installs the Python packages and downloads xtb and g-xTB
   by itself). Your app is then live at `https://<name>.streamlit.app`.

Updating: change a file in the GitHub repository and the app updates itself.
Privacy later: app **Settings -> Sharing** -> "Only specific people can view this app", then invite
people by email.

## Command line (local mode): a cluster compute node or any Linux/Mac computer

    export XTB=/path/to/xtb                       # optional: downloaded automatically if missing
    python3 engine.py "SMILES" name acetonitrile --cpus 4

Results go into `./name/` (same files as the cluster pipeline).

## Web page on your own computer (Linux, Mac, or Windows via WSL)

    pip install -r requirements.txt
    streamlit run app.py

then open http://localhost:8501.

## Web page with Docker (any server)

    docker build -t reactivity-screen .
    docker run -p 7860:7860 reactivity-screen

## Server settings (environment variables)

`MAX_JOBS` (calculations at the same time, default 1), `JOB_CPUS`, `MAX_HEAVY` (largest molecule,
default 60 heavy atoms), `JOB_TIMEOUT` (seconds, default 900), `AUTO_GXTB` (1 = download g-xTB if missing).
Finished runs are deleted after 24 h.

## Methods and limits

GFN2-xTB (Grimme group) geometries, ALPB implicit solvent, optional g-xTB single-point refinement,
relaxed coordinate scans in the spirit of imposed activation (IACTA, Chem. Sci. 2022, 13, 13857).
It is a screening method: barriers are upper-bound estimates along one coordinate and all
predictions should be confirmed with DFT transition-state calculations.
