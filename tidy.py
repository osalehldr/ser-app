#!/usr/bin/env python3
"""
tidy.py FOLDER  -- last step of a run: keep the main results on top, move everything else to details/.

In every folder that holds a report (one per species / isomer / site):
   NAME_REPORT.txt    the report                        (was NAME_report.txt)
   NAME_PICTURE.png   the predicted outcome             (was NAME_break.png)
   NAME_PRODUCT.png   the product with the nucleophile  (was NAME_nuproduct.png)
   NAME_OVERALL.png   start -> final of a sequence      (was NAME_overall.png)
everything else (structures, xtb outputs, scans, logs) -> details/. Summary files of a whole molecule
(NAME_ranking.txt -> NAME_RANKING.txt) stay on top. Stereo comparison files are removed.
"""
import os, sys, glob, shutil

MAIN = (("_report.txt", "_REPORT.txt"), ("_break.png", "_PICTURE.png"), ("_nuproduct.png", "_PRODUCT.png"),
        ("_overall.png", "_OVERALL.png"))


def has_report(d):
    return bool(glob.glob(os.path.join(d, "*_report.txt")) or glob.glob(os.path.join(d, "*_REPORT.txt")))


def species_dirs(top):
    """folders that hold a report, not inside details/"""
    out = []
    for root, dirs, files in os.walk(top):
        dirs[:] = [x for x in dirs if x != "details"]
        if has_report(root):
            out.append(root)
    return out


def tidy_species(d, sp):
    rep = (glob.glob(os.path.join(d, "*_report.txt")) or glob.glob(os.path.join(d, "*_REPORT.txt")))[0]
    name = os.path.basename(rep).rsplit("_", 1)[0]
    keep = set()
    for old, new in MAIN:
        src, dst = os.path.join(d, name + old), os.path.join(d, name + new)
        if os.path.exists(src):
            os.replace(src, dst)
        if old == "_break.png" and not os.path.exists(dst):          # several pictures: keep the first
            alt = sorted(glob.glob(os.path.join(d, name + "_break_*.png")))
            if alt:
                os.replace(alt[0], dst)
        if os.path.exists(dst):
            keep.add(os.path.basename(dst))
    det = os.path.join(d, "details")
    os.makedirs(det, exist_ok=True)
    inner = set(os.path.relpath(x, d).split(os.sep)[0] for x in sp if x != d and x.startswith(d + os.sep))
    for f in os.listdir(d):
        if f in keep or f == "details" or f in inner:
            continue
        if f.endswith(("_RANKING.txt",)):
            continue
        shutil.move(os.path.join(d, f), os.path.join(det, f))


def main(top):
    top = os.path.abspath(top)
    for f in glob.glob(os.path.join(top, "*_stereo.txt")) + glob.glob(os.path.join(top, ".stereo_note")):
        os.remove(f)
    for f in glob.glob(os.path.join(top, "**", "*_ranking.txt"), recursive=True):
        if "/details/" not in f:
            os.replace(f, f[:-len("_ranking.txt")] + "_RANKING.txt")
    sp = species_dirs(top)
    for d in sorted(sp, key=len, reverse=True):                 # deepest first
        tidy_species(d, sp)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        if os.path.isdir(t):
            main(t)
