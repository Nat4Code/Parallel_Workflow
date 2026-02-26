#!/usr/bin/env python3
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="results")
    ap.add_argument("--write-csv", type=str, default="best_results.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.indir, "task_*.json")))
    if not paths:
        raise SystemExit(f"No results found in {args.indir}/task_*.json")

    rows = []
    best = None

    for p in paths:
        with open(p, "r") as f:
            d = json.load(f)
        tid = d["task_id"]
        ix = d["tile"]["ix"]
        iy = d["tile"]["iy"]
        bf = d["best"]["f"]
        bx = d["best"]["x"]
        by = d["best"]["y"]
        rows.append((tid, ix, iy, bx, by, bf, p))
        if best is None or bf < best[5]:
            best = (tid, ix, iy, bx, by, bf, p)

    # write CSV:
    with open(args.write_csv, "w") as f:
        f.write("task_id,ix,iy,best_x,best_y,best_f,path\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.12g},{r[4]:.12g},{r[5]:.12g},{r[6]}\n")

    tid, ix, iy, bx, by, bf, p = best
    print("=== GLOBAL BEST ===")
    print(f"task_id={tid} tile=({ix},{iy}) best_f={bf:.12g} best_xy=({bx:.12g},{by:.12g})")
    print(f"from: {p}")
    print(f"wrote: {args.write_csv}")


if __name__ == "__main__":
    main()