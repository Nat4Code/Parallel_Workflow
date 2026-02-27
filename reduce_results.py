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

    best = None   # min f
    worst = None  # max f

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
        if worst is None or bf > worst[5]:
            worst = (tid, ix, iy, bx, by, bf, p)

    # Write CSV
    with open(args.write_csv, "w") as f:
        f.write("task_id,ix,iy,best_x,best_y,best_f,path\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.12g},{r[4]:.12g},{r[5]:.12g},{r[6]}\n")

    print(f"Scanned {len(paths)} result files in: {args.indir}")

    b_tid, b_ix, b_iy, b_x, b_y, b_f, b_path = best
    print("\n=== GLOBAL MIN (best objective) ===")
    print(f"task_id={b_tid} tile=({b_ix},{b_iy}) f={b_f:.12g} xy=({b_x:.12g},{b_y:.12g})")
    print(f"file: {b_path}")

    w_tid, w_ix, w_iy, w_x, w_y, w_f, w_path = worst
    print("\n=== GLOBAL MAX (worst objective) ===")
    print(f"task_id={w_tid} tile=({w_ix},{w_iy}) f={w_f:.12g} xy=({w_x:.12g},{w_y:.12g})")
    print(f"file: {w_path}")

    print(f"\nwrote: {args.write_csv}")


if __name__ == "__main__":
    main()