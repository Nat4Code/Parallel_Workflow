#!/usr/bin/env python3
import argparse
import json
import math
import sys
import os
import random
from dataclasses import dataclass
from typing import Tuple


########################
# Objective + Gradient #
########################
TWO_PI = 2.0 * math.pi

def f_xy(x: float, y: float) -> float:
    '''function we generate (could be any function we want)'''

    return (
        20.0
        + x * x + y * y
        - 10.0 * (math.cos(TWO_PI * x) + math.cos(TWO_PI * y))
        + 0.5 * math.sin(3.0 * x) * math.sin(3.0 * y)
        + 0.1 * x + 0.1 * y
    )

def grad_f_xy(x: float, y: float) -> Tuple[float, float]:
    '''Gradient of the function - could use a numerical
    technique to compute this automatically in the future...'''

    dfdx = (
        2.0 * x
        + 20.0 * math.pi * math.sin(TWO_PI * x)
        + 1.5 * math.cos(3.0 * x) * math.sin(3.0 * y)
        + 0.1
    )
    dfdy = (
        2.0 * y
        + 20.0 * math.pi * math.sin(TWO_PI * y)
        + 1.5 * math.sin(3.0 * x) * math.cos(3.0 * y)
        + 0.1
    )
    return dfdx, dfdy


#####################################################
# physics inspired optimizer:                       #
#  x'' = -∇f/m - gamma * x'                         #
#  (discretized with simple damped velocity update) #
#####################################################
@dataclass
class DynamicsParams:
    dt: float = 0.02
    mass: float = 1.0

    # friction coeff...
    gamma: float = 0.8

    # safety clamp on speed (to keep dynamics stable)
    max_speed: float = 10.0


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def reflect_into_box(x: float, lo: float, hi: float) -> float:
    # reflect boundary condition (more stable than hard clamp for dynamics).
    if x < lo:
        return lo + (lo - x)
    if x > hi:
        return hi - (x - hi)
    return x


def tile_bounds(task_id: int, nx: int, ny: int, xlo: float, xhi: float, ylo: float, yhi: float):
    if task_id < 0 or task_id >= nx * ny:
        raise ValueError(f"task_id {task_id} out of range for nx={nx}, ny={ny}")
    ix = task_id % nx
    iy = task_id // nx
    dx = (xhi - xlo) / nx
    dy = (yhi - ylo) / ny
    tx0 = xlo + ix * dx
    tx1 = tx0 + dx
    ty0 = ylo + iy * dy
    ty1 = ty0 + dy
    return ix, iy, tx0, tx1, ty0, ty1


def coarse_pick_start(tx0: float, tx1: float, ty0: float, ty1: float, gx: int, gy: int, rng: random.Random):
    # small coarse grid; pick best point as start.
    best = None
    for i in range(gx):
        x = tx0 + (i + 0.5) * (tx1 - tx0) / gx
        for j in range(gy):
            y = ty0 + (j + 0.5) * (ty1 - ty0) / gy
            val = f_xy(x, y)
            if best is None or val < best[0]:
                best = (val, x, y)

    # tiny random jitter helps avoid “grid artifacts”.
    _, x0, y0 = best
    x0 += rng.uniform(-1e-3, 1e-3)
    y0 += rng.uniform(-1e-3, 1e-3)
    x0 = clamp(x0, tx0, tx1)
    y0 = clamp(y0, ty0, ty1)
    return x0, y0


def run_damped_dynamics(x: float, y: float, tx0: float, tx1: float, ty0: float, ty1: float,
                        steps: int, params: DynamicsParams):
    vx, vy = 0.0, 0.0
    best_val = f_xy(x, y)
    best_x, best_y = x, y

    for _ in range(steps):
        gx, gy = grad_f_xy(x, y)

        # acceleration = -grad/m - gamma*v
        ax = -(gx / params.mass) - params.gamma * vx
        ay = -(gy / params.mass) - params.gamma * vy

        # velocity update
        vx += params.dt * ax
        vy += params.dt * ay

        # safety clamp on speed
        speed = math.hypot(vx, vy)
        if speed > params.max_speed:
            s = params.max_speed / speed
            vx *= s
            vy *= s

        # position update
        x += params.dt * vx
        y += params.dt * vy

        # reflect boundaries to keep dynamics sane inside the tile
        x = reflect_into_box(x, tx0, tx1)
        y = reflect_into_box(y, ty0, ty1)

        val = f_xy(x, y)
        if val < best_val:
            best_val = val
            best_x, best_y = x, y

    return best_val, best_x, best_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=None, help="SLURM_ARRAY_TASK_ID (or manual)")
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=10)
    ap.add_argument("--xlo", type=float, default=-5.12)
    ap.add_argument("--xhi", type=float, default=5.12)
    ap.add_argument("--ylo", type=float, default=-5.12)
    ap.add_argument("--yhi", type=float, default=5.12)
    ap.add_argument("--coarse-gx", type=int, default=25)
    ap.add_argument("--coarse-gy", type=int, default=25)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    # task id: prefer arg, else SLURM
    tid = args.task_id

    if tid < 0 or tid >= args.nx * args.ny:
        print(f"ERROR: task_id out of range! task_id={tid} out of range for nx*ny={args.nx*args.ny}")
        print("Exiting with code 42 to indicate specific failure...")
        sys.exit(42)  # non-zero exit code to indicate failure
        
    if tid is None:
        tid_env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if tid_env is None:
            raise SystemExit("Need --task-id or SLURM_ARRAY_TASK_ID")
        tid = int(tid_env)

    if args.nx * args.ny <= 0:
        raise SystemExit("nx*ny must be positive")

    rng = random.Random(args.seed + 1000003 * tid)

    ix, iy, tx0, tx1, ty0, ty1 = tile_bounds(tid, args.nx, args.ny, args.xlo, args.xhi, args.ylo, args.yhi)

    x0, y0 = coarse_pick_start(tx0, tx1, ty0, ty1, args.coarse_gx, args.coarse_gy, rng)

    params = DynamicsParams(dt=args.dt, mass=args.mass, gamma=args.gamma)

    best_val, best_x, best_y = run_damped_dynamics(
        x0, y0, tx0, tx1, ty0, ty1, args.steps, params
    )

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f"task_{tid:05d}.json")

    payload = {
        "task_id": tid,
        "tile": {"ix": ix, "iy": iy, "nx": args.nx, "ny": args.ny},
        "tile_bounds": {"x0": tx0, "x1": tx1, "y0": ty0, "y1": ty1},
        "start": {"x": x0, "y": y0, "f": f_xy(x0, y0)},
        "best": {"x": best_x, "y": best_y, "f": best_val},
        "params": {
            "steps": args.steps,
            "dt": args.dt,
            "mass": args.mass,
            "gamma": args.gamma,
            "coarse_gx": args.coarse_gx,
            "coarse_gy": args.coarse_gy,
            "seed": args.seed,
        },
    }

    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    # single-line summary for logs:
    print(f"[task {tid}] tile=({ix},{iy}) start_f={payload['start']['f']:.6f} best_f={best_val:.6f} "
          f"best_xy=({best_x:.6f},{best_y:.6f}) -> {outpath}")


if __name__ == "__main__":
    main()