"""
Compute per-religion metrics from scenario run outputs.

Loads JSON results and exports line-by-line metrics to CSV. Analysis module.
"""
from __future__ import annotations

import sys
from pathlib import Path
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_json, save_json
from analysis.metrics import per_religion_total_share_metrics


def compute_for(path: Path) -> dict:
    data = load_json(path)
    ode = data["ode"]
    abs_mean = data["abs_mean"]
    religions = sorted(int(k) for k in ode["B"].keys())
    return per_religion_total_share_metrics(ode, abs_mean, religions)


def write_csv(metrics: dict, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["religion", "MAE", "RMSE", "R2"])
        for r, vals in metrics["per_religion"].items():
            w.writerow([r, vals["mae"], vals["rmse"], vals["r2"]])
        s = metrics["summary"]
        w.writerow([])
        w.writerow(["SUMMARY", "mean_mae", s["mean_mae"]])
        w.writerow(["SUMMARY", "mean_rmse", s["mean_rmse"]])
        w.writerow(["SUMMARY", "mean_r2", s["mean_r2"]])
        w.writerow(["SUMMARY", "max_mae", s["max_mae"]])
        w.writerow(["SUMMARY", "max_rmse", s["max_rmse"]])
        w.writerow(["SUMMARY", "min_r2", s["min_r2"]])


def main() -> None:
    runs = Path("outputs/runs")

    for key in ["A", "B"]:
        path = runs / f"scenario_{key}.json"
        m = compute_for(path)

        # save JSON + CSV
        save_json(m, runs / f"scenario_{key}_line_metrics.json")
        write_csv(m, runs / f"scenario_{key}_line_metrics.csv")

        # also print a compact empirical claim
        s = m["summary"]
        print(f"[Scenario {key}] mean MAE={s['mean_mae']:.4g}, mean RMSE={s['mean_rmse']:.4g}, mean R2={s['mean_r2']:.4g}; "
              f"max MAE={s['max_mae']:.4g}, max RMSE={s['max_rmse']:.4g}, min R2={s['min_r2']:.4g}")


if __name__ == "__main__":
    main()





