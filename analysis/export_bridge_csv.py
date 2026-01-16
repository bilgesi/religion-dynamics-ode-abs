"""
Export ODE-ABS bridge metrics to CSV format.

Converts JSON bridge metrics to tabular CSV for further analysis.
Analysis module for paper artifact generation.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json


def export_bridge_metrics_csv(json_path: Path, csv_path: Path) -> None:
    """
    Export bridge_metrics.json to CSV format.
    Each row is a case with: case, mean_mae, mean_r2, and optionally parameter values.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    metrics = data["metrics"]
    
    # Determine which columns to include
    # Check if params exist in first case
    has_params = len(metrics) > 0 and "params" in metrics[0]
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if has_params:
            # Get all parameter keys from first case
            param_keys = []
            if metrics and "params" in metrics[0]:
                params = metrics[0]["params"]
                # Flatten nested params
                for key in ["b", "mu", "dt", "t_max"]:
                    if key in params:
                        param_keys.append(key)
                # Handle religion-specific params
                religion_params = ["beta0", "q", "sigma", "kappa", "tauB", "tauM", "rhoB", "rhoM", "rhoP"]
                if metrics and "params" in metrics[0]:
                    p = metrics[0]["params"]
                    religions = sorted([int(k) for k in p.get("beta0", {}).keys()])
                    for r in religions:
                        for param_name in religion_params:
                            if param_name in p:
                                param_keys.append(f"{param_name}_{r}")
                    # Handle nu if exists
                    if "nu" in p and p["nu"] is not None:
                        for r1 in sorted(p["nu"].keys()):
                            for r2 in sorted(p["nu"][r1].keys()):
                                param_keys.append(f"nu_{r1}_{r2}")
            
            writer = csv.DictWriter(f, fieldnames=["case", "mean_mae", "mean_r2"] + param_keys)
            writer.writeheader()
            
            for row in metrics:
                out_row = {
                    "case": row["case"],
                    "mean_mae": row["mean_mae"],
                    "mean_r2": row["mean_r2"],
                }
                if "params" in row:
                    p = row["params"]
                    # Simple params
                    for key in ["b", "mu", "dt", "t_max"]:
                        if key in p:
                            out_row[key] = p[key]
                    # Religion-specific params
                    for param_name in religion_params:
                        if param_name in p:
                            for r in religions:
                                key = f"{param_name}_{r}"
                                if str(r) in p[param_name]:
                                    out_row[key] = p[param_name][str(r)]
                                else:
                                    out_row[key] = ""
                    # Nu params
                    if "nu" in p and p["nu"] is not None:
                        for r1 in sorted(p["nu"].keys()):
                            for r2 in sorted(p["nu"][r1].keys()):
                                key = f"nu_{r1}_{r2}"
                                out_row[key] = p["nu"][r1][r2]
                    else:
                        # Fill nu columns with empty if nu is None
                        for key in param_keys:
                            if key.startswith("nu_"):
                                if key not in out_row:
                                    out_row[key] = ""
                
                writer.writerow(out_row)
        else:
            # Simple version without params
            writer = csv.DictWriter(f, fieldnames=["case", "mean_mae", "mean_r2"])
            writer.writeheader()
            for row in metrics:
                writer.writerow({
                    "case": row["case"],
                    "mean_mae": row["mean_mae"],
                    "mean_r2": row["mean_r2"],
                })


if __name__ == "__main__":
    json_path = Path("outputs/runs/bridge_metrics.json")
    csv_path = Path("outputs/runs/bridge_metrics.csv")
    
    print(f"Exporting {json_path} to {csv_path}...")
    export_bridge_metrics_csv(json_path, csv_path)
    print(f"CSV exported successfully to {csv_path}")




