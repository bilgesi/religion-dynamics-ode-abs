"""
Find outlier cases from bridge metrics.

Identifies worst-performing parameter configurations by MAE and R-squared.
Legacy/utility script for debugging.
"""
from pathlib import Path
import json

p = Path("outputs/runs/bridge_metrics.json")
data = json.loads(p.read_text(encoding="utf-8"))["metrics"]

worst_mae = max(data, key=lambda d: d["mean_mae"])
worst_r2  = min(data, key=lambda d: d["mean_r2"])

print("Worst by MAE:")
print(f"  Case: {worst_mae['case']}")
print(f"  MAE: {worst_mae['mean_mae']:.6f}")
print(f"  R²: {worst_mae['mean_r2']:.6f}")
if "params" in worst_mae:
    print(f"  Params: {worst_mae['params']}")

print("\nWorst by R2:")
print(f"  Case: {worst_r2['case']}")
print(f"  MAE: {worst_r2['mean_mae']:.6f}")
print(f"  R²: {worst_r2['mean_r2']:.6f}")
if "params" in worst_r2:
    print(f"  Params: {worst_r2['params']}")

