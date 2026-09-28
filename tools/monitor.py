"""
tools/monitor.py — Real-time progress monitor for NEMO YOLO training.
"""
import sys
import time
from pathlib import Path
import pandas as pd

CSV_PATH = Path("runs/detect/runs/train/nemo_yolo11n/results.csv")

def print_status():
    if not CSV_PATH.exists():
        print(f"Results file not found at: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    if len(df) == 0:
        print("Training has started, waiting for first epoch to log...")
        return

    latest = df.iloc[-1]
    best_row = df.loc[df["metrics/mAP50(B)"].idxmax()]

    cur_epoch = int(latest["epoch"])
    total_epochs = 100
    pct = (cur_epoch / total_epochs) * 100

    print("=" * 65)
    print(f" NEMO YOLO11n-1C Detector Training Monitor")
    print("=" * 65)
    print(f" Progress:     Epoch {cur_epoch}/{total_epochs} ({pct:.1f}%)")
    print(f" Current mAP50:       {latest['metrics/mAP50(B)']:.4f}  (mAP50-95: {latest['metrics/mAP50-95(B)']:.4f})")
    print(f" Current Precision:   {latest['metrics/precision(B)']:.4f}  | Recall: {latest['metrics/recall(B)']:.4f}")
    print(f" Best mAP50 so far:   {best_row['metrics/mAP50(B)']:.4f}  (achieved at Epoch {int(best_row['epoch'])})")
    print("-" * 65)
    print(" Recent 5 Epochs:")
    print(f"  {'Epoch':>5} | {'Box Loss':>8} | {'Cls Loss':>8} | {'Precision':>9} | {'Recall':>7} | {'mAP50':>7}")
    print("  " + "-" * 57)
    recent = df.tail(5)
    for _, row in recent.iterrows():
        print(f"  {int(row['epoch']):>5} | {row['train/box_loss']:>8.4f} | {row['train/cls_loss']:>8.4f} | {row['metrics/precision(B)']:>9.4f} | {row['metrics/recall(B)']:>7.4f} | {row['metrics/mAP50(B)']:>7.4f}")
    print("=" * 65)

if __name__ == "__main__":
    if "--watch" in sys.argv or "-w" in sys.argv:
        try:
            while True:
                print("\033[H\033[J", end="")  # clear terminal
                print_status()
                print("\n[Refreshing every 10s... Press Ctrl+C to exit]")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        print_status()
