import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS = [
    REPO_ROOT / "05_logs" / "training.log",
    REPO_ROOT / "04_models" / "adapters" / "output_dpo" / "training.log",
    REPO_ROOT / "training.log",
]


def monitor_training(log_file: str | None = None):
    """Monitor training progress from logs."""
    print("\n" + "=" * 60)
    print("Training Progress Monitor")
    print("=" * 60 + "\n")

    print("Looking for training logs...")

    possible_logs = DEFAULT_LOGS.copy()
    if log_file:
        possible_logs.insert(0, Path(log_file))

    log_path = None
    for path in possible_logs:
        if path and Path(path).exists():
            log_path = Path(path)
            break

    if not log_path:
        print("No log file found. Training might not have started yet.")
        print("\nTip: Run training with logging, e.g.:")
        print("  python 02_src/train_dpo.py 2>&1 | tee 05_logs/training.log")
        print("\nThen in another terminal:")
        print("  python 02_src/monitor_training.py")
        return

    print(f"Found log: {log_path}\n")
    print("Monitoring training... (Ctrl+C to stop)\n")
    print("-" * 60)

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    if any(keyword in line.lower() for keyword in ["loss", "step", "epoch", "error"]):
                        print(line.strip())
                else:
                    time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")


def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else None
    monitor_training(log_file)


if __name__ == "__main__":
    main()
