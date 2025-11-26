"""
Monitor training progress by tailing the logs.

Usage:
    python monitor_training.py
"""

import time
import sys
from pathlib import Path


def monitor_training(log_file: str = None):
    """Monitor training progress from logs."""
    print("\n" + "="*60)
    print("📊 Training Progress Monitor")
    print("="*60 + "\n")
    
    print("Looking for training logs...")
    
    # Look for common log locations
    possible_logs = [
        "training.log",
        "output_llama32_sft/training.log",
        log_file
    ]
    
    log_path = None
    for path in possible_logs:
        if path and Path(path).exists():
            log_path = Path(path)
            break
    
    if not log_path:
        print("⚠️  No log file found. Training might not have started yet.")
        print("\nTip: Run this script after starting training:")
        print("  python train_llama32_sft.py 2>&1 | tee training.log")
        print("\nThen in another terminal:")
        print("  python monitor_training.py")
        return
    
    print(f"✅ Found log: {log_path}\n")
    print("Monitoring training... (Ctrl+C to stop)\n")
    print("-" * 60)
    
    try:
        with open(log_path, 'r') as f:
            # Move to end of file
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    # Print lines containing loss or step info
                    if any(keyword in line.lower() for keyword in ['loss', 'step', 'epoch', 'error']):
                        print(line.strip())
                else:
                    time.sleep(1)
                    
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


def main():
    """Main monitoring function."""
    log_file = sys.argv[1] if len(sys.argv) > 1 else None
    monitor_training(log_file)


if __name__ == "__main__":
    main()
