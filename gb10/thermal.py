"""
gb10/thermal.py - Live GB10 thermal zone reader

Confirmed zone mapping across 6+ independent units, multiple OEMs,
kernel versions 6.11 through 6.17:
  thermal_zone0 -> TSOC  (SoC temperature)
  thermal_zone1 -> TS0E  (CPU cluster 0 efficiency cores)
  thermal_zone2 -> TS0P  (CPU cluster 0 performance cores)
  thermal_zone3 -> TS1E  (CPU cluster 1 efficiency cores)
  thermal_zone4 -> TS1P  (CPU cluster 1 performance cores)
  thermal_zone5 -> TGPU  (GPU die temperature)
  thermal_zone6 -> TUNC  (Unknown component, runs 25-39C at idle)

CX7 NIC ASIC temperature from mlx5 hwmon -- separate device, ~45C idle.
No hardcoded thresholds. Live measurement only.
"""

import os
import sys
import time
import glob
import argparse


ZONE_NAMES = {
    0: "TSOC",
    1: "TS0E",
    2: "TS0P",
    3: "TS1E",
    4: "TS1P",
    5: "TGPU",
    6: "TUNC",
}


def read_thermal_zones():
    zones = {}
    for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            idx = int(path.split("/thermal_zone")[1].split("/")[0])
            raw = open(path).read().strip()
            zones[idx] = int(raw) / 1000.0
        except (ValueError, IOError):
            zones[idx] = None
    return zones


def read_mlx5_temp():
    """Read CX7 NIC ASIC temperature from mlx5 hwmon device."""
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*/name"):
        try:
            name = open(hwmon).read().strip()
            if "mlx5" in name:
                base = os.path.dirname(hwmon)
                for tf in glob.glob(os.path.join(base, "temp*_input")):
                    raw = open(tf).read().strip()
                    return int(raw) / 1000.0
        except (IOError, ValueError):
            pass
    return None


def read_nvme_temp():
    """Read NVMe SSD temperature from nvme hwmon device."""
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*/name"):
        try:
            name = open(hwmon).read().strip()
            if "nvme" in name:
                base = os.path.dirname(hwmon)
                # temp1 is composite (drive), temp2/3 are sensors
                tf = os.path.join(base, "temp1_input")
                if os.path.exists(tf):
                    raw = open(tf).read().strip()
                    return int(raw) / 1000.0
        except (IOError, ValueError):
            pass
    return None


def format_temp(val):
    if val is None:
        return "  N/A "
    return f"{val:5.1f}C"


def print_header():
    cols = ["TSOC", "TS0E", "TS0P", "TS1E", "TS1P", "TGPU", "TUNC", "CX7 ", "NVMe"]
    print("  " + "  ".join(f"{c:>6}" for c in cols))


def print_row(zones, cx7, nvme, timestamp=False):
    vals = []
    for i in range(7):
        vals.append(format_temp(zones.get(i)))
    vals.append(format_temp(cx7))
    vals.append(format_temp(nvme))
    line = "  " + "  ".join(vals)
    if timestamp:
        line = f"{time.strftime('%H:%M:%S')}  " + "  ".join(vals)
    print(line, flush=True)


def run_once():
    zones = read_thermal_zones()
    cx7 = read_mlx5_temp()
    nvme = read_nvme_temp()
    print_header()
    print_row(zones, cx7, nvme, timestamp=True)


def run_loop(interval, timestamps):
    print_header()
    n = 0
    try:
        while True:
            zones = read_thermal_zones()
            cx7 = read_mlx5_temp()
            nvme = read_nvme_temp()
            print_row(zones, cx7, nvme, timestamp=timestamps)
            n += 1
            # reprint header every 40 rows
            if n % 40 == 0:
                print_header()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Live GB10 thermal zone reader. No hardcoded thresholds."
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        help="Poll interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-1", "--once",
        action="store_true",
        help="Print one reading and exit",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Suppress timestamp column",
    )
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_loop(args.interval, timestamps=not args.no_timestamp)


if __name__ == "__main__":
    main()


def thermal_snapshot():
    """Complete thermal state snapshot for swarm fitness evaluation."""
    zones = read_thermal_zones()
    cx7 = read_mlx5_temp()
    nvme = read_nvme_temp()
    named = {ZONE_NAMES.get(i, f"zone{i}"): v for i, v in zones.items()}
    return {
        "zones": named,
        "TGPU":  named.get("TGPU"),
        "TSOC":  named.get("TSOC"),
        "TS1P":  named.get("TS1P"),
        "TS0P":  named.get("TS0P"),
        "TUNC":  named.get("TUNC"),
        "CX7":   cx7,
        "NVMe":  nvme,
    }
