#!/usr/bin/env python3
"""
SmartAir Guardian — Automatic Dataset Collector
================================================
Reads CSV rows from ESP8266 over Serial and saves
them to gas_dataset.csv automatically.

Usage:
    python collect_dataset.py                   # auto-detect port
    python collect_dataset.py --port COM3       # Windows
    python collect_dataset.py --port /dev/ttyUSB0  # Linux

Install dependencies:
    pip install pyserial rich
"""

import serial
import serial.tools.list_ports
import csv
import os
import sys
import time
import argparse
import threading
from datetime import datetime
from pathlib import Path

# ── Try to import rich for nice terminal UI ──────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH = True
except ImportError:
    RICH = False
    print("[INFO] Install 'rich' for a better UI: pip install rich")

# ── Configuration ────────────────────────────────────────────
BAUD_RATE    = 115200
OUTPUT_FILE  = "gas_dataset.csv"
BACKUP_DIR   = "dataset_backups"
LABEL_NAMES  = {0:"Normal", 1:"LPG", 2:"Smoke", 3:"CO", 4:"Methane"}
LABEL_COLORS = {0:"green", 1:"yellow", 2:"red", 3:"bright_red", 4:"magenta"}
TARGET_PER_CLASS = 500

CSV_COLUMNS = [
    "mq135","mq2","mq7","mq4","mq3",
    "temperature","humidity","flame",
    "label","label_name","timestamp_ms"
]

# ── State ────────────────────────────────────────────────────
stats = {
    "total": 0,
    "per_class": {i: 0 for i in range(5)},
    "current_label": 0,
    "collecting": False,
    "port": "",
    "last_row": {},
    "session_start": time.time(),
    "errors": 0,
    "log": []
}
lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
def find_esp_port():
    """Auto-detect ESP8266 connected via USB."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description + p.manufacturer if p.manufacturer else p.description).lower()
        if any(k in desc for k in ["ch340","cp210","ftdi","usb serial","uart"]):
            return p.device
    # Fallback: return first available port
    if ports:
        return ports[0].device
    return None

def load_existing_counts():
    """Count existing rows per class in the CSV if it already exists."""
    if not Path(OUTPUT_FILE).exists():
        return
    with open(OUTPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lbl = int(row["label"])
                if 0 <= lbl <= 4:
                    stats["per_class"][lbl] += 1
                    stats["total"] += 1
            except (KeyError, ValueError):
                pass

def backup_dataset():
    """Save a timestamped backup of the current CSV."""
    if not Path(OUTPUT_FILE).exists():
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{BACKUP_DIR}/gas_dataset_{ts}.csv"
    import shutil
    shutil.copy(OUTPUT_FILE, backup)
    log(f"Backup saved: {backup}")

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    with lock:
        stats["log"].append(f"[{ts}] {msg}")
        if len(stats["log"]) > 20:
            stats["log"].pop(0)

def validate_row(row_dict):
    """Check that a parsed CSV row has valid values."""
    try:
        mq_vals = [int(row_dict[k]) for k in ["mq135","mq2","mq7","mq4","mq3"]]
        temp = float(row_dict["temperature"])
        hum  = float(row_dict["humidity"])
        lbl  = int(row_dict["label"])

        if not all(0 <= v <= 1023 for v in mq_vals):
            return False, "MQ values out of range"
        if not (-10 <= temp <= 80):
            return False, f"Temperature out of range: {temp}"
        if not (0 <= hum <= 100):
            return False, f"Humidity out of range: {hum}"
        if lbl not in range(5):
            return False, f"Invalid label: {lbl}"
        return True, "ok"
    except (KeyError, ValueError) as e:
        return False, str(e)

# ─────────────────────────────────────────────────────────────
def serial_reader(ser, csv_writer, csv_file):
    """Background thread: reads serial, writes valid rows to CSV."""
    header_received = False

    while True:
        try:
            raw = ser.readline().decode("utf-8", errors="replace").strip()
            if not raw:
                continue

            # Comment lines from ESP8266 (start with #)
            if raw.startswith("#"):
                log(raw[2:])  # strip "# "
                # Parse state changes
                if "COLLECTING" in raw:
                    with lock: stats["collecting"] = True
                elif "STOPPED" in raw:
                    with lock: stats["collecting"] = False
                elif "Label changed to:" in raw:
                    try:
                        lbl = int(raw.split("Label changed to:")[1].split()[0])
                        with lock: stats["current_label"] = lbl
                    except: pass
                continue

            # CSV header line
            if raw.startswith("mq135"):
                header_received = True
                # Write header to file only if file is new/empty
                if csv_file.tell() == 0:
                    csv_writer.writerow(CSV_COLUMNS)
                log("CSV header received from ESP8266")
                continue

            # Data row — parse and validate
            if not header_received:
                continue

            parts = raw.split(",")
            if len(parts) != len(CSV_COLUMNS):
                with lock: stats["errors"] += 1
                log(f"Bad column count ({len(parts)}): {raw[:40]}")
                continue

            row_dict = dict(zip(CSV_COLUMNS, parts))
            valid, reason = validate_row(row_dict)

            if not valid:
                with lock: stats["errors"] += 1
                log(f"Invalid row skipped — {reason}")
                continue

            # Write to CSV
            csv_writer.writerow([row_dict[c] for c in CSV_COLUMNS])
            csv_file.flush()  # write immediately, don't buffer

            # Update stats
            lbl = int(row_dict["label"])
            with lock:
                stats["total"] += 1
                stats["per_class"][lbl] += 1
                stats["current_label"] = lbl
                stats["last_row"] = row_dict

        except serial.SerialException as e:
            log(f"Serial error: {e}")
            break
        except Exception as e:
            with lock: stats["errors"] += 1
            log(f"Parse error: {e}")

# ─────────────────────────────────────────────────────────────
def command_sender(ser):
    """Background thread: reads keyboard input and sends commands to ESP."""
    HELP = """
Commands you can type here:
  start        — ESP begins collecting
  stop         — ESP pauses
  label 0      — Normal air
  label 1      — LPG gas
  label 2      — Smoke
  label 3      — Carbon monoxide
  label 4      — Methane
  status       — ESP prints current state
  count        — ESP prints per-class counts
  backup       — Save a CSV backup now
  quit / exit  — Stop and exit
  help         — Show this list
"""
    print(HELP)
    while True:
        try:
            cmd = input().strip().lower()
            if not cmd:
                continue

            if cmd in ("quit","exit","q"):
                backup_dataset()
                log("Exiting — dataset saved.")
                os._exit(0)

            elif cmd == "help":
                print(HELP)

            elif cmd == "backup":
                backup_dataset()

            elif cmd == "start":
                ser.write(b"START\n")

            elif cmd == "stop":
                ser.write(b"STOP\n")

            elif cmd.startswith("label "):
                lbl = cmd.split()[1]
                ser.write(f"LABEL:{lbl}\n".encode())

            elif cmd == "status":
                ser.write(b"STATUS\n")

            elif cmd == "count":
                ser.write(b"COUNT\n")

            else:
                print(f"Unknown command: {cmd}. Type 'help'.")

        except (EOFError, KeyboardInterrupt):
            backup_dataset()
            log("Interrupted — dataset saved.")
            os._exit(0)

# ─────────────────────────────────────────────────────────────
def run_rich_ui(ser, csv_writer, csv_file):
    """Run with live rich terminal dashboard."""
    console = Console()

    def make_display():
        elapsed = int(time.time() - stats["session_start"])
        h,m,s  = elapsed//3600, (elapsed%3600)//60, elapsed%60
        rate   = stats["total"] / max(elapsed, 1)

        # Status panel
        status_color = "green" if stats["collecting"] else "yellow"
        status_text  = "COLLECTING" if stats["collecting"] else "PAUSED"
        lbl = stats["current_label"]

        lines = []
        lines.append(Text.assemble(
            ("Status: ", "bold white"), (status_text, f"bold {status_color}"),
            ("   Label: ", "bold white"),
            (LABEL_NAMES[lbl], f"bold {LABEL_COLORS[lbl]}"),
            (f"  [{lbl}]", "dim")
        ))
        lines.append(Text.assemble(
            ("Session: ", "dim"), (f"{h:02d}:{m:02d}:{s:02d}", "white"),
            ("   Rate: ", "dim"), (f"{rate:.1f} rows/s", "white"),
            ("   Errors: ", "dim"), (str(stats["errors"]), "red" if stats["errors"] else "white")
        ))

        # Last row
        lr = stats.get("last_row", {})
        if lr:
            lines.append(Text.assemble(
                ("Last: ", "dim"),
                (f"MQ135={lr.get('mq135','?')} MQ2={lr.get('mq2','?')} "
                 f"MQ7={lr.get('mq7','?')} MQ4={lr.get('mq4','?')} "
                 f"MQ3={lr.get('mq3','?')} "
                 f"T={lr.get('temperature','?')}°C H={lr.get('humidity','?')}%", "cyan")
            ))

        status_panel = Panel(
            "\n".join(str(l) for l in lines),
            title=f"[bold]SmartAir Guardian — Dataset Collector[/bold]",
            subtitle=f"[dim]Port: {stats['port']}  |  File: {OUTPUT_FILE}[/dim]",
            border_style="blue"
        )

        # Progress table
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        table.add_column("Class", style="bold")
        table.add_column("Collected", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Progress")

        for i in range(5):
            count  = stats["per_class"][i]
            pct    = min(100, int(count / TARGET_PER_CLASS * 100))
            bar    = "█" * (pct // 5) + "░" * (20 - pct // 5)
            color  = LABEL_COLORS[i]
            done   = count >= TARGET_PER_CLASS
            table.add_row(
                f"[{color}]{LABEL_NAMES[i]}[/{color}]",
                f"[{'green' if done else 'white'}]{count}[/{'green' if done else 'white'}]",
                str(TARGET_PER_CLASS),
                f"[{color}]{bar}[/{color}] {pct}%"
            )

        total_done = sum(1 for i in range(5) if stats["per_class"][i] >= TARGET_PER_CLASS)
        table.add_row(
            "[bold white]TOTAL[/bold white]",
            f"[bold]{stats['total']}[/bold]",
            str(TARGET_PER_CLASS * 5),
            f"[bold]{total_done}/5 classes complete[/bold]"
        )

        # Log panel
        log_lines = "\n".join(stats["log"][-8:]) or "Waiting for data..."
        log_panel = Panel(log_lines, title="[dim]Log[/dim]", border_style="dim", height=12)

        # Commands hint
        hint = Panel(
            "[dim]start[/dim]  [dim]stop[/dim]  "
            "[dim]label 0–4[/dim]  [dim]backup[/dim]  [dim]quit[/dim]",
            title="[dim]Commands[/dim]", border_style="dim"
        )

        from rich.columns import Columns
        return Panel(
            "\n".join([
                str(status_panel), str(table), str(log_panel), str(hint)
            ]),
            expand=True
        )

    # Start background threads
    reader_t = threading.Thread(target=serial_reader, args=(ser, csv_writer, csv_file), daemon=True)
    sender_t = threading.Thread(target=command_sender, args=(ser,), daemon=True)
    reader_t.start()
    sender_t.start()

    console.print(Panel(
        "[bold green]Connected![/bold green] Type commands below. "
        "The display updates automatically.",
        border_style="green"
    ))

    with Live(console=console, refresh_per_second=2, screen=False) as live:
        while True:
            elapsed = int(time.time() - stats["session_start"])
            h,m,s   = elapsed//3600, (elapsed%3600)//60, elapsed%60
            rate    = stats["total"] / max(elapsed, 1)
            lbl     = stats["current_label"]
            status  = "COLLECTING" if stats["collecting"] else "PAUSED   "

            # Build progress bars
            rows = []
            for i in range(5):
                count = stats["per_class"][i]
                pct   = min(100, int(count / TARGET_PER_CLASS * 100))
                bar   = "█"*(pct//5) + "░"*(20-pct//5)
                rows.append(f"  {LABEL_NAMES[i]:8s} {count:5d}/{TARGET_PER_CLASS}  [{bar}] {pct}%")

            lr = stats.get("last_row",{})
            last = (f"MQ135={lr.get('mq135','---')} MQ2={lr.get('mq2','---')} "
                    f"MQ7={lr.get('mq7','---')} MQ4={lr.get('mq4','---')} "
                    f"MQ3={lr.get('mq3','---')} "
                    f"T={lr.get('temperature','--')}C H={lr.get('humidity','--%')}%") if lr else "waiting..."

            log_tail = "\n".join(f"  {l}" for l in stats["log"][-5:])

            text = Text()
            text.append(f"\n  Status : ", style="dim")
            text.append(status, style="bold green" if stats["collecting"] else "bold yellow")
            text.append(f"   Label : ", style="dim")
            text.append(f"{LABEL_NAMES[lbl]} [{lbl}]", style=f"bold {LABEL_COLORS[lbl]}")
            text.append(f"\n  Time   : {h:02d}:{m:02d}:{s:02d}", style="dim")
            text.append(f"   Total : ", style="dim")
            text.append(str(stats["total"]), style="bold white")
            text.append(f"   Rate : {rate:.1f}/s   Errors: ", style="dim")
            text.append(str(stats["errors"]), style="bold red" if stats["errors"] else "dim")
            text.append(f"\n  Last   : {last}\n", style="cyan")
            text.append("\n".join(rows))
            text.append(f"\n\n  Log:\n{log_tail}\n", style="dim")

            live.update(Panel(text,
                title=f"[bold blue]SmartAir Dataset Collector[/bold blue]  |  {OUTPUT_FILE}  |  Port: {stats['port']}",
                border_style="blue"))
            time.sleep(0.5)

# ─────────────────────────────────────────────────────────────
def run_simple_ui(ser, csv_writer, csv_file):
    """Fallback UI without rich library."""
    print("\n=== SmartAir Dataset Collector ===")
    print(f"Saving to: {OUTPUT_FILE}")
    print(f"Port: {stats['port']}")
    print("Commands: start | stop | label 0-4 | backup | quit\n")

    reader_t = threading.Thread(target=serial_reader, args=(ser, csv_writer, csv_file), daemon=True)
    sender_t = threading.Thread(target=command_sender, args=(ser,), daemon=True)
    reader_t.start()
    sender_t.start()

    last_print = 0
    while True:
        if time.time() - last_print >= 2:
            last_print = time.time()
            print(f"\r  Total: {stats['total']}  |  "
                  + "  ".join(f"{LABEL_NAMES[i]}: {stats['per_class'][i]}" for i in range(5))
                  + f"  |  {'COLLECTING' if stats['collecting'] else 'PAUSED'}   ", end="")
        time.sleep(0.1)

# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SmartAir Dataset Collector")
    parser.add_argument("--port",   help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud",   type=int, default=BAUD_RATE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--target", type=int, default=TARGET_PER_CLASS,
                        help="Target samples per class (default 500)")
    args = parser.parse_args()

    global OUTPUT_FILE, TARGET_PER_CLASS
    OUTPUT_FILE = args.output
    TARGET_PER_CLASS = args.target

    # Find port
    port = args.port or find_esp_port()
    if not port:
        print("ERROR: No serial port found. Connect ESP8266 and retry, or use --port COM3")
        sys.exit(1)

    stats["port"] = port
    print(f"Connecting to {port} at {args.baud} baud...")

    # Load existing data counts
    load_existing_counts()
    if stats["total"] > 0:
        print(f"Resuming — found {stats['total']} existing rows in {OUTPUT_FILE}")

    # Open serial
    try:
        ser = serial.Serial(port, args.baud, timeout=2)
        time.sleep(2)  # wait for ESP8266 to reset after serial connect
    except serial.SerialException as e:
        print(f"ERROR: Could not open {port}: {e}")
        sys.exit(1)

    # Open CSV (append mode — resume if file exists)
    file_exists = Path(OUTPUT_FILE).exists() and Path(OUTPUT_FILE).stat().st_size > 0
    csv_file = open(OUTPUT_FILE, "a", newline="")
    csv_writer = csv.writer(csv_file)

    if not file_exists:
        csv_writer.writerow(CSV_COLUMNS)
        csv_file.flush()
        print(f"Created new file: {OUTPUT_FILE}")
    else:
        print(f"Appending to existing file: {OUTPUT_FILE}")

    log(f"Connected to {port}")
    log(f"Existing rows loaded: {stats['total']}")

    try:
        if RICH:
            run_rich_ui(ser, csv_writer, csv_file)
        else:
            run_simple_ui(ser, csv_writer, csv_file)
    except KeyboardInterrupt:
        pass
    finally:
        backup_dataset()
        csv_file.close()
        ser.close()
        print(f"\nDone. Total rows saved: {stats['total']}")
        print(f"File: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
