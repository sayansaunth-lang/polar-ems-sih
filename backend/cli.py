#!/usr/bin/env python3
"""
POLAR-EMS terminal client.

A menu-driven command-line client for the POLAR-EMS backend API. Run the
backend first (see README.md), then run this in a second terminal:

    python cli.py

Every action here is a plain HTTP call to the FastAPI server -- this file
has no simulation logic of its own, so it can never drift out of sync with
what the API actually does.
"""

from __future__ import annotations

import os
import sys
import time

import requests
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

BASE_URL = os.environ.get("POLAR_EMS_API", "https://polar-ems-backend.onrender.com").rstrip("/")
console = Console()


def api_get(path: str, **params):
    r = requests.get(f"{BASE_URL}{path}", params=params, timeout=5)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json=None):
    r = requests.post(f"{BASE_URL}{path}", json=json, timeout=5)
    if not r.ok:
        console.print(f"[red]Error {r.status_code}:[/red] {r.text}")
        return None
    return r.json()


def check_connection() -> bool:
    try:
        h = api_get("/health")
        console.print(f"[green]Connected[/green] to {BASE_URL} - sim time {h['sim_time']}")
        return True
    except requests.exceptions.RequestException as exc:
        console.print(f"[red]Cannot reach backend at {BASE_URL}[/red]: {exc}")
        console.print("Start it first with: [bold]uvicorn app.main:app --reload --port 8000[/bold] (from backend/)")
        return False


def render_status(snap: dict, audit: dict | None = None) -> Panel:
    mode_color = "cyan" if snap["mode"] == "AI OPTIMIZED" else "yellow"
    conn_color = "green" if snap["online"] else "red"

    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="dim")
    t.add_column()
    t.add_row("Sim time", snap["sim_time"])
    t.add_row("Mode", f"[{mode_color}]{snap['mode']}[/{mode_color}]")
    t.add_row("Connectivity", f"[{conn_color}]{'ONLINE' if snap['online'] else 'OFFLINE'}[/{conn_color}]"
                               + (f"  (outbox: {snap['outbox']})" if not snap["online"] else ""))
    if snap.get("load"):
        t.add_row("Station load", f"{snap['load']['total']:.0f} kW")
    if snap.get("renewables"):
        r = snap["renewables"]
        t.add_row("Renewables", f"wind {r['wind_kw']:.0f} kW + solar {r['solar_kw']:.0f} kW "
                                 f"({r['fraction_pct']:.0f}% of load)")
    b = snap["battery"]
    t.add_row("Battery", f"{b['soc']:.0f}% SOC, {b['temp']:.1f}degC, health {b['health']:.0f}%, "
                          f"{'charging' if b['charging'] else 'discharging/idle'}")
    d = snap["diesel"]
    t.add_row("Diesel", f"{'RUNNING ' + format(d['output_kw'], '.0f') + ' kW' if d['on'] else 'standby'}, "
                        f"fuel {d['fuel_l']:.0f}/{d['tank_cap_l']} L, runtime {d['runtime_h']:.1f} h")
    t.add_row("Open alerts", str(snap.get("alerts_open", 0)))
    if audit:
        t.add_row("Fuel saved", f"{audit['fuel_saved_l']} L  ({audit['co2_avoided_kg']} kg CO2 avoided)")
    if snap.get("decision", {}).get("reason"):
        t.add_row("Why", snap["decision"]["reason"])
    return Panel(t, title="POLAR-EMS live status", border_style="cyan")


def cmd_status():
    snap = api_get("/api/telemetry/latest")
    audit = api_get("/api/green-audit")
    console.print(render_status(snap, audit))


def cmd_watch():
    console.print("[dim]Watching live status. Press Ctrl+C to stop.[/dim]")
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                snap = api_get("/api/telemetry/latest")
                audit = api_get("/api/green-audit")
                live.update(render_status(snap, audit))
                time.sleep(1)
    except KeyboardInterrupt:
        pass


def cmd_alerts():
    data = api_get("/api/alerts", limit=15)
    if not data["alerts"]:
        console.print("[dim]No alerts currently.[/dim]")
        return
    t = Table(title=f"Alerts ({data['count']})")
    t.add_column("Sev", style="bold")
    t.add_column("Subsystem")
    t.add_column("Detected")
    t.add_column("Expected")
    t.add_column("Explanation", max_width=50)
    for a in data["alerts"]:
        sev_style = {"critical": "red", "warning": "yellow", "info": "blue"}.get(a["severity"], "white")
        t.add_row(f"[{sev_style}]{a['severity'].upper()}[/{sev_style}]", a["subsystem"],
                  a["detected"], a["expected"], a["explanation"])
    console.print(t)


def cmd_audit():
    a = api_get("/api/green-audit")
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="dim")
    t.add_column()
    t.add_row("Diesel consumed (AI)", f"{a['fuel_consumed_ai_l']} L")
    t.add_row("Diesel consumed (baseline)", f"{a['fuel_consumed_baseline_l']} L")
    t.add_row("Fuel saved", f"{a['fuel_saved_l']} L")
    t.add_row("CO2 avoided", f"{a['co2_avoided_kg']} kg")
    t.add_row("Renewable fraction", f"{a['renewable_fraction_pct']}%")
    t.add_row("Generator runtime (AI)", f"{a['generator_runtime_ai_h']} h")
    t.add_row("Generator runtime (baseline)", f"{a['generator_runtime_baseline_h']} h")
    t.add_row("Runtime reduction", f"{a['generator_runtime_reduction_h']} h")
    console.print(Panel(t, title="Green audit (cumulative since last reset)", border_style="green"))


def cmd_scenario():
    scn = api_get("/api/scenarios")["scenarios"]
    console.print("\nAvailable scenarios:")
    for i, s in enumerate(scn):
        console.print(f"  [{i}] [bold]{s['name']}[/bold] - {s['desc']}")
    choice = input("Pick a scenario number (or blank to cancel): ").strip()
    if not choice:
        return
    try:
        idx = int(choice)
        sid = scn[idx]["id"]
    except (ValueError, IndexError):
        console.print("[red]Invalid choice.[/red]")
        return
    result = api_post("/api/simulation/scenario", {"id": sid})
    if result:
        console.print(f"[green]Scenario applied:[/green] {result['scenario']}")


def cmd_sim_control():
    console.print("\n[1] Start  [2] Pause  [3] Reset")
    choice = input("Choice: ").strip()
    if choice == "1":
        api_post("/api/simulation/start")
        console.print("[green]Simulation running.[/green]")
    elif choice == "2":
        api_post("/api/simulation/pause")
        console.print("[yellow]Simulation paused.[/yellow]")
    elif choice == "3":
        api_post("/api/simulation/reset")
        console.print("[green]Simulation reset to defaults.[/green]")
    else:
        console.print("[dim]No change.[/dim]")


def cmd_connectivity():
    snap = api_get("/api/telemetry/latest")
    if snap["online"]:
        api_post("/api/simulation/offline")
        console.print("[red]Simulated internet outage triggered.[/red] Edge autonomy active.")
    else:
        api_post("/api/simulation/online")
        console.print("[green]Connectivity restored.[/green] Syncing queued records...")


def cmd_speed():
    val = input("New speed multiplier (1, 4, or 12): ").strip()
    try:
        speed = int(val)
    except ValueError:
        console.print("[red]Enter a whole number.[/red]")
        return
    result = api_post("/api/simulation/speed", {"speed": speed})
    if result:
        console.print(f"[green]Speed set to {result['speed']}x.[/green]")


MENU = """
[bold cyan]POLAR-EMS terminal client[/bold cyan]  ({base_url})

  1) Live status snapshot
  2) Watch live status (auto-refresh)
  3) Alerts
  4) Green audit summary
  5) Trigger a demo scenario
  6) Simulation control (start/pause/reset)
  7) Toggle internet outage
  8) Set simulation speed
  0) Exit
"""


def main():
    console.print(Panel.fit("POLAR-EMS - synthetic microgrid backend\nThis is a simulation, not live station data.",
                             border_style="cyan"))
    if not check_connection():
        sys.exit(1)

    actions = {
        "1": cmd_status, "2": cmd_watch, "3": cmd_alerts, "4": cmd_audit,
        "5": cmd_scenario, "6": cmd_sim_control, "7": cmd_connectivity, "8": cmd_speed,
    }
    while True:
        console.print(MENU.format(base_url=BASE_URL))
        choice = input("Choose an option: ").strip()
        if choice == "0":
            console.print("Goodbye.")
            break
        action = actions.get(choice)
        if not action:
            console.print("[red]Unknown option.[/red]")
            continue
        try:
            action()
        except requests.exceptions.RequestException as exc:
            console.print(f"[red]Request failed:[/red] {exc}")


if __name__ == "__main__":
    main()
