"""
POLAR-EMS simulation engine.

A Python port of the same physical/dispatch model used by the frontend
simulator (weather & season model, renewable power curves, priority-tiered
load, two parallel dispatch controllers, a hard safety layer, statistical
anomaly detection, and a seasonal-naive forecaster).

This is a synthetic model, not a connection to real station hardware. See
README.md for the same "what's simulated vs real" note that ships with the
frontend.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import count
from typing import Optional

_uid_counter = count(1)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def gauss() -> float:
    return random.gauss(0, 1)


def day_of_year(d: datetime) -> int:
    return d.timetuple().tm_yday


# ---------------------------------------------------------------------------
# Constants (kept numerically identical to the frontend's CFG object so the
# two simulations behave the same way for the same scenario).
# ---------------------------------------------------------------------------
class CFG:
    TURBINE_RATED_KW = 100
    TURBINE_CUT_IN = 3.5
    TURBINE_RATED = 12.0
    TURBINE_CUT_OUT = 25.0
    SOLAR_BASE_KW = 80
    BATT_CAPACITY_KWH = 500
    BATT_MAX_RATE_KW = 150
    BATT_RESERVE_BASE = 20
    BATT_CRITICAL = 10
    DIESEL_RATED_KW = 300
    DIESEL_MIN_STABLE = 0.2
    DIESEL_SFC_BASE = 0.30  # L per kWh
    DIESEL_TANK_CAP_L = 12000
    CO2_PER_LITER = 2.68  # kg CO2 per litre diesel (standard emission factor)
    LOAD_BASE = {
        "life_support": 40,
        "heating": 70,
        "research": 60,
        "general": 50,
        "amenities": 25,
    }


SCENARIOS = [
    {"id": "normal", "name": "Normal operation",
     "desc": "Baseline weather and load, both controllers running normally."},
    {"id": "lowrenew", "name": "Low renewable generation",
     "desc": "Wind and solar output drop sharply for an extended calm, overcast spell."},
    {"id": "lowsoc", "name": "Battery SOC falling",
     "desc": "Battery is forced down toward the reserve threshold to test discharge behaviour."},
    {"id": "highdemand", "name": "High station demand",
     "desc": "Station load spikes well above nominal (lab campaign, extra occupancy)."},
    {"id": "outage", "name": "Internet outage",
     "desc": "Cloud link drops. Edge autonomy takes over telemetry, AI, optimizer and safety."},
    {"id": "battdeg", "name": "Battery degradation anomaly",
     "desc": "Battery temperature and internal resistance drift out of normal range."},
    {"id": "turbine", "name": "Wind turbine underperformance",
     "desc": "One turbine's output falls well below what wind speed predicts (icing, blade fault)."},
    {"id": "aifail", "name": "AI service failure",
     "desc": "The optimizer becomes unavailable. Dispatch falls back to the hard-coded safety controller."},
]
SCENARIO_IDS = {s["id"] for s in SCENARIOS}


def season_val(doy: int) -> float:
    """-1 deep winter (polar night) .. +1 deep summer (midnight sun)."""
    return math.cos(2 * math.pi * (doy - 356) / 365.25)


def solar_potential(doy: int) -> float:
    sv = season_val(doy)
    if sv < -0.32:
        return 0.0
    return clamp((sv + 0.32) / 1.32, 0, 1)


def day_factor(hour: float, doy: int) -> float:
    sv = season_val(doy)
    diurnal = max(0.0, math.sin(math.pi * ((hour - 4) / 16)))
    if sv > 0.55:
        blend = clamp((sv - 0.55) / 0.45, 0, 1)
        return diurnal * (1 - blend) + 0.85 * blend
    return diurnal


def seasonal_temp(doy: int) -> float:
    return -25 + 20 * season_val(doy)


def wind_power_curve(speed: float, capacity_kw: float) -> float:
    if speed < CFG.TURBINE_CUT_IN or speed > CFG.TURBINE_CUT_OUT:
        return 0.0
    if speed >= CFG.TURBINE_RATED:
        return capacity_kw
    frac = (speed - CFG.TURBINE_CUT_IN) / (CFG.TURBINE_RATED - CFG.TURBINE_CUT_IN)
    return capacity_kw * (frac ** 3)


def diesel_sfc(load_frac: float, wear: float) -> float:
    if load_frac < CFG.DIESEL_MIN_STABLE:
        penalty = (CFG.DIESEL_MIN_STABLE - load_frac) * 2.2
    elif load_frac < 0.5:
        penalty = (0.5 - load_frac) * 0.4
    else:
        penalty = 0.0
    return CFG.DIESEL_SFC_BASE * (1 + penalty) * wear


@dataclass
class BatteryDieselState:
    soc: float = 70.0
    batt_temp: float = -5.0
    cycles: float = 0.0
    batt_deg: float = 1.0
    diesel_on: bool = False
    fuel_l: float = 9000.0
    runtime_h: float = 0.0
    wear: float = 1.0
    fuel_used_this_tick: float = 0.0


@dataclass
class Alert:
    id: int
    severity: str
    subsystem: str
    detected: str
    expected: str
    explanation: str
    recommendation: str
    timestamp: str

    def to_dict(self):
        return {
            "id": self.id, "severity": self.severity, "subsystem": self.subsystem,
            "detected": self.detected, "expected": self.expected,
            "explanation": self.explanation, "recommendation": self.recommendation,
            "timestamp": self.timestamp,
        }


class SimulationEngine:
    """Holds all mutable simulation state and advances it tick by tick."""

    def __init__(self):
        self.reset()

    # -- lifecycle ----------------------------------------------------
    def reset(self):
        self.running = True
        self.speed = 4
        self.sim_hours_per_tick_base = 0.25
        self.sim_time = datetime(2026, 9, 1, 6, 0, 0)
        self.online = True
        self.outbox = 0
        self.syncing = False
        self.scenario: Optional[str] = None
        self.turbine_count = 3
        self.solar_cap_kw = float(CFG.SOLAR_BASE_KW)
        self.wind_ar = 9.0
        self.cloud_ar = 0.3
        self.weather = {"temp": -20.0, "wind": 9.0, "irr": 0.0, "cloud": 0.3, "doy": 1, "hour": 6.0}
        self.ai = BatteryDieselState()
        self.base = BatteryDieselState()
        self.audit = {
            "fuel_ai": 0.0, "fuel_base": 0.0, "renew_gen_kwh": 0.0,
            "load_served_kwh": 0.0, "runtime_ai": 0.0, "runtime_base": 0.0,
        }
        self.history: list[dict] = []
        self.forecast_pending: list[dict] = []
        self.forecast_errors = {h: [] for h in (1, 6, 12, 24)}
        self.forecast_points = {h: [] for h in (1, 6, 12, 24)}
        self.alerts: list[Alert] = []
        self.health = {"battery": 100.0, "turbine": 100.0, "generator": 100.0}
        self.mode = "AI OPTIMIZED"
        self.current_decision: dict = {}
        self.decision_log: list[dict] = []
        self.sync_log: list[dict] = []
        # scenario overrides
        self.scenario_wind_target: Optional[float] = None
        self.scenario_cloud: Optional[float] = None
        self.load_multiplier = 1.0
        self.battery_anomaly_active = False
        self.turbine_degradation = 1.0
        self.ai_failure = False
        self.sensor_fault_active = False
        self._sensor_fault_ticks_left = 0
        self.tick_count = 0
        self._last_anomaly_check = 0

    def apply_scenario(self, scenario_id: str):
        if scenario_id not in SCENARIO_IDS:
            raise ValueError(f"unknown scenario id: {scenario_id}")
        # clear overrides first (mirrors resetScenarioOverrides in the frontend)
        self.scenario_wind_target = None
        self.scenario_cloud = None
        self.load_multiplier = 1.0
        self.battery_anomaly_active = False
        self.turbine_degradation = 1.0
        self.ai_failure = False
        if scenario_id != "outage" and not self.online:
            self.set_online(True)
        self.scenario = None if scenario_id == "normal" else scenario_id

        if scenario_id == "lowrenew":
            self.wind_ar = 2.5
            self.scenario_wind_target = 2.5
            self.scenario_cloud = 0.85
        elif scenario_id == "lowsoc":
            self.ai.soc = min(self.ai.soc, 24)
            self.base.soc = min(self.base.soc, 24)
        elif scenario_id == "highdemand":
            self.load_multiplier = 1.9
        elif scenario_id == "outage":
            self.set_online(False)
        elif scenario_id == "battdeg":
            self.battery_anomaly_active = True
        elif scenario_id == "turbine":
            self.turbine_degradation = 0.55
        elif scenario_id == "aifail":
            self.ai_failure = True

    def set_online(self, val: bool):
        if val == self.online:
            return
        self.online = val
        if val:
            self._log_sync(f"Connectivity restored. Beginning batch upload of {self.outbox} queued record(s).")
            self.syncing = True
        else:
            self._log_sync("Connectivity to cloud portal lost. Switching to edge autonomy.")

    def sync_step(self):
        """Advance one step of the (simulated) batch upload; called by the API poll loop."""
        if not self.syncing:
            return
        if self.outbox <= 0:
            self.syncing = False
            self._log_sync("Sync complete. All records confirmed on cloud portal.")
            return
        chunk = max(1, math.ceil(self.outbox / 6))
        self.outbox = max(0, self.outbox - chunk)
        if self.outbox == 0:
            self.syncing = False
            self._log_sync("Sync complete. All records confirmed on cloud portal.")

    def _log_sync(self, msg: str):
        self.sync_log.insert(0, {"time": self.sim_time.isoformat(), "message": msg})
        self.sync_log = self.sync_log[:40]

    # -- weather --------------------------------------------------------
    def _step_weather(self, dt_h: float):
        doy = ((day_of_year(self.sim_time) - 1) % 365) + 1
        hour = self.sim_time.hour + self.sim_time.minute / 60
        wind_target = self.scenario_wind_target if self.scenario_wind_target is not None else 9.5
        self.wind_ar = clamp(self.wind_ar * 0.86 + wind_target * 0.14 + gauss() * 0.55, 0.3, 30)
        cloud_target = self.scenario_cloud if self.scenario_cloud is not None else 0.32
        self.cloud_ar = clamp(self.cloud_ar * 0.9 + cloud_target * 0.1 + gauss() * 0.04, 0, 0.95)
        pot = solar_potential(doy) * day_factor(hour, doy)
        irr = pot * 950 * (1 - self.cloud_ar * 0.85)
        temp = seasonal_temp(doy) + gauss() * 1.4 + math.sin((hour - 15) / 24 * 2 * math.pi) * 2.5
        self.weather = {"temp": temp, "wind": self.wind_ar, "irr": irr, "cloud": self.cloud_ar,
                         "doy": doy, "hour": hour}

    def _compute_renewables(self):
        w = self.weather
        wind_cap_total = self.turbine_count * CFG.TURBINE_RATED_KW
        wind_pow = clamp(wind_power_curve(w["wind"], wind_cap_total) * self.turbine_degradation, 0, wind_cap_total)
        solar_pow = clamp((w["irr"] / 1000) * self.solar_cap_kw * 0.92, 0, self.solar_cap_kw)
        return {"wind_pow": wind_pow, "solar_pow": solar_pow, "total": wind_pow + solar_pow}

    def _compute_load(self):
        w = self.weather
        heat_factor = clamp((-5 - w["temp"]) / 40, 0, 1.6)
        b = CFG.LOAD_BASE
        hour = w["hour"]
        life_support = max(20.0, b["life_support"] + gauss() * 1.5)
        heating = max(10.0, b["heating"] * (0.35 + heat_factor) + gauss() * 4)
        research = max(15.0, b["research"] + math.sin(hour / 24 * 2 * math.pi * 3) * 8 + gauss() * 4)
        general = max(15.0, b["general"] + gauss() * 3)
        amenities = max(5.0, b["amenities"] * (0.6 + 0.5 * math.sin((hour - 13) / 24 * 2 * math.pi)) + gauss() * 2)
        mult = self.load_multiplier
        load = {
            "life_support": life_support * mult, "heating": heating * mult,
            "research": research * mult, "general": general * mult, "amenities": amenities * mult,
        }
        load["total"] = sum(load.values())
        return load

    # -- dispatch ---------------------------------------------------------
    def _dispatch_baseline(self, gen: dict, load: dict, batt: BatteryDieselState, dt_h: float) -> dict:
        surplus = gen["total"] - load["total"]
        renew_to_load = load["total"] if surplus >= 0 else gen["total"]
        remaining = max(0.0, -surplus)
        batt_kw = 0.0
        diesel_kw = 0.0
        shed_kw = 0.0
        reason = ""
        cap_kwh = CFG.BATT_CAPACITY_KWH

        if surplus > 0.01:
            headroom_kwh = (100 - batt.soc) / 100 * cap_kwh
            charge_kw = min(surplus, CFG.BATT_MAX_RATE_KW, headroom_kwh / dt_h)
            batt.soc = clamp(batt.soc + charge_kw * dt_h / cap_kwh * 100, 0, 100)
            batt_kw = charge_kw
            batt.diesel_on = False
            reason = (f"Rule: renewables ({gen['total']:.0f} kW) exceed load ({load['total']:.0f} kW); "
                      f"surplus routed to battery charge.")
        else:
            if batt.soc > CFG.BATT_RESERVE_BASE:
                avail_kwh = (batt.soc - CFG.BATT_RESERVE_BASE) / 100 * cap_kwh
                discharge_kw = min(remaining, CFG.BATT_MAX_RATE_KW, avail_kwh / dt_h)
                batt.soc = clamp(batt.soc - discharge_kw * dt_h / cap_kwh * 100, 0, 100)
                batt_kw = -discharge_kw
                remaining -= discharge_kw
                reason = (f"Rule: SOC above fixed {CFG.BATT_RESERVE_BASE}% reserve, battery discharged "
                          f"{discharge_kw:.0f} kW to cover load.")
            if remaining > 0.5:
                batt.diesel_on = True
                diesel_kw = min(remaining, CFG.DIESEL_RATED_KW)
                remaining -= diesel_kw
                if remaining > 0.5:
                    shed_kw = remaining
                reason += (" " if reason else "") + f"Rule: diesel started at {diesel_kw:.0f} kW to cover remaining shortfall."
            else:
                batt.diesel_on = False
                if not reason:
                    reason = f"Rule: battery covers load within fixed {CFG.BATT_RESERVE_BASE}% reserve; diesel stays off."
            if shed_kw > 0.5:
                reason += f" Rule: non-critical load of {shed_kw:.0f} kW shed; diesel at rated capacity cannot cover the rest."

        reason += f" Battery now at {batt.soc:.0f}%."
        self._burn_fuel(batt, diesel_kw, dt_h, wear=1.0)
        return {"renew_kw": renew_to_load, "batt_kw": batt_kw, "diesel_kw": diesel_kw,
                "shed_kw": shed_kw, "reason": reason}

    def _forecast_renew_avg(self, hours_ahead: int) -> float:
        total, n = 0.0, 0
        for h in range(1, hours_ahead + 1):
            t = self.sim_time + timedelta(hours=h)
            doy2 = ((day_of_year(t) - 1) % 365) + 1
            hour2 = t.hour
            pot = solar_potential(doy2) * day_factor(hour2, doy2)
            solar_est = pot * 950 / 1000 * self.solar_cap_kw * 0.92
            wind_est = wind_power_curve(self.wind_ar, self.turbine_count * CFG.TURBINE_RATED_KW) * 0.9
            total += solar_est + wind_est
            n += 1
        return total / n if n else 0.0

    def _dispatch_ai(self, gen: dict, load: dict, batt: BatteryDieselState, dt_h: float) -> dict:
        cap_kwh = CFG.BATT_CAPACITY_KWH
        renew_fcst = self._forecast_renew_avg(3)
        dyn_reserve = 32 if renew_fcst < load["total"] * 0.45 else 20
        surplus = gen["total"] - load["total"]
        renew_to_load = load["total"] if surplus >= 0 else gen["total"]
        remaining = max(0.0, -surplus)
        batt_kw = 0.0
        diesel_kw = 0.0
        shed_kw = 0.0
        reason = ""

        if surplus > 0.01:
            headroom_kwh = (100 - batt.soc) / 100 * cap_kwh
            charge_kw = min(surplus, CFG.BATT_MAX_RATE_KW, headroom_kwh / dt_h)
            batt.soc = clamp(batt.soc + charge_kw * dt_h / cap_kwh * 100, 0, 100)
            batt_kw = charge_kw
            batt.diesel_on = False
            reason = (f"Renewables ({gen['total']:.0f} kW) exceed load ({load['total']:.0f} kW); surplus of "
                      f"{charge_kw:.0f} kW routed to battery charge, diesel held off.")
        else:
            can_discharge = batt.soc > dyn_reserve
            if can_discharge:
                avail_kwh = (batt.soc - dyn_reserve) / 100 * cap_kwh
                discharge_kw = min(remaining, CFG.BATT_MAX_RATE_KW, avail_kwh / dt_h)
                batt.soc = clamp(batt.soc - discharge_kw * dt_h / cap_kwh * 100, 0, 100)
                batt_kw = -discharge_kw
                remaining -= discharge_kw
                reason = (f"Forecasted renewable output over next 3h averages {renew_fcst:.0f} kW, so battery "
                          f"discharged {discharge_kw:.0f} kW (reserve floor set to {dyn_reserve}% given forecast).")
            if remaining > 0.5:
                batt.diesel_on = True
                target = remaining
                if batt.soc < 75 and renew_fcst < load["total"] * 0.5:
                    target = min(CFG.DIESEL_RATED_KW, remaining + CFG.DIESEL_RATED_KW * 0.25)
                diesel_kw = min(target, CFG.DIESEL_RATED_KW)
                extra_to_batt = max(0.0, diesel_kw - remaining)
                if extra_to_batt > 0.1:
                    headroom_kwh = (100 - batt.soc) / 100 * cap_kwh
                    actual_extra = min(extra_to_batt, headroom_kwh / dt_h)
                    batt.soc = clamp(batt.soc + actual_extra * dt_h / cap_kwh * 100, 0, 100)
                    batt_kw += actual_extra
                covered = min(diesel_kw, remaining)
                remaining -= covered
                if remaining > 0.5:
                    shed_kw = remaining
                tail = (", running above immediate demand to recharge the battery and avoid a second cold start soon."
                        if extra_to_batt > 0.1 else ".")
                reason += (" " if reason else "") + \
                    f"Diesel started at {diesel_kw:.0f} kW to cover the {(remaining + covered):.0f} kW shortfall{tail}"
            elif not reason:
                batt.diesel_on = False
                reason = f"Battery covers the shortfall within the {dyn_reserve}% reserve; diesel stays off."

        if shed_kw > 0.5:
            reason += f" Load shed of {shed_kw:.0f} kW applied to amenities/general tiers to protect life-support and research loads."
        reason += f" Battery now at {batt.soc:.0f}%."
        self._burn_fuel(batt, diesel_kw, dt_h, wear=self.ai.wear)
        return {"renew_kw": renew_to_load, "batt_kw": batt_kw, "diesel_kw": diesel_kw,
                "shed_kw": shed_kw, "reason": reason, "dyn_reserve": dyn_reserve}

    @staticmethod
    def _burn_fuel(batt: BatteryDieselState, diesel_kw: float, dt_h: float, wear: float):
        if batt.diesel_on and diesel_kw > 0:
            load_frac = diesel_kw / CFG.DIESEL_RATED_KW
            sfc = diesel_sfc(load_frac, wear)
            fuel_used = diesel_kw * sfc * dt_h
            batt.fuel_l = max(0.0, batt.fuel_l - fuel_used)
            batt.runtime_h += dt_h
            batt.fuel_used_this_tick = fuel_used
        else:
            batt.fuel_used_this_tick = 0.0

    def _apply_safety(self, decision: dict, batt: BatteryDieselState, load: dict) -> str:
        forced_fallback = self.ai_failure or self.sensor_fault_active
        if batt.soc < CFG.BATT_CRITICAL:
            forced_fallback = True
            batt.diesel_on = True
            decision["diesel_kw"] = CFG.DIESEL_RATED_KW
            still_short = max(0.0, load["total"] - decision["renew_kw"]
                               - abs(min(0.0, decision["batt_kw"])) - decision["diesel_kw"])
            decision["shed_kw"] = max(decision.get("shed_kw", 0.0), still_short)
            decision["reason"] = (f"SAFETY OVERRIDE: battery SOC {batt.soc:.0f}% is below the "
                                   f"{CFG.BATT_CRITICAL}% critical floor. Diesel forced to full output and "
                                   f"non-critical loads (general, amenities) shed to protect life-support and heating.")
        return "SAFE FALLBACK" if forced_fallback else "AI OPTIMIZED"

    # -- anomalies ----------------------------------------------------------
    def _push_alert(self, severity, subsystem, detected, expected, explanation, recommendation):
        a = Alert(next(_uid_counter), severity, subsystem, detected, expected,
                  explanation, recommendation, self.sim_time.isoformat())
        self.alerts.insert(0, a)
        self.alerts = self.alerts[:60]

    def _check_anomalies(self, gen: dict):
        self.tick_count += 1
        if self.tick_count - self._last_anomaly_check < 4:
            return
        self._last_anomaly_check = self.tick_count

        wind_cap_total = self.turbine_count * CFG.TURBINE_RATED_KW
        expected_wind = wind_power_curve(self.weather["wind"], wind_cap_total)
        if expected_wind > 20 and gen["wind_pow"] < expected_wind * 0.75:
            pct = round((1 - gen["wind_pow"] / expected_wind) * 100)
            self.health["turbine"] = clamp(self.health["turbine"] - 1.2, 30, 100)
            if random.random() < 0.5 or self.turbine_degradation < 0.9:
                self._push_alert(
                    "warning", "Wind turbine array", f"{gen['wind_pow']:.0f} kW", f"{expected_wind:.0f} kW",
                    f"Turbine output is {pct}% below the value predicted by the power curve for a "
                    f"{self.weather['wind']:.1f} m/s wind speed.",
                    "Schedule a blade/ice inspection; check pitch and yaw alignment before the next storm window.")
        else:
            self.health["turbine"] = clamp(self.health["turbine"] + 0.2, 30, 100)

        if self.battery_anomaly_active:
            self.ai.batt_temp = clamp(self.ai.batt_temp + random.uniform(0.5, 2.2), -10, 45)
            self.health["battery"] = clamp(self.health["battery"] - 1.5, 20, 100)
            if random.random() < 0.6:
                self._push_alert(
                    "critical", "Battery bank", f"{self.ai.batt_temp:.1f} degC", "-5 to 15 degC",
                    "Battery temperature is drifting outside the normal operating band with an accelerated rise "
                    "rate, consistent with degrading thermal management or cell imbalance.",
                    "Inspect thermal management and cell balance; reduce peak discharge rate until confirmed safe.")
        else:
            self.ai.batt_temp = clamp(self.ai.batt_temp * 0.9 + (-5) * 0.1 + gauss() * 0.4, -15, 40)
            self.health["battery"] = clamp(self.health["battery"] + 0.15, 20, 100)

        if self.ai.diesel_on:
            expected_sfc = CFG.DIESEL_SFC_BASE
            actual_sfc = self.ai.wear * expected_sfc
            if actual_sfc > expected_sfc * 1.15:
                self.health["generator"] = clamp(self.health["generator"] - 1.0, 20, 100)
                if random.random() < 0.4:
                    self._push_alert(
                        "warning", "Diesel generator", f"{actual_sfc:.2f} L/kWh", f"{expected_sfc:.2f} L/kWh",
                        f"Fuel consumption per kWh delivered is {round((actual_sfc / expected_sfc - 1) * 100)}% "
                        f"above the expected specific fuel consumption for this generator.",
                        "Inspect fuel injectors and air filter; schedule a combustion efficiency service.")
            else:
                self.health["generator"] = clamp(self.health["generator"] + 0.1, 20, 100)

        if random.random() < 0.03:
            self.sensor_fault_active = True
            self._sensor_fault_ticks_left = 1
            self._push_alert(
                "info", "Zone load sensor", "0 kW (dropout)", "~50 kW",
                "A zone load sensor reported an implausible dropout to zero for one reading cycle, "
                "inconsistent with surrounding telemetry.",
                "Reading flagged and excluded from dispatch input; monitor for repeat faults on this node.")

        if self.turbine_degradation < 1.0 or self.scenario == "battdeg":
            self.ai.wear = clamp(self.ai.wear + 0.0006, 1, 1.4)

    # -- forecasting ----------------------------------------------------------
    def _generate_forecasts(self, load: dict):
        for h in (1, 6, 12, 24):
            target_t = self.sim_time + timedelta(hours=h)
            decay = math.exp(-h / 14)
            doy_t = ((day_of_year(target_t) - 1) % 365) + 1
            hour_t = target_t.hour
            b = CFG.LOAD_BASE
            heat_est = clamp((-5 - seasonal_temp(doy_t)) / 40, 0, 1.6)
            seasonal_load_est = (b["life_support"] + b["heating"] * (0.35 + heat_est)
                                  + b["research"] + b["general"] + b["amenities"] * 0.85)
            baseline_now = sum(b.values())
            deviation = load["total"] - baseline_now
            pred_load = seasonal_load_est + deviation * decay + gauss() * 3 * (1 + h / 12)

            pot = solar_potential(doy_t) * day_factor(hour_t, doy_t)
            pred_solar_seasonal = pot * 950 / 1000 * self.solar_cap_kw * 0.92
            pot_now = solar_potential(self.weather["doy"]) * day_factor(self.weather["hour"], self.weather["doy"])
            irr_seasonal_now = pot_now * 950
            irr_deviation = self.weather["irr"] - irr_seasonal_now
            pred_solar = clamp(
                pred_solar_seasonal + (irr_deviation / 1000 * self.solar_cap_kw * 0.92) * decay * 0.3,
                0, self.solar_cap_kw,
            )
            wind_decay = 9.5 + (self.wind_ar - 9.5) * decay
            pred_wind = wind_power_curve(wind_decay, self.turbine_count * CFG.TURBINE_RATED_KW)

            self.forecast_pending.append({
                "target_t": target_t, "horizon": h, "pred_load": pred_load,
                "pred_wind": pred_wind, "pred_solar": pred_solar,
            })
        if len(self.forecast_pending) > 400:
            self.forecast_pending = self.forecast_pending[-400:]

    def _resolve_forecasts(self, actual_load: float, actual_wind: float, actual_solar: float):
        now = self.sim_time
        still = []
        for f in self.forecast_pending:
            if f["target_t"] <= now:
                err_l = abs(f["pred_load"] - actual_load)
                arr = self.forecast_errors[f["horizon"]]
                arr.append({"abs": err_l, "sq": err_l * err_l})
                if len(arr) > 200:
                    del arr[0]
                pts = self.forecast_points[f["horizon"]]
                pts.append({
                    "t": f["target_t"].isoformat(), "actual": actual_load, "pred": f["pred_load"],
                    "pred_wind": f["pred_wind"], "actual_wind": actual_wind,
                    "pred_solar": f["pred_solar"], "actual_solar": actual_solar,
                })
                if len(pts) > 60:
                    del pts[0]
            else:
                still.append(f)
        self.forecast_pending = still

    # -- main tick ----------------------------------------------------------
    def tick(self):
        if not self.running:
            return
        dt_h = self.sim_hours_per_tick_base * self.speed
        self.sim_time = self.sim_time + timedelta(hours=dt_h)
        self._step_weather(dt_h)
        gen = self._compute_renewables()
        load = self._compute_load()

        dec_base = self._dispatch_baseline(gen, load, self.base, dt_h)

        ai_unavailable = self.ai_failure or self.sensor_fault_active
        if ai_unavailable:
            decision = self._dispatch_baseline(gen, load, self.ai, dt_h)
            decision["reason"] = "AI optimizer unavailable - hard-coded safety controller engaged. " + decision["reason"]
        else:
            decision = self._dispatch_ai(gen, load, self.ai, dt_h)

        final_mode = self._apply_safety(decision, self.ai, load)
        self.mode = final_mode
        self.current_decision = decision

        self.ai.cycles += abs(decision["batt_kw"]) * dt_h / CFG.BATT_CAPACITY_KWH * 0.5
        self.base.cycles += abs(dec_base["batt_kw"]) * dt_h / CFG.BATT_CAPACITY_KWH * 0.5
        self.ai.batt_deg = clamp(1 - self.ai.cycles * 0.0004, 0.75, 1)

        self.audit["fuel_ai"] += self.ai.fuel_used_this_tick
        self.audit["fuel_base"] += self.base.fuel_used_this_tick
        self.audit["renew_gen_kwh"] += gen["total"] * dt_h
        self.audit["load_served_kwh"] += load["total"] * dt_h
        self.audit["runtime_ai"] = self.ai.runtime_h
        self.audit["runtime_base"] = self.base.runtime_h

        self._generate_forecasts(load)
        self._resolve_forecasts(load["total"], gen["wind_pow"], gen["solar_pow"])
        self._check_anomalies(gen)
        if self.sensor_fault_active:
            self.sensor_fault_active = False  # single-tick fault, mirrors the frontend's setTimeout reset

        frac_renew = clamp(decision["renew_kw"] / load["total"], 0, 1) * 100 if load["total"] > 0 else 0

        self.history.append({
            "t": self.sim_time.isoformat(), "temp": self.weather["temp"], "wind": self.weather["wind"],
            "irr": self.weather["irr"], "load": load, "wind_pow": gen["wind_pow"], "solar_pow": gen["solar_pow"],
            "renew_pow": gen["total"], "soc_ai": self.ai.soc, "soc_base": self.base.soc,
            "diesel_ai": decision["diesel_kw"], "diesel_base": dec_base["diesel_kw"],
            "fuel_ai": self.ai.fuel_l, "frac_renew": frac_renew, "mode": self.mode, "reason": decision["reason"],
            "shed_kw": decision.get("shed_kw", 0), "batt_kw": decision["batt_kw"],
            "fuel_cum_ai": self.audit["fuel_ai"], "fuel_cum_base": self.audit["fuel_base"],
        })
        if len(self.history) > 720:
            self.history = self.history[-720:]

        self.decision_log.insert(0, {
            "t": self.sim_time.isoformat(), "mode": self.mode, "diesel_kw": decision["diesel_kw"],
            "batt_kw": decision["batt_kw"], "reason": decision["reason"],
        })
        self.decision_log = self.decision_log[:80]

        if not self.online:
            self.outbox += 1
        else:
            self.sync_step()

    # -- serialization helpers for the API layer -----------------------------
    def snapshot(self) -> dict:
        last = self.history[-1] if self.history else None
        return {
            "sim_time": self.sim_time.isoformat(),
            "running": self.running,
            "speed": self.speed,
            "online": self.online,
            "syncing": self.syncing,
            "outbox": self.outbox,
            "mode": self.mode,
            "scenario": self.scenario,
            "weather": self.weather,
            "battery": {
                "soc": round(self.ai.soc, 1), "temp": round(self.ai.batt_temp, 1),
                "cycles": round(self.ai.cycles, 1), "health": round(self.health["battery"], 1),
                "charging": self.current_decision.get("batt_kw", 0) > 0 if self.current_decision else False,
            },
            "diesel": {
                "on": self.ai.diesel_on, "output_kw": round(self.current_decision.get("diesel_kw", 0), 1) if self.current_decision else 0,
                "fuel_l": round(self.ai.fuel_l, 1), "tank_cap_l": CFG.DIESEL_TANK_CAP_L,
                "runtime_h": round(self.ai.runtime_h, 2), "health": round(self.health["generator"], 1),
            },
            "load": last["load"] if last else None,
            "renewables": {
                "wind_kw": round(last["wind_pow"], 1) if last else 0,
                "solar_kw": round(last["solar_pow"], 1) if last else 0,
                "total_kw": round(last["renew_pow"], 1) if last else 0,
                "fraction_pct": round(last["frac_renew"], 1) if last else 0,
                "turbine_health": round(self.health["turbine"], 1),
            } if last else None,
            "decision": self.current_decision,
            "alerts_open": len([a for a in self.alerts]),
        }

    def green_audit(self) -> dict:
        fuel_saved = max(0.0, self.audit["fuel_base"] - self.audit["fuel_ai"])
        runtime_reduction = max(0.0, self.audit["runtime_base"] - self.audit["runtime_ai"])
        renew_frac = (clamp(self.audit["renew_gen_kwh"] / self.audit["load_served_kwh"], 0, 1.4) * 100
                      if self.audit["load_served_kwh"] > 0 else 0)
        return {
            "fuel_consumed_ai_l": round(self.audit["fuel_ai"], 1),
            "fuel_consumed_baseline_l": round(self.audit["fuel_base"], 1),
            "fuel_saved_l": round(fuel_saved, 1),
            "co2_avoided_kg": round(fuel_saved * CFG.CO2_PER_LITER, 1),
            "renewable_energy_generated_kwh": round(self.audit["renew_gen_kwh"], 1),
            "renewable_fraction_pct": round(min(100, renew_frac), 1),
            "generator_runtime_ai_h": round(self.audit["runtime_ai"], 2),
            "generator_runtime_baseline_h": round(self.audit["runtime_base"], 2),
            "generator_runtime_reduction_h": round(runtime_reduction, 2),
            "battery_cycles_ai": round(self.ai.cycles, 1),
            "battery_cycles_baseline": round(self.base.cycles, 1),
        }
