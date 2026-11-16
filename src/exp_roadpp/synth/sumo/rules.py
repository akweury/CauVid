"""Rules injected into the drivers' behaviour through TraCI, whose form is known exactly (GT_RULES), and
the rules SUMO's own driver model implies (APPROX_RULES, approximate: their conditions are distances and
speeds, which the labels only partly show).

Every rule is a behaviour of every driver of its classes (the ego included), so the ego is an ordinary
driver. The engine is stepped once per simulation step with the world state of `simulate.query_world`;
each trigger (body newly true) is logged, whether it fires (probability p) and whether the head could be
executed, so the exact hazard of the head is known from the log."""
import math

SIG_RIGHT, SIG_LEFT, SIG_EMERGENCY = 1, 2, 4         # SUMO vehicle signal bits

MOTOR = ("Car", "MedVeh", "LarVeh", "Bus", "Mobike")
HAZARD_CLASSES = ("Car", "MedVeh", "LarVeh")

# R1: a moving driver brakes 1-3 s after a pedestrian waiting to cross appears ahead of it (p 0.8);
# drivers slower than trigger_speed (e.g. queued at the red light) do not react.
R1 = {"range_m": 50.0, "half_angle_deg": 25.0, "lag_s": (1.0, 3.0), "p": 0.8, "trigger_speed": 6.0, "min_speed": 3.0,
      "speed_drop": 5.0, "brake_s": 1.5, "refractory_s": 8.0}
# R2: a driver that turns on the hazard lights brakes to a stop 1-2 s later (p 0.9), stays 5-12 s.
R2 = {"rate_per_s": 0.015, "lag_s": (1.0, 2.0), "p": 0.9, "decel": 3.0, "hold_s": (5.0, 12.0),
      "off_s": (4.0, 8.0), "min_ahead_m": 75.0, "min_speed": 3.0}

GT_RULES = [
    {"id": "R1", "exact": True, "head": {"agent": "self", "label": "Brake", "av_label": "AV-Brake"},
     "body": [{"agent": "self", "label": "moving", "where": f"speed >= {R1['trigger_speed']:g} m/s (not Stop / Brake)"},
              {"agent": "other", "class": "Ped", "label": "Wait2X",
               "where": f"within {R1['range_m']:g} m and {R1['half_angle_deg']:g} deg of the driver's heading"}],
     "lag_s": list(R1["lag_s"]), "p": R1["p"], "classes": list(MOTOR),
     "notes": f"executed only at speed >= {R1['min_speed']:g} m/s (drop {R1['speed_drop']:g} m/s in "
              f"{R1['brake_s']:g} s); one trigger per (driver, pedestrian); refractory {R1['refractory_s']:g} s"},
    {"id": "R2", "exact": True, "head": {"agent": "self", "label": "Brake"},
     "body": [{"agent": "self", "label": "HazLit"}],
     "lag_s": list(R2["lag_s"]), "p": R2["p"], "classes": list(HAZARD_CLASSES),
     "notes": f"hazard lights start at random (rate {R2['rate_per_s']:g}/s per vehicle, not the ego, "
              f">= {R2['min_ahead_m']:g} m before the lane end); the brake ends in Stop for "
              f"{R2['hold_s'][0]:g}-{R2['hold_s'][1]:g} s; without the brake the lights go off after "
              f"{R2['off_s'][0]:g}-{R2['off_s'][1]:g} s"},
]

APPROX_RULES = [
    {"id": "A1", "exact": False, "text": "TL:Red (the ego's signal) -> AV-Stop at the stop line"},
    {"id": "A2", "exact": False, "text": "<vehicle>:Brake / Stop ahead in VehLane -> AV-Brake / AV-Stop (car following)"},
    {"id": "A3", "exact": False, "text": "<vehicle>:IncatLft / IncatRht -> TurLft / TurRht or MovLft / MovRht (SUMO blinkers)"},
    {"id": "A4", "exact": False, "text": "Ped:XingFmLft / XingFmRht in front -> AV-Brake / AV-Stop (yielding)"},
    {"id": "A5", "exact": False, "text": "R2 consequence: <vehicle>:HazLit, Brake -> Stop"},
]


def is_waiting(ped):
    """Ped:Wait2X: standing on a walking area (where SUMO pedestrians wait for a crossing)."""
    return ped["road"].startswith(":") and "_w" in ped["road"] and ped["speed"] < 0.1


def heading(angle_deg):
    """Unit heading of a SUMO angle (degrees, 0 = north, clockwise)."""
    a = math.radians(angle_deg)
    return math.sin(a), math.cos(a)


class RuleEngine:
    def __init__(self, traci, rng):
        self.traci = traci
        self.rng = rng
        self.log = []                   # every trigger
        self.seen_r1 = set()            # (driver, pedestrian)
        self.pending_r1 = {}            # driver -> log entry
        self.last_r1 = {}               # driver -> time of the last executed R1
        self.events_r2 = {}             # driver -> state of its hazard event
        self.had_r2 = set()

    def step(self, t, world):
        self._r1_triggers(t, world)
        self._r1_execute(t, world)
        self._r2(t, world)

    # -------------- R1 --------------
    def _r1_triggers(self, t, world):
        waiting = {pid: p for pid, p in world["ped"].items() if is_waiting(p)}
        if not waiting:
            return
        cos_max = math.cos(math.radians(R1["half_angle_deg"]))
        for vid, v in world["veh"].items():
            if v["cls"] not in MOTOR or v["speed"] < R1["trigger_speed"]:
                continue
            hx, hy = heading(v["angle"])
            for pid, p in waiting.items():
                if (vid, pid) in self.seen_r1:
                    continue
                dx, dy = p["x"] - v["x"], p["y"] - v["y"]
                d = math.hypot(dx, dy)
                if d > R1["range_m"] or d < 1e-6 or (dx * hx + dy * hy) / d < cos_max:
                    continue
                self.seen_r1.add((vid, pid))
                entry = {"rule": "R1", "agent": vid, "body_agent": pid, "t_body": t, "fired": False}
                if self.rng.random() < R1["p"]:
                    busy = vid in self.pending_r1 or t - self.last_r1.get(vid, -1e9) < R1["refractory_s"]
                    if busy:
                        entry["skipped"] = "refractory"
                    else:
                        lag = self.rng.uniform(*R1["lag_s"])
                        entry.update(fired=True, lag=round(lag, 2), t_fire=round(t + lag, 1))
                        self.pending_r1[vid] = entry
                self.log.append(entry)

    def _r1_execute(self, t, world):
        for vid, entry in list(self.pending_r1.items()):
            if t + 1e-6 < entry["t_fire"]:
                continue
            del self.pending_r1[vid]
            v = world["veh"].get(vid)
            entry["executed"] = bool(v and v["speed"] >= R1["min_speed"])
            if entry["executed"]:
                target = max(v["speed"] - R1["speed_drop"], 0.5)
                self.traci.vehicle.slowDown(vid, target, R1["brake_s"])
                self.last_r1[vid] = t

    # -------------- R2 --------------
    def _r2(self, t, world):
        tv = self.traci.vehicle
        dt = world["dt"]
        for vid, v in world["veh"].items():
            if (vid == "ego" or vid in self.had_r2 or v["cls"] not in HAZARD_CLASSES
                    or v["road"].startswith(":") or v["speed"] < R2["min_speed"]
                    or v["lane_len"] - v["lane_pos"] < R2["min_ahead_m"]):
                continue
            if self.rng.random() < R2["rate_per_s"] * dt:
                self.had_r2.add(vid)
                tv.setSignals(vid, SIG_EMERGENCY)
                entry = {"rule": "R2", "agent": vid, "body_agent": vid, "t_body": t, "fired": False}
                if self.rng.random() < R2["p"]:
                    lag = self.rng.uniform(*R2["lag_s"])
                    entry.update(fired=True, lag=round(lag, 2), t_fire=round(t + lag, 1))
                    self.events_r2[vid] = {"phase": "hazard", "entry": entry}
                else:
                    self.events_r2[vid] = {"phase": "lights", "until": t + self.rng.uniform(*R2["off_s"]),
                                           "entry": entry}
                self.log.append(entry)

        for vid, ev in list(self.events_r2.items()):
            v = world["veh"].get(vid)
            if v is None:
                if ev["entry"].get("fired") and "executed" not in ev["entry"]:
                    ev["entry"]["executed"] = False
                del self.events_r2[vid]
                continue
            if ev["phase"] == "lights" and t >= ev["until"]:
                tv.setSignals(vid, -1)
                del self.events_r2[vid]
            elif ev["phase"] == "hazard" and t + 1e-6 >= ev["entry"]["t_fire"]:
                ev["entry"]["executed"] = v["speed"] > 1.0
                if ev["entry"]["executed"]:
                    tv.slowDown(vid, 0.0, max(v["speed"] / R2["decel"], 0.5))
                    ev["phase"] = "braking"
                else:
                    tv.setSignals(vid, -1)
                    del self.events_r2[vid]
            elif ev["phase"] == "braking" and v["speed"] < 0.1:
                tv.setSpeed(vid, 0.0)
                ev.update(phase="stopped", until=t + self.rng.uniform(*R2["hold_s"]))
            elif ev["phase"] == "stopped" and t >= ev["until"]:
                tv.setSpeed(vid, -1)
                tv.setSignals(vid, -1)
                del self.events_r2[vid]
