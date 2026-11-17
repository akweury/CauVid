"""Rules of the synthetic drivers: rules injected through TraCI, whose form is known exactly (GT_RULES), and
rules SUMO's own driver model produces (NATURAL_RULES: their order is fixed, their timing depends on
distances and speeds the labels do not show).

An injected rule is defined on what the labels show, as the driver sees it, and fires at the onset of its
body: R1 and R3 look through the driver's own front camera (`observe`, the ego camera model), so for the
ego their bodies are exactly label rules. (sumo_junction_v0 used distances and speeds instead: no lift in
the labels.) p = 1 for E1 (clean data, deterministic rules); p is a sweep parameter later (E4).

Every rule is a behaviour of every driver of its classes (the ego included), so the ego is an ordinary
driver. The engine is stepped once per simulation step with the world state of `simulate.query_world`;
each trigger (body newly true) is logged, whether it fires (probability p) and whether the head could be
executed, so the exact hazard of the head is known from the log."""
import math

from src.exp_roadpp.synth.sumo import observe

SIG_RIGHT, SIG_LEFT, SIG_EMERGENCY = 1, 2, 4         # SUMO vehicle signal bits

MOTOR = ("Car", "MedVeh", "LarVeh", "Bus", "Mobike")
HAZARD_CLASSES = ("Car", "MedVeh", "LarVeh")

# The labels a rule reads, as `labels` gives them (approximately: the labels add hysteresis, smoothing)
MOV_SPEED = 0.5             # Mov: labels.STOP_OFF; not braking (MOV_ACC = labels.BRAKE_OFF)
MOV_ACC = -1.0
BRAKE_ON, BRAKE_OFF = -1.5, -1.0    # Brake: labels.BRAKE_ON / BRAKE_OFF on the mean acceleration of
ACC_WINDOW = 5                      # the last ACC_WINDOW steps (the labels: a centred mean)

# R1: a driver whose label is Mov brakes 1-2 s after it sees a pedestrian start to cross (Ped:Xing /
# XingFmLft / XingFmRht in its front camera); once per (driver, pedestrian, crossing).
R1 = {"lag_s": (1.0, 2.0), "p": 1.0, "min_speed": 1.5, "speed_drop": 5.0, "decel": 3.3}
# R2: a driver that turns on the hazard lights brakes to a stop 1-2 s later, stays 5-12 s. The lights
# start more often on vehicles the ego sees driving in its direction (rate_seen; oncoming ones leave the
# view within 1-2 s): that only places the events where the camera is, the rule is the same for all.
R2 = {"rate_per_s": 0.003, "rate_seen_per_s": 0.15, "lag_s": (1.0, 2.0), "p": 1.0, "decel": 3.0,
      "hold_s": (5.0, 12.0), "off_s": (4.0, 8.0), "min_ahead_m": 60.0, "min_speed": 3.0}
# R3: a driver whose label is Mov brakes 0.5-1.5 s after it sees the vehicle ahead in its own lane
# (VehLane) start to brake; once per (driver, vehicle ahead, brake onset). Brakes spread back a queue.
R3 = {"lag_s": (0.5, 1.5), "p": 1.0, "min_speed": 1.5, "speed_drop": 4.0, "decel": 3.5}

GT_RULES = [
    {"id": "R1", "exact": True, "form": "other -> self",
     "head": {"agent": "self", "label": "Brake", "av_label": "AV-Brake"},
     "body": [{"agent": "self", "label": "Mov", "av_label": "AV-Mov"},
              {"agent": "other", "class": "Ped", "label": "Xing / XingFmLft / XingFmRht",
               "where": "visible in the driver's front camera"}],
     "onset": "the first frame the driver sees the pedestrian on the crossing while Mov",
     "lag_s": list(R1["lag_s"]), "p": R1["p"], "classes": list(MOTOR),
     "notes": f"executed only at speed >= {R1['min_speed']:g} m/s (drop {R1['speed_drop']:g} m/s at "
              f"{R1['decel']:g} m/s^2)"},
    {"id": "R2", "exact": True, "form": "self -> self",
     "head": {"agent": "self", "label": "Brake"},
     "body": [{"agent": "self", "label": "HazLit"}],
     "onset": "the hazard lights turn on",
     "lag_s": list(R2["lag_s"]), "p": R2["p"], "classes": list(HAZARD_CLASSES),
     "notes": f"hazard lights start at random (rate {R2['rate_per_s']:g}/s per vehicle, "
              f"{R2['rate_seen_per_s']:g}/s if the ego sees it driving in its direction; not the ego, "
              f">= {R2['min_ahead_m']:g} m before the lane end); the brake ends in Stop for "
              f"{R2['hold_s'][0]:g}-{R2['hold_s'][1]:g} s"},
    {"id": "R3", "exact": True, "form": "other -> self",
     "head": {"agent": "self", "label": "Brake", "av_label": "AV-Brake"},
     "body": [{"agent": "self", "label": "Mov", "av_label": "AV-Mov"},
              {"agent": "other", "class": "vehicle", "label": "Brake", "location": "VehLane",
               "where": "ahead in the driver's lane, visible in its front camera"}],
     "onset": "the vehicle ahead starts to brake",
     "lag_s": list(R3["lag_s"]), "p": R3["p"], "classes": list(MOTOR),
     "notes": f"executed only at speed >= {R3['min_speed']:g} m/s (drop {R3['speed_drop']:g} m/s at "
              f"{R3['decel']:g} m/s^2)"},
]

NATURAL_RULES = [
    {"id": "N1", "exact": False, "form": "other -> self",
     "text": "<vehicle>:MovAway in VehLane (the vehicle ahead starts) while AV-Stop -> AV-Mov (queue discharge)"},
    {"id": "N2", "exact": False, "form": "self sequence (common cause: the route)",
     "text": "Mov -> MovRht -> TurRht (vehicles depart in a random lane and change to the turn lane)"},
    {"id": "N3", "exact": False, "form": "self sequence (causal chain)",
     "text": "AV-Mov, TL:Red -> AV-Brake -> AV-Stop"},
    {"id": "N4", "exact": False, "form": "other -> self",
     "text": "<vehicle>:Brake / Stop ahead in VehLane -> Brake / Stop (car following, weaker than R3)"},
    {"id": "N5", "exact": False, "form": "self -> self",
     "text": "IncatLft / IncatRht -> TurLft / TurRht or MovLft / MovRht (SUMO blinkers)"},
]


def is_waiting(ped):
    """Ped:Wait2X: standing on a walking area (where SUMO pedestrians wait for a crossing)."""
    return ped["road"].startswith(":") and "_w" in ped["road"] and ped["speed"] < 0.1


def on_crossing(ped):
    """Ped:Xing / XingFmLft / XingFmRht: on a pedestrian crossing."""
    return ped["road"].startswith(":") and "_c" in ped["road"]


def is_mov(v):
    """The driver's label is Mov (not Stop, not Brake; turning in the junction is not told apart here)."""
    return v["speed"] >= MOV_SPEED and v["acc"] > MOV_ACC


class RuleEngine:
    def __init__(self, traci, rng):
        self.traci = traci
        self.rng = rng
        self.log = []                   # every trigger
        self.pending = []               # log entries of fired, not yet executed triggers (R1, R3)
        self.seen = set()               # (rule, driver, body agent, body episode) already triggered
        self.crossing_episode = {}      # pedestrian -> number of its crossings so far
        self.crossing_before = set()
        self.acc_hist = {}              # vehicle -> its last ACC_WINDOW accelerations
        self.braking = set()            # vehicles in the Brake state
        self.brake_episode = {}         # vehicle -> number of its brake onsets so far
        self.events_r2 = {}             # driver -> state of its hazard event
        self.had_r2 = set()

    def step(self, t, world):
        crossing_new = self._crossings(world)
        brake_new = self._brake_onsets(world)
        self._triggers(t, world, crossing_new, brake_new)
        self._execute(t, world)
        ego = world["veh"].get("ego")
        seen = observe.visible(observe.Camera(ego), observe.scene_agents(world, "ego")) if ego else {}
        ahead = {vid for vid in seen if vid in world["veh"]
                 and abs((world["veh"][vid]["angle"] - ego["angle"] + 180.0) % 360.0 - 180.0) < 45.0}
        self._r2(t, world, ahead)

    # -------------- body states --------------
    def _crossings(self, world):
        crossing = {pid for pid, p in world["ped"].items() if on_crossing(p)}
        for pid in crossing - self.crossing_before:
            self.crossing_episode[pid] = self.crossing_episode.get(pid, 0) + 1
        self.crossing_before = crossing
        return crossing

    def _brake_onsets(self, world):
        """Vehicles whose Brake state turns on in this step."""
        new = set()
        for vid, v in world["veh"].items():
            hist = self.acc_hist.setdefault(vid, [])
            hist.append(v["acc"])
            del hist[:-ACC_WINDOW]
            acc = sum(hist) / len(hist)
            if vid in self.braking:
                if acc > BRAKE_OFF or v["speed"] < MOV_SPEED:
                    self.braking.discard(vid)
            elif acc < BRAKE_ON and v["speed"] >= MOV_SPEED:
                self.braking.add(vid)
                self.brake_episode[vid] = self.brake_episode.get(vid, 0) + 1
                new.add(vid)
        for vid in set(self.acc_hist) - set(world["veh"]):
            del self.acc_hist[vid]
            self.braking.discard(vid)
        return new

    # -------------- R1, R3: triggers seen through the driver's camera --------------
    def _triggers(self, t, world, crossing, brake_new):
        agents, views = None, {}
        for vid, v in world["veh"].items():
            if v["cls"] not in MOTOR or not is_mov(v):
                continue
            cands = []
            for pid in crossing:
                key = ("R1", vid, pid, self.crossing_episode[pid])
                p = world["ped"][pid]
                if key not in self.seen and math.hypot(p["x"] - v["x"], p["y"] - v["y"]) <= observe.MAX_RANGE:
                    cands.append((key, pid))
            for lid in brake_new:
                lead = world["veh"][lid]
                if (lid == vid or lead["lane"] != v["lane"] or lead["lane_pos"] <= v["lane_pos"]
                        or lead["lane_pos"] - v["lane_pos"] > observe.MAX_RANGE):
                    continue
                key = ("R3", vid, lid, self.brake_episode[lid])
                if key not in self.seen:
                    cands.append((key, lid))
            if not cands:
                continue
            if agents is None:
                agents = observe.scene_agents(world, None)
            if vid not in views:
                views[vid] = observe.visible(observe.Camera(v), {k: a for k, a in agents.items() if k != vid})
            for key, body in cands:
                if body not in views[vid]:
                    continue
                self.seen.add(key)
                rule = key[0]
                spec = R1 if rule == "R1" else R3
                entry = {"rule": rule, "agent": vid, "body_agent": body, "t_body": t, "fired": False}
                if self.rng.random() < spec["p"]:
                    lag = self.rng.uniform(*spec["lag_s"])
                    entry.update(fired=True, lag=round(lag, 2), t_fire=round(t + lag, 1))
                    self.pending.append(entry)
                self.log.append(entry)

    def _execute(self, t, world):
        due = [e for e in self.pending if t + 1e-6 >= e["t_fire"]]
        self.pending = [e for e in self.pending if t + 1e-6 < e["t_fire"]]
        for entry in due:
            spec = R1 if entry["rule"] == "R1" else R3
            v = world["veh"].get(entry["agent"])
            entry["executed"] = bool(v and v["speed"] >= spec["min_speed"])
            if entry["executed"]:
                target = max(v["speed"] - spec["speed_drop"], 0.0)
                self.traci.vehicle.slowDown(entry["agent"], target, max(0.5, (v["speed"] - target) / spec["decel"]))

    # -------------- R2 --------------
    def _r2(self, t, world, seen_by_ego):
        tv = self.traci.vehicle
        dt = world["dt"]
        for vid, v in world["veh"].items():
            if (vid == "ego" or vid in self.had_r2 or v["cls"] not in HAZARD_CLASSES
                    or v["road"].startswith(":") or v["speed"] < R2["min_speed"]
                    or v["lane_len"] - v["lane_pos"] < R2["min_ahead_m"]):
                continue
            rate = R2["rate_seen_per_s"] if vid in seen_by_ego else R2["rate_per_s"]
            if self.rng.random() < rate * dt:
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
