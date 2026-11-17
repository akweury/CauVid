"""Rules injected into the drivers' behaviour through TraCI, whose form is known exactly (GT_RULES), and
the rules SUMO's own driver model implies (APPROX_RULES, approximate: their conditions are distances and
speeds, which the labels only partly show).

An exact rule is defined on what the labels show, as the driver sees it, and fires at the onset of its
body: R1 looks through the driver's own front camera (`observe`, the ego camera model), so for the ego
its body is exactly the label rule Ped:Wait2X(other) ∧ AV-Mov(self). (sumo_junction_v0 used distances
and speeds the labels do not show: no lift at the label level.)

Every rule is a behaviour of every driver of its classes (the ego included), so the ego is an ordinary
driver. The engine is stepped once per simulation step with the world state of `simulate.query_world`;
each trigger (body newly true) is logged, whether it fires (probability p) and whether the head could be
executed, so the exact hazard of the head is known from the log."""
import math

from src.exp_roadpp.synth.sumo import observe

SIG_RIGHT, SIG_LEFT, SIG_EMERGENCY = 1, 2, 4         # SUMO vehicle signal bits

MOTOR = ("Car", "MedVeh", "LarVeh", "Bus", "Mobike")
HAZARD_CLASSES = ("Car", "MedVeh", "LarVeh")

# Label Mov of a driver as `labels` gives it (not Stop, not Brake; turning in the junction is not told
# apart here: the labels give TurLft / TurRht there, so such triggers only add unexplained brakes)
MOV_SPEED = 0.5             # labels.STOP_OFF
MOV_ACC = -1.0              # labels.BRAKE_OFF

# R1: a driver whose label is Mov brakes 1-3 s after it first sees a pedestrian waiting to cross
# (Ped:Wait2X in its front camera); once per (driver, pedestrian, waiting episode), several pending
# brakes per driver allowed (a group of pedestrians: one trigger each).
# p = 1 for E1 (clean data, deterministic rules); sumo_junction_v1 (first try) had p 0.8 and a 5 s
# refractory, and its R1 lift was 1.6. p is a sweep parameter later (E4).
R1 = {"lag_s": (1.0, 3.0), "p": 1.0, "min_speed": 1.5, "speed_drop": 5.0, "decel": 3.3}
# R2: a driver that turns on the hazard lights brakes to a stop 1-2 s later (p 0.9), stays 5-12 s.
# The hazard lights start more often on vehicles the ego sees driving in its direction (rate_seen; they
# stay in view for the lag, oncoming ones leave it within 1-2 s): that only places the events where the
# camera is, the rule itself (HazLit -> Brake 1-2 s later) is the same for every vehicle.
R2 = {"rate_per_s": 0.003, "rate_seen_per_s": 0.15, "lag_s": (1.0, 2.0), "p": 1.0, "decel": 3.0,
      "hold_s": (5.0, 12.0), "off_s": (4.0, 8.0), "min_ahead_m": 60.0, "min_speed": 3.0}

GT_RULES = [
    {"id": "R1", "exact": True, "head": {"agent": "self", "label": "Brake", "av_label": "AV-Brake"},
     "body": [{"agent": "self", "label": "Mov", "av_label": "AV-Mov"},
              {"agent": "other", "class": "Ped", "label": "Wait2X", "where": "visible in the driver's front camera"}],
     "onset": "the first frame both body atoms hold (per pedestrian waiting episode)",
     "lag_s": list(R1["lag_s"]), "p": R1["p"], "classes": list(MOTOR),
     "notes": f"executed only at speed >= {R1['min_speed']:g} m/s (drop {R1['speed_drop']:g} m/s at "
              f"{R1['decel']:g} m/s^2)"},
    {"id": "R2", "exact": True, "head": {"agent": "self", "label": "Brake"},
     "body": [{"agent": "self", "label": "HazLit"}],
     "onset": "the hazard lights turn on",
     "lag_s": list(R2["lag_s"]), "p": R2["p"], "classes": list(HAZARD_CLASSES),
     "notes": f"hazard lights start at random (rate {R2['rate_per_s']:g}/s per vehicle, "
              f"{R2['rate_seen_per_s']:g}/s if the ego sees it driving in its direction; not the ego, "
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


def is_mov(v):
    """The driver's label is Mov (approximately: `labels` adds hysteresis and smoothing)."""
    return v["speed"] >= MOV_SPEED and v["acc"] > MOV_ACC


class RuleEngine:
    def __init__(self, traci, rng):
        self.traci = traci
        self.rng = rng
        self.log = []                   # every trigger
        self.wait_episode = {}          # pedestrian -> number of its waiting episodes so far
        self.waiting_before = set()
        self.seen_r1 = set()            # (driver, pedestrian, waiting episode)
        self.pending_r1 = []            # log entries of the fired, not yet executed triggers
        self.events_r2 = {}             # driver -> state of its hazard event
        self.had_r2 = set()

    def step(self, t, world):
        self._r1_triggers(t, world)
        self._r1_execute(t, world)
        ego = world["veh"].get("ego")
        seen = observe.visible(observe.Camera(ego), observe.scene_agents(world, "ego")) if ego else {}
        ahead = {vid for vid in seen if vid in world["veh"]
                 and abs((world["veh"][vid]["angle"] - ego["angle"] + 180.0) % 360.0 - 180.0) < 45.0}
        self._r2(t, world, ahead)

    # -------------- R1 --------------
    def _r1_triggers(self, t, world):
        waiting = {pid for pid, p in world["ped"].items() if is_waiting(p)}
        for pid in waiting - self.waiting_before:
            self.wait_episode[pid] = self.wait_episode.get(pid, 0) + 1
        self.waiting_before = waiting
        if not waiting:
            return
        agents = None
        for vid, v in world["veh"].items():
            if v["cls"] not in MOTOR or not is_mov(v):
                continue
            new = [pid for pid in waiting if (vid, pid, self.wait_episode[pid]) not in self.seen_r1
                   and math.hypot(world["ped"][pid]["x"] - v["x"], world["ped"][pid]["y"] - v["y"]) <= observe.MAX_RANGE]
            if not new:
                continue
            if agents is None:
                agents = observe.scene_agents(world, None)
            seen = observe.visible(observe.Camera(v), {k: a for k, a in agents.items() if k != vid})
            for pid in new:
                if pid not in seen:
                    continue
                self.seen_r1.add((vid, pid, self.wait_episode[pid]))
                entry = {"rule": "R1", "agent": vid, "body_agent": pid, "t_body": t, "fired": False}
                if self.rng.random() < R1["p"]:
                    lag = self.rng.uniform(*R1["lag_s"])
                    entry.update(fired=True, lag=round(lag, 2), t_fire=round(t + lag, 1))
                    self.pending_r1.append(entry)
                self.log.append(entry)

    def _r1_execute(self, t, world):
        due = [e for e in self.pending_r1 if t + 1e-6 >= e["t_fire"]]
        self.pending_r1 = [e for e in self.pending_r1 if t + 1e-6 < e["t_fire"]]
        for entry in due:
            vid = entry["agent"]
            v = world["veh"].get(vid)
            entry["executed"] = bool(v and v["speed"] >= R1["min_speed"])
            if entry["executed"]:
                target = max(v["speed"] - R1["speed_drop"], 0.0)
                self.traci.vehicle.slowDown(vid, target, max(0.5, (v["speed"] - target) / R1["decel"]))

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
