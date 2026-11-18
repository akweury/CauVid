"""One SUMO simulation at 10 Hz: the injected rules act every step; from the ego's departure on, the world
state of every step is recorded (all vehicles, pedestrians and the ego's next signal), until the clip and
a margin are recorded, the ego leaves the network or MAX_RECORD_S pass.

The clip of CLIP_FRAMES (`clip_window`) is cut by one of CLIP_MODES, chosen per video (`run.py` mixes them
so that every event of interest gets enough instances, since sumo_v8):
  depart:   the first 20 s after the departure (sumo_v6 and before): the approach, a light turning red
            while the ego drives (N3); at a red light it ends while the ego still waits (signal cycles of
            90 s, the ego's red lasts 40-50 s)
  junction: ends POST_JUNCTION_S after the ego enters the junction (sumo_v7): the wait at a red light
            and the change to green (N4), the turn; every stop in it is followed by the start
  random:   ends anywhere between 20 s after the departure and POST_JUNCTION_S after entering the
            junction: neither the start nor the end of the clip is tied to an event
Clips start CLIP_FRAMES before their end, at the departure at the earliest. An ego that never enters the
junction keeps its last CLIP_FRAMES frames (junction, random)."""
import math

from src.exp_roadpp.synth.sumo import scenario, traffic
from src.exp_roadpp.synth.sumo.observe import heading
from src.exp_roadpp.synth.sumo.rules import RuleEngine

DT = 0.1                    # s per step = per frame (roadpp: 10 fps)
CLIP_FRAMES = 200           # 20 s, as the roadpp videos
MARGIN_FRAMES = 10          # recorded after the clip: labels that look ahead (lane changes)
MAX_INSERT_WAIT = 60.0      # s the ego may wait for its insertion
MAX_RECORD_S = 150.0        # s recorded at most after the departure (queues at a red light)
POST_JUNCTION_S = (1.5, 5.0)    # s the clip goes on after the ego enters the junction (uniform)
JUNCTION = ":C"             # internal edges of the junction node C (`scenario.build_network`)
CLIP_MODES = ("depart", "junction", "random")


def _traci():
    """libsumo (in-process, fast) if it loads, else the TraCI client (socket)."""
    try:
        import libsumo
        return libsumo
    except (ImportError, OSError):
        import traci
        return traci


def query_world(traci, t, lane_len):
    """World state of the current step; vehicle positions are box centres (SUMO gives the front)."""
    veh = {}
    tv = traci.vehicle
    for vid in tv.getIDList():
        fx, fy = tv.getPosition(vid)
        angle = tv.getAngle(vid)
        length = tv.getLength(vid)
        hx, hy = heading(angle)
        lane = tv.getLaneID(vid)
        if lane not in lane_len:
            lane_len[lane] = traci.lane.getLength(lane)
        cls = traffic.agent_class(tv.getTypeID(vid))
        veh[vid] = {"x": round(fx - hx * length / 2, 3), "y": round(fy - hy * length / 2, 3),
                    "angle": round(angle, 2), "speed": round(tv.getSpeed(vid), 3),
                    "acc": round(tv.getAcceleration(vid), 3), "lane": lane, "lane_index": tv.getLaneIndex(vid),
                    "road": tv.getRoadID(vid), "signals": tv.getSignals(vid), "cls": cls,
                    "length": length, "width": tv.getWidth(vid), "height": traffic.VEHICLE_CLASSES[cls][4],
                    "lane_pos": round(tv.getLanePosition(vid), 2), "lane_len": round(lane_len[lane], 2)}
    ped = {}
    tp = traci.person
    for pid in tp.getIDList():
        x, y = tp.getPosition(pid)
        ped[pid] = {"x": round(x, 3), "y": round(y, 3), "angle": round(tp.getAngle(pid), 2),
                    "speed": round(tp.getSpeed(pid), 3), "road": tp.getRoadID(pid)}
    return {"t": round(t, 1), "dt": DT, "veh": veh, "ped": ped}


FAR_SIDE_BEYOND = 2.0       # m past the far side of the junction: where the ego's signal head hangs


def _far_side(traci, lane, x, y, hx, hy):
    """Distance from the stop line (x, y) across the junction along the heading (hx, hy): the farthest
    start of a lane the ego's lane leads to (straight on: the opposite side; at a T: the cross road's far edge)."""
    best = None
    for link in traci.lane.getLinks(lane):
        sx, sy = traci.lane.getShape(link[0])[0]
        d = (sx - x) * hx + (sy - y) * hy
        best = d if best is None else max(best, d)
    return best


def ego_signal(traci, ego):
    """The ego's next traffic light: its state char, distance, the stop line point (x, y) and the point
    where its signal head hangs (far_x, far_y): over the ego's lane, FAR_SIDE_BEYOND past the far side of
    the junction (as US mast-arm signals), so the ego sees it while it waits at the stop line; or None."""
    nxt = traci.vehicle.getNextTLS("ego")
    if not nxt or ego["road"].startswith(":"):
        return None
    _tls, _link, dist, state = nxt[0]
    if dist > ego["lane_len"] - ego["lane_pos"] + 1.0:
        return None                                         # grids: the next signal is at a later junction
    x, y = traci.lane.getShape(ego["lane"])[-1]
    hx, hy = heading(ego["angle"])
    across = _far_side(traci, ego["lane"], x, y, hx, hy)
    across = max(across or 0.0, 5.0) + FAR_SIDE_BEYOND
    return {"state": state, "dist": round(dist, 2), "x": round(x, 2), "y": round(y, 2),
            "far_x": round(x + hx * across, 2), "far_y": round(y + hy * across, 2)}


def clip_window(num_recorded, enter, post_frames, mode="junction", rng=None, clip_frames=CLIP_FRAMES):
    """(start, end) frame indices of the clip (end exclusive, without the margin) in the recorded frames;
    enter: the first frame of the ego in the junction, or None; rng draws the end of a "random" clip."""
    available = num_recorded - MARGIN_FRAMES
    if mode == "depart":
        end = min(available, clip_frames)
    else:
        latest = available if enter is None else min(available, max(enter + post_frames, clip_frames))
        end = latest                                        # junction; never entered: the last frames
        if mode == "random":
            end = rng.randint(min(clip_frames, latest), latest)
        elif mode != "junction":
            raise ValueError(f"clip mode {mode!r}, known: {CLIP_MODES}")
    return max(0, end - clip_frames), end


def simulate(net_file, route_file, seed, rng, ego_depart, clip_mode="junction", clip_frames=CLIP_FRAMES):
    """{"frames": [world state per step of the clip and its margin], "firings": rule log,
    "clip": {"mode", "start_s": clip start after the departure, "enter_s": ego in the junction (or None)}}.
    clip_frames: CLIP_FRAMES (20 s) for the junction maps; long clips (grids) from the departure on."""
    traci = _traci()
    cmd = [scenario.sumo_binary("sumo"), "-n", str(net_file), "-r", str(route_file),
           "--step-length", str(DT), "--seed", str(seed), "--lanechange.duration", "2",
           "--time-to-teleport", "120", "--collision.action", "warn", "--no-step-log", "true",
           "--no-warnings", "true", "--pedestrian.model", "striping"]
    traci.start(cmd)
    try:
        for tls in traci.trafficlight.getIDList():         # random signal offset per simulation
            phases = traci.trafficlight.getAllProgramLogics(tls)[0].phases
            traci.trafficlight.setPhase(tls, rng.randrange(len(phases)))
        post_frames = int(round(rng.uniform(*POST_JUNCTION_S) / DT))
        engine = RuleEngine(traci, rng)
        lane_len, frames, enter = {}, [], None
        t_end = ego_depart + MAX_INSERT_WAIT + max(MAX_RECORD_S, clip_frames * DT + 30.0)
        while traci.simulation.getTime() < t_end:
            traci.simulationStep()
            world = query_world(traci, traci.simulation.getTime(), lane_len)
            engine.step(world["t"], world)
            if "ego" in world["veh"]:
                world["tl"] = ego_signal(traci, world["veh"]["ego"])
                frames.append(world)
                if enter is None and world["veh"]["ego"]["road"].startswith(JUNCTION):
                    enter = len(frames) - 1
                if clip_mode == "depart" and len(frames) >= clip_frames + MARGIN_FRAMES:
                    break
                if enter is not None and len(frames) >= max(enter + post_frames, clip_frames) + MARGIN_FRAMES:
                    break
            elif frames:
                break                                       # the ego left the network
    finally:
        traci.close()
    start, end = clip_window(len(frames), enter, post_frames, clip_mode, rng, clip_frames)
    clip = {"mode": clip_mode, "start_s": round(start * DT, 1), "enter_s": None if enter is None else round(enter * DT, 1),
            "recorded_s": round(len(frames) * DT, 1)}
    return {"frames": frames[start:end + MARGIN_FRAMES], "firings": engine.log, "clip": clip}
