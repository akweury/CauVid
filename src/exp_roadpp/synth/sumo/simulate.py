"""One SUMO simulation at 10 Hz: the injected rules act every step; from the ego's departure on, the world
state of every step is recorded (all vehicles, pedestrians and the ego's next signal), until the ego
leaves the network or the clip and a margin are recorded."""
import math

from src.exp_roadpp.synth.sumo import scenario, traffic
from src.exp_roadpp.synth.sumo.rules import RuleEngine, heading

DT = 0.1                    # s per step = per frame (roadpp: 10 fps)
CLIP_FRAMES = 200           # 20 s, as the roadpp videos
MARGIN_FRAMES = 10          # recorded after the clip: labels that look ahead (lane changes)
MAX_INSERT_WAIT = 60.0      # s the ego may wait for its insertion


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


def ego_signal(traci, ego):
    """The ego's next traffic light: its state char, distance and the stop line point, or None."""
    nxt = traci.vehicle.getNextTLS("ego")
    if not nxt or ego["road"].startswith(":"):
        return None
    _tls, _link, dist, state = nxt[0]
    x, y = traci.lane.getShape(ego["lane"])[-1]
    return {"state": state, "dist": round(dist, 2), "x": round(x, 2), "y": round(y, 2)}


def simulate(net_file, route_file, seed, rng, ego_depart):
    """{"frames": [world state per step from the ego's departure], "firings": rule log}."""
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
        engine = RuleEngine(traci, rng)
        lane_len, frames = {}, []
        t_end = ego_depart + MAX_INSERT_WAIT + (CLIP_FRAMES + MARGIN_FRAMES) * DT
        while traci.simulation.getTime() < t_end:
            traci.simulationStep()
            world = query_world(traci, traci.simulation.getTime(), lane_len)
            engine.step(world["t"], world)
            if "ego" in world["veh"]:
                world["tl"] = ego_signal(traci, world["veh"]["ego"])
                frames.append(world)
                if len(frames) >= CLIP_FRAMES + MARGIN_FRAMES:
                    break
            elif frames:
                break                                       # the ego left the network
    finally:
        traci.close()
    return {"frames": frames, "firings": engine.log}
