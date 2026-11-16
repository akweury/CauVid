"""World states of one simulation -> one ROAD++ ground-truth video ({"vid", "data": {"numf", "frames",
"agent_tubes", "av_action_tubes", ...}}), as `step_dataset` reads it, with the roadpp label.json.

Labels come from the simulator state with fixed thresholds (hysteresis against flicker). Every label is
relative to the ego where the roadpp definition is (MovAway / MovTow, Xing from left / right, lanes,
pavements). Rare roadpp labels (Rev, PushObj, Ovtak, EmVeh, bus / cycle lanes, parking) are not made."""
import math
from collections import defaultdict

from src.exp_roadpp.synth.sumo import observe, traffic
from src.exp_roadpp.synth.sumo.rules import SIG_EMERGENCY, SIG_LEFT, SIG_RIGHT, is_waiting, heading

STOP_ON, STOP_OFF = 0.3, 0.5        # m/s: Stop below STOP_ON until above STOP_OFF
BRAKE_ON, BRAKE_OFF = -1.5, -1.0    # m/s^2: Brake below BRAKE_ON until above BRAKE_OFF
TURN_RATE = 8.0                     # deg/s yaw rate inside the junction: TurLft / TurRht
YAW_SMOOTH = 2                      # frames on each side of the yaw rate mean
LC_HALF = 10                        # frames before / after a lane index change: MovLft / MovRht
DIR_COS = 0.5                       # |cos(heading, line of sight)| above: MovAway / MovTow, else crossing
FILL_GAP = 3                        # occluded frames between two visible runs kept in the tube
ACC_SMOOTH = 2                      # frames on each side of the acceleration mean (SUMO's driver noise)
DEBOUNCE = 3                        # Stop / Brake: gaps of at most this many frames filled, shorter runs dropped
TL_ID = "__tl__"


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _hysteresis(values, on, off, below=True):
    state, out = False, []
    for v in values:
        state = (v < on if below else v > on) if not state else (v < off if below else v > off)
        out.append(state)
    return out


def _smooth(values, half):
    out = []
    for i in range(len(values)):
        w = values[max(0, i - half):i + half + 1]
        out.append(sum(w) / len(w))
    return out


def _debounce(flags, n):
    """Gaps of at most n False frames inside True runs filled, then True runs shorter than n dropped."""
    out = list(flags)
    idx = [i for i, f in enumerate(out) if f]
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= n + 1:
            out[a:b] = [True] * (b - a)
    i = 0
    while i < len(out):
        if out[i]:
            j = i
            while j < len(out) and out[j]:
                j += 1
            if j - i < n:
                out[i:j] = [False] * (j - i)
            i = j
        else:
            i += 1
    return out


def agent_streams(frames):
    """Per agent (vehicles incl. the ego, pedestrians): {frame index: {"stop", "brake", "yaw", "lc"}}."""
    dt = frames[0]["dt"] if frames else 0.1
    series = defaultdict(list)
    for k, w in enumerate(frames):
        for vid, v in w["veh"].items():
            series[vid].append((k, v))
        for pid, p in w["ped"].items():
            series[pid].append((k, p))
    out = {}
    for aid, seq in series.items():
        ks = [k for k, _ in seq]
        recs = [r for _, r in seq]
        stop = _debounce(_hysteresis([r["speed"] for r in recs], STOP_ON, STOP_OFF), DEBOUNCE)
        acc = _smooth([r.get("acc", 0.0) for r in recs], ACC_SMOOTH)
        brake = _debounce(_hysteresis(acc, BRAKE_ON, BRAKE_OFF), DEBOUNCE)
        rate = [0.0] + [_wrap(recs[i]["angle"] - recs[i - 1]["angle"]) / dt for i in range(1, len(recs))]
        yaw = _smooth(rate, YAW_SMOOTH)
        lc = [None] * len(recs)
        for i in range(1, len(recs)):
            a, b = recs[i - 1], recs[i]
            if ("lane_index" in b and a["road"] == b["road"] and not b["road"].startswith(":")
                    and a["lane_index"] != b["lane_index"] and ks[i] == ks[i - 1] + 1):
                side = "Lft" if b["lane_index"] > a["lane_index"] else "Rht"     # SUMO lane 0 is rightmost
                for j in range(max(0, i - LC_HALF), min(len(recs), i + LC_HALF)):
                    lc[j] = side
        out[aid] = {k: {"stop": stop[i], "brake": brake[i], "yaw": yaw[i], "lc": lc[i]} for i, k in enumerate(ks)}
    return out


def _motion_label(cam, x, y, angle):
    """MovAway / MovTow along the line of sight, else XingFmLft / XingFmRht by the lateral direction."""
    f, l = cam.to_cam(x, y)
    d = math.hypot(f, l) or 1.0
    hx, hy = heading(angle)
    hf, hl = cam.direction(hx, hy)
    c = (hf * f + hl * l) / d
    if c > DIR_COS:
        return "MovAway"
    if c < -DIR_COS:
        return "MovTow"
    return "XingFmLft" if hl < 0 else "XingFmRht"


def vehicle_labels(cam, v, s, ego):
    """(actions, locations) of a vehicle (Car .. Bus, Cyc, Mobike)."""
    acts = []
    internal = v["road"].startswith(":")
    if s["stop"]:
        acts.append("Stop")
    else:
        if internal and abs(s["yaw"]) > TURN_RATE:
            acts.append("TurRht" if s["yaw"] > 0 else "TurLft")          # SUMO angles turn clockwise
        elif s["lc"]:
            acts.append("Mov" + s["lc"])
        else:
            acts.append(_motion_label(cam, v["x"], v["y"], v["angle"]))
        if s["brake"]:
            acts.append("Brake")
    if v["cls"] != "Cyc":
        if v["signals"] & SIG_EMERGENCY:
            acts.append("HazLit")
        else:
            if v["signals"] & SIG_LEFT:
                acts.append("IncatLft")
            if v["signals"] & SIG_RIGHT:
                acts.append("IncatRht")
    if internal:
        loc = "Jun"
    elif v["lane"] == ego["lane"]:
        loc = "VehLane"
    else:
        diff = abs(_wrap(v["angle"] - ego["angle"]))
        if diff < 45:
            loc = "OutgoLane"
        elif diff > 135:
            loc = "IncomLane"
        else:
            loc = "IncomLane" if v["road"].endswith("2C") else "OutgoLane"
    return acts, [loc]


def ped_labels(cam, p, s):
    hx, hy = heading(p["angle"])
    hf, hl = cam.direction(hx, hy)
    on_crossing = p["road"].startswith(":") and "_c" in p["road"]
    if on_crossing:
        if p["speed"] > 0.1 and abs(hl) > abs(hf):
            acts = ["XingFmLft" if hl < 0 else "XingFmRht"]
        else:
            acts = ["Xing"]
    elif is_waiting(p):
        acts = ["Wait2X"]
    elif not s["stop"]:
        m = _motion_label(cam, p["x"], p["y"], p["angle"])
        acts = [m if m in ("MovAway", "MovTow") else "Mov"]
    else:
        acts = ["Stop"]
    if on_crossing:
        loc = "xing"
    else:
        loc = "LftPav" if cam.to_cam(p["x"], p["y"])[1] > 0 else "RhtPav"
    return acts, [loc]


def tl_label(state):
    c = state[0] if state else ""
    return {"r": "Red", "R": "Red", "y": "Amber", "Y": "Amber", "u": "Amber",
            "g": "Green", "G": "Green"}.get(c)


def av_label(ego, s):
    if s["stop"]:
        return "AV-Stop"
    if s["brake"]:
        return "AV-Brake"
    if ego["road"].startswith(":") and abs(s["yaw"]) > TURN_RATE:
        return "AV-TurRht" if s["yaw"] > 0 else "AV-TurLft"
    if s["lc"]:
        return "AV-Mov" + s["lc"]
    return "AV-Mov"


def _scene_agents(w, cam):
    """Agents of one frame as `observe.visible` takes them (the ego excluded), with the ego's signal."""
    agents = {}
    for vid, v in w["veh"].items():
        if vid == "ego":
            continue
        agents[vid] = {"x": v["x"], "y": v["y"], "z0": 0.0, "z1": v["height"], "height": v["height"],
                       "occluder": v["cls"] not in ("Cyc",),
                       "corners": observe.footprint(v["x"], v["y"], v["angle"], v["length"], v["width"])}
    pl, pw, ph = traffic.PED_SIZE
    for pid, p in w["ped"].items():
        agents[pid] = {"x": p["x"], "y": p["y"], "z0": 0.0, "z1": ph, "height": ph, "occluder": False,
                       "corners": observe.footprint(p["x"], p["y"], p["angle"], pl, pw)}
    tl = w.get("tl")
    if tl and tl_label(tl["state"]):
        # the signal head on a pole 3 m right of the ego's stop line point; never occluded (height)
        right_x, right_y = cam.fy, -cam.fx
        x, y = tl["x"] + 3.0 * right_x, tl["y"] + 3.0 * right_y
        sl, sw, sh = traffic.TL_SIZE
        agents[TL_ID] = {"x": x, "y": y, "z0": traffic.TL_BOTTOM, "z1": traffic.TL_BOTTOM + sh,
                         "height": float("inf"), "occluder": False,
                         "corners": observe.footprint(x, y, 0.0, sl, sw)}
    return agents


def _fill_runs(frame_ids, gap):
    """Runs of frame indices with gaps of at most `gap` missing frames filled."""
    runs = []
    for k in sorted(frame_ids):
        if runs and k - runs[-1][-1] <= gap + 1:
            runs[-1].extend(range(runs[-1][-1] + 1, k + 1))
        else:
            runs.append([k])
    return runs


def build_video(vid, frames, labels_json, num_frames):
    """The ROAD++ ground truth of the first num_frames recorded frames; also {agent id: set of visible
    frame indices} (the tubes' frames, 0-based) for the rule log."""
    agent_ix = {n: i for i, n in enumerate(labels_json["agent_labels"])}
    action_ix = {n: i for i, n in enumerate(labels_json["action_labels"])}
    loc_ix = {n: i for i, n in enumerate(labels_json["loc_labels"])}
    av_ix = {n: i for i, n in enumerate(labels_json["av_action_labels"])}
    streams = agent_streams(frames)

    seen = defaultdict(dict)                    # agent -> {frame index: (box, cls, actions, locations)}
    av_ids = []
    for k in range(num_frames):
        w = frames[k]
        ego = w["veh"]["ego"]
        cam = observe.Camera(ego)
        av_ids.append(av_ix[av_label(ego, streams["ego"][k])])
        agents = _scene_agents(w, cam)
        boxes = observe.visible(cam, agents)
        # the frames of filled gaps need a box too: every projectable agent, visible or not
        for aid, a in agents.items():
            box = boxes.get(aid)
            hidden = box is None
            if hidden:
                p = observe.project(cam, a["corners"], a["z0"], a["z1"])
                if p is None:
                    continue
                box = p["box"]
            if aid == TL_ID:
                cls, acts, locs = "TL", [tl_label(w["tl"]["state"])], []
            elif aid in w["veh"]:
                v = w["veh"][aid]
                cls = v["cls"]
                acts, locs = vehicle_labels(cam, v, streams[aid][k], ego)
            else:
                cls = "Ped"
                acts, locs = ped_labels(cam, w["ped"][aid], streams[aid][k])
            seen[aid][k] = (box, cls, acts, locs, hidden)

    frames_out = {str(k + 1): {"annotated": 1, "rgb_image_id": str(k + 1), "width": observe.W_IMG,
                               "height": observe.H_IMG, "annos": {}, "av_action_ids": [av_ids[k]]}
                  for k in range(num_frames)}
    agent_tubes, visible_frames = {}, {}
    for aid, per_frame in seen.items():
        shown = [k for k, rec in per_frame.items() if not rec[4]]
        runs = [[k for k in run if k in per_frame] for run in _fill_runs(shown, FILL_GAP)]
        tube_uid = f"{vid}-{aid}"
        visible_frames[aid] = set()
        for j, run in enumerate(runs):
            annos = {}
            for k in run:
                box, cls, acts, locs, _hidden = per_frame[k]
                box_id = f"b{k + 1}_{aid}"
                frames_out[str(k + 1)]["annos"][box_id] = {
                    "box": box, "agent_ids": [agent_ix[cls]], "loc_ids": [loc_ix[x] for x in locs],
                    "action_ids": [action_ix[x] for x in acts], "duplex_ids": [], "triplet_ids": [],
                    "tube_uid": tube_uid}
                annos[str(k + 1)] = box_id
                visible_frames[aid].add(k)
            agent_tubes[f"{tube_uid}-{j + 1:03d}"] = {"label_id": agent_ix[per_frame[run[0]][1]], "annos": annos}

    av_tubes, start = {}, 0
    for k in range(1, num_frames + 1):
        if k == num_frames or av_ids[k] != av_ids[start]:
            av_tubes[f"{vid}-av-{len(av_tubes):03d}"] = {"label_id": av_ids[start],
                                                         "frames": [str(f + 1) for f in range(start, k)]}
            start = k
    data = {"numf": num_frames, "split_ids": ["all", "train"], "frames": frames_out, "agent_tubes": agent_tubes,
            "action_tubes": {}, "loc_tubes": {}, "duplex_tubes": {}, "triplet_tubes": {}, "av_action_tubes": av_tubes}
    return {"vid": vid, "data": data}, visible_frames
