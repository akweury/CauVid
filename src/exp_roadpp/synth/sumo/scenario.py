"""The road networks: single junctions C with arms N, E, S, W (three or four of them), sidewalks and
pedestrian crossings (netconvert guesses them). Edge ids: "<X>2C" towards the junction, "C2<X>" away
from it; lane 0 of every edge is its sidewalk, lanes 1.. the road (1 = right).

`DEFAULT_SPEC` is the junction of sumo_junction_v0..v3 (one signalised four-way junction, 150 m arms,
two lanes). `random_spec` draws the junction family of the multi-map datasets (sumo_maps_*): T or
four-way, arm directions off the compass by up to ANGLE_JITTER, 1-3 lanes per arm, the speed limit,
the arm length (about 8-11 s of driving to the junction) and the control (signal, priority road,
all-way stop)."""
import json
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

ARM = 150.0             # m, arm length of the default junction
SPEED = 13.8            # m/s, speed limit; at most 13.89 so that netconvert guesses sidewalks
NUM_LANES = 2
COMPASS = {"N": 90.0, "E": 0.0, "S": 270.0, "W": 180.0}      # arm direction from C (deg, from +x, ccw)

DEFAULT_SPEC = {"arms": {a: {"angle": COMPASS[a], "length": ARM, "lanes": NUM_LANES} for a in COMPASS},
                "speed": SPEED, "control": "traffic_light", "main": None}

# the junction family of the multi-map datasets
P_THREE_ARMS = 0.3
ANGLE_JITTER = 20.0                                   # deg; arms stay >= 50 deg apart
LANES = {1: 0.3, 2: 0.5, 3: 0.2}                      # road lanes per arm and direction
SPEEDS = (8.3, 13.8)                                  # m/s, 30-50 km/h
APPROACH_S = (8.0, 11.0)                              # s of driving from the arm's end to the junction
MIN_ARM = 80.0                                        # m (R2 hazards need 60 m ahead)
CONTROLS = {"traffic_light": 0.6, "priority": 0.25, "allway_stop": 0.15}


def sumo_binary(name):
    """Path of a SUMO binary (sumo, netconvert): $SUMO_HOME/bin, the eclipse-sumo pip package, or PATH."""
    candidates = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(Path(os.environ["SUMO_HOME"]) / "bin" / name)
    try:
        import sumo
        candidates.append(Path(sumo.SUMO_HOME) / "bin" / name)
    except ImportError:
        pass
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"SUMO binary {name} not found (pip install eclipse-sumo, or set SUMO_HOME)")


def _pick(rng, weights):
    return rng.choices(list(weights), list(weights.values()))[0]


def random_spec(rng):
    """One junction of the multi-map family."""
    arms = list(COMPASS)
    if rng.random() < P_THREE_ARMS:
        arms.remove(rng.choice(arms))                 # a T junction: one opposite pair stays (the main road)
    speed = round(rng.uniform(*SPEEDS), 1)
    control = _pick(rng, CONTROLS)
    pairs = [p for p in (("N", "S"), ("E", "W")) if all(a in arms for a in p)]
    main = list(rng.choice(pairs))                    # the priority road (control "priority")
    main_lanes = _pick(rng, LANES)
    spec = {"arms": {}, "speed": speed, "control": control, "main": main}
    for a in arms:
        lanes = main_lanes if a in main else _pick(rng, LANES)
        if control == "priority" and a not in main:
            lanes = min(lanes, main_lanes)            # side roads are not wider than the main road
        spec["arms"][a] = {"angle": round(COMPASS[a] + rng.uniform(-ANGLE_JITTER, ANGLE_JITTER), 1),
                           "length": round(max(MIN_ARM, speed * rng.uniform(*APPROACH_S)), 1),
                           "lanes": lanes}
    return spec


def describe(spec):
    """Short name of a junction, e.g. "4arm_traffic_light_2-2-1-2"."""
    return f"{len(spec['arms'])}arm_{spec['control']}_" + "-".join(str(spec["arms"][a]["lanes"]) for a in spec["arms"])


def routes(spec=None):
    """Every vehicle route through the junction: (in edge, out edge), no U-turns (12 at four arms, 6 at three)."""
    arms = list((spec or DEFAULT_SPEC)["arms"])
    return [(f"{a}2C", f"C2{b}") for a in arms for b in arms if a != b]


def build_network(net_dir, spec=None, name="junction"):
    """Writes and converts the network of one junction once; returns the .net.xml path."""
    spec = spec or DEFAULT_SPEC
    net_dir = Path(net_dir)
    net_dir.mkdir(parents=True, exist_ok=True)
    net_file = net_dir / f"{name}.net.xml"
    if net_file.exists():
        return net_file
    nodes = [f'<node id="C" x="0" y="0" type="{spec["control"]}"/>']
    edges = []
    for a, arm in spec["arms"].items():
        rad = math.radians(arm["angle"])
        x, y = arm["length"] * math.cos(rad), arm["length"] * math.sin(rad)
        nodes.append(f'<node id="{a}" x="{x:.2f}" y="{y:.2f}" type="priority"/>')
        prio = 2 if spec.get("main") and a in spec["main"] else 1
        for eid, frm, to in ((f"{a}2C", a, "C"), (f"C2{a}", "C", a)):
            edges.append(f'<edge id="{eid}" from="{frm}" to="{to}" numLanes="{arm["lanes"]}" '
                         f'speed="{spec["speed"]}" priority="{prio}"/>')
    (net_dir / f"{name}.nod.xml").write_text("<nodes>\n  " + "\n  ".join(nodes) + "\n</nodes>\n")
    (net_dir / f"{name}.edg.xml").write_text("<edges>\n  " + "\n  ".join(edges) + "\n</edges>\n")
    subprocess.run([sumo_binary("netconvert"),
                    "--node-files", str(net_dir / f"{name}.nod.xml"),
                    "--edge-files", str(net_dir / f"{name}.edg.xml"),
                    "--sidewalks.guess", "true", "--crossings.guess", "true",
                    "--no-turnarounds", "true", "--tls.default-type", "static",
                    "--output-file", str(net_file)], check=True, capture_output=True)
    return net_file


def build_maps(sumo_dir, num_maps, seed):
    """The junctions of a multi-map dataset: sumo/maps/map_XXX.net.xml and sumo/maps.json
    {map id: {"spec", "name", "net"}} (net relative to sumo_dir). Same seed, same maps."""
    sumo_dir = Path(sumo_dir)
    index_file = sumo_dir / "maps.json"
    if index_file.exists():
        maps = json.loads(index_file.read_text())
        if len(maps) == num_maps:
            return maps
        raise ValueError(f"{index_file} has {len(maps)} maps, not {num_maps}: use another dataset name")
    rng = random.Random(seed * 7_919 + 1)
    maps = {}
    for k in range(num_maps):
        mid = f"map_{k:03d}"
        spec = random_spec(rng)
        net = build_network(sumo_dir / "maps", spec, mid)
        maps[mid] = {"spec": spec, "name": describe(spec), "net": str(net.relative_to(sumo_dir))}
    index_file.write_text(json.dumps(maps, indent=1))
    return maps


def map_split(maps, fractions=(0.7, 0.15, 0.15)):
    """{"train", "val", "test": [map ids]}: the maps sorted by kind (control, arms) and dealt out in turn,
    so that every split has every kind."""
    order = sorted(maps, key=lambda m: (maps[m]["spec"]["control"], len(maps[m]["spec"]["arms"]), m))
    period = round(1 / min(fractions[1:]))            # e.g. every 7th map to val, the next to test
    out = {"train": [], "val": [], "test": []}
    for k, m in enumerate(order):
        r = k % period
        out["val" if r == 0 else "test" if r == 1 else "train"].append(m)
    return out
