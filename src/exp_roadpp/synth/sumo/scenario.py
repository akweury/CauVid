"""The road networks: single junctions C with arms N, E, S, W (three or four of them), sidewalks and
pedestrian crossings (netconvert guesses them). Edge ids: "<X>2C" towards the junction, "C2<X>" away
from it; lane 0 of every edge is its sidewalk, lanes 1.. the road (1 = right).

`DEFAULT_SPEC` is the junction of sumo_junction_v0..v3 (one signalised four-way junction, 150 m arms,
two lanes). `random_spec` draws the junction family of the multi-map datasets (sumo_maps_*): T or
four-way, arm directions off the compass by up to ANGLE_JITTER, 1-3 lanes per arm, the speed limit,
the arm length (about 8-11 s of driving to the junction) and the control (signal, priority road,
all-way stop).

`random_grid_spec` draws the grid family of the long-video datasets (sumo_long_*): R x C junctions
J<r><c> (row and column as letters, e.g. Jab, so that "<from>2<to>" stays unambiguous) on a jittered grid
with blocks of BLOCK_M, a control per junction, lanes per street, and the outer arms ending in boundary
nodes B<k> (k as letters; where traffic enters and leaves); every junction has four arms.
Edge ids "<from>2<to>" as above (`heads_to_junction`)."""
import json
import math
import os
import random
import re
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


# the grid family of the long-video datasets
GRID_ROWS = (4, 5)                                    # junction rows / columns, uniform (3 x 3 grids are too
GRID_COLS = (4, 5)                                    # small: a 2 km ego walk inside them circles left)
BLOCK_M = (120.0, 200.0)                              # m between neighbouring junctions (per row / column gap)
TURN_WEIGHTS = {"straight": 0.5, "right": 0.3, "left": 0.2}   # a free choice at a junction (R4 aside)
FRINGE_M = (90.0, 130.0)                              # m of the outer arms (the ego departs on one)
NODE_JITTER = 8.0                                     # m, junction positions off the grid (arm angles vary)
GRID_CONTROLS = {"traffic_light": 0.5, "priority": 0.3, "allway_stop": 0.2}     # per junction


def heads_to_junction(road):
    """The edge leads into a junction (C of the single-junction maps, J<r><c> of the grids), not to the
    network boundary: a crossing vehicle on it comes towards a junction (IncomLane)."""
    return road.endswith("2C") or re.search(r"2J[a-z]+$", road) is not None


def _letters(k):
    """0 -> "a", 25 -> "z", 26 -> "ba" (no digits in grid node ids)."""
    out = ""
    while True:
        out = chr(ord("a") + k % 26) + out
        k //= 26
        if k == 0:
            return out


def _j(r, c):
    return f"J{_letters(r)}{_letters(c)}"


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


def random_grid_spec(rng):
    """One grid of the long-video family: {"layout": "grid", "rows", "cols", "speed", "main",
    "nodes": {id: {"x", "y", "type"}}, "edges": [{"id", "from", "to", "lanes", "priority"}]} (both
    directions of every street). "main": the priority streets ("rows" or "cols") at priority junctions."""
    rows, cols = rng.randint(*GRID_ROWS), rng.randint(*GRID_COLS)
    speed = round(rng.uniform(*SPEEDS), 1)
    main = rng.choice(("rows", "cols"))
    col_x = [0.0]
    for _ in range(cols - 1):
        col_x.append(col_x[-1] + rng.uniform(*BLOCK_M))
    row_y = [0.0]
    for _ in range(rows - 1):
        row_y.append(row_y[-1] - rng.uniform(*BLOCK_M))       # row 0 on top
    row_lanes = [_pick(rng, LANES) for _ in range(rows)]
    col_lanes = [_pick(rng, LANES) for _ in range(cols)]
    nodes, edges = {}, []

    def jitter():
        return rng.uniform(-NODE_JITTER, NODE_JITTER)

    for r in range(rows):
        for c in range(cols):
            nodes[_j(r, c)] = {"x": round(col_x[c] + jitter(), 2), "y": round(row_y[r] + jitter(), 2),
                                 "type": _pick(rng, GRID_CONTROLS)}

    def street(a, b, lanes, along):
        prio = 2 if along == main else 1
        for x, y in ((a, b), (b, a)):
            edges.append({"id": f"{x}2{y}", "from": x, "to": y, "lanes": lanes, "priority": prio})

    for r in range(rows):
        for c in range(cols - 1):
            street(_j(r, c), _j(r, c + 1), row_lanes[r], "rows")
    for c in range(cols):
        for r in range(rows - 1):
            street(_j(r, c), _j(r + 1, c), col_lanes[c], "cols")
    k = 0
    for r in range(rows):                                     # outer arms: west and east of every row
        for c, dx in ((0, -1), (cols - 1, 1)):
            j = nodes[_j(r, c)]
            nodes[f"B{_letters(k)}"] = {"x": round(j["x"] + dx * rng.uniform(*FRINGE_M), 2), "y": round(j["y"] + jitter(), 2),
                              "type": "priority"}
            street(_j(r, c), f"B{_letters(k)}", row_lanes[r], "rows")
            k += 1
    for c in range(cols):                                     # north and south of every column
        for r, dy in ((0, 1), (rows - 1, -1)):
            j = nodes[_j(r, c)]
            nodes[f"B{_letters(k)}"] = {"x": round(j["x"] + jitter(), 2), "y": round(j["y"] + dy * rng.uniform(*FRINGE_M), 2),
                              "type": "priority"}
            street(_j(r, c), f"B{_letters(k)}", col_lanes[c], "cols")
            k += 1
    return {"layout": "grid", "rows": rows, "cols": cols, "speed": speed, "main": main,
            "nodes": nodes, "edges": edges}


def is_grid(spec):
    return bool(spec) and spec.get("layout") == "grid"


def describe(spec):
    """Short name of a junction, e.g. "4arm_traffic_light_2-2-1-2"; of a grid, e.g. "grid3x4_tl6-pr4-as2"."""
    if is_grid(spec):
        types = [n["type"] for k, n in spec["nodes"].items() if k.startswith("J")]
        short = {"traffic_light": "tl", "priority": "pr", "allway_stop": "as"}
        return (f"grid{spec['rows']}x{spec['cols']}_"
                + "-".join(f"{short[t]}{types.count(t)}" for t in GRID_CONTROLS if t in types))
    return f"{len(spec['arms'])}arm_{spec['control']}_" + "-".join(str(spec["arms"][a]["lanes"]) for a in spec["arms"])


def routes(spec=None):
    """Every vehicle route through the junction: (in edge, out edge), no U-turns (12 at four arms, 6 at three)."""
    arms = list((spec or DEFAULT_SPEC)["arms"])
    return [(f"{a}2C", f"C2{b}") for a in arms for b in arms if a != b]


def build_network(net_dir, spec=None, name="junction"):
    """Writes and converts the network of one junction (or a grid spec) once; returns the .net.xml path."""
    spec = spec or DEFAULT_SPEC
    net_dir = Path(net_dir)
    net_dir.mkdir(parents=True, exist_ok=True)
    net_file = net_dir / f"{name}.net.xml"
    if net_file.exists():
        return net_file
    if is_grid(spec):
        nodes = [f'<node id="{k}" x="{n["x"]:.2f}" y="{n["y"]:.2f}" type="{n["type"]}"/>'
                 for k, n in spec["nodes"].items()]
        edges = [f'<edge id="{e["id"]}" from="{e["from"]}" to="{e["to"]}" numLanes="{e["lanes"]}" '
                 f'speed="{spec["speed"]}" priority="{e["priority"]}"/>' for e in spec["edges"]]
    else:
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


def build_maps(sumo_dir, num_maps, seed, layout="junction"):
    """The junctions (layout "junction") or grids ("grid") of a multi-map dataset: sumo/maps/map_XXX.net.xml
    and sumo/maps.json {map id: {"spec", "name", "net"}} (net relative to sumo_dir). Same seed, same maps."""
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
        spec = random_grid_spec(rng) if layout == "grid" else random_spec(rng)
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


# --- routes on a grid: random walks without U-turns ---
def turn(spec, graph, e_in, e_out):
    """"left", "right" or "straight" from edge e_in onto e_out at the node between them."""
    n = spec["nodes"]
    a, b, c = graph["from"][e_in], graph["to"][e_in], graph["to"][e_out]
    ax, ay = n[b]["x"] - n[a]["x"], n[b]["y"] - n[a]["y"]
    bx, by = n[c]["x"] - n[b]["x"], n[c]["y"] - n[b]["y"]
    ang = math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by))     # ccw positive (y up)
    if abs(ang) < 35.0:
        return "straight"
    return "left" if ang > 0 else "right"


def grid_graph(spec):
    """{"out": {node: [edge ids leaving it]}, "to": {edge: node}, "from": {edge: node}, "length": {edge: m}}."""
    n = spec["nodes"]
    g = {"out": {}, "to": {}, "from": {}, "length": {}}
    for e in spec["edges"]:
        g["out"].setdefault(e["from"], []).append(e["id"])
        g["to"][e["id"]], g["from"][e["id"]] = e["to"], e["from"]
        g["length"][e["id"]] = math.hypot(n[e["to"]]["x"] - n[e["from"]]["x"], n[e["to"]]["y"] - n[e["from"]]["y"])
    return g


def random_walk(spec, graph, rng, first, min_m=0.0, max_hops=None, rule_r4=True):
    """A route from edge `first`: at every junction an edge onward (no U-turn, no edge twice), drawn by
    TURN_WEIGHTS; after a right turn the next junction is a left turn (injected rule R4, `rules.R4`).
    Until min_m metres are driven only edges into junctions are taken (the walk stays inside the grid);
    after that (or max_hops junctions) it leaves at the first chance. Returns (edges, turns), or None if
    it cannot go on (R4 or the edges already driven force it out too early)."""
    route, turns, dist, prev = [first], [], graph["length"][first], None
    while True:
        b = graph["to"][route[-1]]
        if not b.startswith("J"):
            return route, turns                                   # reached the boundary
        a = graph["from"][route[-1]]
        options = [e for e in graph["out"][b] if graph["to"][e] != a and e not in route]
        kinds = {e: turn(spec, graph, route[-1], e) for e in options}
        if rule_r4 and prev == "right":
            options = [e for e in options if kinds[e] == "left"]
        inside = [e for e in options if graph["to"][e].startswith("J")]
        done = dist >= min_m and (max_hops is None or len(turns) >= max_hops)
        if not done and dist < min_m:
            if not inside:
                return None
            options = inside
        elif done or (max_hops is not None and len(turns) >= 3 * max_hops):
            out = [e for e in options if not graph["to"][e].startswith("J")]
            options = out or options
        if not options:
            return None
        e = rng.choices(options, [TURN_WEIGHTS[kinds[x]] for x in options])[0]
        route.append(e)
        turns.append(kinds[e])
        dist += graph["length"][e]
        prev = kinds[e]
        if len(route) > 200:
            return route, turns
