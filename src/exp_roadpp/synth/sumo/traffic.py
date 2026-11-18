"""Random traffic of one simulation: vehicles of the roadpp agent classes on random routes, each with
its own sampled driver (vType), pedestrians walking between random arms, and the ego (a Car) that
departs after the warm-up. Class names are the roadpp agent labels."""
from xml.sax.saxutils import quoteattr

from src.exp_roadpp.synth.sumo import scenario

# agent label: SUMO vClass, share of the vehicles (roughly the roadpp mix: mostly cars), length, width, height (m), max speed (m/s), accel, decel
VEHICLE_CLASSES = {
    "Car":    ("passenger",  0.78,  4.5, 1.8, 1.5, 16.0, 2.6, 4.5),
    "MedVeh": ("delivery",   0.10,  6.0, 2.1, 2.5, 14.0, 2.0, 4.0),
    "LarVeh": ("truck",      0.04, 10.0, 2.5, 3.5, 13.0, 1.3, 4.0),
    "Bus":    ("bus",        0.04, 12.0, 2.5, 3.2, 13.0, 1.2, 4.0),
    "Cyc":    ("bicycle",    0.02,  1.6, 0.6, 1.7,  5.5, 1.2, 3.0),
    "Mobike": ("motorcycle", 0.02,  2.0, 0.8, 1.5, 16.0, 3.0, 6.0),
}
PED_SIZE = (0.5, 0.5, 1.75)             # length, width, height
TL_SIZE = (0.4, 0.4, 1.0)               # signal head; its bottom at TL_BOTTOM
TL_BOTTOM = 4.0

# Vehicles (the ego too) depart in a random lane, so those in the wrong lane for their turn change lanes
# before the junction (natural rule N2: Mov -> MovRht -> TurRht); with "best" they started in the turn lane.
DEFAULTS = {
    "vehicle_rate": 0.8,        # vehicles per s entering the network (all routes)
    "person_rate": 0.15,        # pedestrians per s
    "warmup": 60.0,             # s before the ego departs
    "duration": 60.0,           # s of traffic generated after the warm-up
}


def agent_class(type_id):
    """The roadpp agent label of a vType / vehicle type id "<class>__<k>" (ego: "Car__ego")."""
    return type_id.split("__", 1)[0]


def _vtype(type_id, cls, rng):
    vclass, _share, length, width, height, vmax, accel, decel = VEHICLE_CLASSES[cls]
    attrs = {
        "id": type_id, "vClass": vclass, "length": length, "width": width, "height": height,
        "maxSpeed": vmax, "accel": round(accel * rng.uniform(0.8, 1.2), 2),
        "decel": round(decel * rng.uniform(0.9, 1.1), 2), "sigma": round(rng.uniform(0.2, 0.5), 2),
        "tau": round(rng.uniform(0.8, 1.6), 2), "minGap": round(rng.uniform(1.5, 3.0), 2),
        "speedFactor": round(min(1.25, max(0.75, rng.gauss(1.0, 0.1))), 3),
        "lcAssertive": round(rng.uniform(0.8, 1.5), 2),
    }
    return "<vType " + " ".join(f"{k}={quoteattr(str(v))}" for k, v in attrs.items()) + "/>"


# Grids (long videos): traffic per junction, generated for the whole long clip; every vehicle drives a
# random walk (`scenario.random_walk`, which also applies R4), the ego a long one from a boundary arm.
GRID_DEFAULTS = {
    "vehicle_rate_per_junction": 0.15,  # vehicles per s and junction, departing on random edges into junctions
    "person_rate_per_junction": 0.08,   # pedestrians per s and junction, walking across one junction
    "warmup": 90.0,                     # s before the ego departs (the grid fills)
    "max_hops": (1, 5),                 # junctions a vehicle passes (uniform), then it leaves at the first chance
    "ego_tries": 2000,                  # random walks drawn until one stays inside for the clip
    "ego_route_share": 0.8,             # the ego's route: clip x speed limit x this (+100 m); shorter
}                                       # (stops make the drive longer) biases the turns less towards left


def write_routes(route_file, rng, params=None, spec=None, clip_s=None):
    """Writes the route file of one simulation on the junction spec (`scenario`, default: the v0-v3
    junction) or grid spec (then clip_s: the long clip the ego's route must fill); returns
    {"ego_route": edges, "ego_depart": s}. The vehicle rate is per four arms (a T junction gets 3/4 of it)."""
    if scenario.is_grid(spec):
        return _write_grid_routes(route_file, rng, params, spec, clip_s)
    p = {**DEFAULTS, **(params or {})}
    arms = len((spec or scenario.DEFAULT_SPEC)["arms"])
    vehicle_rate = p["vehicle_rate"] * arms / 4
    end = p["warmup"] + p["duration"]
    classes = list(VEHICLE_CLASSES)
    shares = [VEHICLE_CLASSES[c][1] for c in classes]
    all_routes = scenario.routes(spec)
    types, items = [], []

    t, k = 0.0, 0
    while True:
        t += rng.expovariate(vehicle_rate)
        if t >= end:
            break
        cls = rng.choices(classes, shares)[0]
        type_id = f"{cls}__{k}"
        types.append(_vtype(type_id, cls, rng))
        a, b = rng.choice(all_routes)
        items.append((t, f'<vehicle id="v{k}" type="{type_id}" depart="{t:.1f}" departLane="random" '
                         f'departSpeed="max"><route edges="{a} {b}"/></vehicle>'))
        k += 1

    ego_route = rng.choice(all_routes)
    types.append(_vtype("Car__ego", "Car", rng))
    items.append((p["warmup"], f'<vehicle id="ego" type="Car__ego" depart="{p["warmup"]:.1f}" departLane="random" '
                                f'departSpeed="max"><route edges="{ego_route[0]} {ego_route[1]}"/></vehicle>'))

    edges = [e for r in all_routes for e in r]
    edges = sorted(set(edges))
    t, k = 0.0, 0
    while True:
        t += rng.expovariate(p["person_rate"])
        if t >= end:
            break
        a, b = rng.sample(edges, 2)
        items.append((t, f'<person id="p{k}" depart="{t:.1f}" departPos="random">'
                         f'<walk from="{a}" to="{b}" arrivalPos="random"/></person>'))
        k += 1

    items.sort(key=lambda x: x[0])
    route_file.write_text("<routes>\n  " + "\n  ".join(types) + "\n  "
                          + "\n  ".join(x for _t, x in items) + "\n</routes>\n")
    return {"ego_route": list(ego_route), "ego_depart": p["warmup"]}


def _vehicle_classes(rng):
    classes = list(VEHICLE_CLASSES)
    return rng.choices(classes, [VEHICLE_CLASSES[c][1] for c in classes])[0]


def _write_grid_routes(route_file, rng, params, spec, clip_s):
    """Routes of one simulation on a grid: random-walk vehicles, pedestrians crossing at one junction each,
    and the ego on a walk from a boundary arm that stays inside the grid for ego_route_share of the clip
    at the speed limit (an ego that leaves before the clip ends is dropped by `run.py`). Staying inside
    biases the ego's free turns towards left (rights followed by R4's left near the boundary fail): on
    random grids P(next labelled turn left | left) ~0.54, vs 1 after a right."""
    p = {**GRID_DEFAULTS, **(params or {})}
    graph = scenario.grid_graph(spec)
    junctions = [k for k in spec["nodes"] if k.startswith("J")]
    into_junction = [e["id"] for e in spec["edges"] if e["to"].startswith("J")]
    duration = (clip_s or 120.0) + 120.0                    # the clip, the ego's insertion wait and a margin
    end = p["warmup"] + duration
    types, items = [], []

    t, k = 0.0, 0
    while True:
        t += rng.expovariate(p["vehicle_rate_per_junction"] * len(junctions))
        if t >= end:
            break
        walk = scenario.random_walk(spec, graph, rng, rng.choice(into_junction), max_hops=rng.randint(*p["max_hops"]))
        if walk is None:
            continue
        cls = _vehicle_classes(rng)
        type_id = f"{cls}__{k}"
        types.append(_vtype(type_id, cls, rng))
        items.append((t, f'<vehicle id="v{k}" type="{type_id}" depart="{t:.1f}" departLane="random" '
                         f'departPos="random" departSpeed="max"><route edges="{" ".join(walk[0])}"/></vehicle>'))
        k += 1

    starts = [e["id"] for e in spec["edges"] if e["from"].startswith("B")]
    min_m = (clip_s or 120.0) * spec["speed"] * p["ego_route_share"] + 100.0
    ego = None
    for _ in range(p["ego_tries"]):
        ego = scenario.random_walk(spec, graph, rng, rng.choice(starts), min_m=min_m)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError(f"no ego route of {min_m:.0f} m inside the grid in {p['ego_tries']} tries")
    types.append(_vtype("Car__ego", "Car", rng))
    items.append((p["warmup"], f'<vehicle id="ego" type="Car__ego" depart="{p["warmup"]:.1f}" departLane="random" '
                                f'departSpeed="max"><route edges="{" ".join(ego[0])}"/></vehicle>'))

    t, k = 0.0, 0
    while True:
        t += rng.expovariate(p["person_rate_per_junction"] * len(junctions))
        if t >= end:
            break
        j = rng.choice(junctions)
        a, b = rng.sample(graph["out"][j], 2)                 # two arms of one junction: a walk across it
        items.append((t, f'<person id="p{k}" depart="{t:.1f}" departPos="random">'
                         f'<walk from="{a}" to="{b}" arrivalPos="random"/></person>'))
        k += 1

    items.sort(key=lambda x: x[0])
    route_file.write_text("<routes>\n  " + "\n  ".join(types) + "\n  "
                          + "\n  ".join(x for _t, x in items) + "\n</routes>\n")
    return {"ego_route": ego[0], "ego_turns": ego[1], "ego_depart": p["warmup"]}
