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


def write_routes(route_file, rng, params=None, spec=None):
    """Writes the route file of one simulation on the junction spec (`scenario`, default: the v0-v3
    junction); returns {"ego_route": (in, out), "ego_depart": s}. The vehicle rate is per four arms (a
    T junction gets 3/4 of it)."""
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
