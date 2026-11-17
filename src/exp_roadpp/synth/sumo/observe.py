"""What the ego's front camera sees: a pinhole camera with the roadpp (Waymo front) image size and a
50 deg horizontal field of view, agents within MAX_RANGE, occlusion by nearer vehicles of at least
OCCLUDER_HEIGHT of the agent's height (1D angular z-buffer in the top view), and the agents' 3D boxes
projected to normalised image boxes [x1, y1, x2, y2]."""
import math

from src.exp_roadpp.synth.sumo import traffic

W_IMG, H_IMG = 1920, 1280
HFOV_DEG = 50.0
FX = (W_IMG / 2) / math.tan(math.radians(HFOV_DEG / 2))
CAM_HEIGHT = 2.0            # m above the road
CAM_BACK = 1.5              # m behind the ego's front bumper
MAX_RANGE = 80.0            # m
MIN_DEPTH = 0.3             # m, points nearer to the camera plane are dropped
MIN_BOX_PX = 2
MIN_VISIBLE = 0.3           # share of an agent's angular extent not hidden by nearer occluders
OCCLUDER_HEIGHT = 0.8       # an occluder hides an agent only if at least this share of its height


def heading(angle_deg):
    """Unit heading of a SUMO angle (degrees, 0 = north, clockwise)."""
    a = math.radians(angle_deg)
    return math.sin(a), math.cos(a)


class Camera:
    """The front camera of a vehicle (the ego, or any driver for the injected rules)."""
    def __init__(self, ego):
        self.fx, self.fy = heading(ego["angle"])             # forward
        self.lx, self.ly = -self.fy, self.fx                  # left
        front = ego["length"] / 2 - CAM_BACK
        self.x, self.y = ego["x"] + self.fx * front, ego["y"] + self.fy * front

    def to_cam(self, x, y):
        """(forward, left) of a world point in the camera frame (m)."""
        dx, dy = x - self.x, y - self.y
        return dx * self.fx + dy * self.fy, dx * self.lx + dy * self.ly

    def direction(self, vx, vy):
        """(forward, left) components of a world direction."""
        return vx * self.fx + vy * self.fy, vx * self.lx + vy * self.ly


def footprint(x, y, angle, length, width):
    hx, hy = heading(angle)
    rx, ry = hy, -hx                                          # right of the heading
    out = []
    for a in (length / 2, -length / 2):
        for b in (width / 2, -width / 2):
            out.append((x + hx * a + rx * b, y + hy * a + ry * b))
    return out


def project(cam, corners, z0, z1):
    """Normalised image box of the 3D box over the footprint `corners` and heights z0..z1, the angular
    interval (min, max) of its visible corners and its depth; None if it is not in the image."""
    us, vs, angles, depth = [], [], [], None
    for x, y in corners:
        f, l = cam.to_cam(x, y)
        if f < MIN_DEPTH:
            continue
        angles.append(math.atan2(l, f))
        depth = f if depth is None else min(depth, f)
        for z in (z0, z1):
            us.append(W_IMG / 2 - FX * l / f)
            vs.append(H_IMG / 2 - FX * (z - CAM_HEIGHT) / f)
    if not us:
        return None
    u0, u1 = max(min(us), 0.0), min(max(us), W_IMG)
    v0, v1 = max(min(vs), 0.0), min(max(vs), H_IMG)
    if u1 - u0 < MIN_BOX_PX or v1 - v0 < MIN_BOX_PX:
        return None
    box = [round(u0 / W_IMG, 6), round(v0 / H_IMG, 6), round(u1 / W_IMG, 6), round(v1 / H_IMG, 6)]
    return {"box": box, "a0": min(angles), "a1": max(angles), "depth": depth}


def _covered(a0, a1, intervals):
    """Length of [a0, a1] covered by the union of intervals."""
    parts = sorted((max(a0, b0), min(a1, b1)) for b0, b1 in intervals if b1 > a0 and b0 < a1)
    total, end = 0.0, a0
    for b0, b1 in parts:
        b0 = max(b0, end)
        if b1 > b0:
            total += b1 - b0
            end = b1
    return total


def visible(cam, agents):
    """{agent id: normalised box} of the agents the camera sees. agents: {id: {"corners", "z0", "z1",
    "height", "occluder", "x", "y"}}."""
    proj = {}
    for aid, a in agents.items():
        f, l = cam.to_cam(a["x"], a["y"])
        if math.hypot(f, l) > MAX_RANGE:
            continue
        p = project(cam, a["corners"], a["z0"], a["z1"])
        if p is not None:
            proj[aid] = p
    out = {}
    for aid, p in proj.items():
        width = p["a1"] - p["a0"]
        occluders = [(q["a0"], q["a1"]) for bid, q in proj.items()
                     if bid != aid and agents[bid]["occluder"] and q["depth"] < p["depth"] - 0.5
                     and agents[bid]["height"] >= OCCLUDER_HEIGHT * agents[aid]["height"]]
        if width <= 1e-9 or 1.0 - _covered(p["a0"], p["a1"], occluders) / width >= MIN_VISIBLE:
            out[aid] = p["box"]
    return out


def scene_agents(world, driver):
    """The vehicles (but the driver) and pedestrians of one world state as `visible` takes them."""
    agents = {}
    for vid, v in world["veh"].items():
        if vid == driver:
            continue
        agents[vid] = {"x": v["x"], "y": v["y"], "z0": 0.0, "z1": v["height"], "height": v["height"],
                       "occluder": v["cls"] != "Cyc",
                       "corners": footprint(v["x"], v["y"], v["angle"], v["length"], v["width"])}
    pl, pw, ph = traffic.PED_SIZE
    for pid, p in world["ped"].items():
        agents[pid] = {"x": p["x"], "y": p["y"], "z0": 0.0, "z1": ph, "height": ph, "occluder": False,
                       "corners": footprint(p["x"], p["y"], p["angle"], pl, pw)}
    return agents
