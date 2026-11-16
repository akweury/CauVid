"""The road network: one signalised four-way junction C with arms N, E, S, W of ARM m, two lanes per
direction, sidewalks and pedestrian crossings (netconvert guesses them). Edge ids: "<X>2C" towards the
junction, "C2<X>" away from it; lane 0 of every edge is its sidewalk, lanes 1-2 the road (1 = right)."""
import os
import shutil
import subprocess
from pathlib import Path

ARM = 150.0             # m, arm length (ego clip: ~10 s approach, the junction, the exit)
SPEED = 13.8            # m/s, speed limit; at most 13.89 so that netconvert guesses sidewalks
NUM_LANES = 2
ARMS = {"N": (0.0, ARM), "E": (ARM, 0.0), "S": (0.0, -ARM), "W": (-ARM, 0.0)}


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


def routes():
    """Every vehicle route through the junction: (in edge, out edge), no U-turns (12)."""
    return [(f"{a}2C", f"C2{b}") for a in ARMS for b in ARMS if a != b]


def build_network(net_dir):
    """Writes and converts the junction network once; returns the .net.xml path."""
    net_dir = Path(net_dir)
    net_dir.mkdir(parents=True, exist_ok=True)
    net_file = net_dir / "junction.net.xml"
    if net_file.exists():
        return net_file
    nodes = ['<node id="C" x="0" y="0" type="traffic_light"/>']
    nodes += [f'<node id="{a}" x="{x}" y="{y}" type="priority"/>' for a, (x, y) in ARMS.items()]
    edges = []
    for a in ARMS:
        edges.append(f'<edge id="{a}2C" from="{a}" to="C" numLanes="{NUM_LANES}" speed="{SPEED}"/>')
        edges.append(f'<edge id="C2{a}" from="C" to="{a}" numLanes="{NUM_LANES}" speed="{SPEED}"/>')
    (net_dir / "junction.nod.xml").write_text("<nodes>\n  " + "\n  ".join(nodes) + "\n</nodes>\n")
    (net_dir / "junction.edg.xml").write_text("<edges>\n  " + "\n  ".join(edges) + "\n</edges>\n")
    subprocess.run([sumo_binary("netconvert"),
                    "--node-files", str(net_dir / "junction.nod.xml"),
                    "--edge-files", str(net_dir / "junction.edg.xml"),
                    "--sidewalks.guess", "true", "--crossings.guess", "true",
                    "--no-turnarounds", "true", "--tls.default-type", "static",
                    "--output-file", str(net_file)], check=True, capture_output=True)
    return net_file
