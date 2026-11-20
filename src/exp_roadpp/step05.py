import math
import os
import sys
import getpass
import torch
from tqdm import tqdm

from src.exp_roadpp import utils_data, roles as agent_roles, geometry as agent_geometry, grid as agent_grid
from src.exp_roadpp import motion as agent_motion, ego_motion as agent_ego_motion, ego_flow as agent_ego_flow
from src.exp_roadpp import predict as agent_predict
from src.exp_roadpp import duration as agent_duration
from src.exp_roadpp.cauvid import TemporalRuleNet, memory_bins, slot_spans
from src.exp_roadpp.baselines.onset_mlp import OnsetMLP


def _video_frames(video_atoms):
    # The T frames of `build_video_tensor`: every current frame and the last next frame.
    per_frame_atoms = video_atoms["per_frame_atoms"]
    frames = [current_frame for current_frame, _next_frame in per_frame_atoms]
    if per_frame_atoms:
        frames.append(per_frame_atoms[-1][1])
    return frames


def video_agent_ids(video_atoms):
    """{tube_uid: agent index n} of `build_video_tensor`, in order of first appearance."""
    agent_ids = {}
    for frame_atoms in _video_frames(video_atoms):
        for _atom_index, _tag, tube_uid in frame_atoms:
            agent_ids.setdefault(tube_uid, len(agent_ids))
    return agent_ids


def _video_cells(X):
    # (T, N, A) bool tensor of `build_video_tensor` -> its compact form on CPU:
    # the (C, 3) int32 (t, n, a) of its True cells and its (T, N) shape.
    return {"cells": torch.nonzero(X).to(torch.int32).cpu(), "shape": tuple(X.shape[:2])}


def _video_from_cells(video, num_atoms, device):
    # Inverse of `_video_cells`; with num_atoms below the stored atoms (`label_atom_count`:
    # the label atoms only) the atoms from num_atoms on (the geometry atoms) are left out.
    X = torch.zeros((*video["shape"], num_atoms), dtype=torch.bool, device=device)
    t, n, a = video["cells"].to(device).long().T
    keep = a < num_atoms
    X[t[keep], n[keep], a[keep]] = True
    return X


def label_atom_count(data):
    """The number of GT label atoms of `prepare_videos` data: its first atoms; the
    geometry atoms (`geometry.py`) and the grid atoms (`grid.py`) follow them."""
    return data.get("num_label_atoms", data["num_atoms"])


def load_vocabulary(cfg):
    """The atom signatures of every atom index: the label vocabulary of
    cfg["unique_atoms_file"], then the geometry atoms of cfg["geometry"] (`geometry.vocabulary`),
    then the grid position atoms of cfg["grid"] (`grid.vocabulary`), then the motion atoms of
    cfg["motion"] (`motion.vocabulary`), then the ego motion atoms of cfg["ego_motion"]
    (`ego_motion.vocabulary`), then the ego flow atoms of cfg["ego_flow"] (`ego_flow.vocabulary`), then the
    duration atoms of cfg["durations"] (`duration.vocabulary`), then the prediction atoms of cfg["predict"]
    (`predict.vocabulary`)."""
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    out = unique_atoms + agent_geometry.vocabulary(unique_atoms, cfg.get("geometry", ()))
    if cfg.get("grid"):
        out += agent_grid.vocabulary()
    if cfg.get("motion"):
        out += agent_motion.vocabulary()
    if cfg.get("ego_motion"):
        out += agent_ego_motion.vocabulary()
    if cfg.get("ego_flow"):
        out += agent_ego_flow.vocabulary()
    if cfg.get("durations"):
        out += agent_duration.vocabulary(unique_atoms, duration_labels(cfg))
    if cfg.get("predict"):
        out += agent_predict.vocabulary(unique_atoms)
    return out


def is_input_atom(signature):
    """Geometry, grid, motion, ego motion, ego flow, duration and prediction atoms are inputs only: no head
    learns or is evaluated on them."""
    return len(signature) > 1 and signature[1] in ("geometry", "grid", "motion", "ego_motion", "ego_flow", "duration",
                                                   "predict")


# -------------- Grid atoms (`grid.py`) --------------
def grid_tag(cfg):
    """File tag of the box atoms: the grid settings and the motion atoms ("" without both). The
    grid spec file is named by the grid part only (`grid_tag({"grid": ...})`)."""
    tag = ""
    if cfg.get("grid"):
        tag += f"_grid{agent_grid.GRID_ROWS}x{agent_grid.GRID_COLS}q{agent_grid.OVERLAP:g}pos"
    if cfg.get("motion"):
        tag += agent_motion.tag()
    if cfg.get("ego_motion"):
        tag += agent_ego_motion.tag()
    if cfg.get("ego_flow"):
        tag += agent_ego_flow.tag()
    if cfg.get("durations"):
        tag += agent_duration.tag(duration_labels(cfg))
    if cfg.get("predict"):
        tag += agent_predict.tag()
    return tag


def _agent_boxes(boxes, video_id, video_atoms, X_labels):
    """(T, N, 4) boxes of one video's agents (`roles.video_boxes`) and (T, N) has_box: a box
    where the agent holds some label atom; the ego (no box) never has one."""
    agent_ids = video_agent_ids(video_atoms)
    video_boxes, classes = agent_roles.video_boxes(boxes.get(video_id), agent_ids, X_labels.shape[0])
    has_box = ~torch.isnan(video_boxes).any(dim=2) & X_labels.any(dim=2).cpu()
    ego = agent_ids.get(None)
    if ego is not None:
        has_box[:, ego] = False
    return video_boxes, classes, has_box


def fit_grid(cfg):
    """The grid of the train split (`grid.py`): the coordinate quantiles of each box corner (the
    rank map of the rows and columns)."""
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    A = len(unique_atoms)
    if not cfg.get("train_box_file") or not os.path.exists(cfg["train_box_file"]):
        raise FileNotFoundError(f"The grid needs the train box file {cfg.get('train_box_file')} (step_dataset).")
    boxes = torch.load(cfg["train_box_file"], map_location="cpu")
    atoms = utils_data.load_json(cfg["train_atom_file"])
    points, num_videos = [], 0
    for video_id, video_atoms in tqdm(atoms.items(), desc="Grid: train boxes"):
        X = build_video_tensor(video_atoms, A)
        if X.shape[0] < 2:
            continue
        video_boxes, _classes, has_box = _agent_boxes(boxes, video_id, video_atoms, X)
        num_videos += 1
        points.append(agent_grid.corner_points(video_boxes)[has_box])          # (K, 2 corners, 2 axes)
    del atoms
    points = torch.cat(points)
    sample = torch.randperm(len(points), generator=torch.Generator().manual_seed(0))[:2_000_000]
    knots = agent_grid.axis_knots(points[sample])
    print(f"[Grid]\t {agent_grid.GRID_ROWS}x{agent_grid.GRID_COLS} cells, {num_videos} train videos, {len(points)} boxes")
    return {"rows": agent_grid.GRID_ROWS, "cols": agent_grid.GRID_COLS, "overlap": agent_grid.OVERLAP,
            "knots": knots, "num_train_videos": num_videos}


def load_grid(cfg):
    """The grid of cfg["grid_file"] (fitted on the train split by `fit_grid` if missing), None without the grid."""
    if not cfg.get("grid"):
        return None
    if not os.path.exists(cfg["grid_file"]):
        utils_data.save_json(fit_grid(cfg), cfg["grid_file"])
    return utils_data.load_json(cfg["grid_file"])


# -------------- Motion atoms (`motion.py`) --------------
def fit_motion(cfg):
    """The motion thresholds of the train split (`motion.fit`): the signals of every train box, per class."""
    if not cfg.get("train_box_file") or not os.path.exists(cfg["train_box_file"]):
        raise FileNotFoundError(f"The motion atoms need the train box file {cfg.get('train_box_file')} (step_dataset).")
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    A = len(unique_atoms)
    boxes = torch.load(cfg["train_box_file"], map_location="cpu")
    atoms = utils_data.load_json(cfg["train_atom_file"])
    samples = {}
    for video_id, video_atoms in tqdm(atoms.items(), desc="Motion: train boxes"):
        X = build_video_tensor(video_atoms, A)
        if X.shape[0] <= agent_motion.K:
            continue
        video_boxes, classes, has_box = _agent_boxes(boxes, video_id, video_atoms, X)
        sig = agent_motion.signals(video_boxes, has_box)                         # (T, N, 4)
        for n, cls in enumerate(classes):
            v = sig[:, n][~torch.isnan(sig[:, n, 0])]
            if len(v):
                samples.setdefault(str(cls), []).append(v)
    del atoms
    spec = agent_motion.fit({cls: torch.cat(v) for cls, v in samples.items()})
    print(f"[Motion]\t thresholds per class (own fit for >= {agent_motion.MIN_CLASS_SAMPLES} train signals): "
          + ", ".join(f"{c} {n}{'' if c in spec['classes'] else ' (all)'}" for c, n in sorted(spec["counts"].items())))
    for cls, t in [("all", spec["all"])] + sorted(spec["classes"].items()):
        print(f"[Motion]\t {cls}: " + ", ".join(f"{s} {[round(v, 4) for v in t[s]]}" for s in agent_motion.SIGNALS))
    return {**spec, "k": agent_motion.K, "debounce": agent_motion.DEBOUNCE, "version": agent_motion.VERSION}


def load_motion(cfg):
    """The motion thresholds of cfg["motion_file"] (fitted on the train split by `fit_motion` if
    missing), None without the motion atoms."""
    if not cfg.get("motion"):
        return None
    if not os.path.exists(cfg["motion_file"]):
        utils_data.save_json(fit_motion(cfg), cfg["motion_file"])
    return utils_data.load_json(cfg["motion_file"])


# -------------- Ego motion atoms (`ego_motion.py`) --------------
def fit_ego_motion(cfg):
    """The ego motion level cuts of the train split (`ego_motion.fit`): the signals of every train video's ego."""
    if not cfg.get("train_box_file") or not os.path.exists(cfg["train_box_file"]):
        raise FileNotFoundError(f"The ego motion atoms need the train box file {cfg.get('train_box_file')} (step_dataset).")
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    A = len(unique_atoms)
    boxes = torch.load(cfg["train_box_file"], map_location="cpu")
    atoms = utils_data.load_json(cfg["train_atom_file"])
    samples = {s: [] for s in agent_ego_motion.SIGNALS}
    for video_id, video_atoms in tqdm(atoms.items(), desc="Ego motion: train videos"):
        ego = video_agent_ids(video_atoms).get(None)
        X = build_video_tensor(video_atoms, A)
        if ego is None or X.shape[0] <= agent_motion.K:
            continue
        video_boxes, classes, has_box = _agent_boxes(boxes, video_id, video_atoms, X)
        sig = agent_ego_motion.signals(video_boxes, has_box, classes, ego, X, unique_atoms)
        seen = X[:, ego].any(dim=1)
        for s, v in sig.items():
            samples[s].append(v[seen])
    del atoms
    spec = agent_ego_motion.fit({s: torch.cat(v) if v else torch.zeros(0) for s, v in samples.items()})
    for s, cuts in spec["cuts"].items():
        print(f"[Ego motion]\t {s}: cuts {[round(c, 4) for c in cuts]} ({spec['counts'][s]} train frames)")
    for s, band in spec.get("bands", {}).items():
        print(f"[Ego motion]\t {s}: dead band +-{band:.5f} ({spec['counts'][s]} train frames)")
    return {**spec, "smooth": agent_ego_motion.SMOOTH, "version": agent_ego_motion.VERSION}


def load_ego_motion(cfg):
    """The ego motion cuts of cfg["ego_motion_file"] (fitted on the train split by `fit_ego_motion`
    if missing), None without the ego motion atoms."""
    if not cfg.get("ego_motion"):
        return None
    if not os.path.exists(cfg["ego_motion_file"]):
        utils_data.save_json(fit_ego_motion(cfg), cfg["ego_motion_file"])
    return utils_data.load_json(cfg["ego_motion_file"])


# -------------- Duration atoms (`duration.py`) --------------
def duration_labels(cfg):
    """The labels that get duration atoms: cfg["duration_labels"] (comma separated or a list), default
    `duration.LABELS`."""
    labels = cfg.get("duration_labels") or agent_duration.LABELS
    if isinstance(labels, str):
        labels = [x for x in labels.split(",") if x]
    return tuple(labels)


def load_predict(cfg):
    """The prediction atom settings (`predict.py`: fixed camera constants, nothing fitted), None without them."""
    if not cfg.get("predict"):
        return None
    return {"version": agent_predict.VERSION, "camera": agent_predict.CAMERA}


def load_durations(cfg):
    """The duration atom settings ({"labels"}), None without the duration atoms."""
    if not cfg.get("durations"):
        return None
    return {"labels": duration_labels(cfg)}


# -------------- Ego flow atoms (`ego_flow.py`) --------------
def load_ego_flow(cfg):
    """The ego flow settings (where the per-video flow cache lives and how to fill it), None without
    the ego flow atoms. The level cuts are fixed in `ego_flow` (from `ego_flow_diag`), not fitted."""
    if not cfg.get("ego_flow"):
        return None
    return {"cache_dir": str(cfg["ego_flow_cache_dir"]), "frame_dir": str(cfg.get("frame_dir") or ""),
            "video_dir": str(cfg.get("video_dir") or ""), "fps": int(cfg.get("frame_rate", 10)),
            "workers": int(cfg.get("ego_flow_workers", 16)), "tag": agent_ego_flow.tag()}


def _ensure_ego_flow(atoms, boxes, spec):
    """Fill the flow cache for every video of one split (`ego_flow.ensure_cache`, decodes the videos once)."""
    jobs = []
    for video_id, video_atoms in atoms.items():
        T = len(_video_frames(video_atoms))
        if T < 2:
            continue
        by_frame = {}
        for tube in ((boxes or {}).get(video_id) or {}).values():
            for f, b in zip(tube["frames"].tolist(), tube["boxes"].tolist()):
                by_frame.setdefault(int(f), []).append(tuple(b))
        jobs.append((video_id, T, by_frame))
    agent_ego_flow.ensure_cache(jobs, spec["cache_dir"], spec["frame_dir"], spec["video_dir"], spec["fps"],
                                spec["workers"])


def rule_input_mask(data, cfg, num_scopes=2):
    """(M,) bool: the memory inputs a TemporalRuleNet rule may learn, None for all. With the grid
    or the motion atoms: the agent itself reads its label, position and motion atoms; the other
    agents (and roles) only the label atoms (some other agent's position or motion, pooled over
    the agents, says little). The ego motion atoms are read in every scope: only the ego holds them,
    so the other-agent scope does not pool them ("the ego slows down")."""
    layout = data.get("grid_layout")
    if not layout:
        return None
    J, A = cfg["num_slots"], data["num_atoms"]
    atom = torch.arange(A)
    own_only = torch.zeros(A, dtype=torch.bool)
    for key in ("position", "motion"):
        if key in layout:
            own_only |= (atom >= layout[key][0]) & (atom < layout[key][1])
    allowed = torch.ones((num_scopes, J, A), dtype=torch.bool)
    allowed[1:][:, :, own_only] = False
    return allowed.flatten()


def video_roles(video, device):
    """(T, N, K) long agent roles of a stored video (`prepare_videos`), or None if
    the data was prepared without roles."""
    roles = video.get("roles")
    return None if roles is None else roles.to(device).long()


def group_clauses_by_body(clauses):
    clauses_by_body = {}
    for c in clauses.values():
        anchor_body = tuple(c['anchor']["body"][0])
        clauses_by_body[anchor_body] = clauses_by_body.get(anchor_body, [])
        clauses_by_body[anchor_body].append(tuple(c['anchor']["head"]))
    
    for body_group in clauses_by_body.values():
        body_group[:] = tuple(sorted(body_group, key=repr))

    print(f"[Grouping] Clauses grouped by body: {len(clauses)} -> {len(clauses_by_body)}")
    return clauses_by_body


def _atom_name(signature):
    # ["pred", "action", "action_id", 3, "action_id_label", "MovAway",
    #  "agent_class", 1, "agent_class_label", "Car"] -> "Car:MovAway"
    fields = dict(zip(signature[2::2], signature[3::2]))
    labels = [str(v) for k, v in fields.items() if k.endswith("_label") and k != "agent_class_label"]
    return f"{fields.get('agent_class_label', '?')}:{'/'.join(labels) or signature[1]}"


def atom_classes(vocab, num_label_atoms, num_atoms):
    """(A,) long: the agent class index of every label atom (the prefix of its name, e.g. "Car" of
    Car:Stop, "av" of av:AV-Mov), -1 for the other atoms (geometry, grid, ...); and the class names."""
    names, index = [], {}
    out = torch.full((num_atoms,), -1, dtype=torch.long)
    for a in range(min(num_label_atoms, num_atoms, len(vocab))):
        c = _atom_name(vocab[a]).split(":", 1)[0]
        out[a] = index.setdefault(c, len(index))
        if len(names) < len(index):
            names.append(c)
    return out, names


def possible_cells(video, atom_class, num_classes):
    """(N, A) bool of one stored video: the atom can hold for the agent at all, i.e. its class
    (`atom_classes`, on atom_class's device) is the agent's class (the class of most of the label atoms the
    agent holds in the video; the ego holds the av: atoms) or the atom is not a label atom. Agents without
    a label atom: no label atom is possible."""
    device = atom_class.device
    N, C = int(video["shape"][1]), max(num_classes, 1)
    _t, n, a = video["cells"].to(device).long().T
    keep = a < len(atom_class)
    n, cls = n[keep], atom_class[a[keep]]
    n, cls = n[cls >= 0], cls[cls >= 0]
    counts = torch.zeros(N * C, device=device).index_add_(0, n * C + cls, torch.ones(len(n), device=device))
    counts = counts.view(N, C)
    agent = torch.where(counts.sum(1) > 0, counts.argmax(1), torch.full((N,), -2, dtype=torch.long, device=device))
    return (atom_class[None, :] == agent[:, None]) | (atom_class[None, :] < 0)


def class_possible(data, vocab, device):
    """possible(video_id) -> (N, A) bool (`possible_cells`) of the videos of data. A Car never starts
    Ped:Mov, only the ego starts av:AV-Mov: counted over every agent, "bodies -> not head" and exclusion
    rates mostly measure the agent's class (sumo_maps_v7 run ad9e0b: inhibitors such as
    Car:Stop(X0) -> not AV-Mov with violation 0)."""
    atom_class, names = atom_classes(vocab, label_atom_count(data), data["num_atoms"])
    atom_class = atom_class.to(device)
    cache = {}

    def possible(video_id):
        if video_id not in cache:
            cache.clear()                                   # one video at a time
            cache[video_id] = possible_cells(data["videos"][video_id], atom_class, len(names))
        return cache[video_id]
    return possible


_WANDB_CLOUD_HOST = "https://api.wandb.ai"


def _wandb_login(wandb):
    """Log in to the hosted W&B cloud; returns False if the user skips tracking.

    Asks the user to pick a login method: the saved credentials (~/.netrc or
    WANDB_API_KEY) or an API key typed in (hidden), from
    https://wandb.ai/authorize. Without a terminal, only saved credentials are tried.
    A WANDB_API_KEY in the environment (e.g. forwarded into Docker) is used without asking.
    """
    if not sys.stdin.isatty() or os.environ.get("WANDB_API_KEY"):
        return wandb.login(host=_WANDB_CLOUD_HOST)
    print(
        "[wandb] Choose a login method for https://wandb.ai:\n"
        "  1) use saved credentials (~/.netrc or WANDB_API_KEY)\n"
        "  2) enter an API key (from https://wandb.ai/authorize)\n"
        "  3) skip wandb tracking"
    )
    choice = input("[wandb] Select 1/2/3 [1]: ").strip() or "1"
    if choice == "3":
        return False
    if choice == "1" and wandb.login(host=_WANDB_CLOUD_HOST, relogin=False, verify=True):
        return True
    if choice == "1":
        print("[wandb] No valid saved credentials, enter an API key instead.")
    key = getpass.getpass("[wandb] API key (input hidden): ").strip()
    if not key:
        return False
    # The key is verified against the server and saved to ~/.netrc for later runs.
    return wandb.login(key=key, host=_WANDB_CLOUD_HOST, relogin=True, verify=True)


def _init_wandb(cfg, extra_config=None, job_type="step05_extension"):
    """Online run on the hosted W&B cloud, or None if tracking is off or fails.

    Its config holds the plain settings of cfg (numbers, strings, lists; not the
    file paths) and extra_config.

    Tracking is fail-open: a missing package, skipped or failed login, or
    network error only prints a warning.
    """
    if not cfg.get("wandb", True):
        return None
    try:
        import wandb
        # Always the hosted cloud, never a stale WANDB_BASE_URL of a local server.
        os.environ["WANDB_BASE_URL"] = _WANDB_CLOUD_HOST
        if not _wandb_login(wandb):
            print("[wandb] Tracking disabled: not logged in.")
            return None
        settings = {k: v for k, v in cfg.items() if isinstance(v, (bool, int, float, str, list))}
        run = wandb.init(
            project=cfg.get("wandb_project", "cauvid-roadpp"),
            name=cfg.get("wandb_run_name"),
            job_type=job_type,
            mode="online",
            config={**settings, **(extra_config or {})},
        )
        print(f"[wandb] Run: {run.url}")
        return run
    except Exception as exc:
        print(f"[wandb] Tracking disabled: {type(exc).__name__}: {exc}")
        return None


def build_video_tensor(video_atoms, num_atoms, device="cpu"):
    """(T, N, A) bool tensor of one video: X[t, n, a] is True if agent n holds atom a in frame t.

    video_atoms is one video of the compressed atom file; its per_frame_atoms[t]
    is the (frame t, frame t + 1) pair of [atom_index, "tube_uid", tube_uid]
    atoms, so T = len(per_frame_atoms) + 1. Agents are the tube_uids (the ego
    vehicle has tube_uid None), numbered 0..N-1 in order of first appearance.
    num_atoms is A, the size of the shared atom vocabulary.
    """
    frames = _video_frames(video_atoms)
    agent_ids = video_agent_ids(video_atoms)
    cells = [
        (t, agent_ids[tube_uid], atom_index)
        for t, frame_atoms in enumerate(frames)
        for atom_index, _tag, tube_uid in frame_atoms
    ]
    X = torch.zeros((len(frames), len(agent_ids), num_atoms), dtype=torch.bool, device=device)
    if cells:
        t, n, a = torch.tensor(cells, dtype=torch.long, device=device).T
        X[t, n, a] = True
    return X


def atom_onsets(X, min_off_frames=1):
    """(T, N, A) bool: atom a switches on for agent n in frame t, after being off
    for at least the min_off_frames frames before t."""
    held = torch.zeros_like(X)
    for k in range(1, min_off_frames + 1):
        held[k:] |= X[:-k]
    return X & ~held


def is_fresh(output_file, *source_files):
    """True if output_file exists and is newer than every existing source file."""
    if not os.path.exists(output_file):
        return False
    return all(os.path.getmtime(output_file) >= os.path.getmtime(f) for f in source_files if os.path.exists(f))


def prepare_videos(atom_file, num_atoms, device, min_off_frames=1, box_file=None, roles=(), geometry=(),
                   unique_atoms=None, grid_spec=None, motion_spec=None, ego_motion_spec=None, ego_flow_spec=None,
                   duration_spec=None, predict_spec=None):
    """Compressed atoms and frame lists of every video of one split.
    With roles (`roles.ROLES`), each video also gets "roles" (T, N, K) int16: the
    agent filling each role for each agent and frame (-1: none), from the boxes of
    box_file (step_dataset `_tracks_to_boxes`); a video without boxes has none.
    With geometry (`geometry.GEOMETRY` groups), the geometry atoms of the boxes
    (`geometry.video_geometry`) follow the num_atoms label atoms in every video.
    "ahead" and geometry need unique_atoms, the label vocabulary (num_atoms of them).
    With a box file, each video also gets "groups" (T, N) int16 (`grid.agent_groups`, -1: no box)
    and the data "group_tag" (`grid.GROUP_TAG`).
    With grid_spec (`fit_grid`), the grid position atoms (`grid.grid_atoms`) follow the geometry
    atoms; with motion_spec (`fit_motion`), the motion atoms (`motion.motion_atoms`) follow them;
    with ego_motion_spec (`fit_ego_motion`), the ego motion atoms (`ego_motion.ego_atoms`, the ego's
    column only) follow them; with ego_flow_spec (`load_ego_flow`), the ego flow atoms (`ego_flow.flow_atoms`,
    the ego's column only, read from the camera image) follow them; with duration_spec (`load_durations`), the
    duration atoms (`duration.duration_atoms`, from the label atoms) follow them; with predict_spec
    (`load_predict`), the prediction atoms (`predict.predict_atoms`: reach / conflict times of pedestrians
    and cyclists, from the boxes) follow them; "grid_layout" gives their [start, end) atom ranges
    ("position", "motion", "ego_motion", "ego_flow", "duration", "predict").

    Returns {
      "num_atoms": A, the label atoms and the geometry atoms after them,
      "num_label_atoms": the label atoms (num_atoms),
      "min_off_frames": frames an atom must be off before it can switch on again,
      "videos": {video_id: `_video_cells` of its (T, N, A) atom tensor},
      "frames": {video_id: (F,) long, frames with some visible agent},
      "positive_frames": {video_id: (P,) long, frames where some agent already
                          visible in the previous frame switches an atom on},
    }; videos shorter than 2 frames are left out.
    An agent entering the scene switches on all of its atoms at once; those
    are appearances, not events, so they do not make a frame positive. Neither
    does an atom back on after a gap shorter than min_off_frames (label flicker).
    """
    atoms = utils_data.load_json(atom_file)
    boxes = None
    if (roles or geometry or grid_spec is not None or motion_spec is not None or ego_motion_spec is not None
            or predict_spec is not None):
        if box_file is None or not os.path.exists(box_file):
            raise FileNotFoundError(f"Agent roles {roles}, geometry {geometry} and the grid need the box file "
                                    f"{box_file} (step_dataset).")
        if unique_atoms is None or len(unique_atoms) != num_atoms:
            raise ValueError("Agent roles and geometry need the label vocabulary (unique_atoms, num_atoms long).")
    if ego_flow_spec is not None and (box_file is None or not os.path.exists(box_file)):
        raise FileNotFoundError(f"The ego flow atoms mask the agents' boxes: need the box file {box_file} (step_dataset).")
    if box_file is not None and os.path.exists(box_file):
        # With boxes, every video also gets its groups of adjacent same-class agents (step 06 group-level metrics).
        boxes = torch.load(box_file, map_location="cpu")
    geometry_atoms = agent_geometry.vocabulary(unique_atoms, geometry) if geometry else []
    grid_start = num_atoms + len(geometry_atoms)
    num_grid = 2 * agent_grid.num_cells() if grid_spec is not None else 0
    motion_start = grid_start + num_grid
    num_motion = len(agent_motion.ATOMS) if motion_spec is not None else 0
    motion_on = torch.zeros(num_motion)
    ego_motion_start = motion_start + num_motion
    num_ego_motion = len(agent_ego_motion.ATOMS) if ego_motion_spec is not None else 0
    ego_motion_on, ego_frames = torch.zeros(num_ego_motion), 0
    ego_flow_start = ego_motion_start + num_ego_motion
    num_ego_flow = len(agent_ego_flow.ATOMS) if ego_flow_spec is not None else 0
    ego_flow_on, flow_frames, flow_missing = torch.zeros(num_ego_flow), 0, 0
    duration_start = ego_flow_start + num_ego_flow
    if duration_spec is not None and (unique_atoms is None or len(unique_atoms) != num_atoms):
        raise ValueError("The duration atoms need the label vocabulary (unique_atoms, num_atoms long).")
    num_duration = (len(agent_duration.vocabulary(unique_atoms, duration_spec["labels"]))
                    if duration_spec is not None else 0)
    duration_on = 0.0
    predict_start = duration_start + num_duration
    num_predict = len(agent_predict.vocabulary(unique_atoms)) if predict_spec is not None else 0
    predict_on = 0.0
    turn_atoms = agent_predict.ego_turn_atoms(unique_atoms) if predict_spec is not None else []
    if ego_flow_spec is not None:
        _ensure_ego_flow(atoms, boxes, ego_flow_spec)
    grid_rows = 0
    videos, frames, positive_frames = {}, {}, {}
    role_stats = torch.zeros(len(roles))
    geometry_stats = torch.zeros(len(agent_geometry.fact_names(geometry)))
    num_rows = 0
    for video_id, video_atoms in tqdm(atoms.items(), desc="Preparing videos"):
        X = build_video_tensor(video_atoms, num_atoms, device=device)
        if X.shape[0] < 2:
            continue
        X_labels = X
        # Rows, onsets and positive frames from the label atoms only.
        visible = X.any(dim=2)                                      # (T, N)
        stayed = torch.zeros_like(visible)
        stayed[1:] = visible[1:] & visible[:-1]                     # visible in t and t - 1
        has_onset = (atom_onsets(X, min_off_frames).any(dim=2) & stayed).any(dim=1)
        num_rows += int(visible.sum())

        if boxes is not None:
            agent_ids = video_agent_ids(video_atoms)
            ego = agent_ids.get(None)
            video_boxes, classes, has_box = _agent_boxes(boxes, video_id, video_atoms, X)
        if roles:
            in_ego_lane = agent_roles.ego_lane(X, unique_atoms) if "ahead" in roles else None
            crossing = agent_roles.on_crossing(X, unique_atoms) if "xing_ped" in roles else None
            partners = agent_roles.role_partners(video_boxes, classes, ego, roles, in_ego_lane, crossing)
            partners[~visible.cpu()] = -1
            role_stats += (partners >= 0).sum(dim=(0, 1)).float()
        if geometry:
            G = agent_geometry.video_geometry(video_boxes, classes, X, unique_atoms, geometry, ego)
            geometry_stats += G.view(*G.shape[:2], -1, len(geometry_stats)).sum(dim=(0, 1, 2)).float()
            X = torch.cat([X, G.to(X.device)], dim=2)
        if grid_spec is not None:
            cells = agent_grid.corner_cells(video_boxes, has_box, grid_spec["knots"])
            grid_rows += int(has_box.sum())
            X = torch.cat([X, agent_grid.grid_atoms(cells).to(X.device)], dim=2)
        if motion_spec is not None:
            M_atoms = agent_motion.motion_atoms(video_boxes, has_box, classes, motion_spec)
            motion_on += M_atoms.sum(dim=(0, 1)).float()
            X = torch.cat([X, M_atoms.to(X.device)], dim=2)
        if ego_motion_spec is not None:
            E_atoms = torch.zeros((*X.shape[:2], num_ego_motion), dtype=torch.bool)
            if ego is not None and X.shape[0] > agent_motion.K:
                sig = agent_ego_motion.signals(video_boxes, has_box, classes, ego, X_labels, unique_atoms)
                seen = X_labels[:, ego].any(dim=1).cpu()
                E_atoms[:, ego] = agent_ego_motion.ego_atoms(sig, ego_motion_spec) & seen[:, None]
                ego_motion_on += E_atoms[:, ego].sum(dim=0).float()
                ego_frames += int(seen.sum())
            X = torch.cat([X, E_atoms.to(X.device)], dim=2)
        if ego_flow_spec is not None:
            F_atoms = torch.zeros((*X.shape[:2], num_ego_flow), dtype=torch.bool)
            ego_n = video_agent_ids(video_atoms).get(None)
            raw = agent_ego_flow.load_cache(ego_flow_spec["cache_dir"], video_id)
            if raw is None:
                flow_missing += 1
            elif ego_n is not None:
                seen = X_labels[:, ego_n].any(dim=1).cpu()
                F_atoms[:, ego_n] = torch.from_numpy(agent_ego_flow.flow_atoms(raw, X.shape[0])) & seen[:, None]
                ego_flow_on += F_atoms[:, ego_n].sum(dim=0).float()
                flow_frames += int(seen.sum())
            X = torch.cat([X, F_atoms.to(X.device)], dim=2)
        if duration_spec is not None:
            D_atoms = agent_duration.duration_atoms(X_labels, unique_atoms, duration_spec["labels"])
            duration_on += float(D_atoms.any(dim=2).sum())
            X = torch.cat([X, D_atoms.to(X.device)], dim=2)
        if predict_spec is not None:
            turning = (X_labels[:, ego][:, turn_atoms].any(dim=1).cpu().numpy() if ego is not None and turn_atoms
                       else None)
            P_atoms = torch.from_numpy(agent_predict.predict_atoms(video_boxes.numpy(), has_box.numpy(), classes,
                                                                   turning, unique_atoms, ego,
                                                                   predict_spec["camera"]))
            predict_on += float(P_atoms.any(dim=2).sum())
            X = torch.cat([X, P_atoms.to(X.device)], dim=2)
        videos[video_id] = _video_cells(X)
        if roles:
            videos[video_id]["roles"] = partners.to(torch.int16)
        if boxes is not None:
            videos[video_id]["groups"] = agent_grid.agent_groups(video_boxes, has_box, classes).to(torch.int16)
        frames[video_id] = torch.nonzero(visible.any(dim=1)).flatten().cpu()
        positive_frames[video_id] = torch.nonzero(has_onset).flatten().cpu()
    name = atom_file.name if hasattr(atom_file, "name") else atom_file
    if roles:
        print(f"[Roles]\t {name}: share of visible agents (rows) with a partner: "
              + ", ".join(f"{r} {role_stats[k] / max(num_rows, 1):.2f}" for k, r in enumerate(roles)))
    if geometry:
        print(f"[Geometry]\t {name}: share of visible agents (rows) holding each fact: "
              + ", ".join(f"{f} {geometry_stats[k] / max(num_rows, 1):.2f}"
                          for k, f in enumerate(agent_geometry.fact_names(geometry))))
    if grid_spec is not None:
        print(f"[Grid]\t {name}: share of visible agents (rows) with a box: {grid_rows / max(num_rows, 1):.2f}; "
              f"{num_grid} grid position atoms")
    if motion_spec is not None:
        print(f"[Motion]\t {name}: share of visible agents (rows) holding each motion atom: "
              + ", ".join(f"{a} {motion_on[i] / max(num_rows, 1):.3f}" for i, a in enumerate(agent_motion.ATOMS)))
    if ego_motion_spec is not None:
        print(f"[Ego motion]\t {name}: share of the ego's frames holding each ego motion atom: "
              + ", ".join(f"{a} {ego_motion_on[i] / max(ego_frames, 1):.3f}" for i, a in enumerate(agent_ego_motion.ATOMS)))
    if ego_flow_spec is not None:
        print(f"[Ego flow]\t {name}: share of the ego's frames holding each ego flow atom: "
              + ", ".join(f"{a} {ego_flow_on[i] / max(flow_frames, 1):.3f}" for i, a in enumerate(agent_ego_flow.ATOMS))
              + (f"; {flow_missing} videos without a flow cache (no video file?)" if flow_missing else ""))
    if duration_spec is not None:
        print(f"[Durations]\t {name}: {num_duration} duration atoms of the labels {list(duration_spec['labels'])}; "
              f"share of visible agents (rows) holding one: {duration_on / max(num_rows, 1):.2f}")
    if predict_spec is not None:
        print(f"[Predict]\t {name}: {num_predict} prediction atoms; share of visible agents (rows) holding one: "
              f"{predict_on / max(num_rows, 1):.3f}")
    out = {"num_atoms": num_atoms + len(geometry_atoms) + num_grid + num_motion + num_ego_motion + num_ego_flow
                        + num_duration + num_predict,
           "num_label_atoms": num_atoms,
           "min_off_frames": min_off_frames, "roles": list(roles), "geometry": list(geometry), "videos": videos,
           "frames": frames, "positive_frames": positive_frames}
    if boxes is not None:
        out["group_tag"] = agent_grid.GROUP_TAG
    if (grid_spec is not None or motion_spec is not None or ego_motion_spec is not None or ego_flow_spec is not None
            or duration_spec is not None or predict_spec is not None):
        out["grid_layout"] = {}
    if grid_spec is not None:
        out["grid_layout"]["position"] = [grid_start, grid_start + num_grid]
    if motion_spec is not None:
        out["grid_layout"]["motion"] = [motion_start, motion_start + num_motion]
    if ego_motion_spec is not None:
        out["grid_layout"]["ego_motion"] = [ego_motion_start, ego_motion_start + num_ego_motion]
    if ego_flow_spec is not None:
        out["grid_layout"]["ego_flow"] = [ego_flow_start, ego_flow_start + num_ego_flow]
    if duration_spec is not None:
        out["grid_layout"]["duration"] = [duration_start, duration_start + num_duration]
    if predict_spec is not None:
        out["grid_layout"]["predict"] = [predict_start, predict_start + num_predict]
    return out


# Marks how training_data.pt was split, so a file made another way is rebuilt.
SPLIT_SOURCE = "dataset train / dataset val"


def prepare_training_data(cfg):
    """Everything the TemporalRuleNet training samples its batches from: the
    `prepare_videos` data of the dataset's train split ("train_videos", all of
    them trained on) and of its val split ("val_videos": model selection, the
    cascade thresholds and the step 06 alarm thresholds), in one dict."""
    unique_atoms = utils_data.load_json(cfg['unique_atoms_file'])
    num_atoms = len(unique_atoms)
    grid_spec, motion_spec, ego_motion_spec = load_grid(cfg), load_motion(cfg), load_ego_motion(cfg)
    ego_flow_spec, duration_spec, predict_spec = load_ego_flow(cfg), load_durations(cfg), load_predict(cfg)
    train = prepare_videos(cfg['train_atom_file'], num_atoms, cfg["device"], cfg["min_off_frames"],
                           cfg.get("train_box_file"), cfg["roles"], cfg["geometry"], unique_atoms, grid_spec,
                           motion_spec, ego_motion_spec, ego_flow_spec, duration_spec, predict_spec)
    val = prepare_videos(cfg['val_atom_file'], num_atoms, cfg["device"], cfg["min_off_frames"],
                         cfg.get("val_box_file"), cfg["roles"], cfg["geometry"], unique_atoms, grid_spec,
                         motion_spec, ego_motion_spec, ego_flow_spec, duration_spec, predict_spec)
    shared = set(train["videos"]) & set(val["videos"])
    if shared:
        raise ValueError(f"{len(shared)} videos are in both the train and the val split, e.g. {sorted(shared)[:3]}.")
    data = {"num_atoms": train["num_atoms"], "num_label_atoms": num_atoms, "min_off_frames": cfg["min_off_frames"],
            "split": SPLIT_SOURCE, "roles": list(cfg["roles"]), "geometry": list(cfg["geometry"]),
            "grid_tag": grid_tag(cfg)}
    if "grid_layout" in train:
        data["grid_layout"] = train["grid_layout"]
    for key in ("videos", "frames", "positive_frames"):
        data[key] = {**train[key], **val[key]}
    train_videos, val_videos = sorted(train["videos"]), sorted(val["videos"])

    num_frames = sum(len(f) for f in data["frames"].values())
    num_positive = sum(len(f) for f in data["positive_frames"].values())
    print(f"[Videos]\t {len(train_videos)} train, {len(val_videos)} val (the dataset splits)")
    print(f"[Frames]\t {num_frames} ({num_positive} with onsets)")
    return {**data, "train_videos": train_videos, "val_videos": val_videos}


def save_data(data, path):
    """torch.save to a temporary file, then renamed: a run reading the shared file never sees half of it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(data, tmp)
    os.replace(tmp, path)


def shared_data_file(local_file, cfg, kind):
    """The prepared dataset file of a run: its own local_file if it has one (runs made before the shared
    cache), else <exp>/data_cache/<kind>_data_<data_key>.pt, shared by every run with the same data
    settings (min_off_frames, roles, geometry, grid / motion / ego motion atoms)."""
    if os.path.exists(local_file) or not cfg.get("data_cache_dir"):
        return local_file
    return cfg["data_cache_dir"] / f"{kind}_data_{cfg['data_key']}.pt"


def subsample_train_videos(data, cfg):
    """Learning curve (--train_fraction f < 1): keep only a share f of the train videos, nested (the
    videos of a smaller share are in every larger one: one shuffle by train_subset_seed, its first f N
    videos). The val videos and the event types of the heads (counted over all train + val videos, before
    this) stay the same, so every share is scored on the same heads and events."""
    f = float(cfg.get("train_fraction", 1.0))
    data["train_fraction"] = f
    if f >= 1.0:
        return data
    import random
    videos = sorted(data["train_videos"])
    random.Random(int(cfg.get("train_subset_seed", 0))).shuffle(videos)
    keep = max(1, int(round(f * len(videos))))
    data["train_videos"] = sorted(videos[:keep])
    print(f"[Videos]\t learning curve: {keep} of {len(videos)} train videos (train_fraction {f:g}, "
          f"seed {cfg.get('train_subset_seed', 0)})")
    return data


def load_training_data(cfg):
    """training_data.pt, (re)built by `prepare_training_data` if it is missing,
    has another min_off_frames, other agent roles or geometry or was split another way. Adds the event
    types of the heads ("event_types", `head_event_types`) and "target_heads" (A,) bool,
    the heads the models learn: never the geometry heads (inputs only), and not the noise
    heads if cfg["exclude_noise_heads"]."""
    data = torch.load(cfg["training_data_file"], map_location="cpu") if os.path.exists(cfg["training_data_file"]) else {}
    if (data.get("min_off_frames") != cfg["min_off_frames"] or data.get("split") != SPLIT_SOURCE
            or data.get("roles", []) != list(cfg["roles"]) or data.get("geometry", []) != list(cfg["geometry"])
            or data.get("grid_tag", "") != grid_tag(cfg)):
        data = prepare_training_data(cfg)
        save_data(data, cfg["training_data_file"])
    data["event_types"] = head_event_types(data, load_vocabulary(cfg), cfg)
    subsample_train_videos(data, cfg)
    types = data["event_types"]["types"]
    noise, inputs = types == EVENT_TYPES.index("noise"), types == EVENT_TYPES.index("input")
    data["target_heads"] = ~inputs & (~noise if cfg["exclude_noise_heads"] else torch.ones_like(noise))
    data["pos_weight_scale"] = head_pos_weight_scale(data["event_types"], cfg)
    print(f"[Heads]\t {int(data['target_heads'].sum())} of {data['num_atoms']} heads are learned"
          + (f" ({int(noise.sum())} noise heads, < {cfg['noise_below']} train + val onsets, left out)"
             if cfg["exclude_noise_heads"] else "")
          + (f" ({int(inputs.sum())} geometry / grid atoms, inputs only)" if inputs.any() else ""))
    return data


# Event types by the frequency of their head (onsets in the train + val videos), see
def head_pos_weight_scale(event_types, cfg):
    """(A,) factor of every head's onset weight (pos_weight): with cfg["type_balance"], an
    event type whose heads have fewer onsets each (mean train + val onsets per head) weighs
    its onsets up, (M_max / M_type) ** power capped at type_balance_max, M_max the mean of
    the type with the most onsets per head (trivial), so the frequent trivial heads no longer
    dominate the loss, the sparsity budget and the rules; else 1 for every head. The mean per
    head, not the type's total: a type of few heads (medium: 2 trivial heads, 278 onsets, vs
    17 regular heads, 485) is still the frequent one. The noise heads (no targets) keep 1."""
    types = event_types["types"]
    scale = torch.ones(len(types))
    if not cfg.get("type_balance"):
        return scale
    means = {t: sum(h["onsets"] for h in event_types["heads"][t]) / len(event_types["heads"][t])
             for t in EVENT_TYPES if t not in ("noise", "input") and event_types["heads"][t]}
    reference = max(means.values())
    for g, t in enumerate(EVENT_TYPES):
        if t in means and means[t] > 0:
            scale[types == g] = min((reference / means[t]) ** cfg["type_balance_power"], cfg["type_balance_max"])
    print("[Type balance]\t onset weight x " + ", ".join(
        f"{t} {scale[types == EVENT_TYPES.index(t)][0]:.2f} ({means[t]:.1f} onsets per head)"
        for t in means if (types == EVENT_TYPES.index(t)).any()))
    return scale


def pos_weight_of(training_data, cfg):
    """The onset weight of the loss: cfg["pos_weight"], times the per-head type balance
    (`head_pos_weight_scale`, (A,) on cfg["device"]) only for the main rule net
    (cfg["balanced"], set by `train_temporal_rule_net`); every other model is unweighted."""
    scale = training_data.get("pos_weight_scale")
    if scale is None or not cfg.get("balanced"):
        return cfg["pos_weight"]
    return cfg["pos_weight"] * scale.to(cfg["device"])


def type_balance_tag(cfg):
    """File tag suffix of the type balance (per-head mean), "" without it."""
    return f"_tbh{cfg['type_balance_power']:g}m{cfg['type_balance_max']:g}" if cfg.get("type_balance") else ""


# `head_event_types`: heads of the geometry atoms are "input" (inputs only, never targets),
# the others with < noise_below onsets are "noise", < rare_below "rare",
# > trivial_above "trivial", the rest "regular". The rare heads are training targets like the
# regular ones; step 06 only reports them apart (too few onsets for reliable rules).
EVENT_TYPES = ("noise", "rare", "regular", "trivial", "input")       # "input" last: step 06 reports the others


def head_event_types(training_data, unique_atoms, cfg):
    """The event type of every head by its frequency: its onsets (on after
    min_off_frames off, the agent visible in the frame before) in the train + val
    videos: fewer than cfg["noise_below"] onsets (every head never seen included)
    is "noise", fewer than cfg["rare_below"] "rare", more than cfg["trivial_above"]
    "trivial", the rest "regular"; the geometry atoms (inputs only) are "input".
    Returns {"types": (A,) long EVENT_TYPES index, "heads": {type: [{"head",
    "label", "onsets"}], most frequent first}, the two cutoffs and the onset count}."""
    A = training_data["num_atoms"]
    counts = torch.zeros(A)
    for video in training_data["videos"].values():
        X = _video_from_cells(video, A, "cpu")
        onsets = atom_onsets(X, training_data["min_off_frames"])
        onsets[0] = False
        onsets[1:] &= X[:-1].any(dim=2, keepdim=True)
        counts += onsets.sum(dim=(0, 1)).float()
    types = torch.full((A,), EVENT_TYPES.index("regular"), dtype=torch.long)
    types[counts < cfg.get("rare_below", 0)] = EVENT_TYPES.index("rare")
    types[counts < cfg["noise_below"]] = EVENT_TYPES.index("noise")
    types[counts > cfg["trivial_above"]] = EVENT_TYPES.index("trivial")
    types[torch.tensor([is_input_atom(s) for s in unique_atoms[:A]], dtype=torch.bool)] = EVENT_TYPES.index("input")
    order = torch.argsort(counts, descending=True, stable=True)
    heads = {t: [{"head": int(a), "label": _atom_name(unique_atoms[a]), "onsets": int(counts[a])}
                 for a in order.tolist() if types[a] == g]
             for g, t in enumerate(EVENT_TYPES)}
    return {"types": types, "heads": heads, "noise_below": cfg["noise_below"], "rare_below": cfg.get("rare_below"),
            "trivial_above": cfg["trivial_above"],
            "count_source": "train + val onsets", "num_onsets": int(counts.sum())}


def frames_to_sample(training_data, video_id, cfg):
    """Target frames to sample from one video per epoch: cfg["frames_per_video"] per
    cfg["frames_per_video_ref"] frames (32 per 20 s), so a long video gets as many per second as a
    short one (sumo_long_*: a 122 s video and its seven 20 s cuts get the same rows per second). Videos
    up to the reference length keep frames_per_video (the roadpp and 20 s SUMO runs are unchanged)."""
    T = training_data["videos"][video_id]["shape"][0]
    n = cfg["frames_per_video"]
    return max(n, round(n * T / cfg.get("frames_per_video_ref", 200)))


def _sample_frames(training_data, video_id, num_frames, positive_ratio, generator, horizon=1):
    """(<= num_frames,) sorted target frames of one video, a share positive_ratio
    of them drawn from its frames with onsets and the rest from all its frames.
    A positive is moved 0..horizon - 1 frames before its onset, where the onset
    is still ahead within the largest window."""
    def pick(pool, k):
        if len(pool) == 0 or k <= 0:
            return pool[:0]
        return pool[torch.randint(len(pool), (k,), generator=generator)]

    positives = pick(training_data["positive_frames"][video_id], round(positive_ratio * num_frames))
    positives = (positives - torch.randint(horizon, positives.shape, generator=generator)).clamp(min=0)
    others = pick(training_data["frames"][video_id], num_frames - len(positives))
    return torch.unique(torch.cat([positives, others]))


def _video_rows(model, training_data, video_id, frames, device, history=None):
    """`video_rows` of the target frames of one stored video, rebuilt from its atoms.
    Heads outside training_data["target_heads"] (the noise heads, see
    `load_training_data`) are never at risk and never targets, so no loss, count
    or clause covers them."""
    video = training_data["videos"][video_id]
    X = _video_from_cells(video, training_data["num_atoms"], device)
    rows = video_rows(model, X, frames.to(device), training_data.get("min_off_frames", 1),
                      None if history is None else history.to(device), video_roles(video, device))
    if "target_heads" in training_data:
        target_heads = training_data["target_heads"].to(device)
        rows["at_risk"] = rows["at_risk"] & target_heads
        rows["targets"] = rows["targets"] & target_heads[None, None, :]
    return rows


def video_rows(model, X, frames, min_off_frames=1, history=None, roles=None):
    """`TemporalRuleNet.training_rows` (a dict of inputs, rows, at_risk, first,
    targets, known) of the target frames of one (T, N, A) video tensor, without
    the agents entering the scene: their atoms all switch on at once in t, which
    are not events. history (F,): the frames known before each target frame
    (default all; older ones are unknown inputs). roles (T, N, K): the agent roles
    (`video_roles`), passed on only to models that read roles."""
    if getattr(model, "roles", ()):
        if roles is None:
            raise ValueError(f"The model reads the agent roles {model.roles}, the data has none: "
                             "prepare it with the same cfg['roles'].")
        rows = model.training_rows(X, frames, min_off_frames, history, roles=roles)
    else:
        rows = model.training_rows(X, frames, min_off_frames, history)
    t, n = rows["rows"].T
    seen = (t > 0) & X[(t - 1).clamp(min=0), n].any(dim=1)
    return {key: value[seen] for key, value in rows.items()}


def sample_history(frames, cfg, generator):
    """(F,) known history of target frames for training: with probability
    truncate_prob a random 0..memory frames (the model learns to predict from
    partial facts, with lower confidence), else everything before the frame.
    With slot_ranges the memory is every known frame (the coarse bins 3-6, 6-12, > 12 s),
    so a truncated history is drawn from 0..t (all of it known at most): capped at the
    3 s of the single slots, half of the rows would never see a long bin known (a rule
    "seen 3-12 s before" could not learn from them) and its unknown chance would be fit on
    the all-unknown case only. Without slot_ranges the draws are as before."""
    truncated = torch.rand(len(frames), generator=generator) < cfg["truncate_prob"]
    if cfg.get("slot_ranges"):
        full = frames.cpu()
        drawn = (torch.rand(len(frames), generator=generator) * (full + 1).float()).long().clamp(max=full)
        return torch.where(truncated, drawn, full)
    memory = cfg["slot_size"] * cfg["num_slots"]
    return torch.where(truncated, torch.randint(memory + 1, (len(frames),), generator=generator), frames.cpu())


def concat_rows(parts):
    """One rows dict of several `video_rows` dicts."""
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def coarsen_video(X, block_size):
    """(ceil(T / s), N, A) bool: the atom holds in some frame of each block of
    s = block_size frames (frames past the video end count as empty).
    Used by step05_layer / step06_layer."""
    pad = (-X.shape[0]) % block_size
    X = torch.cat([X, X.new_zeros((pad, *X.shape[1:]))])
    return X.view(-1, block_size, *X.shape[1:]).any(dim=1)


@torch.no_grad()
def _validation_loss(model, training_data, val_frames, cfg, candidates=None):
    """Mean window loss over the fixed target frames of the validation videos
    (only the candidate cells, if candidates is given; see `fit_onset_model`)."""
    model.eval()
    losses = []
    for video_id, (frames, history) in val_frames.items():
        rows = _video_rows(model, training_data, video_id, frames, cfg["device"], history)
        if len(rows["inputs"]) == 0:
            continue
        at_risk = rows["at_risk"] if candidates is None else rows["at_risk"] & candidates(rows["inputs"])
        hazards, _rules = model(rows["inputs"])
        losses.append(model.loss(hazards, rows["targets"], at_risk, rows["known"],
                                 pos_weight_of(training_data, cfg)).item())
    return sum(losses) / len(losses) if losses else float("nan")


def anchor_rules(cfg):
    """[((atom, slot), [head atom])] of the anchor clauses, one per body, in the
    order their rules are seeded: rules i * B..i * B + B - 1 belong to anchor i."""
    body_groups = group_clauses_by_body(utils_data.load_json(cfg["causal_anchor_clause_file"]))
    return [(body, [head[0] for head in heads]) for body, heads in body_groups.items()]


def fit_onset_model(model, training_data, cfg, optimizer, history_file, regularization=None, run=None,
                    model_stats=None, candidates=None):
    """Train an onset model (TemporalRuleNet or the OnsetMLP baseline) on the
    training videos; returns it with the weights of the lowest validation loss.

    One epoch visits every training video once, in batches of videos_per_batch
    videos with frames_per_video sampled target frames each. The input rows are
    rebuilt from the stored atoms per batch, never saved. regularization(epoch):
    the extra loss term of an epoch, or None; run: the W&B run the losses of each
    epoch are logged to, or None; model_stats(): extra values logged per epoch;
    candidates(inputs) -> (B, A) bool: if given, the loss only covers these
    (row, head) cells (the cascade verifier learns on the stage-1 candidates).
    The targets are the onsets within each of cfg["windows"] frames; the known
    history of each target frame is truncated at random (`sample_history`).
    """
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    device = cfg["device"]
    # Optional: gradient-norm clipping and halving the learning rate (the OnsetMLP sets them)
    # after lr_patience epochs without a better val loss; early stopping after patience (both models).
    grad_clip, patience = cfg.get("grad_clip"), cfg.get("patience")
    # The best val epoch (and early stopping) only counts from epoch select_from_epoch + 1 on: the
    # rule net sets it to l0_warmup_epochs, so the kept model has trained under the sparsity loss
    # (medium 080f05 / 88be20: the best val epoch was 10, the last one without it).
    select_from = cfg.get("select_from_epoch", 0)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=cfg["lr_patience"])
                 if cfg.get("lr_patience") else None)
    best_epoch = 0
    pos_weight = pos_weight_of(training_data, cfg)                                   # scalar or (A,)
    # Fixed validation frames and known histories, so the losses of the epochs are comparable.
    val_frames = {}
    for video_id in training_data["val_videos"]:
        frames = _sample_frames(training_data, video_id, frames_to_sample(training_data, video_id, cfg),
                                cfg["positive_frame_ratio"], generator, max(cfg["windows"]))
        val_frames[video_id] = (frames, sample_history(frames, cfg, generator))
    # Learning curve: each video of a share f of the train data round(1 / f) times per epoch (`get_cfg`).
    train_videos = list(training_data["train_videos"]) * cfg.get("epoch_repeat", 1)
    best_val_loss, best_state = float("inf"), None
    history = []
    for epoch in range(cfg["num_epochs"]):
        model.train()
        order = torch.randperm(len(train_videos), generator=generator).tolist()
        epoch_losses = []
        # The epoch's target frames, in chunks of frames_per_video (a long video gives several,
        # `frames_to_sample`); a batch holds videos_per_batch chunks, so the batch size and the steps
        # per second of video are those of 20 s videos. The draws are in the order of the former
        # per-batch loop: with one chunk per video (videos up to 20 s) the epoch is unchanged.
        fpv, items = cfg["frames_per_video"], []
        for i in order:
            frames = _sample_frames(training_data, train_videos[i],
                                    frames_to_sample(training_data, train_videos[i], cfg),
                                    cfg["positive_frame_ratio"], generator, max(cfg["windows"]))
            known = sample_history(frames, cfg, generator)
            items += [(train_videos[i], frames[k:k + fpv], known[k:k + fpv]) for k in range(0, max(len(frames), 1), fpv)]
        if len(items) > len(order):                         # long videos: their chunks spread over the batches
            items = [items[j] for j in torch.randperm(len(items), generator=generator).tolist()]
        for start in range(0, len(items), cfg["videos_per_batch"]):
            parts = [_video_rows(model, training_data, video_id, frames, device, known)
                     for video_id, frames, known in items[start:start + cfg["videos_per_batch"]]]
            rows = concat_rows(parts)
            if len(rows["inputs"]) == 0:
                continue
            at_risk = rows["at_risk"]
            if candidates is not None:
                at_risk = at_risk & candidates(rows["inputs"])

            hazards, _hidden = model(rows["inputs"])
            onset_loss = model.loss(hazards, rows["targets"], at_risk, rows["known"], pos_weight)
            loss = onset_loss if regularization is None else onset_loss + regularization(epoch)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(onset_loss.item())

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_loss = _validation_loss(model, training_data, val_frames, cfg, candidates)
        print(f"[Epoch {epoch + 1}/{cfg['num_epochs']}] train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if run is not None:
            run.log({"train/onset_loss": train_loss, "val/onset_loss": val_loss,
                     **(model_stats() if model_stats else {})}, step=epoch)
        selectable = epoch + 1 > select_from
        if selectable and val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if scheduler is not None:
            scheduler.step(val_loss)
            history[-1]["lr"] = optimizer.param_groups[0]["lr"]
        if patience and selectable and epoch + 1 - max(best_epoch, select_from) >= patience:
            print(f"[Training] Early stop: no better val loss for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print(f"[Training] No epoch after {select_from} (num_epochs {cfg['num_epochs']}): the last one is kept")
    print(f"[Training] Best val loss {best_val_loss:.4f} (epoch {best_epoch})")
    utils_data.save_json(history, history_file)
    if run is not None:
        run.summary["best_val_onset_loss"] = best_val_loss
    return model


MOTOR_CLASSES = ("Car", "MedVeh", "LarVeh", "Bus", "Mobike")


def literal_groups(training_data, cfg):
    """[(label, [atom index])] of the group literals (`TemporalRuleNet` groups): every label atom
    ("Brake", "VehLane", ...) held by >= 2 motor vehicle classes, as "any motor vehicle X"; [] without
    cfg["literal_groups"]."""
    if not cfg.get("literal_groups"):
        return []
    vocab = load_vocabulary(cfg)
    num_labels = training_data.get("num_label_atoms", len(vocab))
    by_label = {}
    for i, sig in enumerate(vocab[:num_labels]):
        cls, _, label = _atom_name(sig).partition(":")
        if cls in MOTOR_CLASSES:
            by_label.setdefault(label, []).append(i)
    return [(label, atoms) for label, atoms in sorted(by_label.items()) if len(atoms) >= 2]


def build_rule_net(training_data, cfg, num_rules, roles=None, literals=True):
    """A TemporalRuleNet of cfg (slots, windows, roles, input mask, self gate) with num_rules rule nodes,
    and with literals the negated / group literals of cfg["negation"] / cfg["literal_groups"]."""
    roles = tuple(cfg["roles"]) if roles is None else tuple(roles)
    return TemporalRuleNet(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"], num_rules,
                           cfg["init_log_alpha"], cfg["mask_type"], cfg["windows"], roles=roles,
                           input_mask=rule_input_mask(training_data, cfg, 2 + len(roles)),
                           require_self=cfg.get("require_self", False), ranges=cfg.get("slot_ranges", False),
                           negation=bool(cfg.get("negation")) and literals,
                           groups=literal_groups(training_data, cfg) if literals else None)


def _seeded_rule_net(training_data, cfg):
    """A TemporalRuleNet with one rule node per (anchor body, slot offset), plus
    num_free_rules unseeded ones, on cfg["device"]; with anchor_source "mined" one rule per mined seed
    (`step05_mine`) plus free_rules_per_head free rules per learned head."""
    if cfg.get("anchor_source") == "mined":
        from src.exp_roadpp import step05_mine
        heads = torch.nonzero(training_data["target_heads"]).flatten().tolist()
        seeds = step05_mine.load_or_mine(training_data, cfg)
        num_rules = TemporalRuleNet.mined_rule_count(seeds, heads, cfg["free_rules_per_head"])
        model = build_rule_net(training_data, cfg, num_rules)
        generator = torch.Generator().manual_seed(cfg["seed"])
        num_seeded = model.seed_rules_mined(seeds, heads, cfg["free_rules_per_head"], cfg["seed_head_weight"],
                                            generator)
        print(f"[Model] mined rules: {num_seeded} seeded (`step05_mine`), {num_rules - num_seeded} free "
              f"({cfg['free_rules_per_head']} per head, {len(heads)} heads); {model.num_literals} literals "
              f"({model.num_inputs} inputs, negation {model.negation}, {model.num_groups} groups), "
              f"{cfg['mask_type']} mask")
        return model.to(cfg["device"])
    anchors = anchor_rules(cfg)
    per_head = cfg.get("per_head_rules", False)
    if per_head:
        # One rule per (anchor body, learned head, slot, copy), plus free rules per learned head.
        heads = torch.nonzero(training_data["target_heads"]).flatten().tolist()
        num_rules = TemporalRuleNet.per_head_rule_count(anchors, heads, cfg["num_slots"], cfg["rule_copies"],
                                                        cfg["free_rules_per_head"])
    else:
        num_rules = len(anchors) * cfg["num_slots"] + cfg["num_free_rules"]
    model = build_rule_net(training_data, cfg, num_rules)
    if per_head:
        generator = torch.Generator().manual_seed(cfg["seed"])
        num_seeded = model.seed_rules_per_head(anchors, heads, cfg["rule_copies"], cfg["free_rules_per_head"],
                                               cfg["seed_head_weight"], cfg["copy_input_share"], generator)
        print(f"[Model] per-head rules: {num_seeded} seeded ({len(anchors)} anchor bodies x their learned heads x "
              f"{cfg['num_slots']} slots x {cfg['rule_copies']} copies), {num_rules - num_seeded} free "
              f"({cfg['free_rules_per_head']} per head, {len(heads)} heads), copies > 0 learn a random "
              f"{cfg['copy_input_share']:g} of the inputs, {cfg['mask_type']} mask")
    else:
        num_seeded = model.seed_rules(anchors, cfg["seed_head_weight"])
        print(f"[Model] {num_seeded} rules seeded from {len(anchors)} anchor bodies, {cfg['num_free_rules']} free, "
              f"{cfg['mask_type']} mask")
    return model.to(cfg["device"])


def _fit_rule_net(model, training_data, cfg, history_file, run=None, candidates=None):
    """`fit_onset_model` of a TemporalRuleNet, with its sparsity loss after
    l0_warmup_epochs; the sigmoid gates get their own learning rate (gate_lr,
    and a tiny gate_eps: their gradients start at ~1e-12, sigmoid' of -11)."""
    def regularization(epoch):
        # No sparsity pressure in the first epochs, so useful inputs can grow first.
        return (cfg["l0_lambda"] if epoch >= cfg["l0_warmup_epochs"] else 0.0) * model.sparsity_loss()

    @torch.no_grad()
    def model_stats():
        active = (model.rule_mask() > cfg["mask_threshold"]) & ~model.fixed_inputs
        extra = active.sum(dim=1).float()
        gates = model.log_alpha[~model.fixed_inputs]
        return {"model/sparsity_loss": model.sparsity_loss().item(),
                "model/extra_inputs_per_rule": extra.mean().item(),
                "model/max_extra_inputs": extra.max().item(),
                "model/share_rules_extended": (extra > 0).float().mean().item(),
                "gates/max_log_alpha": gates.max().item(),
                "gates/count_above_0": (gates > 0).sum().item()}

    if cfg["mask_type"] == "sigmoid":
        others = [p for name, p in model.named_parameters() if name != "log_alpha"]
        optimizer = torch.optim.Adam([{"params": [model.log_alpha], "lr": cfg["gate_lr"], "eps": cfg["gate_eps"]},
                                      {"params": others, "lr": cfg["lr"]}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # The kept (best val) epoch must have trained under the sparsity loss.
    cfg = {**cfg, "select_from_epoch": cfg["l0_warmup_epochs"]}
    return fit_onset_model(model, training_data, cfg, optimizer, history_file, regularization, run, model_stats,
                           candidates)


def train_temporal_rule_net(training_data, cfg, run=None):
    """Stage 1 of the cascade: the recall model, a TemporalRuleNet seeded from the
    anchor clauses plus free rules for the heads without an anchor."""
    # Only the main rule net has the type-balanced onset weights (`pos_weight_of`).
    return _fit_rule_net(_seeded_rule_net(training_data, cfg), training_data, {**cfg, "balanced": True},
                         cfg["history_file"], run)


def _saved_roles(num_inputs, num_atoms, cfg, per_scope=None):
    """The roles of a saved model with num_inputs memory inputs: none for a model of
    the two scopes (own, others; trained before the roles), else cfg["roles"]."""
    # per_scope: the inputs per scope and atom (the rule net: its spans; the MLP: its packed slots)
    per_scope = per_scope or len(slot_spans(cfg["num_slots"], cfg.get("slot_ranges", False)))
    num_scopes = num_inputs // (per_scope * num_atoms)
    if num_scopes == 2:
        return ()
    if num_scopes != 2 + len(cfg["roles"]):
        raise ValueError(f"The saved model has {num_scopes} input scopes, cfg['roles'] {cfg['roles']} gives "
                         f"{2 + len(cfg['roles'])}.")
    return tuple(cfg["roles"])


def load_temporal_rule_net(training_data, cfg, model_file=None):
    """The trained TemporalRuleNet of model_file (default cfg["model_file"]), in
    eval mode on cfg["device"].

    The model is rebuilt with the same sizes it was trained with (num_atoms of
    the training data, slots, mask_type and windows of cfg, the number of rules of the
    saved weights) before its weights, and its seeded anchor inputs, are loaded.
    """
    state = torch.load(model_file or cfg["model_file"], map_location="cpu")
    R, width = state["log_alpha"].shape
    model = build_rule_net(training_data, cfg, R)
    if width != model.num_literals:
        # A model of the inputs only (trained before the negated / group literals), possibly without roles.
        roles = _saved_roles(width, training_data["num_atoms"], cfg)
        model = build_rule_net(training_data, cfg, R, roles=roles, literals=False)
    model.load_state_dict(state)
    return model.to(cfg["device"]).eval()


# -------------- Cascade: stage-1 candidates and the stage-2 verifier --------------
@torch.no_grad()
def candidate_thresholds(model, training_data, cfg):
    """Per-head stage-1 score thresholds that keep cascade_recall of the onset
    rows of the validation videos, on the largest window (rows with the head
    switching on within max(windows) frames); a head with fewer than
    cascade_min_positives such rows gets the threshold of all heads together.

    Returns {"thresholds": (A,) CPU, "recall": the target, "val_row_recall" and
    "val_candidate_share": the share of the onset rows and of all at-risk
    (row, head) cells with an observed largest window at or above the thresholds}.
    """
    model.eval()
    device, A = cfg["device"], training_data["num_atoms"]

    def rows(video_id):
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            r = _video_rows(model, training_data, video_id, frames, device)
            if len(r["inputs"]):
                scores = model.window_probs(model(r["inputs"])[0])[:, -1]
                yield scores, r["targets"][:, -1], r["at_risk"] & r["known"][:, -1]

    heads, values = [], []
    for video_id in tqdm(training_data["val_videos"], desc="Stage-1 scores of the val onsets"):
        for scores, targets, _cells in rows(video_id):
            r, h = torch.nonzero(targets, as_tuple=True)
            heads.append(h)
            values.append(scores[r, h])
    heads, values = (torch.cat(x).cpu() if x else torch.zeros(0) for x in (heads, values))
    q = 1 - cfg["cascade_recall"]
    thresholds = torch.full((A,), torch.quantile(values, q).item() if len(values) else 0.0)
    for h in heads.unique().tolist():
        if (heads == h).sum() >= cfg["cascade_min_positives"]:
            thresholds[h] = torch.quantile(values[heads == h], q)

    kept, onsets, candidates, cells = 0, 0, 0, 0
    on_device = thresholds.to(device)
    for video_id in tqdm(training_data["val_videos"], desc="Stage-1 candidates of val"):
        for scores, targets, at_risk in rows(video_id):
            flagged = scores >= on_device
            kept += (flagged & targets).sum().item()
            onsets += targets.sum().item()
            candidates += (flagged & at_risk).sum().item()
            cells += at_risk.sum().item()
    return {"thresholds": thresholds, "recall": cfg["cascade_recall"],
            "val_row_recall": kept / max(onsets, 1), "val_candidate_share": candidates / max(cells, 1)}


def make_candidates(model, thresholds):
    """candidates(inputs) -> (B, A) bool: the stage-1 model flags the head (its
    probability of an onset within the largest window is at or above the head's threshold)."""
    model.eval()

    @torch.no_grad()
    def candidates(inputs):
        return model.window_probs(model(inputs)[0])[:, -1] >= thresholds.to(inputs.device)
    return candidates


def train_verifier(training_data, cfg, candidates, run=None):
    """Stage 2 of the cascade: a TemporalRuleNet seeded like stage 1, trained only
    on the stage-1 candidate cells with verifier_pos_weight, so its rules (the
    anchors extended by the atoms that separate true from false candidates)
    learn to reject the stage-1 false alarms."""
    verifier_cfg = {**cfg, "pos_weight": cfg["verifier_pos_weight"]}
    return _fit_rule_net(_seeded_rule_net(training_data, cfg), training_data, verifier_cfg,
                         cfg["verifier_history_file"], run, candidates)


# -------------- Black-box baseline --------------
def train_onset_mlp(training_data, cfg, run=None):
    """Train the OnsetMLP baseline (`fit_onset_model`) on the same input rows,
    targets and batches as the TemporalRuleNet, with weight decay instead of
    the L0 loss and no anchor seeding. Its hidden size defaults to the number of
    rule nodes of the TemporalRuleNet, so both have about as many parameters."""
    hidden_size = cfg["mlp_hidden_size"] or len(anchor_rules(cfg)) * cfg["num_slots"] + cfg["num_free_rules"]
    model = OnsetMLP(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"], hidden_size, cfg["mlp_dropout"],
                     cfg["windows"], cfg["mlp_activation"], roles=cfg["roles"],
                     open_memory=cfg.get("slot_ranges", False)).to(cfg["device"])
    print(f"[Baseline] OnsetMLP with {hidden_size} {cfg['mlp_activation']} hidden units, lr {cfg['mlp_lr']}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["mlp_lr"], weight_decay=cfg["mlp_weight_decay"])
    mlp_cfg = {**cfg, "grad_clip": cfg["mlp_grad_clip"], "lr_patience": cfg["mlp_lr_patience"],
               "patience": cfg["mlp_patience"]}

    @torch.no_grad()
    def model_stats():
        # Hidden units that never vary would leave one constant score per head (the prior).
        return {"model/hidden_weight_norm": model.hidden.weight.norm().item()}
    return fit_onset_model(model, training_data, mlp_cfg, optimizer, cfg["mlp_history_file"], run=run,
                           model_stats=model_stats)


def load_onset_mlp(training_data, cfg):
    """The trained OnsetMLP of cfg["mlp_model_file"], in eval mode on cfg["device"],
    or None if it was not trained."""
    if not os.path.exists(cfg["mlp_model_file"]):
        return None
    state = torch.load(cfg["mlp_model_file"], map_location="cpu")
    model = OnsetMLP(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"], state["hidden.weight"].shape[0],
                     windows=cfg["windows"], activation=cfg["mlp_activation"],
                     roles=_saved_roles(state["hidden.weight"].shape[1] // 2, training_data["num_atoms"], cfg,
                                        len(memory_bins(cfg["num_slots"], cfg.get("slot_ranges", False)))),
                     open_memory=cfg.get("slot_ranges", False))
    model.load_state_dict(state)
    return model.to(cfg["device"]).eval()


@torch.no_grad()
def _rule_head_counts(model, training_data, video_ids, cfg, desc="Counting rule firings", candidates=None,
                      possible=None):
    """Counts of every rule r and head atom h over all frames of the given videos
    (with their full history; only the cells candidates(inputs) flags, if given:
    the verifier's rules are judged on the stage-1 candidates they filter), for
    each alignment f = 0..J-1: the head in future slot f, so the bodies with
    j > f are in the memory and the f slots of bodies closest to the head have
    not happened yet. f = 0 is the whole clause (every body seen, head within the
    next slot); a larger f is the same clause known only up to its earlier bodies.

    Returns a dict of CPU tensors:
      fires (R, J, A): rows where every body of r with j > f holds (r has one) and
          h is at risk, has not switched on before slot f and slot f is observed;
      hits (R, J, A): the fires whose first onset of h falls in slot f;
      videos (R, J, A): the videos with a hit;
      anchor_fires / anchor_hits (R, J, A): the same for the anchor input of r
          alone (its fixed input, without the learned extra atoms; 0 for free rules);
      onsets (J, A) / at_risk (J, A): the hits and fires of an empty body (the base rate).
    possible(video_id) -> (N, A) bool (`class_possible`), if given: also possible_fires, possible_hits
    (R, J, A) and possible_onsets, possible_at_risk (J, A), the same counted only on the rows of agents
    the head can hold at all (its class; the inhibitors are judged on them).
    """
    J, S = model.num_slots, model.slot_size
    num_rules, num_atoms = model.log_alpha.shape[0], model.num_atoms
    device = cfg["device"]
    # A hit: the head switches on within clause_slots slots from slot f (2: within 2 s, the reaction lag
    # of the injected rules; 1: the next slot only).
    w = int(cfg.get("clause_slots", 1))
    shape = (num_rules, J * num_atoms)
    counts = {name: torch.zeros(shape, device=device)
              for name in ("fires", "hits", "videos", "anchor_fires", "anchor_hits")}
    counts["onsets"] = torch.zeros(J * num_atoms, device=device)
    counts["at_risk"] = torch.zeros(J * num_atoms, device=device)
    if possible is not None:
        counts["possible_fires"] = torch.zeros(shape, device=device)
        counts["possible_hits"] = torch.zeros(shape, device=device)
        counts["possible_onsets"] = torch.zeros(J * num_atoms, device=device)
        counts["possible_at_risk"] = torch.zeros(J * num_atoms, device=device)

    offset = getattr(model, "literal_offset", model.input_offset)

    def in_memory(rule_literals, f):
        # Body literals with j > f (in the memory for a head in slot f); `literal_values(inputs, f)` moves the
        # inputs f slots back, so these columns read the past slots j - f.
        return (rule_literals & (offset > f)).float()

    body = model.rule_mask() > cfg["mask_threshold"]
    known_bodies = [in_memory(body, f) for f in range(J)]
    known_anchors = [in_memory(model.fixed_inputs, f) for f in range(J)]
    for video_id in tqdm(video_ids, desc=desc):
        video_hits = torch.zeros(shape, device=device)
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            rows = _video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            at_risk = rows["at_risk"]
            if candidates is not None:
                at_risk = at_risk & candidates(rows["inputs"])
            first = rows["first"]
            can = possible(video_id)[rows["rows"][:, 1]].float() if possible is not None else None    # (B, A)
            last_window = rows["known"].shape[1] - 1
            for f in range(J):
                x = model.literal_values(rows["inputs"], f) if hasattr(model, "literal_values") \
                    else model.split_inputs(rows["inputs"])[0]
                risk = (at_risk & (first >= f * S) & rows["known"][:, min(f + w - 1, last_window)]).float()
                onset = (at_risk & (first >= f * S) & (first < (f + w) * S)).float()
                cols = slice(f * num_atoms, (f + 1) * num_atoms)
                for prefix, b in (("", known_bodies[f]), ("anchor_", known_anchors[f])):
                    size = b.sum(dim=1)
                    fired = ((x @ b.T >= size) & (size > 0)).float()                      # (B, R)
                    counts[prefix + "fires"][:, cols] += fired.T @ risk
                    if prefix:
                        counts["anchor_hits"][:, cols] += fired.T @ onset
                    else:
                        video_hits[:, cols] += fired.T @ onset
                counts["onsets"][cols] += onset.sum(dim=0)
                counts["at_risk"][cols] += risk.sum(dim=0)
                if can is not None:
                    b = known_bodies[f]
                    size = b.sum(dim=1)
                    fired = ((x @ b.T >= size) & (size > 0)).float()
                    counts["possible_fires"][:, cols] += fired.T @ (risk * can)
                    counts["possible_hits"][:, cols] += fired.T @ (onset * can)
                    counts["possible_onsets"][cols] += (onset * can).sum(dim=0)
                    counts["possible_at_risk"][cols] += (risk * can).sum(dim=0)
        counts["hits"] += video_hits
        counts["videos"] += video_hits > 0
    per_head = ("onsets", "at_risk", "possible_onsets", "possible_at_risk")
    out = {name: value.view(num_rules, J, num_atoms) for name, value in counts.items() if name not in per_head}
    out.update({name: counts[name].view(J, num_atoms) for name in per_head if name in counts})
    return {name: value.cpu() for name, value in out.items()}


def scope_names(model):
    """Names of the input scopes (slot types) of a model: X0 the agent itself, X1
    some other agent, then its roles (e.g. "ahead")."""
    return ["X0", "X1", *getattr(model, "roles", ())]


def _temporal_clause_text(clause, unique_atoms):
    # Bodies earliest first, in frames before the head's 10-frame slot:
    # "Car:MovAway(X0)@-30..-20f -> Car:Brake(ahead)@-10..0f => Car:Stop(X0)".
    names = clause.get("scope_names") or ["X0", "X1"]

    def atom(atom_index, slot, lit=None):
        scope = names[slot] if slot < len(names) else f'X{slot}'
        if lit and lit.get("kind") in ("G", "!G"):
            return f"{'!' if lit['kind'] == '!G' else ''}veh:{lit['group']}({scope})"
        neg = "!" if lit and lit.get("kind") == "!X" else ""
        return f"{neg}{_atom_name(unique_atoms[atom_index])}({scope})"
    size = clause.get("slot_size", 10)
    frames = clause.get("span_frames") or [(size * (j - 1), size * j) for _a, _slot, j in clause["bodies"]]
    lits = clause.get("literals") or [None] * len(clause["bodies"])
    body = " -> ".join(f"{atom(a, slot, lit)}@-{'inf' if far is None else far}..-{near}f"
                       for (a, slot, _j), (near, far), lit in zip(clause["bodies"], frames, lits))
    return f"{body} => {atom(*clause['head'])}"


def extract_temporal_clauses(model, training_data, cfg, run=None, candidates=None, inhibitor_file=None):
    """Clauses of the trained model, one per (rule node, head) pair with a large
    head weight, kept if the rule hits the head in at least min_support training
    frames of at least min_videos videos (every body seen, head in the next slot).

    Each clause: bodies [(atom, slot, j)] earliest first (the atom holds in the
    slot j slots before the head's; slot 0 the head agent, 1 some other agent),
    body_scopes their slots, head (atom, 0), anchor (atom, slot, j) its seeded
    anchor input or None, weight, and on the training videos support (hits),
    fires, confidence = support / fires, num_videos, anchor_confidence (of the
    anchor input alone). The val_ fields are the same on the validation videos,
    plus val_recall (the share of the head's onsets the clause hits) and
    val_base_rate. partial (train and val): per alignment f, the confidence of
    the head f slots ahead knowing only the bodies with j > f (known_bodies of
    them), with the base rate: how the confidence grows as more bodies are seen.
    Sorted by confidence. With candidates (the verifier), all counts are over
    the stage-1 candidate cells.
    inhibitor_file: also the inhibitors (`inhibitor_clauses`, from the same counts) are saved there.
    """
    def ratio(a, b):
        return float(a) / float(b) if b > 0 else None

    rules = model.extract_rules(cfg["mask_threshold"], cfg["weight_threshold"])
    possible = (class_possible(training_data, load_vocabulary(cfg), cfg["device"])
                if inhibitor_file is not None else None)
    train = _rule_head_counts(model, training_data, training_data["train_videos"], cfg,
                              desc="Counting rule firings (train)", candidates=candidates, possible=possible)
    val = _rule_head_counts(model, training_data, training_data["val_videos"], cfg,
                            desc="Counting rule firings (val)", candidates=candidates, possible=possible)

    clauses = []
    for rule in rules:
        r = rule["rule"]
        anchor_inputs = torch.nonzero(model.fixed_inputs[r]).flatten().tolist()
        anchor = list(model.decode_input(anchor_inputs[0])) if anchor_inputs else None
        offsets = [j for _atom, _slot, j in rule["bodies"]]
        for h, weight in rule["heads"]:
            support, num_videos = int(train["hits"][r, 0, h]), int(train["videos"][r, 0, h])
            if support < cfg["min_support"] or num_videos < cfg["min_videos"]:
                continue
            partial = []
            for f in range(min(max(offsets), model.num_slots)):
                partial.append({
                    "unseen_slots": f,
                    "known_bodies": sum(j > f for j in offsets),
                    "confidence": ratio(train["hits"][r, f, h], train["fires"][r, f, h]),
                    "val_confidence": ratio(val["hits"][r, f, h], val["fires"][r, f, h]),
                    "base_rate": ratio(train["onsets"][f, h], train["at_risk"][f, h]),
                })
            clauses.append({
                "bodies": [list(b) for b in rule["bodies"]],
                # The slots a..b before the head's each body may hold in (a == b == j without slot ranges).
                "spans": [list(sp) for sp in rule.get("spans", [(j, j) for _a, _s, j in rule["bodies"]])],
                # The frames near..far before the head's slot of each body (far None: any known time before).
                "span_frames": [list(sp) for sp in rule.get("span_frames", [])] or None,
                "body_scopes": [slot for _atom, slot, _j in rule["bodies"]],
                "scope_names": scope_names(model),
                "head": [h, 0],
                "slot_size": model.slot_size,
                "anchor": anchor,
                # Each body literal: kind "X" (holds), "!X" (known absent), "G" / "!G" (a group: any motor vehicle
                # class holds / none does), with its group label.
                "literals": rule.get("literals"),
                "seed": [model.literal_info(m) for m in anchor_inputs] if hasattr(model, "literal_info") else None,
                "clause_slots": int(cfg.get("clause_slots", 1)),
                "rule": r,
                "weight": weight,
                "support": support,
                "fires": int(train["fires"][r, 0, h]),
                "confidence": support / max(int(train["fires"][r, 0, h]), 1),
                "num_videos": num_videos,
                "anchor_confidence": ratio(train["anchor_hits"][r, 0, h], train["anchor_fires"][r, 0, h]),
                "val_support": int(val["hits"][r, 0, h]),
                "val_fires": int(val["fires"][r, 0, h]),
                "val_confidence": ratio(val["hits"][r, 0, h], val["fires"][r, 0, h]),
                "val_anchor_confidence": ratio(val["anchor_hits"][r, 0, h], val["anchor_fires"][r, 0, h]),
                "val_recall": ratio(val["hits"][r, 0, h], val["onsets"][0, h]),
                "val_base_rate": ratio(val["onsets"][0, h], val["at_risk"][0, h]),
                "partial": partial,
            })
    clauses.sort(key=lambda c: (-c["confidence"], -c["support"]))
    if inhibitor_file is not None:
        utils_data.save_json(inhibitor_clauses(model, train, val, cfg), inhibitor_file)

    print(f"[Rules] {len(rules)} rule nodes with heads -> {len(clauses)} clauses kept")
    if run is not None:
        import wandb
        table = wandb.Table(columns=["clause", "confidence", "support", "fires", "num_videos", "weight", "length"])
        unique_atoms = load_vocabulary(cfg)
        for clause in clauses:
            table.add_data(_temporal_clause_text(clause, unique_atoms), clause["confidence"], clause["support"],
                           clause["fires"], clause["num_videos"], clause["weight"], len(clause["bodies"]))
        run.log({"clauses": table})
        run.summary["num_clauses"] = len(clauses)
        run.summary["num_rules_with_heads"] = len(rules)
    return clauses


def inhibitor_kind(suppression, cfg):
    """"inhibitor" (the head's rate after the bodies <= max_inhibitor_suppression x its base rate),
    "correction" (>= 1: the head is not rarer after the bodies; the negative weight offsets other rules)
    or "weak"; None without a rate."""
    if suppression is None:
        return None
    if suppression <= cfg.get("max_inhibitor_suppression", 0.5):
        return "inhibitor"
    return "correction" if suppression >= 1.0 else "weak"


def inhibitor_clauses(model, train, val, cfg):
    """Inhibitors "bodies -> not head": the (rule node, head) pairs with a head weight below
    -weight_threshold (`TemporalRuleNet.extract_rules(negative=True)`), judged on the possible_ counts of
    `_rule_head_counts` (only the rows of agents of the head's class: a body that only says "this agent
    is of another class" fires there never and is dropped). Kept if the bodies fire in >= min_inhibitor_fires
    such training rows where the head could start (at risk) and the head is not rare there (base rate >=
    min_inhibitor_base: ruling out what hardly ever happens says nothing). Each: the clause fields of
    `extract_temporal_clauses` with violation_rate = P(head starts in the next slot | bodies) (its
    confidence), base_rate and suppression = violation_rate / base_rate (0: the bodies rule the head
    out), on train and val, and kind (`inhibitor_kind`, by val suppression). Sorted by val suppression,
    then train fires."""
    def ratio(a, b):
        return float(a) / float(b) if b > 0 else None

    rules = model.extract_rules(cfg["mask_threshold"], cfg["weight_threshold"], negative=True)
    out, dropped = [], 0
    for rule in rules:
        r = rule["rule"]
        for h, weight in rule["heads"]:
            fires = int(train["possible_fires"][r, 0, h])
            base = ratio(train["possible_onsets"][0, h], train["possible_at_risk"][0, h])
            if fires < cfg.get("min_inhibitor_fires", 50) or not base or base < cfg.get("min_inhibitor_base", 0.01):
                dropped += 1
                continue
            rate = ratio(train["possible_hits"][r, 0, h], fires)
            val_rate = ratio(val["possible_hits"][r, 0, h], val["possible_fires"][r, 0, h])
            val_base = ratio(val["possible_onsets"][0, h], val["possible_at_risk"][0, h])
            val_suppression = val_rate / val_base if val_rate is not None and val_base else None
            out.append({
                "bodies": [list(b) for b in rule["bodies"]],
                "spans": [list(sp) for sp in rule.get("spans", [(j, j) for _a, _s, j in rule["bodies"]])],
                "span_frames": [list(sp) for sp in rule.get("span_frames", [])] or None,
                "body_scopes": [slot for _atom, slot, _j in rule["bodies"]],
                "scope_names": scope_names(model),
                "head": [h, 0], "negated": True, "slot_size": model.slot_size, "rule": r, "weight": weight,
                "literals": rule.get("literals"),
                "fires": fires, "support": int(train["possible_hits"][r, 0, h]), "confidence": rate,
                "violation_rate": rate, "base_rate": base, "suppression": rate / base if rate is not None else None,
                "val_fires": int(val["possible_fires"][r, 0, h]), "val_support": int(val["possible_hits"][r, 0, h]),
                "val_confidence": val_rate, "val_violation_rate": val_rate, "val_base_rate": val_base,
                "val_suppression": val_suppression, "kind": inhibitor_kind(val_suppression, cfg),
                "partial": [],
            })
    out.sort(key=lambda c: (c["val_suppression"] if c["val_suppression"] is not None else 9.0, -c["fires"]))
    kinds = {k: sum(1 for c in out if c["kind"] == k) for k in ("inhibitor", "weak", "correction")}
    print(f"[Inhibitors] {sum(len(r['heads']) for r in rules)} negative (rule, head) pairs; {dropped} dropped (bodies "
          f"fire < {cfg.get('min_inhibitor_fires', 50)} times on agents of the head's class, or the head is rare "
          f"there); kept {len(out)}: by val suppression {kinds}")
    return out


# -------------- Plot helpers (shared with step 06) --------------
# Confidences, precisions and base rates span 1e-6..1, so they go on log axes;
# a 0 cannot be shown there and is drawn on the lower edge instead.
def log_floor(values, floor=1e-6):
    """Lower limit of a log axis: the decade at or below the smallest positive value."""
    positive = [v for v in values if v is not None and v > floor]
    return 10 ** math.floor(math.log10(min(positive))) if positive else floor


def clip_low(values, low):
    """Values with None / 0 / anything below low raised to low (drawn on the axis edge)."""
    return [max(v or 0.0, low) for v in values]


def clause_scatter(ax, rows, x_key, y_key):
    """Log-log scatter of two confidences of every clause, one color per body
    length, dashed y = x; 0 is drawn on the lower edges."""
    xs, ys = [c[x_key] for c in rows], [c[y_key] for c in rows]
    low = log_floor(xs + ys)
    xs, ys = clip_low(xs, low), clip_low(ys, low)
    lengths = [len(c["bodies"]) for c in rows]
    for length in sorted(set(lengths)):
        picked = [i for i, n in enumerate(lengths) if n == length]
        ax.scatter([xs[i] for i in picked], [ys[i] for i in picked], s=16, alpha=0.7,
                   label=f"{length} atom{'s' if length > 1 else ''}", clip_on=False)
    ax.plot([low, 1], [low, 1], "k--", linewidth=0.8)
    ax.set(xscale="log", yscale="log", xlim=(low, 1), ylim=(low, 1))
    ax.legend(title="body length", fontsize=7, title_fontsize=7)


def head_dot_plot(ax, labels, series, reference=None):
    """Per-row dots on a log x axis: series [(label, values, color, marker)], and
    a gray line from reference (e.g. the base rate) to the first series, the lift."""
    low = log_floor([v for _label, values, _c, _m in series for v in values] + list(reference or []))
    y = list(range(len(labels)))
    if reference is not None:
        ax.hlines(y, clip_low(reference, low), clip_low(series[0][1], low), color="lightgray", linewidth=2, zorder=1)
    for label, values, color, marker in series:
        ax.scatter(clip_low(values, low), y, color=color, marker=marker, s=22, label=label, zorder=3, clip_on=False)
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set(xscale="log", xlim=(low, 1))
    ax.grid(axis="x", which="major", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7)


def _plot_results(clauses, cfg, num_heads=30):
    """Save the step 05 figures, one PDF each, to cfg["figure_dir"]:
      training_curves.pdf: train / val onset loss per epoch;
      extension_gain.pdf: val confidence of the anchor alone vs the extended clause;
      train_vs_val_confidence.pdf: overfitting check of every clause;
      precision_recall_val.pdf: val confidence vs recall, size = train support;
      best_clause_per_head.pdf: best val confidence per head vs its base rate;
      delay_map.pdf: clauses per (head, slot of the anchor before the head);
      partial_evidence.pdf: clause confidence (val, over the base rate) by how many of its bodies are known.
    A figure is skipped (with a message) when its data is missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_atoms = load_vocabulary(cfg)
    figure_dir = cfg["figure_dir"]
    os.makedirs(figure_dir, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figure_dir / name)
        plt.close(fig)
        print(f"[Figure]\t {figure_dir / name}")

    def has(clause, *keys):
        return all(clause.get(k) is not None for k in keys)

    # Training curves.
    if os.path.exists(cfg["history_file"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, file, color in (("rule net", cfg["history_file"], "tab:blue"),
                                  ("MLP baseline", cfg["mlp_history_file"], "tab:green")):
            if not os.path.exists(file):
                continue
            history = utils_data.load_json(file)
            ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], color=color,
                    linestyle="--", label=f"{name} train")
            ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], color=color,
                    label=f"{name} val")
        ax.set(title="Onset loss per epoch", xlabel="epoch", ylabel="weighted BCE")
        ax.legend()
        save(fig, "training_curves.pdf")
    else:
        print(f"[Figure]\t skipped training_curves.pdf: no {cfg['history_file']}")

    if not clauses:
        print("[Figure]\t skipped the clause figures: no clauses")
        return

    # Extension gain: the extended clauses only, above the diagonal = the extra atoms help.
    rows = [c for c in clauses if len(c["bodies"]) > 1 and has(c, "val_confidence", "val_anchor_confidence")]
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        clause_scatter(ax, rows, "val_anchor_confidence", "val_confidence")
        better = sum(c["val_confidence"] > c["val_anchor_confidence"] for c in rows)
        ax.set(title=f"Extension gain on val ({better}/{len(rows)} above y = x)",
               xlabel="confidence of the anchor alone", ylabel="confidence of the extended clause")
        save(fig, "extension_gain.pdf")
    else:
        print("[Figure]\t skipped extension_gain.pdf: no extended clause fires on val")

    # Train vs val confidence: far below the diagonal = overfitted.
    rows = [c for c in clauses if has(c, "val_confidence")]
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        clause_scatter(ax, rows, "confidence", "val_confidence")
        ax.set(title="Clause confidence, train vs val", xlabel="train confidence", ylabel="val confidence")
        save(fig, "train_vs_val_confidence.pdf")

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sizes = [4 + 60 * c["support"] / max(r["support"] for r in rows) for c in rows]
        low = log_floor([c["val_confidence"] for c in rows])
        ax.scatter([c["val_recall"] or 0 for c in rows], clip_low([c["val_confidence"] for c in rows], low),
                   s=sizes, alpha=0.5, clip_on=False)
        ax.set(title="Val confidence vs recall (size = train support)",
               xlabel="recall (share of head onsets hit)", ylabel="confidence (precision, log)",
               xlim=(0, 1), ylim=(low, 1), yscale="log")
        save(fig, "precision_recall_val.pdf")
    else:
        print("[Figure]\t skipped train_vs_val_confidence.pdf / precision_recall_val.pdf: no clause fires on val")

    # Best clause per head, against the head's base rate.
    best = {}
    for c in rows:
        h = c["head"][0]
        if h not in best or c["val_confidence"] > best[h]["val_confidence"]:
            best[h] = c
    if best:
        heads = sorted(best, key=lambda h: best[h]["val_confidence"], reverse=True)[:num_heads]
        fig, ax = plt.subplots(figsize=(7, 0.25 * len(heads) + 1.5))
        head_dot_plot(ax, [_atom_name(unique_atoms[h]) for h in heads],
                      [("best clause", [best[h]["val_confidence"] for h in heads], "tab:blue", "o"),
                       ("base rate", [best[h]["val_base_rate"] for h in heads], "tab:red", "|")],
                      reference=[best[h]["val_base_rate"] for h in heads])
        ax.set(title=f"Best val confidence per head (top {len(heads)}), gray = lift over base rate",
               xlabel="val confidence (log)")
        save(fig, "best_clause_per_head.pdf")

    # Delay map: how many slots before the head the anchor of the clauses of each head lies.
    size = cfg["slot_size"]
    bins = [(size * (j - 1), size * j) for j in range(1, cfg["num_slots"] + 1)]
    rows = [c for c in clauses if c.get("anchor")]
    if rows:
        heads = sorted({c["head"][0] for c in rows},
                       key=lambda h: -sum(c["head"][0] == h for c in rows))[:num_heads]
        grid = torch.zeros((len(heads), len(bins)))
        for c in rows:
            if c["head"][0] in heads:
                grid[heads.index(c["head"][0]), c["anchor"][2] - 1] += 1
        fig, ax = plt.subplots(figsize=(7, 0.25 * len(heads) + 1.5))
        image = ax.imshow(grid.numpy(), aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(bins)), [f"{lo}-{hi}" for lo, hi in bins], fontsize=7)
        ax.set_yticks(range(len(heads)), [_atom_name(unique_atoms[h]) for h in heads], fontsize=7)
        ax.set(title="Clauses per head and anchor slot", xlabel="anchor held this many frames before the head's slot")
        fig.colorbar(image, ax=ax, label="clauses")
        save(fig, "delay_map.pdf")

    # Partial evidence: confidence of the clauses with 2+ bodies by how many of their bodies are known.
    rows = [c for c in clauses if len(c["bodies"]) > 1 and c.get("partial")]
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        lift = {}
        for c in rows:
            points = [(p["known_bodies"], p["val_confidence"] / p["base_rate"]) for p in c["partial"]
                      if p["val_confidence"] is not None and p["base_rate"]]
            if len(points) > 1:
                ax.plot(*zip(*sorted(points)), color="tab:blue", alpha=0.15, linewidth=0.8)
            for known, value in points:
                lift.setdefault(known, []).append(value)
        known = sorted(lift)
        ax.plot(known, [sorted(lift[k])[len(lift[k]) // 2] for k in known], "o-", color="black", label="median")
        ax.axhline(1, color="tab:red", linestyle="--", label="base rate")
        ax.set(title="Clause confidence by bodies known (val)", xlabel="bodies of the clause known",
               ylabel="confidence / base rate (log)", yscale="log")
        ax.legend()
        save(fig, "partial_evidence.pdf")


def _print_summary(clauses, training_data, cfg, num_examples=15, title="Step 05 Result", clause_file=None):
    """Terminal summary of the step 05 result, printed on every run (cached or not)."""
    unique_atoms = load_vocabulary(cfg)
    print(f"----------------- {title} -----------------")
    print(f"[Videos]\t {len(training_data['train_videos'])} train, {len(training_data['val_videos'])} val")
    print(f"[Clauses]\t {len(clauses)} in {clause_file or cfg['extended_clause_file']}")
    if not clauses:
        print("[Clauses]\t none passed min_support / min_videos / weight_threshold")
        return
    lengths = {}
    for clause in clauses:
        lengths[len(clause["bodies"])] = lengths.get(len(clause["bodies"]), 0) + 1
    confidences = sorted(clause["confidence"] for clause in clauses)
    print(f"[Heads]\t\t {len({clause['head'][0] for clause in clauses})} distinct head atoms")
    print(f"[Body length]\t " + ", ".join(f"{n} atoms: {count}" for n, count in sorted(lengths.items())))
    print(f"[Confidence]\t median {confidences[len(confidences) // 2]:.3f}, max {confidences[-1]:.3f} (train)")
    val_confidences = sorted(c["val_confidence"] for c in clauses if c.get("val_confidence") is not None)
    if val_confidences:
        print(f"[Confidence]\t median {val_confidences[len(val_confidences) // 2]:.3f}, "
              f"max {val_confidences[-1]:.3f} (val, {len(val_confidences)} clauses firing on val)")
    gains = sorted(c["val_confidence"] - c["val_anchor_confidence"] for c in clauses
                   if len(c["bodies"]) > 1 and c.get("val_confidence") is not None
                   and c.get("val_anchor_confidence") is not None)
    if gains:
        print(f"[Extension]\t {sum(g > 0 for g in gains)}/{len(gains)} extended clauses beat their anchor on val, "
              f"median gain {gains[len(gains) // 2]:+.3f}")
    print(f"[Figures]\t {cfg['figure_dir']}")
    print(f"[Top {min(num_examples, len(clauses))}]")
    for clause in clauses[:num_examples]:
        print(f"  conf {clause['confidence']:.2f}  support {clause['support']:>5}  "
              f"{_temporal_clause_text(clause, unique_atoms)}")


MINE_VERSION = "m1"            # `step05_mine`: bump when the mining changes (new seeds, new model files)


def anchor_file_of(input_data, source, seed=0):
    """The anchor clause file of an anchor source ({clause_id: {"anchor": {"head", "body"}}}, the step 04
    format): "causal" the step 04 file itself; the others are written once into the run's step05_output:
    "all" every step 03 LLM anchor, "random" as many of them as the causal file holds, drawn with seed,
    "none" no anchor."""
    causal = input_data["causal_anchor_clause_file"]
    if source in ("causal", "mined"):                   # mined: the anchors are not used to seed (`step05_mine`)
        return causal
    out = input_data["output_dir"] / (f"anchors_{source}" + (f"{seed}" if source == "random" else "") + ".json")
    if not os.path.exists(out):
        import random
        llm = utils_data.load_json(input_data["anchor_clause_file"])            # step 03: [{"head", "body"}]
        if source == "none":
            chosen = []
        elif source == "all":
            chosen = llm
        else:
            k = len(utils_data.load_json(causal))
            chosen = random.Random(seed).sample(llm, min(k, len(llm)))
        utils_data.save_json({str(i): {"anchor": c} for i, c in enumerate(chosen)}, out)
        print(f"[Anchors]\t {source}: {len(chosen)} of {len(llm)} LLM anchor clauses -> {out}")
    return out


def get_cfg(input_data):
    """Settings and files of step 05; step 06 uses them to load the trained model."""
    # Memory: 3 slots of 10 frames (3 s). Medium 6b02b3 / full 78e420 scored with only the last h s of
    # their 5 s memory known: AP at 1 s >= 90% of the full memory, best at 2-3 s, no gain beyond 3 s.
    slot_size, num_slots = 10, input_data.get("num_slots", 3)       # --num_slots (default 3)
    windows = [slot_size * (f + 1) for f in range(10)]
    mask_type, num_free_rules = "sigmoid", 128
    # Files are named by the settings that change the model, so a model trained
    # for other settings is never reused.
    # Noise heads (fewer than noise_below onsets in the train + val videos) are no training
    # targets; step 06 reports their events separately.
    exclude_noise_heads, noise_below = True, 5
    # Type balance (`head_pos_weight_scale`), main rule net only: the onsets of the event types
    # with fewer onsets per head weigh (M_max / M_type) ** power, at most type_balance_max.
    # Off: no clear gain (type balance inflates the scores of the regular / rare heads into
    # false alarms; roles left the val loss unchanged; geometry made it worse). A module is kept
    # only with a clear effect in a clean ablation; set these to re-run one.
    type_balance = {"type_balance": input_data.get("type_balance", False), "type_balance_power": 0.5,
                    "type_balance_max": 4.0}                                     # --type_balance
    # Agent roles (`roles.py`): one more input scope per role, the atoms of the agent filling
    # it for the row's agent (e.g. the agent ahead of it), from the GT boxes; () for none.
    roles = tuple(input_data.get("roles", ()))  # --roles ahead,near_ped (default none)
    # Geometry atoms (`geometry.py`): facts of the GT boxes that hold without depth (bearing
    # sector, at the image edge, bearing drift), inputs only, after the label atoms; () for none.
    geometry = agent_geometry.GEOMETRY if input_data.get("geometry") else ()   # --geometry (default False)
    # Grid atoms (`grid.py`): the cells of the agent's own two box corners in the 2D image
    # (position atoms; the compound atoms of the other agents are removed, no rule used them).
    grid = input_data.get("grid", True)         # --grid (default True)
    # Motion atoms (`motion.py`): how the agent's own box grows, shrinks and moves (approach, recede,
    # closing in faster / slower, left / right, still), own scope only. Off since 2026-11-15: no effect
    # (medium suite 20261113-1635: every metric unchanged without them; full 78e420: the own-label lookup
    # matched the models without them). --motion True for the ablation or to resume a run that used them.
    motion = input_data.get("motion", False)    # --motion (default False)
    # Ego motion atoms (`ego_motion.py`): the ego's speed level read from the scene, the size and
    # closing speed of its lead agent and of its nearest agent; the timing diagnostic
    # (`step05_timing_diag`) found the AV-Stop timing there, none in the ego's labels. Off until an
    # ablation on medium and full shows a clear gain.
    ego_motion = input_data.get("ego_motion", False)          # --ego_motion (default False)
    # Ego flow atoms (`ego_flow.py`): the ego's yaw (left / straight / right, strong) and speed levels read
    # from the camera image (sparse optical flow, denoised), defined in an empty scene too. ego_flow_diag
    # (medium_s, 2026-11-16): the yaw passes the straight band 1-2 s before the AV-TurRht / TurLft labels
    # (80% / 94% of the onsets 1-2 s before, 15% of the controls), the speed falls before AV-Stop.
    # Off until an ablation shows a clear gain.
    ego_flow = input_data.get("ego_flow", False)              # --ego_flow (default False)
    # Duration atoms (`duration.py`): how long a label has held, log-scale age bins and their unions
    # ("the agent ahead has braked for 0.5-2 s"). Off until an ablation shows a clear gain.
    durations = input_data.get("durations", False)            # --durations (default False)
    dur_labels = input_data.get("duration_labels") or None    # --duration_labels (default duration.LABELS)
    # Slot ranges (`TemporalRuleNet` ranges): a body reads a span of slots a..b ("slowing down 2-4 s before")
    # instead of one 1 s slot, so the steps of a sequence may last more or less long. Off until an
    # ablation shows a clear gain.
    slot_ranges = input_data.get("slot_ranges", False)        # --slot_ranges (default False)
    # Prediction atoms (`predict.py`): when the ego reaches a pedestrian / cyclist and whether their paths
    # cross then, from the boxes; read in every scope. Off until an ablation shows a clear gain.
    predict = input_data.get("predict", False)                # --predict (default False)
    # Every rule must hold an own-agent atom (`TemporalRuleNet` require_self): without it, 75% of the
    # rules read only the other agents and fire for every agent of a scene alike (medium, grid 3x3).
    require_self = input_data.get("require_self", True)       # --require_self (default True)
    # Anchor clauses that seed the rules (`anchor_file`, the causal step ablation): "causal" (the step 04
    # causal filter of the LLM anchors, the main model), "all" (every LLM anchor, no causal filter),
    # "random" (a random subset of the LLM anchors as large as the causal one, anchor_subset_seed),
    # "none" (no anchors: free rules only, as many more as the causal anchors seed, the same capacity).
    anchor_source = input_data.get("anchor_source", "causal")
    anchor_seed = input_data.get("anchor_subset_seed", 0)
    if anchor_source not in ("causal", "all", "random", "none", "mined"):
        raise ValueError(f"Unknown anchor_source {anchor_source}")
    # Literals of the rule bodies (`TemporalRuleNet` negation / groups): "!X" (X known absent: "the car ahead
    # has not stopped", "no pedestrian crossed 2-3 s ago") and "any motor vehicle X". The SUMO v7 rule mining
    # (`rule_tree`, 2026-11-20) needed both for the injected rules R1 / R3 (precision 0.47 -> 0.73).
    negation = bool(input_data.get("negation", True))                     # --negation (default True)
    groups_on = bool(input_data.get("literal_groups", True))              # --literal_groups (default True)
    if slot_ranges and (negation or groups_on):
        raise ValueError("--slot_ranges does not support the negated / group literals: pass --negation False "
                         "--literal_groups False")
    # A clause hits its head if the head switches on within clause_slots slots (2: 2 s, the reaction lag of the
    # injected rules; 1 s cut R3 off: its precision 0.46 at 1 s, 0.61 at 2 s).
    clause_slots = int(input_data.get("clause_slots", 2))                 # --clause_slots (default 2)
    anchor_file = anchor_file_of(input_data, anchor_source, anchor_seed)
    if input_data.get("num_free_rules") is not None:
        num_free_rules = int(input_data["num_free_rules"])
    elif anchor_source == "none":
        causal_bodies = len(group_clauses_by_body(utils_data.load_json(input_data["causal_anchor_clause_file"])))
        num_free_rules += causal_bodies * num_slots
    # Partial facts: the share of training frames with a truncated history (--truncate_prob, default 0.5).
    truncate_prob = float(input_data.get("truncate_prob", 0.5))
    # Learning curve (--train_fraction, default 1): train on a share f of the train videos
    # (`subsample_train_videos`). An epoch then visits each of them round(1 / f) times, so it has as many
    # gradient steps as one on all the data: the epoch counts, early stopping, warm-up, lr schedule and
    # the val evaluations (once per epoch) are the same for every share.
    train_fraction = float(input_data.get("train_fraction", 1.0))
    train_subset_seed = int(input_data.get("train_subset_seed", 0))
    epoch_repeat = 1 if train_fraction >= 1 else max(1, int(round(1 / train_fraction)))
    # Fit diagnostics (`ablation.py` preset fit): the rule net's lr, sparsity weight and initial gate logit,
    # and the seed of both models (init and frame sampling); the defaults are the main model.
    rule_lr = float(input_data["rule_lr"]) if input_data.get("rule_lr") is not None else 0.01
    l0_lambda = float(input_data["l0_lambda"]) if input_data.get("l0_lambda") is not None else 1e-3
    init_log_alpha = float(input_data["init_log_alpha"]) if input_data.get("init_log_alpha") is not None else -11.0
    train_seed = int(input_data.get("train_seed") or 0)
    # Per-head rules (`TemporalRuleNet.seed_rules_per_head`, --per_head_rules): every rule serves one head;
    # rule_copies rules per (anchor body, head, slot), the copies after the first learn a random
    # copy_input_share of the inputs; free_rules_per_head free rules per learned head (num_free_rules unused).
    # The MLP keeps its hidden size (the shared-rule count), so its model and file stay the same.
    per_head_rules = bool(input_data.get("per_head_rules", False))
    rule_copies = int(input_data.get("rule_copies") or 2)
    free_rules_per_head = int(input_data.get("free_rules_per_head") if input_data.get("free_rules_per_head")
                              is not None else 4)
    copy_input_share = float(input_data.get("copy_input_share") or 0.5)
    videos_per_batch = int(input_data.get("videos_per_batch") or 4)        # --videos_per_batch (default 4)
    base_tag = f"slots{slot_size}x{num_slots}_w{len(windows)}_{mask_type}_f{num_free_rules}_partial"
    if truncate_prob != 0.5:
        base_tag += f"{truncate_prob:g}"
    if slot_ranges:
        base_tag += "H"         # truncated histories drawn from all the known memory (`sample_history`)
    if anchor_source not in ("causal", "mined"):
        base_tag += f"_a{anchor_source}" + (str(anchor_seed) if anchor_source == "random" else "")
    if train_fraction < 1:
        base_tag += f"_tf{train_fraction:g}s{train_subset_seed}"
    if train_seed:
        base_tag += f"_seed{train_seed}"
    if exclude_noise_heads:
        base_tag += f"_tn{noise_below}"
    if roles:
        base_tag += "_r" + "-".join(roles)
        if "ahead" in roles:
            base_tag += "L"                     # "ahead": the ego's lead in its lane (roles.py)
    if geometry:
        base_tag += "_g" + "-".join(geometry)
    base_tag += grid_tag({"grid": grid, "motion": motion, "ego_motion": ego_motion, "ego_flow": ego_flow,
                          "durations": durations, "duration_labels": dur_labels, "predict": predict})
    # The prepared datasets depend only on these settings (`shared_data_file`).
    min_off_frames = 10
    data_key = (f"m{min_off_frames}" + ("_r" + "-".join(roles) if roles else "") + ("_g" + "-".join(geometry) if geometry else "")
                + grid_tag({"grid": grid, "motion": motion, "ego_motion": ego_motion, "ego_flow": ego_flow,
                          "durations": durations, "duration_labels": dur_labels, "predict": predict}))
    # The main rule net (and what is derived from it: its clauses, cascade, pruning, rule set)
    # carries the type balance; the unweighted models (the MLP) keep base_tag.
    tag = base_tag + type_balance_tag(type_balance) + ("_self" if require_self else "") + ("_rng" if slot_ranges else "")
    if rule_lr != 0.01:
        tag += f"_lr{rule_lr:g}"
    if l0_lambda != 1e-3:
        tag += f"_l0{l0_lambda:g}"
    if init_log_alpha != -11.0:
        tag += f"_ia{init_log_alpha:g}"
    if anchor_source == "mined":
        tag += f"_mined{MINE_VERSION}f{free_rules_per_head}"
    elif per_head_rules:
        tag += f"_ph{rule_copies}c{free_rules_per_head}f{copy_input_share:g}s"
    if negation:
        tag += "_neg"
    if groups_on:
        tag += "_grp"
    if clause_slots != 1:
        tag += f"_cs{clause_slots}"
    if videos_per_batch != 4:
        tag += f"_vb{videos_per_batch}"
    # OnsetMLP baseline: lr 0.01 (the rule net's) with ReLU let every hidden unit die in the
    # first epochs (a constant score per head); a smaller lr, GELU, clipping and early stopping.
    mlp_lr, mlp_activation = 1e-3, "gelu"
    mlp_tag = f"{base_tag}_lr{mlp_lr:g}_{mlp_activation}" + ("_older" if slot_ranges else "")
    if videos_per_batch != 4:
        mlp_tag += f"_vb{videos_per_batch}"
    return {
            'device': input_data['device'],
            'seed': train_seed,
            # An atom switches on only after being off for this many frames for the
            # agent; shorter gaps are label flicker (re-onsets), not new events.
            'min_off_frames': min_off_frames,
            'data_key': data_key,
            'data_cache_dir': input_data.get("data_cache_dir"),
            # Event types by head frequency (onsets in the train + val videos, `head_event_types`):
            # fewer than noise_below onsets are noise, more than trivial_above trivial, the rest
            # regular. With exclude_noise_heads the noise heads are no training targets.
            'exclude_noise_heads': exclude_noise_heads,
            'noise_below': noise_below,
            # Heads with noise_below..rare_below - 1 onsets are learned but reported as "rare".
            'rare_below': 10,
            'trivial_above': 100,
            **type_balance,
            # Train only the main rule net and the MLP baseline (training time): the motif net is
            # not trained, its cached model is evaluated if it exists; the models derived from the
            # main rule net (cascade, causal pruning, rule set) are skipped. The MLP is always
            # trained (mlp_baseline), as the comparison of the rule net on the same inputs.
            'main_only': True,
            # Targets: the atom switches on within the next W frames, for each window W
            # (10, 20, ..., 100 frames = 1..10 s at 10 fps); the model gives one
            # probability per window, never decreasing with W.
            'windows': windows,
            # Model: time in slots of slot_size frames; the memory holds num_slots past
            # slots (3 x 10 = 30 frames), a rule is a pattern over the num_slots slots
            # before its head, and each window is one more future slot. The rule net's
            # hazards depend on the memory for the first num_slots future slots only (its
            # rules shifted by f slots); later slots get a constant hazard per head, so its
            # windows beyond num_slots slots add only that constant (the MLP reads every
            # window from the inputs). The videos are ~200 frames, so 30 frames of memory
            # leave the largest window (100 frames) observable at t = 30..~100.
            'slot_size': slot_size,
            'num_slots': num_slots,
            # Partial facts: a share truncate_prob of the training frames know only a
            # random 0..memory frames of history (the older memory is unknown), so the
            # model learns calibrated, lower confidence from partial facts.
            'truncate_prob': truncate_prob,
            # Rules: one per (anchor body, lag bin) with the anchor as a fixed input
            # and seed_head_weight to its heads, plus num_free_rules unseeded ones,
            # so heads without an anchor (about a quarter of the onsets) can be recalled.
            'num_free_rules': num_free_rules,
            'seed_head_weight': 1.0,
            'per_head_rules': per_head_rules,
            'negation': negation,
            'literal_groups': groups_on,
            'clause_slots': clause_slots,
            # Rule mining (`step05_mine`, anchor_source "mined"): per learned head, short rules (<= mine_depth
            # literals after a root own-agent literal) mined on a sample of the train rows (<= mine_pos_cap hit
            # rows, <= mine_neg_cap others, weighted back), kept if they reach mine_min_lift x the root's
            # precision with >= mine_min_events onsets in >= min_videos videos, and hold on val (mine_val_lift).
            'mine_pos_cap': int(input_data.get("mine_pos_cap") or 5000),
            'mine_neg_cap': int(input_data.get("mine_neg_cap") or 10000),
            'mine_depth': int(input_data.get("mine_depth") or 3),
            'mine_min_events': int(input_data.get("mine_min_events") or 20),
            'mine_min_lift': float(input_data.get("mine_min_lift") or 2.0),
            'mine_val_lift': float(input_data.get("mine_val_lift") or 1.5),
            'mine_max_rules': int(input_data.get("mine_max_rules") or 8),
            'mine_roots': int(input_data.get("mine_roots") or 2),
            'mine_root_min_cover': 0.2,
            'mine_min_gain': 1e-3,
            'mined_seed_file': input_data["output_dir"] / f"mined_seeds_{base_tag}_{MINE_VERSION}"
                                                          f"{'_neg' if negation else ''}{'_grp' if groups_on else ''}"
                                                          f"_cs{clause_slots}.json",
            'rule_copies': rule_copies,
            'free_rules_per_head': free_rules_per_head,
            'copy_input_share': copy_input_share,
            # Mask of the rule bodies: "sigmoid" (deterministic, init_log_alpha -11 so
            # every input gets a gradient; its gates learn at gate_lr) or
            # "hard_concrete" (the stochastic L0 mask; use init_log_alpha -6).
            'mask_type': mask_type,
            'init_log_alpha': init_log_alpha,
            'gate_lr': 0.05,
            'gate_eps': 1e-20,
            # Training: batches of target frames sampled from a few videos at a time.
            'num_epochs': input_data['num_epochs'],
            'train_fraction': train_fraction,
            'train_subset_seed': train_subset_seed,
            'epoch_repeat': epoch_repeat,
            'lr': rule_lr,
            # Chunks of frames_per_video target frames per gradient step (default 4; larger: better GPU use, fewer
            # and larger steps per epoch, a different optimisation: compare before changing the default).
            'videos_per_batch': videos_per_batch,
            'frames_per_video': 32,
            # ... per frames_per_video_ref frames (`frames_to_sample`): longer videos get more, at the same rate.
            'frames_per_video_ref': 200,
            # Share of sampled frames where some atom switches on (onsets are rare).
            'positive_frame_ratio': 0.5,
            'pos_weight': 20.0,
            'l0_lambda': l0_lambda,
            'l0_warmup_epochs': 10,
            # Early stopping: stop after patience epochs without a better val loss (num_epochs is
            # the maximum); the best val epoch after l0_warmup_epochs is kept. Medium 080f05 (grid + self gate): best val
            # epoch 10, val loss +43% by epoch 100 while train loss fell 4x.
            'patience': 10,
            # Rule extraction thresholds of the trained model.
            'mask_threshold': 0.5,
            'weight_threshold': 0.1,
            # A clause must hit its head in min_support training frames of min_videos videos.
            'min_support': 10,
            'min_videos': 2,
            # Cascade. Stage 1 (the model above) is the recall model: per head, the
            # threshold keeping cascade_recall of the val onset rows flags the
            # candidates (heads with < cascade_min_positives val onset rows use the
            # threshold of all heads). Stage 2, the verifier, is a rule net trained
            # only on those candidates (pos_weight verifier_pos_weight) to reject
            # the false ones; step 06 scores the cascade.
            'cascade': True,
            'cascade_recall': 0.9,
            'cascade_min_positives': 20,
            'verifier_pos_weight': 5.0,
            # Black-box baseline: an OnsetMLP on the same inputs and targets, trained after the
            # rule net; hidden size None = the number of shared-rule nodes (anchor bodies x slots + free rules).
            # Fixed at 830 since the step 03 v2 anchors (own-agent clauses): about that count with them
            # (~236 bodies x 3 + 128), and no longer tied to the anchors or to --per_head_rules.
            'mlp_baseline': True,
            'mlp_hidden_size': 830,
            'mlp_dropout': 0.2,
            'mlp_weight_decay': 1e-4,
            'mlp_lr': mlp_lr,
            'mlp_activation': mlp_activation,
            'mlp_grad_clip': 1.0,
            # Halve the lr after mlp_lr_patience epochs without a better val loss; stop after
            # mlp_patience (num_epochs is the maximum); the best val epoch is kept.
            'mlp_lr_patience': 5,
            'mlp_patience': 20,
            'wandb': True,
            'wandb_project': 'cauvid-roadpp',
            'train_atom_file': input_data['train_atom_file'],
            'val_atom_file': input_data['val_atom_file'],
            'roles': roles,
            'geometry': geometry,
            'grid': grid,
            'motion': motion,
            # The motion thresholds fitted on the train split (`fit_motion`).
            'motion_file': input_data["output_dir"] / f"motion_spec{agent_motion.tag()}.json",
            'ego_motion': ego_motion,
            # The ego motion level cuts fitted on the train split (`fit_ego_motion`).
            'ego_motion_file': input_data["output_dir"] / f"ego_motion_spec{agent_ego_motion.tag()}.json",
            'ego_flow': ego_flow,
            'durations': durations,
            'duration_labels': dur_labels,
            'slot_ranges': slot_ranges,
            'predict': predict,
            # The raw flow samples per video (`ego_flow.ensure_cache`), shared by every exp and run.
            'ego_flow_cache_dir': input_data.get("ego_flow_cache_dir"),
            'ego_flow_workers': int(input_data.get("ego_flow_workers", 16)),
            'frame_dir': input_data.get("frame_dir"),
            'video_dir': input_data.get("video_dir"),
            'frame_rate': input_data.get("frame_rate", 10),
            'require_self': require_self,
            # The grid fitted on the train split (`fit_grid`): the corner coordinate quantiles.
            'grid_file': input_data["output_dir"] / f"grid_spec{grid_tag({'grid': grid})}.json",
            'train_box_file': input_data.get('train_box_file'),
            'val_box_file': input_data.get('val_box_file'),
            # The anchor clauses the rules are seeded from (`anchor_file_of`; the causal ones by default).
            'causal_anchor_clause_file': anchor_file,
            'anchor_source': anchor_source,
            'anchor_subset_seed': anchor_seed,
            "unique_atoms_file": input_data["unique_atoms_file"],
            'training_data_file': shared_data_file(input_data["output_dir"] / "training_data.pt",
                                                   {"data_cache_dir": input_data.get("data_cache_dir"),
                                                    "data_key": data_key}, "training"),
            'model_file': input_data["output_dir"] / f"temporal_rule_net_{tag}.pt",
            "extended_clause_file": input_data["output_dir"] / f"extended_clauses_{tag}.json",
            # Inhibitors "bodies -> not head" (negative head weights, `inhibitor_clauses`); v2: counted on the
            # agents of the head's class only (`class_possible`).
            "inhibitor_clause_file": input_data["output_dir"] / f"inhibitor_clauses_v2_{tag}.json",
            # kind "inhibitor": val suppression <= this (else "weak", or "correction" if >= 1).
            'max_inhibitor_suppression': 0.5,
            'min_inhibitor_fires': 50,
            'min_inhibitor_base': 0.01,
            'history_file': input_data["output_dir"] / f"training_history_{tag}.json",
            'mlp_model_file': input_data["output_dir"] / f"onset_mlp_{mlp_tag}.pt",
            'mlp_history_file': input_data["output_dir"] / f"onset_mlp_history_{mlp_tag}.json",
            'candidate_threshold_file': input_data["output_dir"] / f"cascade_thresholds_{tag}.pt",
            'verifier_model_file': input_data["output_dir"] / f"verifier_rule_net_{tag}.pt",
            'verifier_clause_file': input_data["output_dir"] / f"verifier_clauses_{tag}.json",
            'verifier_history_file': input_data["output_dir"] / f"verifier_history_{tag}.json",
            'figure_dir': input_data["output_dir"] / "figures",
    }


def main(input_data):
    print(f"----------------- Step 05: Temporal Rule Learning -----------------")
    cfg = get_cfg(input_data)

    # 1. Prepare once: compressed atoms of each video of the dataset's train and
    #    val splits and the frames to sample (all / with onsets); redone when
    #    min_off_frames or the split changes. The model and clauses are redone after it.
    training_data = load_training_data(cfg)
    model_fresh = is_fresh(cfg["model_file"], cfg["training_data_file"])
    clauses_fresh = (model_fresh and is_fresh(cfg["extended_clause_file"], cfg["model_file"])
                     and is_fresh(cfg["inhibitor_clause_file"], cfg["model_file"]))

    # One W&B run for the training and extraction below, only if any is not cached.
    run = None
    if not clauses_fresh:
        run = _init_wandb(cfg, {"num_train_videos": len(training_data["train_videos"]),
                                "num_val_videos": len(training_data["val_videos"]),
                                "num_atoms": training_data["num_atoms"]},
                          job_type="step05_temporal_rule_net")
    try:
        # 2. Train the TemporalRuleNet; its input rows are built from the atoms per batch.
        if not model_fresh:
            model = train_temporal_rule_net(training_data, cfg, run)
            torch.save(model.state_dict(), cfg["model_file"])
        model = load_temporal_rule_net(training_data, cfg)

        # 3. Read the learned rules (ordered bodies with lags -> heads) out of the model.
        if not clauses_fresh:
            extended_clauses = extract_temporal_clauses(model, training_data, cfg, run,
                                                        inhibitor_file=cfg["inhibitor_clause_file"])
            utils_data.save_json(extended_clauses, cfg["extended_clause_file"])
    finally:
        if run is not None:
            run.finish()

    # 4. The black-box baseline on the same data, its own W&B run; step 06 evaluates it.
    if cfg["mlp_baseline"] and not is_fresh(cfg["mlp_model_file"], cfg["training_data_file"]):
        run = _init_wandb(cfg, {"num_atoms": training_data["num_atoms"]}, job_type="step05_onset_mlp")
        try:
            mlp = train_onset_mlp(training_data, cfg, run)
            torch.save(mlp.state_dict(), cfg["mlp_model_file"])
        finally:
            if run is not None:
                run.finish()

    # 5. Cascade: stage-1 candidate thresholds on val, then the verifier on the candidates, its own W&B run.
    if cfg["cascade"] and not cfg["main_only"]:
        run_cascade(model, training_data, cfg)

    # 6. Figures (one PDF each) and the terminal summary, on every run.
    extended_clauses = utils_data.load_json(cfg["extended_clause_file"])
    _plot_results(extended_clauses, cfg)
    _print_summary(extended_clauses, training_data, cfg)
    if cfg["cascade"] and not cfg["main_only"]:
        print_cascade_summary(training_data, cfg)


def run_cascade(model, training_data, cfg):
    """Stage-1 thresholds (`candidate_thresholds`), the verifier trained on the
    candidates (`train_verifier`) and its clauses, each redone when its inputs change."""
    if not is_fresh(cfg["candidate_threshold_file"], cfg["model_file"], cfg["training_data_file"]):
        torch.save(candidate_thresholds(model, training_data, cfg), cfg["candidate_threshold_file"])
    candidates = make_candidates(model, torch.load(cfg["candidate_threshold_file"])["thresholds"])

    verifier_fresh = is_fresh(cfg["verifier_model_file"], cfg["candidate_threshold_file"])
    clauses_fresh = verifier_fresh and is_fresh(cfg["verifier_clause_file"], cfg["verifier_model_file"])
    run = None
    if not clauses_fresh:
        run = _init_wandb(cfg, {"num_atoms": training_data["num_atoms"]}, job_type="step05_verifier")
    try:
        if not verifier_fresh:
            verifier = train_verifier(training_data, cfg, candidates, run)
            torch.save(verifier.state_dict(), cfg["verifier_model_file"])
        verifier = load_temporal_rule_net(training_data, cfg, cfg["verifier_model_file"])
        if not clauses_fresh:
            utils_data.save_json(extract_temporal_clauses(verifier, training_data, cfg, run, candidates),
                                 cfg["verifier_clause_file"])
    finally:
        if run is not None:
            run.finish()


def print_cascade_summary(training_data, cfg):
    stage1 = torch.load(cfg["candidate_threshold_file"])
    thresholds = stage1["thresholds"]
    print(f"----------------- Step 05 Cascade -----------------")
    print(f"[Stage 1]\t target recall {stage1['recall']:.2f}: val onset rows kept {stage1['val_row_recall']:.3f}, "
          f"candidates {stage1['val_candidate_share']:.4f} of the at-risk cells")
    print(f"[Stage 1]\t thresholds median {thresholds.median():.3g}, min {thresholds.min():.3g}, "
          f"max {thresholds.max():.3g}")
    _print_summary(utils_data.load_json(cfg["verifier_clause_file"]), training_data, cfg,
                   title="Step 05 Verifier (stage 2, counts over the candidates)",
                   clause_file=cfg["verifier_clause_file"])
