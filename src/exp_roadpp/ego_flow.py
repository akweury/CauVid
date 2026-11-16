"""Ego motion from the camera image (sparse optical flow), defined also in an empty scene.

Why: the ego's own labels say nothing about the drift to the right before a right turn (it is
labelled AV-Mov), and `ego_motion.py` reads the ego's motion from the other agents' boxes, so it
is undefined without agents. The global image motion is there in every frame.

Per sampled frame t (every STEP frames), from the frame pair (t - 1, t) (causal: never a later frame):
  1. both frames grey, resized to WIDTH;
  2. up to MAX_CORNERS corners outside the GT boxes of the agents (moving objects, padded by
     BOX_PAD), tracked by pyramidal Lucas-Kanade;
  3. a similarity transform (scale, rotation, shift) fitted by RANSAC, the coordinates centred on
     the image centre; a fit with fewer than MIN_INLIERS inliers or an inlier share below
     MIN_INLIER_RATIO is dropped (NaN).
Raw signals (per second):
  yaw:   -shift_x / image width * fps; > 0 the scene moves left = the ego turns / drifts right;
  speed: (scale - 1) * fps, the expansion of the scene; ~ the forward speed (0 when standing);
  pitch: shift_y / image height * fps (bumps; reported, not used).
Denoising (`denoise`, causal, per signal, over the sampled series):
  1. spikes: a sample farther than HAMPEL_K robust sd (1.4826 MAD, at least FLOOR) from the
     median of the last HAMPEL_WIN samples is replaced by that median (Hampel filter);
  2. smoothing: the mean of the last SMOOTH_WIN despiked samples (NaN ignored).
Delay (synthetic check, STEP 2 at 10 fps): a gradual change passes the spike test and is delayed
~0.4 s by the trailing mean; an abrupt step larger than HAMPEL_K * FLOOR is first taken for a spike
(2 samples), ~0.8 s in all.
"""
import os

import numpy as np

STEP = 2                     # frames between sampled pairs (5 Hz at 10 fps)
WIDTH = 480                  # flow computed at this image width
MAX_CORNERS = 300
BOX_PAD = 0.1                # GT boxes padded by this share of their size before masking
MIN_INLIERS = 20
MIN_INLIER_RATIO = 0.4
RANSAC_PX = 1.0
HAMPEL_WIN = 5               # samples (1 s at STEP 2, 10 fps)
HAMPEL_K = 3.0
SMOOTH_WIN = 5               # samples
SIGNALS = ("yaw", "speed", "pitch")
FLOOR = {"yaw": 0.02, "speed": 0.01, "pitch": 0.02}   # minimum robust sd of the spike test
VERSION = "flow_v1"


def pair_motion(cv2, g0, g1, boxes=()):
    """(yaw, speed, pitch) per frame (multiply by fps for per second) of grey frames g0 -> g1,
    the agent boxes (normalized x1, y1, x2, y2) masked out; None if the fit is unreliable."""
    h, w = g0.shape
    mask = np.full((h, w), 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        px, py = BOX_PAD * (x2 - x1), BOX_PAD * (y2 - y1)
        a, b = int(max(0.0, x1 - px) * w), int(max(0.0, y1 - py) * h)
        c, d = int(np.ceil(min(1.0, x2 + px) * w)), int(np.ceil(min(1.0, y2 + py) * h))
        mask[b:d, a:c] = 0
    p0 = cv2.goodFeaturesToTrack(g0, MAX_CORNERS, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < MIN_INLIERS:
        return None
    p1, st, _err = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
    if p1 is None:
        return None
    ok = st.ravel() == 1
    a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
    if len(a) < MIN_INLIERS:
        return None
    c = np.array([w / 2, h / 2], np.float32)
    M, inliers = cv2.estimateAffinePartial2D(a - c, b - c, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    if M is None or inliers is None:
        return None
    n = int(inliers.sum())
    if n < MIN_INLIERS or n / len(a) < MIN_INLIER_RATIO:
        return None
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    return -float(M[0, 2]) / w, scale - 1.0, float(M[1, 2]) / h


def _grey(cv2, img):
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (WIDTH, int(round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)


def read_frames(cv2, video_id, frame_dir, video_dir, wanted, frame_offset=1):
    """Yield (t, grey image) for the sorted frame ids `wanted` (tensor frame t = image %05d.jpg t, or
    decoded video frame t - frame_offset); frames not wanted are skipped without decoding to memory."""
    wanted = sorted(set(int(t) for t in wanted))
    d = os.path.join(frame_dir, video_id) if frame_dir else None
    if d and os.path.isdir(d):
        for t in wanted:
            img = cv2.imread(os.path.join(d, f"{t:05d}.jpg"))
            if img is not None:
                yield t, _grey(cv2, img)
        return
    path = next((os.path.join(video_dir, video_id + e) for e in (".mp4", ".avi", ".mov")
                 if video_dir and os.path.exists(os.path.join(video_dir, video_id + e))), None)
    if path is None:
        return
    cap = cv2.VideoCapture(path)
    i, k = 0, 0
    try:
        while k < len(wanted):
            target = wanted[k] - frame_offset
            if target < i:
                k += 1
                continue
            if not cap.grab():
                break
            if i == target:
                ok, img = cap.retrieve()
                if ok:
                    yield wanted[k], _grey(cv2, img)
                k += 1
            i += 1
    finally:
        cap.release()


def raw_series(cv2, video_id, frame_dir, video_dir, sample_frames, boxes_at, fps):
    """{t: (yaw, speed, pitch) per second} of the sampled frames t (pair t - 1, t); NaN where the fit
    failed or a frame is missing. boxes_at(t) -> the agent boxes of frame t."""
    sample_frames = sorted(set(int(t) for t in sample_frames if t >= 1))
    need = sorted({t - 1 for t in sample_frames} | set(sample_frames))
    imgs, out = {}, {}
    pending = set(sample_frames)
    for t, g in read_frames(cv2, video_id, frame_dir, video_dir, need):
        imgs[t] = g
        if t in pending and t - 1 in imgs:
            m = pair_motion(cv2, imgs[t - 1], g, boxes_at(t - 1))
            out[t] = (np.nan,) * 3 if m is None else tuple(v * fps for v in m)
            pending.discard(t)
        for old in [s for s in imgs if s < t - 1]:
            del imgs[old]
    for t in pending:
        out[t] = (np.nan,) * 3
    return out


def denoise(x, floor):
    """Causal Hampel despiking then a trailing mean (module doc) of one sampled series (n,) (NaN: no value)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    clean = x.copy()
    for i in range(n):
        if np.isnan(x[i]):
            continue
        w = x[max(0, i - HAMPEL_WIN + 1):i + 1]
        w = w[~np.isnan(w)]
        if len(w) < 3:
            continue
        med = np.median(w)
        sd = max(1.4826 * np.median(np.abs(w - med)), floor)
        if abs(x[i] - med) > HAMPEL_K * sd:
            clean[i] = med
    out = np.full(n, np.nan)
    for i in range(n):
        w = clean[max(0, i - SMOOTH_WIN + 1):i + 1]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = w.mean()
    return out


def denoised_series(raw, step=STEP):
    """{signal: (times (n,), raw (n,), denoised (n,))} on the regular grid of `raw`'s sample times;
    grid points without a sample are NaN, so a gap longer than the windows restarts the filters."""
    if not raw:
        return {s: (np.zeros(0, int), np.zeros(0), np.zeros(0)) for s in SIGNALS}
    ts = sorted(raw)
    grid = np.arange(ts[0], ts[-1] + 1, step)
    out = {}
    for k, s in enumerate(SIGNALS):
        r = np.array([raw[t][k] if t in raw else np.nan for t in grid])
        out[s] = (grid, r, denoise(r, FLOOR[s]))
    return out


# -------------- Symbolic atoms (`step05`, --ego_flow) --------------
# Level cuts of the denoised signals (medium_s train + val, ego_flow_diag 2026-11-16):
#   yaw: "straight" = the control (straight AV-Mov) q10..q90 (-0.018, 0.024), rounded to +-0.025;
#        strong = the q5 / q95 of all samples (-0.149, 0.165), rounded to +-0.15 (lane changes peak
#        ~0.08, turns ~0.48).
#   speed: still < 0.02 (all-sample q10 0.004); the AV-Stop precursor (< control q10 0.062) falls in
#        slow; the control median 0.16 in cruise.
YAW_CUTS = (-0.15, -0.025, 0.025, 0.15)
SPEED_CUTS = (0.02, 0.08, 0.20)
YAW_ATOMS = ("yaw_left_strong", "yaw_left", "yaw_straight", "yaw_right", "yaw_right_strong")
SPEED_ATOMS = ("speed_still", "speed_slow", "speed_cruise", "speed_fast")
ATOMS = YAW_ATOMS + SPEED_ATOMS
DEBOUNCE = 2                 # samples a new level must hold before it replaces the held one (0.4 s)


def tag():
    return f"_ef{VERSION.replace('flow_', '')}y{len(YAW_ATOMS)}s{len(SPEED_ATOMS)}d{DEBOUNCE}"


def vocabulary():
    """Signatures of the ego flow atoms; `step05._atom_name` reads them as "av:yaw_right"."""
    return [["pred", "ego_flow", "ego_flow_label", a, "agent_class_label", "av"] for a in ATOMS]


def _levels(v, cuts):
    """(n, len(cuts) + 1) bool: the level of each value (NaN: none)."""
    ok = ~np.isnan(v)
    edges = [-np.inf, *cuts, np.inf]
    return np.stack([ok & (v >= lo) & (v < hi) for lo, hi in zip(edges[:-1], edges[1:])], axis=1)


def _debounce(L, n=DEBOUNCE):
    """(n_samples, K) one-hot levels -> debounced: the held level switches to a new one only after the new
    one held n samples in a row (no flicker at a cut, no gap between two levels); a sample without a
    level (NaN) clears it, and the first level after a gap also needs n samples."""
    out = np.zeros_like(L)
    cur, cand, run = None, None, 0
    for i in range(len(L)):
        k = int(L[i].argmax()) if L[i].any() else None
        if k is None:
            cur, cand, run = None, None, 0
        elif k == cur:
            cand, run = None, 0
        else:
            run = run + 1 if k == cand else 1
            cand = k
            if run >= n:
                cur, cand, run = k, None, 0
        if cur is not None:
            out[i, cur] = True
    return out


def flow_atoms(raw, T, step=STEP):
    """(T, len(ATOMS)) bool atoms of one video from its cached raw samples {"frames": (n,), "raw": (n, 3)}:
    denoised (`denoised_series`), cut into levels, debounced over the samples, and held from each
    sample frame t to the next one (frames t .. t + step - 1; a sample at t reads frames t - 1, t)."""
    out = np.zeros((T, len(ATOMS)), dtype=bool)
    frames = np.asarray(raw["frames"]).astype(int)
    if len(frames) == 0:
        return out
    ser = denoised_series({int(t): tuple(v) for t, v in zip(frames, np.asarray(raw["raw"]))}, step)
    grid = ser["yaw"][0]
    L = np.concatenate([_debounce(_levels(ser["yaw"][2], YAW_CUTS)),
                        _debounce(_levels(ser["speed"][2], SPEED_CUTS))], axis=1)
    for i, t in enumerate(grid):
        a, b = int(t), min(int(t) + step, T)
        if a < T:
            out[a:b] = L[i]
    return out


# -------------- Cache (raw samples per video, shared by the exps) --------------
def cache_file(cache_dir, video_id):
    return os.path.join(cache_dir, f"{video_id}.npz")


def _cache_job(job):
    import cv2
    cv2.setNumThreads(1)
    video_id, T, boxes_by_frame, frame_dir, video_dir, fps, out = job
    raw = raw_series(cv2, video_id, frame_dir, video_dir, range(STEP, T, STEP),
                     lambda t: boxes_by_frame.get(t, ()), fps)
    ts = sorted(raw)
    arr = np.array([raw[t] for t in ts], dtype=np.float32).reshape(-1, 3)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, frames=np.array(ts, dtype=np.int32), raw=arr)
    os.replace(tmp, out)
    return video_id, len(ts), int(np.isnan(arr[:, 0]).sum()) if len(arr) else 0


def ensure_cache(jobs, cache_dir, frame_dir, video_dir, fps, workers):
    """Compute the raw samples of the videos missing in cache_dir. jobs: [(video_id, T, {frame: [boxes]})],
    the boxes (normalized, the agents' GT boxes) masked out of the flow."""
    from multiprocessing import Pool
    from tqdm import tqdm
    os.makedirs(cache_dir, exist_ok=True)
    todo = [(v, T, b, str(frame_dir), str(video_dir), fps, cache_file(cache_dir, v)) for v, T, b in jobs
            if not os.path.exists(cache_file(cache_dir, v))]
    if not todo:
        return
    samples = failed = 0
    with Pool(max(1, workers)) as pool:
        for _v, n, nan in tqdm(pool.imap_unordered(_cache_job, todo), total=len(todo), desc="Ego flow (decode)"):
            samples, failed = samples + n, failed + nan
    print(f"[Ego flow]\t {len(todo)} videos computed into {cache_dir}: {samples} samples, "
          f"{failed / max(samples, 1):.1%} without a valid fit")


def load_cache(cache_dir, video_id):
    f = cache_file(cache_dir, video_id)
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return {"frames": z["frames"], "raw": z["raw"]}
