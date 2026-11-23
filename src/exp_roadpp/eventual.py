"""The "eventual" target (2026-11-23, the user: first see whether the model can predict *whether* an agent does
something, then whether it predicts *when*).

The window target (default) asks: does the atom switch on for the agent within the rule window (a few seconds)?
The eventual target asks: does it switch on at any later frame, before the agent leaves the scene? Example: a car
on the oncoming lane (Car:IncomLane) at a traffic light will enter the junction (Car:Jun), whether it waits 1 s
or 20 s at the red light.

Per row (t, agent n) and atom a:
  - eventual (B, A): an onset of a in t..T-1 (as `TemporalRuleNet.first_onsets`: on in f, off in the
    min_off_frames before, the agent visible in f - 1);
  - eventual_known (B, A): an onset was seen, or the agent is gone in the video's last frame (it left the scene
    without one). An agent still visible at the end without an onset is censored (unknown), not a negative;
  - first (B, A): the frames to that onset (the event's identity, t + first), else the window's horizon.

`install()` wraps `step05._video_rows`, so every user of the rows (`step05_mine._head_rows`, the miner, the rule
lab's counts and evaluation, the atom probe) sees the eventual target.
"""
import torch

from src.exp_roadpp import step05

_ORIGINAL = step05._video_rows
_CACHE = {}


def next_onsets(X, min_off_frames):
    """(T, N, A) long: the first frame f >= t with an onset of the atom for the agent; T if none."""
    T, N, A = X.shape
    Xi = X.int()
    cum = torch.cat([torch.zeros((1, N, A), dtype=torch.int32, device=X.device), Xi.cumsum(dim=0)], dim=0)
    f = torch.arange(T, device=X.device)
    lo = (f - min_off_frames).clamp(min=0)
    held_before = (cum[f] - cum[lo]) > 0                                     # on somewhere in f-m..f-1
    seen = torch.zeros((T, N), dtype=torch.bool, device=X.device)
    seen[1:] = X[:-1].any(dim=2)                                             # visible in f - 1
    onset = X & ~held_before & seen[..., None]
    onset[0] = False
    nxt = torch.full((T + 1, N, A), T, dtype=torch.long, device=X.device)
    for t in range(T - 1, -1, -1):
        nxt[t] = torch.where(onset[t], torch.full_like(nxt[t], t), nxt[t + 1])
    return nxt[:T]


def _video_info(training_data, video_id, device):
    key = (video_id, str(device))
    if key not in _CACHE:
        _CACHE.clear()
        X = step05._video_from_cells(training_data["videos"][video_id], training_data["num_atoms"], device)
        _CACHE[key] = (next_onsets(X, int(training_data.get("min_off_frames", 1))), X[-1].any(dim=1), X.shape[0])
    return _CACHE[key]


def _video_rows(model, training_data, video_id, frames, device, history=None):
    rows = _ORIGINAL(model, training_data, video_id, frames, device, history)
    if len(rows["rows"]) == 0:
        rows["eventual"] = rows["at_risk"].clone()
        rows["eventual_known"] = rows["at_risk"].clone()
        return rows
    nxt, visible_at_end, T = _video_info(training_data, video_id, device)
    t, n = rows["rows"].T
    f = nxt[t, n]                                                            # (B, A)
    ev = f < T
    rows["eventual"] = rows["at_risk"] & ev
    rows["eventual_known"] = ev | ~visible_at_end[n][:, None]
    rows["first"] = torch.where(ev, f - t[:, None], rows["first"])
    return rows


def install():
    """Switch every user of `step05._video_rows` to the eventual target (`step05_mine._head_rows` reads it)."""
    step05._video_rows = _video_rows
