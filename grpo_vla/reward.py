"""GRPO reward function for the nuScenes planning VLA (5-dim composite).

This module implements the reward described in
``docs/f1_grpo_design.md`` Section 3, generalised from the original
3-dim (L2 + collision + progress) to a 5-dim composite per the
F.1 implementation task (2026-05-24):

    r_total = alpha * r_l2 + beta * r_smooth + gamma * r_collision
              + delta * r_intent + epsilon * r_speed

with weights:

    alpha   = 1.0  (L2 fidelity to GT waypoints)
    beta    = 0.1  (trajectory smoothness; second-difference penalty)
    gamma   = 2.0  (collision penalty: any bbox within 2 m of a waypoint)
    delta   = 0.5  (intent / direction alignment, cosine similarity)
    epsilon = 0.3  (average-speed match)

All component rewards live in roughly [-X, 0] except r_intent which is
a cosine similarity in [-1, +1]. The veRL trainer consumes ``r_total``
as a flat float scalar; the per-component breakdown is returned alongside
for logging / debugging.

The function is process-safe (pure numpy + a single TrajectoryTokenizer
decode); veRL spawns Ray workers and imports the reward module per worker,
so we keep dependency surface minimal (no torch, no HF processor).

Inputs
------
response_token_ids : list[int]
    The model's generated response token ids (post-prompt). May contain
    boundary tokens <traj_start>/<traj_end>; decoder is robust to noise.
gt_waypoints : np.ndarray, shape (6, 2), dtype float
    Ground-truth future waypoints in the *current* ego frame
    (x forward, y left), metres. Same convention as
    ``planning_dataset.PlanningDataset._compute_waypoints``.
bbox_3d_list : list[dict]
    Per-object dicts with keys {"cx","cy","cz","l","w","h","yaw","cls"}
    (and optional "vx","vy"), all in the *current* ego frame. The
    convenience helper ``parse_bbox_text`` in this module reconstructs
    these dicts from the bbox_text field of ``bbox_egostate_*.jsonl``.
ego_state : dict
    Required key: "speed_mps" (CAN-bus scalar, m/s, clamped >= 0).
    Optional: "ego_l", "ego_w" (vehicle box dims; default UniAD 4.084 x 1.85).

Output
------
dict with keys:
    r_total      : float  -- flat scalar that veRL consumes
    r_l2         : float
    r_smooth     : float
    r_collision  : float
    r_intent     : float
    r_speed      : float
    n_collisions : int    -- diagnostic
    pred_wp      : list   -- decoded predicted waypoints (for logging)
    malformed    : bool   -- True iff response decoded to wrong shape

Malformed responses (wrong waypoint count, no bin tokens, etc.) return
r_total = MALFORMED_REWARD (-2.0, per the design doc Section 3.2).
"""
from __future__ import annotations

import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

# Make scripts/ importable so we can reuse the canonical TrajectoryTokenizer
# rather than re-deriving bin layout (single source of truth).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from trajectory_tokenizer import (  # noqa: E402 -- after sys.path mutation
    TrajectoryTokenizer,
    TrajectoryTokenizerConfig,
)


# ---------------------------------------------------------------------------
# Weights (per F.1 design + 2026-05-24 task spec)
# ---------------------------------------------------------------------------
ALPHA_L2 = 1.0
BETA_SMOOTH = 0.1
GAMMA_COLLISION = 2.0
DELTA_INTENT = 0.5
EPSILON_SPEED = 0.3

# Fixed reward returned for malformed responses (per design doc Sec 3.2).
MALFORMED_REWARD = -2.0

# Default planning horizon and step time (3 s / 6 waypoints @ 2 Hz).
DEFAULT_NUM_WP = 6
DEFAULT_HORIZON_S = 3.0

# Collision threshold (m): if any bbox centre falls within this radius of any
# predicted waypoint, count as 1 collision. Matches the simple proximity check
# in the design doc footnote (NOT the full UniAD rectangle intersection — we
# defer that to the optional uniad_collision_check hook if available).
COLLISION_RADIUS_M = 2.0

# Default ego footprint (UniAD canonical) — kept here for forward-compat with a
# future rectangle-overlap variant. Not used in the L2 proximity variant.
DEFAULT_EGO_L = 4.084
DEFAULT_EGO_W = 1.85


# Module-level tokenizer (cheap to construct, but reused for many reward calls
# in a veRL rollout loop).
_TRAJ_TOK: Optional[TrajectoryTokenizer] = None


def _get_traj_tok() -> TrajectoryTokenizer:
    global _TRAJ_TOK
    if _TRAJ_TOK is None:
        _TRAJ_TOK = TrajectoryTokenizer(TrajectoryTokenizerConfig())
    return _TRAJ_TOK


# ---------------------------------------------------------------------------
# Public: bbox text parser (reconstructs (cx,cy,cz,l,w,h,yaw,vx,vy,cls) dicts
# from a "Detected objects in ego frame:\n- car at (...) ..." block).
# ---------------------------------------------------------------------------

_BBOX_LINE_RE = re.compile(
    r"^-\s*(?P<cls>\w+)\s+at\s*"
    r"\((?P<cx>-?\d+\.?\d*),\s*(?P<cy>-?\d+\.?\d*),\s*(?P<cz>-?\d+\.?\d*)\)\s*m,\s*"
    r"size\s*(?P<l>-?\d+\.?\d*)x(?P<w>-?\d+\.?\d*)x(?P<h>-?\d+\.?\d*),\s*"
    r"yaw\s*(?P<yaw>-?\d+\.?\d*)\s*rad"
    r"(?:,\s*vel\s*\((?P<vx>-?\d+\.?\d*),\s*(?P<vy>-?\d+\.?\d*)\)\s*m/s)?"
    r"\s*$"
)


def parse_bbox_text(bbox_text: str) -> List[dict]:
    """Parse a ``bbox_egostate_*.jsonl`` ``bbox_text`` block into structured
    bbox dicts.

    The serializer (``scripts/prep_bbox_egostate.serialize_bboxes``) formats
    each line as e.g.::

        - car at (-1.9, -5.7, -1.3) m, size 4.3x1.8x1.4, yaw -2.64 rad, vel (-0.0, 0.4) m/s

    All coords are already in the current ego frame (x forward, y left, z up).
    Returns [] for the "none" placeholder strings.
    """
    if not bbox_text or "none" in bbox_text.lower():
        return []
    out: List[dict] = []
    for raw in bbox_text.splitlines():
        line = raw.strip()
        if not line.startswith("-"):
            continue
        m = _BBOX_LINE_RE.match(line)
        if not m:
            # Be lenient — silently skip lines that the regex doesn't match
            # rather than raising. The reward must never crash a rollout.
            continue
        d = {
            "cls": m.group("cls"),
            "cx": float(m.group("cx")),
            "cy": float(m.group("cy")),
            "cz": float(m.group("cz")),
            "l": float(m.group("l")),
            "w": float(m.group("w")),
            "h": float(m.group("h")),
            "yaw": float(m.group("yaw")),
        }
        if m.group("vx") is not None:
            d["vx"] = float(m.group("vx"))
            d["vy"] = float(m.group("vy"))
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Per-component reward helpers (all return non-positive floats, larger = better)
# ---------------------------------------------------------------------------


def _r_l2(pred_wp: np.ndarray, gt_wp: np.ndarray) -> float:
    """Negative mean per-waypoint L2 distance (m)."""
    diff = pred_wp - gt_wp                       # (T, 2)
    l2 = np.linalg.norm(diff, axis=1)            # (T,)
    return float(-l2.mean())


def _r_smooth(pred_wp: np.ndarray) -> float:
    """Negative mean second-difference magnitude squared (m^2/step^2).

    second_diff[t] = pred[t+2] - 2 pred[t+1] + pred[t]   for t in 0..T-3
    Lower = smoother. Returns 0 if fewer than 3 waypoints.
    """
    if pred_wp.shape[0] < 3:
        return 0.0
    sd = pred_wp[2:] - 2.0 * pred_wp[1:-1] + pred_wp[:-2]   # (T-2, 2)
    pen = float(np.mean(np.sum(sd * sd, axis=1)))           # mean ||.||^2
    return -pen


def _r_collision(
    pred_wp: np.ndarray,
    bbox_3d_list: Sequence[dict],
    radius_m: float = COLLISION_RADIUS_M,
) -> tuple:
    """Negative count of bbox centres within ``radius_m`` of any predicted
    waypoint. Returns (reward, n_collisions)."""
    if not bbox_3d_list or pred_wp.shape[0] == 0:
        return 0.0, 0
    # Stack bbox xy centres -> (B, 2). We compare in ego-frame xy plane;
    # z is ignored (planning waypoints are 2-D, vehicles are on the road).
    centres = np.array([[b["cx"], b["cy"]] for b in bbox_3d_list], dtype=np.float32)
    # Pairwise distance (T, B)
    diff = pred_wp[:, None, :] - centres[None, :, :]   # (T, B, 2)
    dist = np.linalg.norm(diff, axis=2)                # (T, B)
    # For each bbox, check if it gets hit by ANY predicted waypoint.
    # The "count any bbox within 2m of any pred[t]" wording in the task
    # spec is naturally interpreted as "number of bboxes that come within
    # 2m of the trajectory" — that's bbox-unique, NOT per-(t, bbox) pair,
    # so a single bbox tracked across 6 frames still counts as 1 collision.
    hit_per_bbox = (dist <= radius_m).any(axis=0)      # (B,)
    n = int(hit_per_bbox.sum())
    return float(-n), n


def _r_intent(pred_wp: np.ndarray, gt_wp: np.ndarray) -> float:
    """Cosine similarity between the predicted and GT net-displacement
    vectors (last - first waypoint). Range: [-1, +1]. Returns 0 if either
    vector is degenerate (norm < 1e-6 m, i.e. stationary)."""
    if pred_wp.shape[0] < 2 or gt_wp.shape[0] < 2:
        return 0.0
    v_pred = pred_wp[-1] - pred_wp[0]
    v_gt = gt_wp[-1] - gt_wp[0]
    n_pred = float(np.linalg.norm(v_pred))
    n_gt = float(np.linalg.norm(v_gt))
    if n_pred < 1e-6 or n_gt < 1e-6:
        # Stationary case: define cosine as 0 (no penalty, no bonus).
        return 0.0
    return float(np.dot(v_pred, v_gt) / (n_pred * n_gt))


def _r_speed(
    pred_wp: np.ndarray,
    gt_wp: np.ndarray,
    horizon_s: float = DEFAULT_HORIZON_S,
) -> float:
    """Negative absolute average-speed difference (m/s).

    speed = norm(wp[-1]) / horizon_s   (matches the design doc's
    progress-shaping speed proxy — net displacement over horizon, NOT
    sum of inter-step distances. This rewards "ended up where GT ended up"
    rather than encouraging zig-zag speed-matching.)
    """
    if pred_wp.shape[0] == 0 or gt_wp.shape[0] == 0:
        return 0.0
    pred_speed = float(np.linalg.norm(pred_wp[-1])) / horizon_s
    gt_speed = float(np.linalg.norm(gt_wp[-1])) / horizon_s
    return float(-abs(pred_speed - gt_speed))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_reward(
    response_token_ids: List[int],
    gt_waypoints: np.ndarray,
    bbox_3d_list: List[dict],
    ego_state: dict,
    *,
    horizon_s: float = DEFAULT_HORIZON_S,
    weights: Optional[dict] = None,
    traj_tok: Optional[TrajectoryTokenizer] = None,
) -> Dict[str, float]:
    """5-dim composite GRPO reward.

    See module docstring for input/output schema.
    """
    w = {
        "alpha": ALPHA_L2,
        "beta": BETA_SMOOTH,
        "gamma": GAMMA_COLLISION,
        "delta": DELTA_INTENT,
        "epsilon": EPSILON_SPEED,
    }
    if weights:
        w.update({k: float(v) for k, v in weights.items()})

    gt_wp = np.asarray(gt_waypoints, dtype=np.float32)
    if gt_wp.ndim != 2 or gt_wp.shape[1] != 2:
        raise ValueError(
            f"gt_waypoints must be (T,2); got {gt_wp.shape}"
        )
    num_wp = int(gt_wp.shape[0])

    tok = traj_tok if traj_tok is not None else _get_traj_tok()
    pred_wp = tok.decode(list(response_token_ids))   # (T_pred, 2)

    # Malformed: decoder returned fewer waypoints than GT (typically 0 if
    # the response had no bin tokens, or T_pred < num_wp if generation
    # was truncated). We don't try to pad-and-score; a truncated response
    # is a real failure mode that the policy should learn to avoid.
    malformed = pred_wp.shape[0] != num_wp
    if malformed:
        return {
            "r_total": float(MALFORMED_REWARD),
            "r_l2": float(MALFORMED_REWARD),
            "r_smooth": 0.0,
            "r_collision": 0.0,
            "r_intent": 0.0,
            "r_speed": 0.0,
            "n_collisions": 0,
            "pred_wp": pred_wp.tolist(),
            "malformed": True,
        }

    r_l2 = _r_l2(pred_wp, gt_wp)
    r_smooth = _r_smooth(pred_wp)
    r_coll, n_coll = _r_collision(pred_wp, bbox_3d_list)
    r_intent = _r_intent(pred_wp, gt_wp)
    r_speed = _r_speed(pred_wp, gt_wp, horizon_s=horizon_s)

    r_total = (
        w["alpha"] * r_l2
        + w["beta"] * r_smooth
        + w["gamma"] * r_coll
        + w["delta"] * r_intent
        + w["epsilon"] * r_speed
    )

    return {
        "r_total": float(r_total),
        "r_l2": float(r_l2),
        "r_smooth": float(r_smooth),
        "r_collision": float(r_coll),
        "r_intent": float(r_intent),
        "r_speed": float(r_speed),
        "n_collisions": int(n_coll),
        "pred_wp": pred_wp.tolist(),
        "malformed": False,
    }


# ---------------------------------------------------------------------------
# veRL-style scalar entry point
# ---------------------------------------------------------------------------


def compute_reward_scalar(
    response_token_ids: List[int],
    extra_info: dict,
    *,
    weights: Optional[dict] = None,
) -> float:
    """Convenience wrapper for veRL's ``compute_score`` interface.

    veRL's custom_reward callback signature is roughly::

        def compute_score(data_source, solution_str, ground_truth, extra_info) -> float

    Our adapter packs everything the reward needs into ``extra_info``:

        extra_info = {
            "gt_waypoints": list-of-list float (T, 2),
            "bbox_3d_list": list-of-dict (see parse_bbox_text),
            "ego_state":    {"speed_mps": float, ...},
            "horizon_s":    float (optional, default 3.0),
        }

    The scalar return is exactly ``compute_reward(...)["r_total"]``.
    """
    gt = np.asarray(extra_info["gt_waypoints"], dtype=np.float32)
    bboxes = extra_info.get("bbox_3d_list", [])
    ego = extra_info.get("ego_state", {})
    horizon_s = float(extra_info.get("horizon_s", DEFAULT_HORIZON_S))
    out = compute_reward(
        response_token_ids,
        gt,
        bboxes,
        ego,
        horizon_s=horizon_s,
        weights=weights,
    )
    return float(out["r_total"])


# ---------------------------------------------------------------------------
# veRL naive-reward-manager entry point
#
# verl/workers/reward_manager/naive.py calls
#   score = self.compute_score(data_source=..., solution_str=<response_text>,
#                              ground_truth=..., extra_info=...)
# i.e. it hands us the DECODED response TEXT, not token ids.  We re-tokenize
# with the canonical TrajectoryTokenizer so that the reward sees the SAME
# bin tokens it was trained on (the response text may include the literal
# "<traj_xxx>" tokens or the assistant boilerplate; the tokenizer handles
# both cases via its decode() being tolerant to noise).
#
# Glue added by Agent D (overnight 2026-05-23 → 24).  Wired in
# configs/grpo_b5prime_3cam.yaml as custom_reward_function.name=planning_reward.
# ---------------------------------------------------------------------------


_TRAJ_BIN_RE = re.compile(r"<traj_bin_(\d+)>")
_TRAJ_START_RE = re.compile(r"<traj_start>")
_TRAJ_END_RE = re.compile(r"<traj_end>")


def _tokenize_solution_str(solution_str: str) -> List[int]:
    """Convert a response text emitted by the policy back to the
    bin-token id stream expected by TrajectoryTokenizer.decode().

    Surface forms (registered as HF special tokens in B.5'):
       <traj_start>            -> TRAJ_START_ID
       <traj_bin_xxx>          -> BIN_BASE + int(xxx)
       <traj_end>              -> TRAJ_END_ID

    We scan left-to-right and emit ids in order. The decoder is robust to
    missing/extra non-bin tokens. If the response has zero bin markers we
    return [] so decode() yields shape (0,2) -> reward marks malformed."""
    s = solution_str or ""
    tok = _get_traj_tok()
    # Scan tokens in source order
    ids: List[int] = []
    pos = 0
    while pos < len(s):
        m_start = _TRAJ_START_RE.match(s, pos)
        m_end = _TRAJ_END_RE.match(s, pos)
        m_bin = _TRAJ_BIN_RE.match(s, pos)
        if m_start:
            ids.append(tok.cfg.traj_start_id)
            pos = m_start.end()
        elif m_end:
            ids.append(tok.cfg.traj_end_id)
            pos = m_end.end()
        elif m_bin:
            try:
                ids.append(tok.bin_token_id(int(m_bin.group(1))))
            except ValueError:
                pass
            pos = m_bin.end()
        else:
            pos += 1
    return ids


def planning_reward(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth=None,
    extra_info: dict | None = None,
    **_unused,
) -> float:
    """veRL naive-reward-manager entry point.

    Args mirror `verl/workers/reward_manager/naive.py`. We:
      1. Re-tokenize `solution_str` into bin-token ids.
      2. Pull `gt_waypoints`, `bbox_3d_list`, `ego_state`, `horizon_s` from
         `extra_info` (packed by dataset_adapter.VeRLNuScenesDataset).
      3. Return the scalar `r_total` from `compute_reward`.

    On any input shape error we return `MALFORMED_REWARD` (-2.0) rather
    than raising — a single bad sample must not crash a rollout batch.
    """
    if extra_info is None:
        return float(MALFORMED_REWARD)
    try:
        gt = ground_truth if ground_truth is not None else extra_info.get("gt_waypoints")
        gt_arr = np.asarray(gt, dtype=np.float32)
        bboxes = extra_info.get("bbox_3d_list", [])
        ego = extra_info.get("ego_state", {"speed_mps": 0.0})
        horizon_s = float(extra_info.get("horizon_s", DEFAULT_HORIZON_S))
        resp_ids = _tokenize_solution_str(solution_str or "")
        out = compute_reward(
            resp_ids,
            gt_arr,
            bboxes,
            ego,
            horizon_s=horizon_s,
        )
        return float(out["r_total"])
    except Exception:
        return float(MALFORMED_REWARD)
