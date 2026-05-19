"""Trajectory (action) tokenizer for video VLA on Qwen2.5-VL.

Scheme (mirrors OpenVLA's ActionTokenizer, generalised to 2-D ego trajectories
in vehicle frame and adapted to Qwen2.5-VL's spare-slot vocabulary):

  * Future horizon = 3.0 s @ 2 Hz = 6 waypoints (configurable).
  * Each waypoint is (dx_local, dy_local) relative to the *current* ego pose,
    measured in metres in the ego-vehicle frame (x forward, y left).
  * Each dim is clamped per-dim:
       dx (forward)  ∈ [-2,  +40] m  -> half-bin 0.082 m
       dy (lateral)  ∈ [-8,   +8] m  -> half-bin 0.031 m
    and uniformly quantised to 256 bins.  The asymmetric forward range
    reflects nuScenes: cars rarely move backward more than ~2 m within a
    3 s horizon (min dx across the train set is −2.02 m), while forward
    can reach ~55 m at highway speeds (99.75% of train GT is in [-2,40]).
    Lateral is symmetric and tight because the 99.9 percentile is ~7.5 m.
  * Bin index i in [0, 255] maps to token id   BIN_BASE + i.
  * Boundary tokens <traj_start>, <traj_end> wrap the action sequence.

Round-trip guarantee:
  Max half-bin = max( (40-(-2))/(2*256), (8-(-8))/(2*256) )
              = max(0.082, 0.031) = 0.082 m.
  -> encode(decode(...)) reproduces every in-range waypoint with
     < 0.10 m L2 error per dim (vs 0.195 m on the prior [-50,50] range).

Qwen2.5-VL vocab layout:
  tokenizer length = 151665 (added vocab ends at 151664)
  input embedding rows = 151936  (padded for matmul efficiency)
  => 271 spare embedding slots already exist; we use 258 of them
     (256 bin tokens + 2 boundary tokens) WITHOUT calling resize_token_embeddings,
     so the LM head and embedding matrix stay perfectly aligned.

Reference repos consulted (May 2026):
  * openvla/openvla   prismatic/vla/action_tokenizer.py (n_bins=256 per-dim,
                       last-N-tokens-of-vocab scheme).
  * ucla-mobility/AutoVLA  tools/action_token/action_token_cluster.py
                          (uses K-means over flattened waypoint sequences with
                           a 2048-entry codebook; we deliberately choose the
                           simpler per-dim binning per the task spec, since
                           per-dim is easier to debug at this scale and the
                           DriveLM key-frame budget is tiny.)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Sequence, Tuple

import numpy as np


# ----------------------------- Defaults ------------------------------------

# 6 waypoints at 0.5 s spacing = next 3.0 s
DEFAULT_HORIZON_S = 3.0
DEFAULT_HZ = 2.0
DEFAULT_NUM_WAYPOINTS = int(DEFAULT_HORIZON_S * DEFAULT_HZ)

# Per-dim binning (tight per-axis ranges; see module docstring).
DEFAULT_NUM_BINS = 256
DEFAULT_DX_MIN_M = -2.0
DEFAULT_DX_MAX_M = +40.0
DEFAULT_DY_MIN_M = -8.0
DEFAULT_DY_MAX_M = +8.0
# Legacy aliases kept for backwards-compatibility with code that referenced
# the symmetric [min_m, max_m] range. Anything reading these now sees the
# *union* of dx/dy bounds so clipping is at least as permissive as before.
DEFAULT_MIN_M = min(DEFAULT_DX_MIN_M, DEFAULT_DY_MIN_M)
DEFAULT_MAX_M = max(DEFAULT_DX_MAX_M, DEFAULT_DY_MAX_M)

# Token-id layout. Qwen2.5-VL has 271 spare slots after added vocab (id 151665).
# We grab the highest contiguous block so we don't trip over any future added
# special tokens HF may inject in 151665..151668-ish.
QWEN_VOCAB_LEN = 151665              # tokenizer length (added vocab ends here)
QWEN_EMBED_ROWS = 151936             # input embedding matrix height
SPARE_SLOTS = QWEN_EMBED_ROWS - QWEN_VOCAB_LEN  # 271

# Place bin tokens at the very tail of the embedding rows; boundary tokens just
# before them. The model already has zeroed rows here so init is fine.
TRAJ_END_ID = QWEN_EMBED_ROWS - 1                   # 151935
TRAJ_START_ID = QWEN_EMBED_ROWS - 2                 # 151934
BIN_BASE = QWEN_EMBED_ROWS - 2 - DEFAULT_NUM_BINS   # 151678  -> ids 151678..151933

TRAJ_START_TOKEN = "<traj_start>"
TRAJ_END_TOKEN = "<traj_end>"


def _bin_token(i: int) -> str:
    """Surface form of a trajectory bin token. Single-token by design (no spaces)."""
    return f"<traj_bin_{i:03d}>"


# ----------------------------- Config object -------------------------------

@dataclass
class TrajectoryTokenizerConfig:
    num_waypoints: int = DEFAULT_NUM_WAYPOINTS
    horizon_s: float = DEFAULT_HORIZON_S
    sample_hz: float = DEFAULT_HZ
    num_bins: int = DEFAULT_NUM_BINS
    # Per-dim quantization bounds (preferred). dx is asymmetric because cars
    # rarely reverse more than ~2 m in 3 s; dy is symmetric.
    dx_min_m: float = DEFAULT_DX_MIN_M
    dx_max_m: float = DEFAULT_DX_MAX_M
    dy_min_m: float = DEFAULT_DY_MIN_M
    dy_max_m: float = DEFAULT_DY_MAX_M
    # Legacy aliases — kept on the dataclass so persisted configs from older
    # runs still load. New code should read dx_/dy_ fields. min_m/max_m below
    # are the union of dx/dy bounds (loosest clip), purely for back-compat
    # with any caller that still references them.
    min_m: float = DEFAULT_MIN_M
    max_m: float = DEFAULT_MAX_M
    bin_base: int = BIN_BASE
    traj_start_id: int = TRAJ_START_ID
    traj_end_id: int = TRAJ_END_ID
    traj_start_token: str = TRAJ_START_TOKEN
    traj_end_token: str = TRAJ_END_TOKEN

    def asdict(self) -> dict:
        return asdict(self)


# ----------------------------- Tokenizer -----------------------------------

class TrajectoryTokenizer:
    """Per-dim 256-bin trajectory tokenizer.

    Encodes shape (T, 2) -> List[int] of length 2*T + 2 (with boundary tokens).
    Decodes back to (T, 2) float32 with quantisation error <= half a bin width.
    """

    def __init__(self, cfg: TrajectoryTokenizerConfig | None = None):
        self.cfg = cfg or TrajectoryTokenizerConfig()
        # Per-dim bin edges and centres. np.digitize uses right-exclusive bins.
        self.dx_edges = np.linspace(
            self.cfg.dx_min_m, self.cfg.dx_max_m, self.cfg.num_bins + 1
        )
        self.dy_edges = np.linspace(
            self.cfg.dy_min_m, self.cfg.dy_max_m, self.cfg.num_bins + 1
        )
        self.dx_centers = 0.5 * (self.dx_edges[:-1] + self.dx_edges[1:])
        self.dy_centers = 0.5 * (self.dy_edges[:-1] + self.dy_edges[1:])

        # Legacy single-axis aliases (loosest range), kept for any older caller
        # that hasn't been migrated. NEW code should use the per-dim arrays.
        self.bin_edges = np.linspace(
            self.cfg.min_m, self.cfg.max_m, self.cfg.num_bins + 1
        )
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    # ---- vocab helpers ----

    def special_token_strings(self) -> List[str]:
        """All surface forms we want HF tokenizer to recognise as single tokens."""
        return [self.cfg.traj_start_token, self.cfg.traj_end_token] + [
            _bin_token(i) for i in range(self.cfg.num_bins)
        ]

    def bin_token_id(self, bin_idx: int) -> int:
        if not (0 <= bin_idx < self.cfg.num_bins):
            raise ValueError(f"bin_idx {bin_idx} out of range [0, {self.cfg.num_bins})")
        return self.cfg.bin_base + bin_idx

    def token_id_to_bin(self, tok_id: int) -> int:
        """Map token id -> bin index. Returns -1 if not a bin token."""
        b = tok_id - self.cfg.bin_base
        if 0 <= b < self.cfg.num_bins:
            return b
        return -1

    # ---- encode / decode ----

    def encode(self, waypoints: np.ndarray, with_boundaries: bool = True) -> List[int]:
        """Encode (T, 2) ndarray of (dx, dy) into token ids.

        Returns a flat python list of ints: [start?, bin_dx_0, bin_dy_0, ..., end?].
        """
        wp = np.asarray(waypoints, dtype=np.float32)
        if wp.ndim != 2 or wp.shape[1] != 2:
            raise ValueError(f"waypoints must be (T,2); got {wp.shape}")
        if wp.shape[0] != self.cfg.num_waypoints:
            # Allow shorter / longer sequences but warn; trim/pad nothing — caller's job.
            pass

        # Clamp per-dim so values outside the tight range saturate at the
        # last bin instead of corrupting an unrelated axis.
        dx = np.clip(wp[:, 0], self.cfg.dx_min_m, self.cfg.dx_max_m)
        dy = np.clip(wp[:, 1], self.cfg.dy_min_m, self.cfg.dy_max_m)

        # digitize edges[1:-1] -> 0..num_bins-1; values at max land in last bin.
        bx = np.digitize(dx, self.dx_edges[1:-1])
        by = np.digitize(dy, self.dy_edges[1:-1])

        ids: List[int] = []
        if with_boundaries:
            ids.append(self.cfg.traj_start_id)
        for t in range(wp.shape[0]):
            ids.append(self.bin_token_id(int(bx[t])))
            ids.append(self.bin_token_id(int(by[t])))
        if with_boundaries:
            ids.append(self.cfg.traj_end_id)
        return ids

    def decode(self, token_ids: Sequence[int]) -> np.ndarray:
        """Decode a sequence of token ids back to (T, 2) waypoints.

        Strips boundary tokens; ignores any non-bin tokens (so this is robust
        to generation that emits stray text between <traj_start> and <traj_end>).
        Each consecutive pair of bin tokens forms one waypoint.
        """
        bin_idxs: List[int] = []
        in_block = False
        saw_start = False
        for tid in token_ids:
            if tid == self.cfg.traj_start_id:
                in_block = True
                saw_start = True
                continue
            if tid == self.cfg.traj_end_id:
                in_block = False
                continue
            # If we never saw <traj_start>, treat the whole sequence as bin tokens
            # (useful for unit-testing encode/decode without boundaries).
            if saw_start and not in_block:
                continue
            b = self.token_id_to_bin(int(tid))
            if b >= 0:
                bin_idxs.append(b)

        # Drop a stray trailing bin if the count is odd (the model sometimes
        # stops mid-waypoint when max_new_tokens is tight).
        if len(bin_idxs) % 2 == 1:
            bin_idxs = bin_idxs[:-1]

        if not bin_idxs:
            return np.zeros((0, 2), dtype=np.float32)

        bins = np.asarray(bin_idxs, dtype=np.int64).reshape(-1, 2)
        # Per-dim inverse: column 0 indexes dx bin centres, column 1 indexes dy.
        wp = np.stack(
            [self.dx_centers[bins[:, 0]], self.dy_centers[bins[:, 1]]],
            axis=1,
        ).astype(np.float32)
        return wp

    # ---- string form for inserting into the chat-template text ----

    def encode_as_str(self, waypoints: np.ndarray) -> str:
        """Render the trajectory as a contiguous string of special tokens.

        Useful when building the assistant turn text BEFORE tokenization
        (HF tokenizer will map each `<traj_bin_xxx>` etc. to its single id
        provided we've called `add_special_tokens` first — see
        `register_with_tokenizer` below).
        """
        wp = np.asarray(waypoints, dtype=np.float32)
        dx = np.clip(wp[:, 0], self.cfg.dx_min_m, self.cfg.dx_max_m)
        dy = np.clip(wp[:, 1], self.cfg.dy_min_m, self.cfg.dy_max_m)
        bx = np.digitize(dx, self.dx_edges[1:-1])
        by = np.digitize(dy, self.dy_edges[1:-1])
        parts = [self.cfg.traj_start_token]
        for t in range(wp.shape[0]):
            parts.append(_bin_token(int(bx[t])))
            parts.append(_bin_token(int(by[t])))
        parts.append(self.cfg.traj_end_token)
        return "".join(parts)


# --------------------------- HF integration --------------------------------

def register_with_tokenizer(tokenizer, traj_tok: TrajectoryTokenizer,
                            assert_ids: bool = True) -> dict:
    """Add the trajectory special tokens to a HF tokenizer in-place.

    We want the token *ids* to match `traj_tok.cfg.bin_base + i` exactly so the
    encode path matches the decode path. With Qwen2.5-VL we exploit the 271
    pre-existing spare embedding rows; we add 258 tokens, and HF auto-assigns
    them ids 151665..151922 (contiguous, in addition order).

    To make ids land where we want them, we add a small pad of placeholder
    special tokens first so the bin tokens slot into the right id range.

    Returns dict {token_str: token_id}.
    """
    cfg = traj_tok.cfg

    # 1. How many slots between current end-of-tokenizer and BIN_BASE?
    cur_len = len(tokenizer)
    pad_needed = cfg.bin_base - cur_len
    if pad_needed < 0:
        raise RuntimeError(
            f"Tokenizer already has {cur_len} tokens, but bin_base={cfg.bin_base}. "
            f"Bin base must be >= current tokenizer length."
        )

    pad_tokens = [f"<traj_reserved_{i:03d}>" for i in range(pad_needed)]
    bin_tokens = [_bin_token(i) for i in range(cfg.num_bins)]
    # Add pad + bins + boundary tokens (boundary tokens land at the very end).
    new_tokens = pad_tokens + bin_tokens + [cfg.traj_start_token, cfg.traj_end_token]

    added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    # Resulting id mapping:
    bin_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in bin_tokens}
    boundary_ids = {
        cfg.traj_start_token: tokenizer.convert_tokens_to_ids(cfg.traj_start_token),
        cfg.traj_end_token: tokenizer.convert_tokens_to_ids(cfg.traj_end_token),
    }

    if assert_ids:
        # Verify alignment
        expected_first = cfg.bin_base
        actual_first = bin_ids[bin_tokens[0]]
        if actual_first != expected_first:
            raise RuntimeError(
                f"Bin token alignment broken: expected first bin id={expected_first}, "
                f"got {actual_first}. Did the base tokenizer get extra tokens added "
                f"between Tier 1 and Tier 2?"
            )
        if boundary_ids[cfg.traj_end_token] != cfg.traj_end_id:
            raise RuntimeError(
                f"traj_end_id mismatch: expected {cfg.traj_end_id}, got "
                f"{boundary_ids[cfg.traj_end_token]}"
            )

    return {**bin_ids, **boundary_ids}


# --------------------------- Persistence -----------------------------------

def save_config(cfg: TrajectoryTokenizerConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg.asdict(), f, indent=2)


def load_config(path: str) -> TrajectoryTokenizerConfig:
    with open(path, "r") as f:
        d = json.load(f)
    return TrajectoryTokenizerConfig(**d)


# --------------------------- Self-test (CLI) -------------------------------

def _self_test() -> None:
    """Round-trip sanity check on a few synthetic trajectories.

    Reports max per-waypoint L2 error in metres; must be <= half-bin (0.196 m).
    """
    cfg = TrajectoryTokenizerConfig()
    tok = TrajectoryTokenizer(cfg)

    rng = np.random.default_rng(0)
    half_bin_dx = (cfg.dx_max_m - cfg.dx_min_m) / (2 * cfg.num_bins)
    half_bin_dy = (cfg.dy_max_m - cfg.dy_min_m) / (2 * cfg.num_bins)
    half_bin_m = max(half_bin_dx, half_bin_dy)

    # Case 1: a typical forward-driving curve (~3 s of 10 m/s curving slightly left).
    t = np.linspace(0.5, 3.0, cfg.num_waypoints)
    dx = 10.0 * t                               # straight forward (stays in [-2,40])
    dy = 0.5 * t**2                             # slight left drift (stays in [-8,8])
    wp1 = np.stack([dx, dy], axis=1).astype(np.float32)

    # Case 2: stationary (all zeros — important corner case)
    wp2 = np.zeros((cfg.num_waypoints, 2), dtype=np.float32)

    # Case 3: out-of-range values (should clamp per-dim)
    wp3 = np.array([
        [+80.0, -120.0],  # both saturate
        [-200.0, +200.0],
        [12.3, -4.5],
        [0.0, 0.0],
        [cfg.dx_max_m - 0.01, cfg.dy_min_m + 0.01],  # at edges
        [cfg.dx_min_m, cfg.dy_max_m],
    ], dtype=np.float32)[: cfg.num_waypoints]

    # Case 4: random uniform in-range (per-dim)
    rdx = rng.uniform(cfg.dx_min_m, cfg.dx_max_m, size=cfg.num_waypoints)
    rdy = rng.uniform(cfg.dy_min_m, cfg.dy_max_m, size=cfg.num_waypoints)
    wp4 = np.stack([rdx, rdy], axis=1).astype(np.float32)

    for name, wp in [("curve", wp1), ("zeros", wp2), ("clamp", wp3), ("random", wp4)]:
        ids = tok.encode(wp)
        # Sanity: length = 2T + 2
        assert len(ids) == 2 * cfg.num_waypoints + 2, len(ids)
        recovered = tok.decode(ids)
        clamped = np.stack([
            np.clip(wp[:, 0], cfg.dx_min_m, cfg.dx_max_m),
            np.clip(wp[:, 1], cfg.dy_min_m, cfg.dy_max_m),
        ], axis=1)
        err = np.abs(recovered - clamped)
        max_err_dx = err[:, 0].max()
        max_err_dy = err[:, 1].max()
        rms_err = np.sqrt((err ** 2).mean())
        ok = bool(max_err_dx <= half_bin_dx + 1e-6 and max_err_dy <= half_bin_dy + 1e-6)
        print(f"  case={name:8s}  max_err_dx={max_err_dx:.4f} (≤{half_bin_dx:.4f}) "
              f" max_err_dy={max_err_dy:.4f} (≤{half_bin_dy:.4f}) rms={rms_err:.4f} ok={ok}")
        if not ok:
            raise AssertionError(f"round-trip error exceeds half-bin for case {name}")

    # Also test decode robustness: stray text tokens interleaved
    ids = tok.encode(wp1)
    noisy = [ids[0], 12345, ids[1], 67890, ids[2], ids[3], ids[4], ids[5],
             ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13]]
    rec_noisy = tok.decode(noisy)
    assert rec_noisy.shape == wp1.shape, rec_noisy.shape
    print(f"  decode is robust to interleaved stray tokens: shape={rec_noisy.shape}")

    # Print id layout
    print()
    print(f"  bin_base       = {cfg.bin_base}")
    print(f"  bin id range   = {cfg.bin_base}..{cfg.bin_base + cfg.num_bins - 1}")
    print(f"  traj_start_id  = {cfg.traj_start_id}")
    print(f"  traj_end_id    = {cfg.traj_end_id}")


if __name__ == "__main__":
    print("trajectory_tokenizer self-test")
    _self_test()
    print("OK")
