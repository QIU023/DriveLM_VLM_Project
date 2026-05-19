"""Verbatim port of UniAD's planning collision metric.

Source: https://github.com/OpenDriveLab/UniAD
  projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py
  (lines 1-149, PlanningMetric class; full file)

  projects/mmdet3d_plugin/datasets/pipelines/occflow_label.py
  (lines 1-190, GenerateOccFlowLabels.__call__ + reframe_boxes; the BEV
  segmentation builder that produces the grids consumed by PlanningMetric.)

  projects/mmdet3d_plugin/uniad/dense_heads/occ_head_plugin/utils.py
  (lines 1-42, gen_dx_bx + calculate_birds_eye_view_parameters; BEV
  parameterization constants.)

  projects/mmdet3d_plugin/uniad/apis/test.py
  (lines 90-103; call-site that wires PlanningMetric to the segmentation grid)

Adaptation only on I/O glue:
  - We do not have an mmdet3d LiDARInstance3DBoxes wrapper; we replace
    `boxes.corners[:, [0,3,7,4], :2]` with a numpy `_box_corners_lwhy()`
    helper that produces the four ground corners from (x, y, l, w, yaw).
  - We do not have the `reframe_boxes` rotate/translate API; we replace it
    with a numpy round-trip "future-LiDAR -> global -> current-ego" identical
    in semantics to UniAD's compound transform (lidar2ego @ ego2global @
    inv(ego2global_cur) @ inv(lidar2ego_cur)) — see `_reframe_lidar_to_curego`.
    UniAD uses the SAME compound (see reframe_boxes lines 56-72).
  - PyTorch tensors are replaced with numpy throughout (offline eval, CPU OK).

Hard rules: the math here is byte-equivalent to UniAD's. Constants, grid
bounds, vehicle filter, visibility filter, +0.5 m forward shift, axis-aligned
ego footprint, [-1, 1] x-flip, [0,1]<->[1,0] swap all preserved.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from skimage.draw import polygon


# ---------------------------------------------------------------------------
# UniAD constants (verbatim from PlanningMetric.__init__ lines 22-33 of
# planning_metrics.py)
# ---------------------------------------------------------------------------

# gen_dx_bx([-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0])
# UniAD only keeps the first two dims (dx[:2], bx[:2]).
DX = np.array([0.5, 0.5], dtype=np.float64)
BX = np.array([-50.0 + 0.5 / 2.0, -50.0 + 0.5 / 2.0], dtype=np.float64)  # = [-49.75, -49.75]
BEV_DIMENSION = np.array([200, 200], dtype=np.int64)

# Ego footprint — Renault Zoe per UniAD/VAD/ST-P3
W = 1.85
H = 4.084

# ---------------------------------------------------------------------------
# UniAD vehicle/visibility filter (verbatim from GenerateOccFlowLabels.__init__
# lines 28-43 of occflow_label.py — only_vehicle=True is the default in all
# UniAD configs; filter_invisible=True ditto).
# ---------------------------------------------------------------------------

NUSC_CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
VEHICLE_CLASSES = ['car', 'bus', 'construction_vehicle',
                   'bicycle', 'motorcycle', 'truck', 'trailer']
VEHICLE_CLASS_SET = set(VEHICLE_CLASSES)


# ---------------------------------------------------------------------------
# Box corners (replacement for LiDARInstance3DBoxes.corners[:, [0,3,7,4], :2])
# Returns 4 ground corners in (x, y) in same coord frame as (cx, cy, yaw).
# ---------------------------------------------------------------------------

def _box_corners_lwhy(cx: float, cy: float, length: float, width: float, yaw: float
                      ) -> np.ndarray:
    """Mirrors mmdet3d LiDARInstance3DBoxes.corners selection. Returns
    (4, 2) ground corners in same frame as cx, cy."""
    hl, hw = length * 0.5, width * 0.5
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.array([[+hl, +hw], [+hl, -hw], [-hl, -hw], [-hl, +hw]],
                   dtype=np.float64)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts @ R.T) + np.array([cx, cy], dtype=np.float64)


# ---------------------------------------------------------------------------
# Reframe agent box from future-LiDAR frame to current-ego frame.
# Mirrors GenerateOccFlowLabels.reframe_boxes (occflow_label.py lines 45-74).
# The order of operations is identical: lidar2ego @ ego2global at the curr
# (future) frame, then inverse ego2global @ inverse lidar2ego at the init
# (reference) frame.
# ---------------------------------------------------------------------------

def _quat_to_R(q_wxyz) -> np.ndarray:
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w),       s * (x * z + y * w)],
            [s * (x * y + z * w),       1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w),       s * (y * z + x * w),       1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _heading(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def _reframe_lidar_to_curego(boxes_lidar_fut: np.ndarray,
                             fut_info: dict, cur_info: dict
                             ) -> List[Tuple[float, float, float, float, float]]:
    """Mirrors GenerateOccFlowLabels.reframe_boxes lines 56-72:

        boxes.rotate(l2e_r_curr.T);   boxes.translate(l2e_t_curr)
        boxes.rotate(e2g_r_curr.T);   boxes.translate(e2g_t_curr)
        boxes.translate(-e2g_t_init); boxes.rotate(inv(e2g_r_init).T)
        boxes.translate(-l2e_t_init); boxes.rotate(inv(l2e_r_init).T)

    boxes is in the LiDAR frame of the FUTURE keyframe; output is in the
    LiDAR frame of the CURRENT (=reference) keyframe. Yaw is composed via
    the same chain. Returns list of (x, y, l, w, yaw_in_cur_lidar)."""
    if boxes_lidar_fut.size == 0:
        return []
    R_le_f = _quat_to_R(fut_info["lidar2ego_rotation"])
    t_le_f = np.asarray(fut_info["lidar2ego_translation"], dtype=np.float64)
    R_e2g_f = _quat_to_R(fut_info["ego2global_rotation"])
    t_e2g_f = np.asarray(fut_info["ego2global_translation"], dtype=np.float64)
    R_le_c = _quat_to_R(cur_info["lidar2ego_rotation"])
    t_le_c = np.asarray(cur_info["lidar2ego_translation"], dtype=np.float64)
    R_e2g_c = _quat_to_R(cur_info["ego2global_rotation"])
    t_e2g_c = np.asarray(cur_info["ego2global_translation"], dtype=np.float64)

    # Compound rotation: future-LiDAR -> current-LiDAR
    R_compound = R_le_c.T @ R_e2g_c.T @ R_e2g_f @ R_le_f
    yaw_offset = _heading(R_compound)

    out: List[Tuple[float, float, float, float, float]] = []
    for b in boxes_lidar_fut:
        # gt_boxes columns: (x, y, z, w, l, h, yaw) per our pkl (verified
        # 2026-05-19 — converter writes dims[:, [1,0,2]] but the version of
        # nuscenes-devkit used to build OUR pkl emits (w, l, h) directly).
        x_l, y_l, z_l = float(b[0]), float(b[1]), float(b[2])
        w_box, l_box = float(b[3]), float(b[4])
        yaw_box = float(b[6])
        p = np.array([x_l, y_l, z_l], dtype=np.float64)
        # future lidar -> future ego
        p = R_le_f @ p + t_le_f
        # future ego -> global
        p = R_e2g_f @ p + t_e2g_f
        # global -> current ego
        p = R_e2g_c.T @ (p - t_e2g_c)
        # current ego -> current lidar
        p = R_le_c.T @ (p - t_le_c)
        out.append((float(p[0]), float(p[1]), l_box, w_box,
                    yaw_box + yaw_offset))
    return out


# ---------------------------------------------------------------------------
# Build the BEV segmentation grid for a single future timestep.
# Mirrors GenerateOccFlowLabels.__call__ lines 99-160:
#   * only_vehicle filter (line 130-136)
#   * filter_invisible: visibility token != 1 (line 138-143)
#   * box corners -> grid indices via (bx, dx) (line 147-150)
#   * cv2.fillPoly(segmentation, [poly_region], 1.0) (line 157)
# We use skimage.draw.polygon instead of cv2 — same semantics.
# ---------------------------------------------------------------------------

def _build_seg_one_frame(fut_info: dict, cur_info: dict,
                         only_vehicle: bool = True,
                         filter_invisible: bool = True) -> np.ndarray:
    """Returns (200, 200) uint8 segmentation. Cells set to 1 where any
    filtered agent box lies; rasterized at future-frame keyframe annotations,
    reframed into current-ego (=reference) LiDAR frame."""
    seg = np.zeros((BEV_DIMENSION[1], BEV_DIMENSION[0]), dtype=np.uint8)
    boxes = np.asarray(fut_info["gt_boxes"], dtype=np.float64)
    if boxes.size == 0:
        return seg
    names = np.asarray(fut_info["gt_names"])
    vis = np.asarray(fut_info["visibility_tokens"])
    keep = np.ones(len(boxes), dtype=bool)
    if only_vehicle:
        keep &= np.array([n in VEHICLE_CLASS_SET for n in names], dtype=bool)
    if filter_invisible:
        keep &= (vis != 1)
    if not keep.any():
        return seg
    boxes = boxes[keep]
    reframed = _reframe_lidar_to_curego(boxes, fut_info, cur_info)
    for (cx, cy, l, w, yaw) in reframed:
        corners = _box_corners_lwhy(cx, cy, l, w, yaw)  # (4, 2) in (x, y)
        # UniAD: bbox_corners (already (x,y) in BEV plane). Map to grid indices.
        #   poly = round((corners - bx[:2] + dx[:2]/2) / dx[:2])
        poly = np.round((corners - BX + DX / 2.0) / DX).astype(np.int32)
        # UniAD passes [poly_region] to cv2.fillPoly with poly shape (4, 2)
        # where col 0 is x_index (col), col 1 is y_index (row). cv2.fillPoly
        # treats it as (x, y) image coords. skimage.draw.polygon takes (rows,
        # cols), so we pass (poly[:, 1], poly[:, 0]).
        rr, cc = polygon(poly[:, 1], poly[:, 0],
                         shape=(BEV_DIMENSION[1], BEV_DIMENSION[0]))
        seg[rr, cc] = 1
    return seg


# ---------------------------------------------------------------------------
# PlanningMetric.evaluate_single_coll — VERBATIM (numpy port, semantics
# identical to planning_metrics.py lines 43-82).
# ---------------------------------------------------------------------------

def evaluate_single_coll(traj: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    """traj: (n_future, 2) — after the same flip+swap that UniAD applies
    in evaluate_coll.

    segmentation: (n_future, 200, 200) uint8.

    Returns: (n_future,) bool of collisions."""
    pts = np.array([
        [-H / 2. + 0.5,  W / 2.],
        [ H / 2. + 0.5,  W / 2.],
        [ H / 2. + 0.5, -W / 2.],
        [-H / 2. + 0.5, -W / 2.],
    ])
    pts = (pts - BX) / DX
    pts[:, [0, 1]] = pts[:, [1, 0]]
    rr, cc = polygon(pts[:, 1], pts[:, 0])
    rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

    n_future = traj.shape[0]
    trajs = traj.reshape(n_future, 1, 2).copy()
    trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]  # mirrors UniAD line 62
    trajs = trajs / DX
    trajs = trajs + rc  # (n_future, K, 2)

    r = trajs[:, :, 0].astype(np.int32)
    r = np.clip(r, 0, BEV_DIMENSION[0] - 1)
    c = trajs[:, :, 1].astype(np.int32)
    c = np.clip(c, 0, BEV_DIMENSION[1] - 1)

    collision = np.full(n_future, False)
    for t in range(n_future):
        rr_t = r[t]
        cc_t = c[t]
        I = np.logical_and(
            np.logical_and(rr_t >= 0, rr_t < BEV_DIMENSION[0]),
            np.logical_and(cc_t >= 0, cc_t < BEV_DIMENSION[1]),
        )
        collision[t] = bool(np.any(segmentation[t, rr_t[I], cc_t[I]]))
    return collision


# ---------------------------------------------------------------------------
# PlanningMetric.evaluate_coll — VERBATIM (numpy port, planning_metrics.py
# lines 84-118). Inputs in standard ego frame (+x forward, +y left); this
# function applies the [-1, 1] flip internally just like UniAD.
# Returns (obj_box_coll_sum_per_step) where each step is 1 if collision.
# We only need `obj_box_coll` because we don't have the dense BEV
# occupancy-network prediction `obj_col` requires.
# ---------------------------------------------------------------------------

def evaluate_coll(trajs: np.ndarray, gt_trajs: np.ndarray,
                  segmentation: np.ndarray) -> np.ndarray:
    """trajs: (n_future, 2) predicted ego waypoints in +x forward, +y left.
    gt_trajs: (n_future, 2) same convention.
    segmentation: (n_future, 200, 200) uint8.

    Returns: (n_future,) bool — collision per step (with the gt_box_coll
    subtraction applied per UniAD line 114). Steps where the GT trajectory
    itself collides are reported as 0 (not counted), to filter degenerate
    annotations — this is what UniAD does."""
    n_future = trajs.shape[0]
    trajs = trajs * np.array([-1.0, 1.0])      # line 91
    gt_trajs = gt_trajs * np.array([-1.0, 1.0])

    gt_box_coll = evaluate_single_coll(gt_trajs, segmentation)
    box_coll = evaluate_single_coll(trajs, segmentation)
    # UniAD line 114: m2 = NOT gt_box_coll; obj_box_coll_sum[m2] += box_coll[m2]
    # i.e. for steps where GT already collides, don't count pred.
    keep = np.logical_not(gt_box_coll)
    return box_coll & keep


# ---------------------------------------------------------------------------
# Top-level helper: per-sample collision rate at horizon indices.
# Mirrors what apis/test.py + PlanningMetric.update do for a single sample.
# ---------------------------------------------------------------------------

def _ego_disp_to_lidar(disp_ego_xy: np.ndarray, cur_info: dict) -> np.ndarray:
    """Convert a (N, 2) displacement vector (Δx, Δy) from current EGO frame
    (+x forward, +y left) into the current LiDAR frame, which is the frame
    UniAD's PlanningMetric operates in (sdc_planning is built by
    `traj_api.get_sdc_planning_label`, ending in current lidar frame —
    trajectory_api.py lines 256-258). Displacements transform by R_le.T only
    (no translation; we use the 2x2 z-restricted rotation)."""
    R_le = _quat_to_R(cur_info["lidar2ego_rotation"])
    R = R_le[:2, :2].T  # ego -> lidar (rotation, 2D)
    return disp_ego_xy @ R.T  # equivalent to (R @ p.T).T


def compute_collision_per_sample(pred_wp_ego: np.ndarray,
                                 gt_wp_ego: np.ndarray,
                                 future_infos: List[dict],
                                 cur_info: dict,
                                 horizon_indices: Tuple[int, ...]
                                 ) -> List[int]:
    """pred_wp_ego, gt_wp_ego: (n_future, 2) in EGO frame (+x forward,
    +y left). future_infos: list of pkl info dicts at future timesteps
    (each has gt_boxes, gt_names, visibility_tokens, lidar2ego_*,
    ego2global_*). cur_info: pkl info at current keyframe.

    The trajectories are converted to current-LiDAR frame for the BEV
    metric (UniAD's PlanningMetric operates in lidar frame — see comments
    in `_ego_disp_to_lidar`).

    Returns list of 0/1 collision flags, one per requested horizon index."""
    n_future = pred_wp_ego.shape[0]
    n_avail = min(n_future, len(future_infos))
    pred_lid = _ego_disp_to_lidar(pred_wp_ego, cur_info)
    gt_lid = _ego_disp_to_lidar(gt_wp_ego, cur_info)
    # Build segmentation for available future frames (boxes already in
    # current lidar frame via _reframe_lidar_to_curego).
    seg = np.zeros((n_future, BEV_DIMENSION[1], BEV_DIMENSION[0]), dtype=np.uint8)
    for t in range(n_avail):
        seg[t] = _build_seg_one_frame(future_infos[t], cur_info)
    per_step = evaluate_coll(pred_lid, gt_lid, seg)
    out: List[int] = []
    for h_idx in horizon_indices:
        if h_idx >= n_avail:
            out.append(0)
        else:
            out.append(int(bool(per_step[h_idx])))
    return out
