"""
BAS Autonomous HAR System - 3D Skeleton & Joint Visualizer
Upgraded from HAR-detection-system-main/visualize_joints.py:
Loads 3D joints from .npz / .json datasets and renders anatomical 3D bones,
joint nodes, and activity annotations in an interactive 3D space.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Standard COCO 17-Keypoint Bone Connections
COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Face
    (5, 6),                                # Shoulders
    (5, 7), (7, 9),                        # Left Arm (Shoulder -> Elbow -> Wrist)
    (6, 8), (8, 10),                       # Right Arm (Shoulder -> Elbow -> Wrist)
    (5, 11), (6, 12),                      # Torso (Shoulders to Hips)
    (11, 12),                              # Pelvis
    (11, 13), (13, 15),                    # Left Leg (Hip -> Knee -> Ankle)
    (12, 14), (14, 16)                     # Right Leg (Hip -> Knee -> Ankle)
]

JOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shldr", "r_shldr", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle"
]


def visualize_3d_joints(
    npz_path: str = "dataset/har_dataset/har_dataset_3d_joints.npz",
    frame_idx: int = 0,
    save_path: str = None,
    elevation: float = 20.0,
    azimuth: float = -60.0
):
    if not os.path.exists(npz_path):
        # Check fallback to HMR2 sample if har_dataset not yet created
        fallback_candidates = [
            "dataset/HAR-detection-system-main/output/hmr2_result.npz",
            "dataset/HAR-detection-system-main/HAR-detection-system-main/output/hmr2_result.npz"
        ]
        found_fallback = None
        for candidate in fallback_candidates:
            if os.path.exists(candidate):
                found_fallback = candidate
                break
        if found_fallback:
            print(f"[Visualizer] Specified file not found. Loading fallback: {found_fallback}")
            npz_path = found_fallback
        else:
            raise FileNotFoundError(f"NPZ dataset file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    joints_all = data["joints_3d"]
    activities = data["activities"] if "activities" in data else None
    timestamps = data["timestamps"] if "timestamps" in data else None

    total_frames = len(joints_all)
    frame_idx = max(0, min(frame_idx, total_frames - 1))
    joints = joints_all[frame_idx]

    activity_str = str(activities[frame_idx]) if activities is not None else "UNKNOWN"
    time_str = f"{timestamps[frame_idx]:.2f}s" if timestamps is not None else f"Frame {frame_idx}"

    print("=" * 65)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - 3D POSE SKELETON VIEWER")
    print(f"   Dataset File : {npz_path}")
    print(f"   Selected Frame: {frame_idx} / {total_frames} (Time: {time_str})")
    print(f"   Activity State: {activity_str}")
    print(f"   Joints Shape  : {joints.shape}")
    print("=" * 65)

    fig = plt.figure(figsize=(10, 9), facecolor="#0e1726")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0e1726")

    # Invert Y/Z appropriately for standard camera projection coordinate visualization
    xs = joints[:, 0]
    ys = joints[:, 2] # Depth as Y axis in 3D plot
    zs = -joints[:, 1] # Upwards as Z axis in 3D plot

    # Plot Bone Links
    num_joints = joints.shape[0]
    if num_joints == 17:
        for j1, j2 in COCO_BONES:
            if j1 < num_joints and j2 < num_joints:
                # Check that both joints have valid non-zero data
                if np.any(joints[j1]) and np.any(joints[j2]):
                    # Distinguish arms vs legs vs torso
                    if j1 in (5, 6, 7, 8, 9, 10) and j2 in (5, 6, 7, 8, 9, 10):
                        color = "#00f0ff" # Cyan arms
                        lw = 2.5
                    elif j1 in (11, 12, 13, 14, 15, 16) and j2 in (11, 12, 13, 14, 15, 16):
                        color = "#10b981" # Lime legs
                        lw = 2.5
                    elif (j1 in (5, 6) and j2 in (11, 12)) or (j1 == 11 and j2 == 12) or (j1 == 5 and j2 == 6):
                        color = "#f43f5e" # Coral torso
                        lw = 3.0
                    else:
                        color = "#a855f7" # Violet face
                        lw = 1.5

                    ax.plot(
                        [xs[j1], xs[j2]],
                        [ys[j1], ys[j2]],
                        [zs[j1], zs[j2]],
                        color=color,
                        linewidth=lw,
                        alpha=0.85
                    )

    # Plot Joint Nodes
    ax.scatter(xs, ys, zs, c="#38bdf8", s=60, edgecolors="white", depthshade=True, alpha=0.95)

    # Add Joint Labels for key anatomical nodes
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        if np.any(joints[i]):
            name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else str(i)
            # Only annotate key joints to prevent clutter
            if i in (0, 5, 6, 7, 8, 9, 10, 11, 12):
                ax.text(x, y, z + 0.03, f" {name}", color="#e2e8f0", fontsize=8, fontweight="bold")

    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_title(
        f"BAS Astronaut 3D Skeleton HAR\nActivity: [{activity_str}] | Timestamp: {time_str}",
        color="white",
        fontsize=12,
        fontweight="bold",
        pad=15
    )

    # Set dark-themed axes
    ax.set_xlabel("Lateral X (m)", color="#94a3b8", labelpad=8)
    ax.set_ylabel("Depth Z (m)", color="#94a3b8", labelpad=8)
    ax.set_zlabel("Height Y (m)", color="#94a3b8", labelpad=8)

    ax.tick_params(colors="#94a3b8")
    ax.grid(color="#334155", linestyle="--", linewidth=0.5)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#1e293b")
        pane.set_edgecolor("#334155")

    # Equalize aspect ratio
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min(), 0.5) / 2.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"[Visualizer] 3D Skeleton rendering saved to: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D Pose & Skeleton Joints")
    parser.add_argument("--npz", default="dataset/har_dataset/har_dataset_3d_joints.npz", help="Path to .npz file")
    parser.add_argument("--frame", type=int, default=100, help="Frame index to view")
    parser.add_argument("--save-img", default=None, help="Optional image path to save visualization")
    parser.add_argument("--elev", type=float, default=20.0, help="Elevation angle")
    parser.add_argument("--azim", type=float, default=-60.0, help="Azimuth angle")
    args = parser.parse_args()

    visualize_3d_joints(
        npz_path=args.npz,
        frame_idx=args.frame,
        save_path=args.save_img,
        elevation=args.elev,
        azimuth=args.azim
    )
