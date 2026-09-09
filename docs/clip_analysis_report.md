# Bharatiya Antariksh Station (BAS) - Video Analysis & Procedural Audit
### Video Source: `clip.mp4` | Duration: 28.26s | Frames: 673 | Resolution: 1920x1080

---

## 1. Executive Summary
This document provides the authoritative step-by-step procedural breakdown of the test captured in `clip.mp4`.
The test executes the **Box Object Extraction & Return Procedure** (`BAS-EXP-BOX-RETURN`), which validates the multi-agent system's ability to track container opening, object extraction, object return, and container sealing with strict deterministic debouncing and anomaly safety gating.

---

## 2. Deterministic Step-by-Step Process

| Step | State Name | Video Timestamp | Frame Window | Physical Action Observed | On-Screen & Voice Guidance | Safety Verification Gate |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **S0** | `IDLE` | `00:00 - 00:04` | Frames 0 - 95 | Operator seated in resting posture. Outer box flaps closed. | *"System initialized. Please open the box."* | System initialized, baseline 3D rack frame locked. |
| **S1** | `BOX_OPENED` | `00:04 - 00:08` | Frames 96 - 190 | Operator reaches forward and opens the container flaps to ~70° elevation. | *"Box opened. Next step: Please take out the object."* | Lid angle >= 35° sustained for >= 12 consecutive frames. |
| **S2** | `OBJECT_EXTRACTED` | `00:08 - 00:18` | Frames 191 - 428 | Operator reaches inside, grasps inner box, and lifts it completely outside the container. | *"Object extracted. Next step: Please return the object into the box."* | Object centroid translated outside container volume, debounced >= 12 frames. |
| **S3** | `OBJECT_RETURNED` | `00:18 - 00:25` | Frames 429 - 595 | Operator re-inserts the inner box back into the container cavity between foam guides. | *"Object returned. Next step: Please close the box."* | Object returned inside container boundary while lid remains open (>= 30°), debounced >= 12 frames. |
| **S4** | `COMPLETE` | `00:25 - 00:28` | Frames 596 - 673 | Operator folds flaps closed over container (lid angle drops < 20°). | *"Box closed. Experiment successfully completed."* | Lid closure sustained >= 12 frames. Experiment procedure finalized. |

---

## 3. Biomechanical & Hand-Object Interaction (HOI) Primitives

1. **Approach Phase (T=00:04, 00:10, 00:22)**: 3D hand keypoints transition within the 0.28m proximity threshold relative to the active object.
2. **Contact & Grasp (T=00:05, 00:12, 00:25)**: Metric hand-object Euclidean distance drops below 0.12m for >= 5 consecutive frames.
3. **Extraction & Translation (T=00:13 - 00:18)**: Metric distance offset exceeds container boundary, shifting `EntityState` from `DOCKED` -> `EXTRACTED`.
4. **Docking & Release (T=00:23 - 00:25)**: Object coordinates re-enter container bounding volume, shifting `EntityState` back to `DOCKED`.

---

## 4. Anomaly Gates & Safety Rule Verification

The test adheres strictly to nominal procedure:
* **No Premature Close (`ERROR_SKIP`)**: The operator did not attempt to close the container while the object was outside.
* **No Object Abandonment (`ERROR_SEQ`)**: The object was returned to its designated container rack before the lid was closed.
* **No Inactivity Timeout (`STALL_TIMEOUT`)**: Each state transition completed within the allowed 45-second window.

---

## 5. Local LLM Synthesis (Model: `qwen3.5:9b`)

```
Report generated directly from 8-agent blackboard telemetry.
```
