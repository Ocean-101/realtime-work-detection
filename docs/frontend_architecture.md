# Frontend UI Documentation: BAS Mission Control

This document provides a detailed overview of the graphical user interface (GUI) and frontend architecture for the Autonomous HAR & Digital Twin System. It maps directly to the expected project deliverables and explains how to modify or extend the UI.

## 1. Fulfillment of Expected Solution Requirements

The frontend UI is purpose-built to monitor and interact with the multi-agent backend. Here is how the UI maps to the expected solution:

| Expected Requirement | UI Implementation & Element |
| :--- | :--- |
| **Continuously process local video feeds** | The UI contains a primary **Video Hero Section** that displays a real-time MJPEG stream (`/stream`) of the bounding boxes, skeletal meshes, and 3D overlays processed by the backend. |
| **Suggest the next step to be performed** | Below the video stream is the **Action & Guidance Strip**. The `NEXT REQUIRED ACTION` card dynamically updates to provide the astronaut with context-aware, on-screen guidance. |
| **Voice-based alerts for skipped/out of sequence steps** | When an anomaly (e.g., `ERROR_SEQ`) is detected, the UI instantly highlights the `NEXT REQUIRED ACTION` card in flashing red, while the backend offline TTS engine generates the voice alert. |
| **Generate timestamped text file** | The UI displays the live **Procedural Step Checklist** in the right sidebar. As the backend writes to the compressed `.jsonl` telemetry file, the UI checklist syncs instantly to show step outcomes and completion status. |
| **Stream video to specific IP & store locally** | The UI itself acts as the viewer for the `HTTP/RTSP` network stream (port 8080). It fetches the stream and displays it directly in the browser via standard `<img src="/stream">` tags. |
| **Graphical User Interface for monitoring** | The entire `index.html` page serves as the unified Mission Control Console, combining the 2D/3D stream, real-time step validation, anomaly alerts, and experiment selection into one dashboard. |
| **Offline standalone system** | The UI is a zero-dependency, vanilla HTML/JS single-page application served locally via the python orchestrator. No internet connection is required. |

---

## 2. UI Layout & Component Breakdown

The `index.html` interface is organized into a highly functional CSS Grid layout:

### A. Top Navigation & Controls (`<header>`)
* **Brand Header:** Displays the system title.
* **Feed Selector Pills:** Buttons that trigger backend API calls to switch video sources.
  * *To Add a New Feed:* Add a new `<button class="pill-btn">` and update the JavaScript `selectFeed(type)` function to call the appropriate backend API endpoint.
* **Reset Test Button:** Fires a `GET /reset` command to the backend to reset the FSM.
* **Status Indicators:** Displays the active source and a live FPS counter fetched from telemetry.

### B. Main Hero Grid (Left Column)
* **Live Video Stream (`#stream-box`):** 
  * Displays the MJPEG stream from the `/stream` endpoint.
  * Uses a robust fallback polling mechanism (`/snapshot`) that activates if the primary stream connection drops.
* **Step Verdict Badge (`#step-verdict-badge`):** 
  * A dynamic pill above the video that flashes green (`NOMINAL`) or red (`ERROR`) based on the current step's validity.
* **Action & Guidance Strip (`.action-grid`):** 
  * **Current Action Card:** Displays what the system detects the user is currently doing (e.g., "HOLDING COMPONENT BOX").
  * **Next Action Card:** Displays instructions for the next required step (e.g., "Extract Red box").

### C. Sidebar Step Checklist (Right Column)
* **Checklist Container (`#step-checklist`):** 
  * A dynamic list of steps populated from the `PROTOCOLS` JavaScript dictionary.
  * *To Add a New Experiment Protocol:* Add a new dictionary entry to the `PROTOCOLS` object in the JavaScript block at the bottom of `index.html`. Define the step IDs, titles, and descriptions.
* **Dynamic Styling:** 
  * As telemetry arrives, steps are dynamically styled as `.completed` (Green checkmark), `.active` (Cyan glow), or default based on the `data.step` variable from the backend.

---

## 3. How to Extend & Modify the UI

The UI is driven by a high-frequency (25 Hz) polling loop. If you need to add new data visualizations, follow these steps:

### 1. Update the Backend Telemetry
Ensure the backend `src/core/shared_memory.py` is outputting your new data point in the `get_snapshot()` dictionary.

### 2. Add the UI Element in HTML
Add your new HTML element (e.g., a new div for "Distance to Object") in `index.html`. Give it a unique ID (e.g., `id="distance-readout"`).

### 3. Update the JavaScript Polling Loop
Locate the `updateTelemetry()` function inside `index.html`. Add logic to parse the new data and update the DOM element:
```javascript
async function updateTelemetry() {
  const resp = await fetch('/telemetry?t=' + Date.now());
  const data = await resp.json();
  
  // Your new UI update logic here:
  if (data.new_metric !== undefined) {
    document.getElementById('distance-readout').textContent = data.new_metric + " mm";
  }
}
```

### 4. Custom Styling (CSS)
Use the pre-defined CSS variables (e.g., `var(--accent-cyan)`, `var(--panel-bg)`) at the top of the `<style>` block to maintain visual consistency with the spaceflight dashboard theme.
