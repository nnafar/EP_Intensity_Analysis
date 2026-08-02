#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
======================================================
--- GUV CIRCLE SELECTOR ---
======================================================
Interactive tool for defining GUV starting positions.

Usage (standalone):
    python interactive_select.py

Usage (from run_analysis.py):
    from interactive_select import run_selector
    circles = run_selector(roi_frame, save_path)

Controls:
    Left-drag  →  Draw a new circle (centre = click point, edge = release)
    Right-click →  Remove the nearest circle
    Enter / Space  →  Confirm and save
    Escape  →  Cancel (returns empty list)

Output:
    JSON file at config.CIRCLES_JSON_PATH, e.g.:
    [
        {"id": 1, "x": 1048, "y": 2181, "r": 119},
        {"id": 2, "x":  450, "y":  780, "r":  62}
    ]
"""

import cv2
import numpy as np
import json
import os
import glob
import sys
from natsort import natsorted


# ── Display constants ───────────────────────────────────────────────────────
DISPLAY_MAX_PX  = 1200   # Longest edge of the display window (pixels)
WIN_TITLE       = "GUV Selector  |  Drag=Draw  RClick=Remove  Enter=Confirm  Esc=Cancel"
COLOR_CONFIRMED = (0, 220, 0)     # Green  – confirmed circles
COLOR_LIVE      = (0, 220, 255)   # Yellow – circle being drawn
COLOR_TEXT      = (255, 255, 255) # White  – labels
COLOR_INSTRUCT  = (0, 255, 255)   # Cyan   – instructions

INSTRUCTIONS = [
    "Left-click 1: select first edge",
    "Left-click 2: select opposite edge",
    "Right-click : cancel drawing or remove nearest",
    "Enter / Space : confirm",
    "Escape : cancel",
]


class _CircleSelector:
    """
    Internal OpenCV mouse-event handler.
    Stores circles in *display* pixel coordinates; converts to *original*
    image coordinates only at the end via get_circles_original().
    """

    def __init__(self, display_image: np.ndarray, scale: float):
        self.base   = display_image          # BGR 8-bit, already scaled
        self.scale  = scale                  # original → display factor
        self._confirmed: list[tuple] = []    # (cx, cy, r) in display coords
        self._live:  tuple | None = None
        self._edge_point: tuple | None = None  # Stores the first click

    # ── Mouse callback ──────────────────────────────────────────────────────

    def mouse_cb(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._edge_point is None:
                # First click sets the starting edge
                self._edge_point = (x, y)
                self._live = (x, y, 0)
            else:
                # Second click finishes the circle
                cx = int((self._edge_point[0] + x) / 2)
                cy = int((self._edge_point[1] + y) / 2)
                r = int(np.hypot(x - self._edge_point[0], y - self._edge_point[1]) / 2)
                if r > 4:
                    self._confirmed.append((cx, cy, r))
                
                self._edge_point = None
                self._live = None

        elif event == cv2.EVENT_MOUSEMOVE:
            # Update live preview between first and second clicks
            if self._edge_point is not None:
                cx = int((self._edge_point[0] + x) / 2)
                cy = int((self._edge_point[1] + y) / 2)
                r = int(np.hypot(x - self._edge_point[0], y - self._edge_point[1]) / 2)
                self._live = (cx, cy, r)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._edge_point is not None:
                # Cancel the active draw operation
                self._edge_point = None
                self._live = None
            elif self._confirmed:
                # Remove the nearest confirmed circle
                dists = [np.hypot(cx - x, cy - y) for cx, cy, _ in self._confirmed]
                self._confirmed.pop(int(np.argmin(dists)))

    # ── Rendering ───────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        canvas = self.base.copy()

        # Instruction panel
        for i, line in enumerate(INSTRUCTIONS):
            cv2.putText(canvas, line, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        COLOR_INSTRUCT, 1, cv2.LINE_AA)

        # Confirmed circles
        for idx, (cx, cy, cr) in enumerate(self._confirmed):
            cv2.circle(canvas, (cx, cy), cr, COLOR_CONFIRMED, 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 3,  COLOR_CONFIRMED, -1)
            cv2.putText(canvas, str(idx + 1), (cx + cr + 4, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLOR_CONFIRMED, 1, cv2.LINE_AA)

        # Live circle and active edge point
        if self._edge_point is not None:
            cv2.circle(canvas, self._edge_point, 3, COLOR_LIVE, -1)
            
        if self._live and self._live[2] > 0:
            cx, cy, cr = self._live
            cv2.circle(canvas, (cx, cy), cr, COLOR_LIVE, 1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 3,  COLOR_LIVE, -1)

        # Counter
        h = canvas.shape[0]
        cv2.putText(canvas, f"GUVs selected: {len(self._confirmed)}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLOR_TEXT, 1, cv2.LINE_AA)
        return canvas

    # ── Output ──────────────────────────────────────────────────────────────

    def get_circles_original(self) -> list[dict]:
        """Convert display-space circles back to original image coordinates."""
        out = []
        for i, (cx, cy, cr) in enumerate(self._confirmed):
            out.append({
                "id": i + 1,
                "x":  int(round(cx / self.scale)),
                "y":  int(round(cy / self.scale)),
                "r":  int(round(cr / self.scale)),
            })
        return out


# ── Public API ──────────────────────────────────────────────────────────────

def run_selector(roi_frame: np.ndarray, save_path: str) -> list[dict]:
    """
    Open an interactive window, let the user draw circles, and save them.

    Parameters
    ----------
    roi_frame : np.ndarray
        First ROI frame (any bit-depth, greyscale).
    save_path : str
        Path where the resulting JSON file should be written.

    Returns
    -------
    list[dict]
        List of {"id", "x", "y", "r"} dicts in original image coordinates.
        Returns [] if the user pressed Escape.
    """
    # ── Convert to 8-bit BGR for display ────────────────────────────────────
    # Clip extreme bright/dark pixels to boost the contrast of the dim structures
    p_low, p_high = np.percentile(roi_frame, (1, 99.5))
    clipped_frame = np.clip(roi_frame, p_low, p_high)
    
    img_8u = cv2.normalize(clipped_frame, None, 0, 255,
                           cv2.NORM_MINMAX, cv2.CV_8U).astype(np.uint8)
    if img_8u.ndim == 2:
        img_bgr = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_8u

    h_orig, w_orig = img_bgr.shape[:2]

    # ── Scale down if necessary ──────────────────────────────────────────────
    scale = min(DISPLAY_MAX_PX / max(h_orig, w_orig), 1.0)
    dw = int(w_orig * scale)
    dh = int(h_orig * scale)
    display_base = cv2.resize(img_bgr, (dw, dh), interpolation=cv2.INTER_AREA)

    selector = _CircleSelector(display_base, scale)

    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TITLE, dw, dh)
    cv2.setMouseCallback(WIN_TITLE, selector.mouse_cb)

    while True:
        cv2.imshow(WIN_TITLE, selector.render())
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):          # Enter or Space → confirm
            break
        elif key == 27:              # Escape → cancel
            selector._confirmed.clear()
            break

    cv2.destroyAllWindows()

    circles = selector.get_circles_original()

    if circles:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(circles, f, indent=2)
        print(f"[CircleSelector] Saved {len(circles)} circle(s) → {save_path}")
    else:
        print("[CircleSelector] No circles saved (cancelled or empty).")

    return circles


def load_circles(path: str) -> list[dict]:
    """Load previously saved circle definitions from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ── Standalone entry point ──────────────────────────────────────────────────

def _main():
    # Lazy import so this script can be imported without config present
    import config as cfg
    import sys

    save_path = cfg.CIRCLES_JSON_PATH
    input_format = getattr(cfg, 'INPUT_FORMAT', 'TIFF').upper()
    
    if input_format == 'ND2':
        import nd2
        nd2_path = os.path.join(cfg.DATA_FOLDER, getattr(cfg, 'ND2_FILE_NAME', f"{cfg.EXPERIMENT_BASE_NAME}.nd2"))
        if not os.path.exists(nd2_path):
            sys.exit(f"ERROR: ND2 file not found: {nd2_path}")
        # NO getattr DEFAULT HERE. A fallback of 0 silently selects the
        # ACTIN channel, which for an Empty (no-cortex) sample is featureless
        # noise -- you would draw circles on nothing and the run would fail
        # later as though tracking were bad. validate_channel_config() raises
        # if the index is missing or the roles collide.
        import guv_analysis_utils as _utils
        _utils.validate_channel_config()
        with nd2.ND2File(nd2_path) as f:
            idx_mem = cfg.ND2_CHANNEL_IDX_MEMBRANE
            data = f.asarray()
            frame = data[0, idx_mem, :, :] if data.ndim == 4 else data[0, :, :]
        print(f"Opening selector on ND2: {nd2_path}")
    else:
        from natsort import natsorted
        import glob
        roi_pattern = os.path.join(cfg.DATA_FOLDER, cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX)
        roi_files = natsorted(glob.glob(roi_pattern))
        if not roi_files:
            sys.exit(f"ERROR: No ROI files found matching: {roi_pattern}")
        frame = cv2.imread(roi_files[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        print(f"Opening selector on: {roi_files[0]}")

    if frame is None:
        sys.exit("ERROR: Could not load first ROI frame.")

    print(f"Will save to: {save_path}")

    circles = run_selector(frame, save_path)
    print(f"\nSelected {len(circles)} GUV(s):")
    for c in circles:
        print(f"  GUV {c['id']:2d}  centre=({c['x']}, {c['y']})  r={c['r']} px")

if __name__ == "__main__":
    _main()