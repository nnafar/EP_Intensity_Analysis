# -*- coding: utf-8 -*-
"""
======================================================
--- GUV ANALYSIS UTILITIES ---
======================================================
This file contains all the helper functions for the GUV analysis.
You should not need to edit this file.
All user-configurable settings are in 'config.py'.
"""

# --- Core Packages ---
import numpy as np
import cv2
import pandas as pd
import os

# --- Packages for Time Extraction ---
import re
from datetime import datetime
import tifffile
from PIL import Image
from PIL.ExifTags import TAGS
from typing import List, Dict, Any, Optional

# -------------------------------------------------------------------
# --- 1. TIME EXTRACTION FUNCTIONS ---
# -------------------------------------------------------------------
# (All time extraction functions: _extract_from_imagej_metadata,
# _extract_from_exif_data, _extract_from_tiff_tags,
# _extract_from_filename_timestamps, extract_timestamps_from_metadata,
# create_manual_timestamps... are UNCHANGED)
# ... [Omitted for brevity, they are the same as the previous version] ...

_FILENAME_PATTERNS = {
    "YYYY-MM-DD_HH-MM-SS": r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',
    "YYYYMMDD_HHMMSS": r'(\d{8}_\d{6})',
    "milliseconds_t": r'_t(\d+)ms',
    "seconds_s": r'_(\d+)s_',
}

def _extract_from_imagej_metadata(tif_files: List[str]) -> Optional[List[float]]:
    print("Trying ImageJ metadata extraction...")
    timestamps = []
    frame_interval = None
    for i, filepath in enumerate(tif_files):
        try:
            with tifffile.TiffFile(filepath) as tif:
                if tif.is_imagej and tif.imagej_metadata:
                    metadata = tif.imagej_metadata
                    if frame_interval is None:
                        for key in ['finterval', 'frame_interval', 'spacing']:
                            if key in metadata:
                                frame_interval = float(metadata[key])
                                print(f"      Found frame interval: {frame_interval} seconds")
                                break
                    if frame_interval is not None:
                        timestamps.append(i * frame_interval)
                    else: return None
                else: return None
        except Exception: return None
    if timestamps:
        global metadata_method_used
        metadata_method_used = "ImageJ metadata"
        return timestamps
    return None

def _extract_from_exif_data(tif_files: List[str]) -> Optional[List[float]]:
    print("Trying EXIF timestamp extraction...")
    timestamps = []
    first_timestamp = None
    for filepath in tif_files:
        try:
            with Image.open(filepath) as img:
                exif_data = img.getexif()
                if exif_data:
                    datetime_str = None
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                            datetime_str = str(value)
                            break
                    if datetime_str:
                        try: dt = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S.%f")
                        except ValueError: dt = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
                        if first_timestamp is None:
                            first_timestamp = dt
                            timestamps.append(0.0)
                        else:
                            time_diff = (dt - first_timestamp).total_seconds()
                            timestamps.append(time_diff)
                    else: return None
                else: return None
        except Exception: return None
    if timestamps:
        global metadata_method_used
        metadata_method_used = "EXIF timestamps"
        return timestamps
    return None

def _extract_from_tiff_tags(tif_files: List[str]) -> Optional[List[float]]:
    print("Trying TIFF tag extraction...")
    timestamps = []
    frame_interval = None
    for i, filepath in enumerate(tif_files):
        try:
            with tifffile.TiffFile(filepath) as tif:
                page = tif.pages[0]
                for tag in page.tags:
                    tag_name = tag.name.lower()
                    if 'interval' in tag_name or 'time' in tag_name:
                        try:
                            interval = float(tag.value)
                            if interval > 0:
                                frame_interval = interval
                                print(f"      Found timing in tag {tag.name}: {frame_interval}")
                                break
                        except (ValueError, TypeError): continue
                if frame_interval is not None:
                    timestamps.append(i * frame_interval)
                else: return None
        except Exception: return None
    if timestamps:
        global metadata_method_used
        metadata_method_used = "TIFF tags"
        return timestamps
    return None

def _extract_from_filename_timestamps(tif_files: List[str]) -> Optional[List[float]]:
    print("Trying filename timestamp patterns...")
    for name, pattern in _FILENAME_PATTERNS.items():
        try:
            time_values = []
            first_dt = None
            for filepath in tif_files:
                filename = os.path.basename(filepath)
                match = re.search(pattern, filename)
                if match:
                    time_str = match.group(1)
                    if 'ms' in pattern: time_values.append(float(time_str) / 1000.0)
                    elif 's' in pattern: time_values.append(float(time_str))
                    else:
                        fmt = "%Y-%m-%d_%H-%M-%S" if '-' in time_str else "%Y%m%d_%H%M%S"
                        dt = datetime.strptime(time_str, fmt)
                        if first_dt is None: first_dt = dt
                        time_values.append((dt - first_dt).total_seconds())
                else: break
            if len(time_values) == len(tif_files):
                global metadata_method_used
                metadata_method_used = f"Filename pattern: '{name}'"
                return time_values
        except Exception: continue
    return None

def extract_timestamps_from_metadata(tif_files: List[str]) -> Optional[List[float]]:
    global metadata_method_used
    metadata_method_used = "None"
    methods_to_try = [
        _extract_from_imagej_metadata,
        _extract_from_exif_data,
        _extract_from_tiff_tags,
        _extract_from_filename_timestamps
    ]
    for method in methods_to_try:
        try:
            timestamps = method(tif_files)
            if timestamps is not None and len(timestamps) == len(tif_files):
                if np.all(np.diff(timestamps) >= 0): return timestamps
                else: print(f"   Method {method.__name__} produced non-monotonic timestamps. Discarding.")
        except Exception as e:
            print(f"   Method {method.__name__} failed: {e}")
            continue
    return None

def create_manual_timestamps(num_files: int, frame_interval: float = 0.2) -> List[float]:
    print(f"Using manual frame interval: {frame_interval} seconds")
    global metadata_method_used
    metadata_method_used = f"Manual calculation ({frame_interval}s interval)"
    return [i * frame_interval for i in range(num_files)]


# -------------------------------------------------------------------
# --- 2. ANALYSIS HELPER FUNCTIONS ---
# -------------------------------------------------------------------

def load_vesicles_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Loads DisGUVery CSV and returns the entire DataFrame of vesicles.
    """
    print(f"Loading GUV coordinates from: {csv_path}")
    try:
        detected_vesicles = pd.read_csv(csv_path)
        if detected_vesicles.empty:
            raise ValueError("No vesicles found in CSV.")
    except Exception as e:
        raise FileNotFoundError(f"Error loading DisGUVery CSV: {e}")
    
    print(f"Found {len(detected_vesicles)} GUV(s) in CSV file.")
    return detected_vesicles

def create_ideal_mask(shape: tuple, xc: int, yc: int, radius: int) -> np.ndarray:
    # (This function is unchanged)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (xc, yc), radius, 1, -1)
    return mask

def apply_guided_filter(img: np.ndarray, mask: np.ndarray, radius: int = 2, eps: float = 1e-6) -> np.ndarray:
    # (This function is unchanged)
    guided_filter = cv2.ximgproc.guidedFilter(guide=img, src=mask, radius=radius, eps=eps)
    return guided_filter

def dyn_model(t: np.ndarray, A: float, b: float) -> np.ndarray:
    # (This function is unchanged)
    return A * (1 - np.exp(-t / b))

# -------------------------------------------------------------------
# --- 3. FIGURE & EXPORT FUNCTIONS ---
# -------------------------------------------------------------------
# (find_closest_frame and style_image are unchanged)

def find_closest_frame(time_data: list, target_time: float) -> (int, float):
    """
    Finds the index and value of the closest time in a list.
    """
    time_array = np.asarray(time_data)
    idx = (np.abs(time_array - target_time)).argmin()
    return idx, time_array[idx]

def style_image(frame: np.ndarray, time_label: str, microns_per_pixel: float, scale_bar_microns: int) -> np.ndarray:
    """
    Applies the red colormap and adds annotations.
    """
    # 1. Normalize to 8-bit (0-255) for display
    img_8bit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 2. Apply red colormap
    img_color = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    img_color[:, :, 0] = 0  # Zero out Blue channel
    img_color[:, :, 1] = 0  # Zero out Green channel
    img_color[:, :, 2] = img_8bit # Set Red channel to intensity
    
    # 3. Add time stamp
    cv2.putText(
        img_color,
        time_label,
        (20, 40),  # Position (from top-left)
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,  # Font scale
        (255, 255, 255),  # Color (white)
        2,  # Thickness
        cv2.LINE_AA
    )
    
    # 4. Add scale bar (if pixel size is set)
    if microns_per_pixel is not None and microns_per_pixel > 0:
        bar_len_px = int(scale_bar_microns / microns_per_pixel)
        
        # Position: 20 pixels from bottom-right
        x0 = img_color.shape[1] - bar_len_px - 20
        y0 = img_color.shape[0] - 20
        
        cv2.rectangle(
            img_color,
            (x0, y0 - 2),
            (x0 + bar_len_px, y0 + 2),
            (255, 255, 255),  # Color (white)
            -1  # -1 = filled
        )
        
    return img_color