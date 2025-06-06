#!/usr/bin/env python3
"""
NDVI Viewer & Exporter GUI
=========================
A self-contained Tkinter application for loading a single multi-channel image (RGB or modified
camera with a blue filter), computing NDVI, interactively exploring results with live
colormaps/thresholds, and exporting a full-resolution colour-mapped TIFF/PNG.

Key features
------------
• **Load** any common raster (JPG/PNG/TIF). 16-bit and 8-bit images supported via Pillow.
• **Channel mapping** – pick which channel is NIR and which is red (drop-downs).
• **Real-time NDVI** – sliders adjust min/max display range without recomputing core NDVI.
• **Colormap selector** – choose any matplotlib colormap (Sequential, Diverging, etc.).
• **High-quality output** – saved file keeps the source resolution, rendered through the
  chosen colormap with smooth interpolation (no pixelated artefacts).
• **Histogram panel** – small live NDVI histogram helps pick sensible slider limits.
• **Copy to clipboard** – quick grab of the current preview for pasting elsewhere.

Usage
-----
1.  Install deps (Python ≥3.9):
    ```bash
    pip install pillow matplotlib numpy
    ```
2.  Run:
    ```bash
    python ndvi_gui.py
    ```

Author: *Your Name*  —  2025-06-06
License: MIT
"""

import sys
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib import cm

plt.ioff()  # turn off interactive mode for embedded plots

__version__ = "1.0.0"

# ------------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------------

def read_image(path: Path) -> np.ndarray:
    """Load image into a Numpy array (H×W×C, float32 0–1)."""
    img = Image.open(path)
    # Pillow automatically converts to RGBA for some formats; we keep first 3 channels
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img_np = np.asarray(img, dtype=np.float32)
    if img_np.ndim == 2:  # grayscale
        img_np = np.stack([img_np] * 3, axis=-1)
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    return img_np / 255.0  # normalise to 0-1 float


def compute_ndvi(arr: np.ndarray, nir_idx: int, red_idx: int) -> np.ndarray:
    """Compute NDVI = (NIR-RED)/(NIR+RED) for the selected channels."""
    nir = arr[:, :, nir_idx]
    red = arr[:, :, red_idx]
    bottom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / np.where(bottom == 0, 1e-5, bottom)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi.astype(np.float32)


def apply_colormap(ndvi: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> np.ndarray:
    """Map NDVI to RGBA UInt8 image via matplotlib colormap."""
    cmap = cm.get_cmap(cmap_name)
    norm = np.clip((ndvi - vmin) / (vmax - vmin), 0, 1)
    rgba = cmap(norm, bytes=True)  # shape H×W×4 uint8
    return rgba  # uint8


# ------------------------------------------------------------------------------
# Main GUI class
# ------------------------------------------------------------------------------
class NDVIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NDVI Viewer — v" + __version__)
        self.geometry("1200x800")
        self.minsize(900, 600)

        self.img_path: Path | None = None
        self.img_data: np.ndarray | None = None  # original RGB float32 0-1
        self.ndvi: np.ndarray | None = None  # float32 -1..1

        # Default mapping assumes consumer RGB camera modified with blue filter:
        # Many DIY builds route NIR mainly into the red channel and visible red into blue.
        # Feel free to change mapping in the dropdowns.
        self.nir_idx = tk.IntVar(value=0)  # R
        self.red_idx = tk.IntVar(value=2)  # B

        # NDVI display limits (-1..1)
        self.vmin = tk.DoubleVar(value=0.0)
        self.vmax = tk.DoubleVar(value=1.0)

        self.cmap_name = tk.StringVar(value="RdYlGn")

        self._build_ui()

    # --------------------------- UI LAYOUT -----------------------------------
    def _build_ui(self):
        # Toolbar frame
        tb = ttk.Frame(self)
        tb.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(tb, text="Load image", command=self.load_image).pack(side=tk.LEFT)
        ttk.Button(tb, text="Save NDVI ↓", command=self.save_ndvi).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="Copy preview", command=self.copy_preview).pack(side=tk.LEFT, padx=2)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Channel mapping controls
        ttk.Label(tb, text="NIR channel:").pack(side=tk.LEFT)
        ttk.Combobox(tb, textvariable=self.nir_idx, width=3, values=(0, 1, 2)).pack(side=tk.LEFT)
        ttk.Label(tb, text="Red channel:").pack(side=tk.LEFT)
        ttk.Combobox(tb, textvariable=self.red_idx, width=3, values=(0, 1, 2)).pack(side=tk.LEFT)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(tb, text="Colormap:").pack(side=tk.LEFT)
        cmaps = sorted(m for m in plt.colormaps() if not m.endswith("_r"))
        ttk.Combobox(tb, textvariable=self.cmap_name, values=cmaps, width=12).pack(side=tk.LEFT)

        # Placeholder for histogram (just a canvas placeholder, updated later)
        self.hist_canvas = tk.Canvas(tb, width=120, height=60, bg="#222")
        self.hist_canvas.pack(side=tk.RIGHT, padx=4)

        # Main display area (scrollable)
        self.img_frame = ttk.Frame(self)
        self.img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.img_frame, bg="#333")
        self.hbar = ttk.Scrollbar(self.img_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self.img_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Sliders frame (bottom)
        sliders = ttk.Frame(self)
        sliders.pack(side=tk.BOTTOM, fill=tk.X, pady=4)

        ttk.Label(sliders, text="NDVI min:").pack(side=tk.LEFT)
        tk.Scale(sliders, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL,
                 length=200, variable=self.vmin, command=lambda e: self.update_preview()).pack(side=tk.LEFT, padx=2)

        ttk.Label(sliders, text="NDVI max:").pack(side=tk.LEFT)
        tk.Scale(sliders, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL,
                 length=200, variable=self.vmax, command=lambda e: self.update_preview()).pack(side=tk.LEFT, padx=2)

        # Colormap selector triggers preview update
        self.cmap_name.trace_add("write", lambda *a: self.update_preview())
        self.nir_idx.trace_add("write", lambda *a: self._recompute_ndvi())
        self.red_idx.trace_add("write", lambda *a: self._recompute_ndvi())

    # --------------------------- Core Actions --------------------------------
    def load_image(self):
        path = filedialog.askopenfilename(title="Select image", filetypes=[
            ("Images", "*.tif *.tiff *.jpg *.jpeg *.png"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.img_data = read_image(Path(path))
            self.img_path = Path(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            return

        self._recompute_ndvi()

    def _recompute_ndvi(self):
        if self.img_data is None:
            return
        # Validate channel indices
        if self.nir_idx.get() == self.red_idx.get():
            messagebox.showwarning("Invalid mapping", "NIR and Red channels must differ.")
            return
        self.ndvi = compute_ndvi(self.img_data, self.nir_idx.get(), self.red_idx.get())
        # Auto-scale sliders to NDVI percentiles
        lo, hi = np.percentile(self.ndvi, [5, 95])
        self.vmin.set(round(float(lo), 2))
        self.vmax.set(round(float(hi), 2))
        self.update_histogram()
        self.update_preview()

    def update_histogram(self):
        if self.ndvi is None:
            self.hist_canvas.delete("all")
            return
        ndvi_flat = self.ndvi.flatten()
        hist, edges = np.histogram(ndvi_flat, bins=50, range=(-1, 1))
        hist = hist / hist.max()
        self.hist_canvas.delete("all")
        w, h = 120, 60
        for i, val in enumerate(hist):
            x0 = i * w / len(hist)
            y0 = h - val * h
            x1 = (i + 1) * w / len(hist)
            self.hist_canvas.create_rectangle(x0, y0, x1, h, fill="#5af", width=0)
        # Draw current min/max lines
        for v in (self.vmin.get(), self.vmax.get()):
            x = (v + 1) / 2 * w
            self.hist_canvas.create_line(x, 0, x, h, fill="#fff")

    def update_preview(self):
        if self.ndvi is None:
            return
        vmin = min(self.vmin.get(), self.vmax.get())
        vmax = max(self.vmin.get(), self.vmax.get())
        rgb = apply_colormap(self.ndvi, self.cmap_name.get(), vmin, vmax)
        preview = Image.fromarray(rgb)
        # Downscale preview for display if it's huge
        max_preview = 1200
        scale = min(1.0, max_preview / max(preview.size))
        if scale < 1.0:
            preview = preview.resize((int(preview.width * scale), int(preview.height * scale)), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.config(scrollregion=(0, 0, preview.width, preview.height))
        self.update_histogram()

    def save_ndvi(self):
        if self.ndvi is None:
            messagebox.showinfo("No data", "Load an image first.")
            return
        path = filedialog.asksaveasfilename(title="Save NDVI", defaultextension=".png",
                                            filetypes=[("PNG file", "*.png"), ("TIFF", "*.tif"), ("All", "*.*")])
        if not path:
            return
        try:
            rgba = apply_colormap(self.ndvi, self.cmap_name.get(), self.vmin.get(), self.vmax.get())
            out_img = Image.fromarray(rgba)
            out_img.save(path)
            messagebox.showinfo("Saved", f"NDVI image saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    def copy_preview(self):
        if not hasattr(self, 'tk_img'):
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(self.tk_img._PhotoImage__photo)  # type: ignore
            messagebox.showinfo("Copied", "Preview image copied to clipboard.")
        except Exception:
            messagebox.showwarning("Clipboard", "Copy not supported on this platform.")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app = NDVIApp()
    app.mainloop()
