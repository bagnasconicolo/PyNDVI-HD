# PyNDVI-HD

This repository contains two Tkinter-based graphical applications for computing and visualising the Normalised Difference Vegetation Index (NDVI) from RGB or modified cameras.

## Directory layout

- `scripts/ndvi_gui.py` – basic NDVI viewer with export capabilities.
- `scripts/pyndvi_hd.py` – extended "HD" viewer with additional smoothing, sharpness, and export scaling options.
- `examples/` – a collection of sample images used for testing the GUIs.

## Requirements

- Python 3.9 or later
- [Pillow](https://pillow.readthedocs.io/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/)
- Tkinter (usually installed with Python, but may require `python3-tk` on Linux)

Install the Python dependencies with:

```bash
pip install pillow matplotlib numpy
```

## Running the applications

### Basic NDVI viewer

```bash
python scripts/ndvi_gui.py
```

Features:

- Load JPG/PNG/TIF images (16-bit and 8-bit supported).
- Choose which channel represents NIR and which represents Red.
- Adjust NDVI display range interactively.
- Switch between any Matplotlib colormap.
- Save the colour-mapped NDVI image at original resolution.
- Histogram panel and clipboard copy.

### PyNDVI HD viewer

```bash
python scripts/pyndvi_hd.py
```

In addition to all features of the basic viewer, the HD version provides:

- Gaussian smoothing of the NDVI array.
- Adjustable sharpness after applying the colormap.
- Export scaling from 0.1× to 2.0× the original resolution.

## Sample images

The `examples/` directory includes a few small photos (JPG) and pre-computed NDVI renders (PNG). They are solely for demonstration/testing and carry no particular meaning.

## License

The code in this repository is released under the MIT License.

