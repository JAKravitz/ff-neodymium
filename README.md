# Nd continuum-removal pipeline (v1)

CLI tools and helpers for neodymium (~740 nm) detection on hyperspectral Firefly/ENVI-style data using continuum removal, NDVI masking, and noise-cleaned classification overlays.

## Quick start

1) Install dependencies (rasterio, numpy, matplotlib; optional: scipy for better denoising/dilation):
```bash
pip install rasterio numpy matplotlib scipy
```

2) Run the CLI on your scene (example from current workflow):
```bash
python run_nd_cli.py \
  --image /Users/jeremy/Desktop/ff1-2512-siwana/FF01_20251108_00501045_0000002512_L2A.tif \
  --hdr   /Users/jeremy/Desktop/ff1-2512-siwana/FF01_20251108_00501045_0000002512_L2A.tif.hdr \
  --center-lat 25.569517 --center-lon 72.358346 --diameter-km 10 \
  --feature-center 739.7 --left-shoulder 733.6 --right-shoulder 755.3 \
  --ndvi-thresh 0.3 --tile 128 --depth-threshold 0.05 \
  --min-neighbors 2 --min-cluster-pixels 20 --dilate-pixels 2 \
  --full-rgb --out-prefix nd_roi
```

Defaults: outputs are written next to the image; if you omit `--out-prefix`, the image stem is used.
```

Outputs:
- `nd_roi_depth.png` – band-depth heatmap
- `nd_roi_class_on_nir.png` – cleaned classification over dimmed NIR
- `nd_roi_depth_on_nir.png` – depth heatmap over dimmed NIR
- `nd_roi_full_rgb_aoi.png` – full-scene RGB with AOI box (when `--full-rgb`)

## CLI argument reference
- `--image` (path): Hyperspectral raster (.tif/.img) with bands-first or standard rasterio layout.
- `--hdr` (path): ENVI header with wavelengths and scale factor.
- `--center-lat`, `--center-lon` (float): ROI center in degrees (EPSG:4326).
- `--diameter-km` (float): ROI width in kilometers (square window in source CRS).
- `--feature-center` (float): Target absorption center wavelength (nm), default 739.7.
- `--left-shoulder`, `--right-shoulder` (float): Continuum shoulders (nm), default 733.6/755.3.
- `--ndvi-thresh` (float): NDVI mask threshold; pixels above are set to NaN (vegetation mask).
- `--tile` (int): Tile size (pixels) for streaming; lower for less RAM.
- `--depth-threshold` (float): Minimum band depth to classify Nd.
- `--min-neighbors` (int): Require this many active neighbors (8-neighborhood) to keep a pixel (denoise).
- `--min-cluster-pixels` (int): Drop connected components smaller than this size (requires scipy; 0 to disable).
- `--dilate-pixels` (int): Visual-only dilation iterations to thicken the displayed mask.
- `--full-rgb` (flag): Also save a downsampled RGB of the full scene with the AOI box.
- `--out-prefix` (string): Prefix for outputs. If omitted, uses the image stem in the image folder.

## Key files
- `nd_continuum.py` – continuum removal, NDVI mask, tiling, overlays, RGB AOI helper.
- `run_nd_cli.py` – CLI wrapper for ROI processing and PNG outputs. See header for current example.

## Notes
- Classification cleaning combines an 8-neighbor majority filter and optional connected-component size filter; `--dilate-pixels` thickens clusters for display only.
- NIR underlay is dimmed (alpha 0.3) to emphasize overlays.
- Tiled processing (`--tile`) controls memory; reduce tile if RAM is limited.
