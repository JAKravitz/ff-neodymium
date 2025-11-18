"""
Core utilities for neodymium (Nd) continuum-removal analysis around ~740 nm.

All functions assume hyperspectral cubes are read with rasterio and use a
bands-first convention: (bands, height, width). Utilities handle reshaping
when needed and expose helpers for ROI selection, continuum removal, depth
mapping, classification, visualization, and saving outputs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform as warp_transform

# Nudge threading down to avoid kernel crashes on constrained environments
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def read_wavelengths_from_hdr(hdr_path: str) -> np.ndarray:
    """
    Parse wavelength values (nanometers) from an ENVI-style .hdr file.

    Parameters
    ----------
    hdr_path : str
        Path to the HDR file.

    Returns
    -------
    np.ndarray
        1D array of wavelengths in nanometers.
    """
    text = Path(hdr_path).read_text()

    import re

    # Prefer the explicit "wavelength =" entry (avoid matching "wavelength unit").
    match = re.search(
        r"\bwavelength\s*=\s*{([^}]*)}", text, flags=re.IGNORECASE | re.DOTALL
    )
    content = None
    if match:
        content = match.group(1)
    else:
        # Fallback: scan lines starting with wavelength
        for line in text.splitlines():
            if line.lower().strip().startswith("wavelength"):
                brace_start = line.find("{")
                brace_end = line.rfind("}")
                if brace_start != -1 and brace_end != -1:
                    content = line[brace_start + 1 : brace_end]
                else:
                    # If no braces, take everything after '='
                    parts = line.split("=", maxsplit=1)
                    if len(parts) == 2:
                        content = parts[1]
                break
    if content is None:
        raise ValueError("Could not find 'wavelength' entry in HDR file.")

    # Split on commas or whitespace; ignore non-numeric tokens (e.g., band names)
    tokens = content.replace(",", " ").split()
    wavelengths = []
    for tok in tokens:
        try:
            wavelengths.append(float(tok))
        except ValueError:
            continue
    if not wavelengths:
        raise ValueError("No wavelengths parsed from HDR file.")
    return np.asarray(wavelengths, dtype=float)


def _parse_hdr_scalar(text: str, key: str) -> Optional[float]:
    """
    Parse a scalar numeric value from an ENVI header by key.
    Returns None if not found.
    """
    key_lower = key.lower()
    for line in text.splitlines():
        if line.lower().strip().startswith(key_lower):
            parts = line.split("=", maxsplit=1)
            if len(parts) == 2:
                try:
                    return float(parts[1].strip().split()[0])
                except ValueError:
                    return None
    return None


def load_hyperspectral_cube(
    image_path: str, hdr_path: str, window: Optional[Window] = None
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Load a hyperspectral cube and wavelength list. If `window` is provided,
    only that spatial window is read (bands-first).

    The cube is returned bands-first with shape (bands, height, width).

    Parameters
    ----------
    image_path : str
        Path to the hyperspectral raster (.tif, .img, etc.).
    hdr_path : str
        Path to the associated ENVI header containing wavelengths.
    window : rasterio.windows.Window, optional
        Spatial window to read. Reads full image when None.

    Returns
    -------
    data : np.ndarray
        Hyperspectral cube, shape (bands, height, width).
    profile : dict
        Rasterio profile for georeferencing/saving.
    wavelengths_nm : np.ndarray
        Wavelengths in nanometers.
    """
    wavelengths_nm = read_wavelengths_from_hdr(hdr_path)
    hdr_text = Path(hdr_path).read_text()
    scale_factor = _parse_hdr_scalar(hdr_text, "reflectance scale factor")
    nodata_hdr = _parse_hdr_scalar(hdr_text, "data ignore value")

    with rasterio.open(image_path) as src:
        if window is not None:
            data = src.read(window=window).astype(np.float32)  # bands-first
            profile = src.profile
            profile.update(
                {
                    "height": int(window.height),
                    "width": int(window.width),
                    "transform": rasterio.windows.transform(window, src.transform),
                }
            )
        else:
            data = src.read().astype(np.float32)  # bands-first
            profile = src.profile
        nodata_profile = src.nodata

    nodata_value = nodata_hdr if nodata_hdr is not None else nodata_profile
    if nodata_value is not None:
        data = np.where(data == nodata_value, np.nan, data)

    if scale_factor is not None:
        data = data * scale_factor

    # If data came back as (height, width, bands) (unlikely for rasterio), reorder
    if data.ndim == 3 and data.shape[0] != len(wavelengths_nm):
        if data.shape[-1] == len(wavelengths_nm):
            data = np.moveaxis(data, -1, 0)
        else:
            raise ValueError("Unexpected cube shape; cannot align with wavelengths.")

    return data, profile, wavelengths_nm


# -----------------------------------------------------------------------------
# ROI helpers
# -----------------------------------------------------------------------------


def _is_geographic(crs: Any) -> bool:
    """Return True if the CRS is geographic (lat/lon)."""
    try:
        return crs and crs.is_geographic
    except AttributeError:
        return False


def compute_roi_window(
    profile: Dict[str, Any],
    center_lat: float,
    center_lon: float,
    diameter_km: float,
) -> Window:
    """
    Compute a pixel window for a square ROI centered on geographic coordinates.

    Parameters
    ----------
    profile : dict
        Rasterio profile from the dataset.
    center_lat : float
        ROI center latitude (degrees).
    center_lon : float
        ROI center longitude (degrees).
    diameter_km : float
        Desired ROI width (kilometers).

    Returns
    -------
    rasterio.windows.Window
        Pixel window covering the ROI.
    """
    transform = profile["transform"]
    crs = profile.get("crs")
    half_km = diameter_km / 2.0

    if _is_geographic(crs):
        # Approximate degrees per kilometer
        deg_per_km_lat = 1.0 / 111.0
        deg_per_km_lon = 1.0 / (111.0 * np.cos(np.deg2rad(center_lat)))

        delta_lat = half_km * deg_per_km_lat
        delta_lon = half_km * deg_per_km_lon

        min_lon, max_lon = center_lon - delta_lon, center_lon + delta_lon
        min_lat, max_lat = center_lat - delta_lat, center_lat + delta_lat

        window = from_bounds(
            west=min_lon,
            south=min_lat,
            east=max_lon,
            north=max_lat,
            transform=transform,
            height=profile["height"],
            width=profile["width"],
            boundless=False,
        )
    else:
        # Project geographic center into dataset CRS
        center_x, center_y = warp_transform(
            "EPSG:4326", crs, [center_lon], [center_lat]
        )
        center_x, center_y = center_x[0], center_y[0]

        # Determine ROI size in pixels using pixel size from affine transform
        pix_x = abs(transform.a)
        pix_y = abs(transform.e)
        half_width_m = half_km * 1000.0
        half_cols = half_width_m / pix_x
        half_rows = half_width_m / pix_y

        row_c, col_c = rowcol(transform, center_x, center_y)

        col_off = col_c - half_cols
        row_off = row_c - half_rows
        width = half_cols * 2
        height = half_rows * 2
        window = Window(col_off=col_off, row_off=row_off, width=width, height=height)

    # Clip window to dataset bounds
    col_start = max(0, int(np.floor(window.col_off)))
    row_start = max(0, int(np.floor(window.row_off)))
    col_end = min(profile["width"], int(np.ceil(window.col_off + window.width)))
    row_end = min(profile["height"], int(np.ceil(window.row_off + window.height)))

    return Window(
        col_off=col_start,
        row_off=row_start,
        width=col_end - col_start,
        height=row_end - row_start,
    )


def subset_cube_to_window(
    data: np.ndarray, profile: Dict[str, Any], window: Window
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Subset a cube to the provided window and update the profile.
    """
    row0, col0 = int(window.row_off), int(window.col_off)
    h, w = int(window.height), int(window.width)
    subset = data[:, row0 : row0 + h, col0 : col0 + w]

    new_profile = profile.copy()
    new_profile.update(
        {
            "height": h,
            "width": w,
            "transform": rasterio.windows.transform(window, profile["transform"]),
        }
    )
    return subset, new_profile


# -----------------------------------------------------------------------------
# Spectral processing
# -----------------------------------------------------------------------------


def select_band_index_for_wavelength(
    wavelengths_nm: np.ndarray, target_nm: float
) -> int:
    """Return index of wavelength closest to target_nm."""
    return int(np.argmin(np.abs(wavelengths_nm - target_nm)))


def continuum_removal(
    spectra: np.ndarray,
    wavelengths_nm: np.ndarray,
    feature_center_nm: float = 739.7,
    left_continuum_nm: float = 733.6,
    right_continuum_nm: float = 755.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform continuum removal and band-depth calculation for a spectral feature.

    Parameters
    ----------
    spectra : np.ndarray
        Array shaped (..., nbands) with spectra in the last dimension.
    wavelengths_nm : np.ndarray
        1D wavelength list (nm).
    feature_center_nm : float
        Center wavelength of absorption feature (nm).
    left_continuum_nm : float
        Left continuum shoulder (nm).
    right_continuum_nm : float
        Right continuum shoulder (nm).

    Returns
    -------
    continuum_removed : np.ndarray
        Continuum-removed reflectance for bands between shoulders; same shape as
        spectra cropped to those bands.
    depth : np.ndarray
        Band depth at feature center, shaped like spectra without the spectral dimension.
    """
    spectra = np.asarray(spectra)
    wavelengths_nm = np.asarray(wavelengths_nm)
    nbands = spectra.shape[-1]
    if nbands != wavelengths_nm.size:
        raise ValueError("Spectra last dimension must match wavelength list.")

    left_idx = select_band_index_for_wavelength(wavelengths_nm, left_continuum_nm)
    right_idx = select_band_index_for_wavelength(wavelengths_nm, right_continuum_nm)
    center_idx = select_band_index_for_wavelength(wavelengths_nm, feature_center_nm)

    if left_idx == right_idx:
        raise ValueError("Left and right continuum indices are identical; adjust shoulders.")
    if left_idx > right_idx:
        left_idx, right_idx = right_idx, left_idx

    # Slice wavelengths and spectra for the continuum span
    wl_slice = wavelengths_nm[left_idx : right_idx + 1]
    spec_slice = spectra[..., left_idx : right_idx + 1]

    wl_left = wavelengths_nm[left_idx]
    wl_right = wavelengths_nm[right_idx]
    left_val = np.take(spectra, indices=left_idx, axis=-1)
    right_val = np.take(spectra, indices=right_idx, axis=-1)

    # Continuum line via linear interpolation
    slope = (right_val - left_val) / (wl_right - wl_left)
    continuum = left_val[..., None] + slope[..., None] * (wl_slice - wl_left)

    with np.errstate(divide="ignore", invalid="ignore"):
        continuum_removed = np.where(continuum != 0, spec_slice / continuum, np.nan)

    # Extract continuum-removed reflectance at feature center
    rel_center = center_idx - left_idx
    rel_center = np.clip(rel_center, 0, continuum_removed.shape[-1] - 1)
    r_center = np.take(continuum_removed, indices=rel_center, axis=-1)
    depth = 1.0 - r_center

    return continuum_removed, depth


def compute_depth_map(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    feature_center_nm: float = 739.7,
    left_continuum_nm: float = 733.6,
    right_continuum_nm: float = 755.3,
) -> np.ndarray:
    """
    Compute a 2D band-depth map from a hyperspectral cube.

    Accepts bands-first (bands, height, width) or bands-last (height, width, bands).
    """
    cube = np.asarray(cube)
    wavelengths_nm = np.asarray(wavelengths_nm)
    if cube.ndim != 3:
        raise ValueError("Cube must be 3D.")

    if cube.shape[0] == wavelengths_nm.size:
        spectrally_last = False
        bands, height, width = cube.shape
        reshaped = cube.reshape(bands, -1).T  # (num_pixels, bands)
    elif cube.shape[-1] == wavelengths_nm.size:
        spectrally_last = True
        height, width, bands = cube.shape
        reshaped = cube.reshape(-1, bands)
    else:
        raise ValueError("Spectral dimension does not match wavelengths length.")

    _, depth_flat = continuum_removal(
        reshaped,
        wavelengths_nm,
        feature_center_nm=feature_center_nm,
        left_continuum_nm=left_continuum_nm,
        right_continuum_nm=right_continuum_nm,
    )
    depth_map = depth_flat.reshape(height, width)

    # Propagate NaNs if nodata was present (zero continuum or invalid spectra)
    return depth_map.astype(np.float32)


# -----------------------------------------------------------------------------
# Tiled Nd continuum removal (based on FF_emit_recal_V1 workflow)
# -----------------------------------------------------------------------------


def integrate_area(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized trapezoidal integration over axis=0 for y shaped (K, N)."""
    return np.trapz(y, x=x, axis=0)


def continuum_remove_tile(subcube: np.ndarray, wavelengths_nm: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuum removal on a spectral tile.

    Returns:
      min_wl (R,C): wavelength at minimum (nm)
      depth  (R,C): 1 - min(continuum_removed)
      area   (R,C): integral of (1 - continuum_removed) over window (nm)
    """
    w = wavelengths_nm[idx]
    win = subcube[idx, :, :]  # (K,R,C)
    r0 = win[0]
    r1 = win[-1]
    w0, w1 = float(w[0]), float(w[-1])
    alpha = (w - w0) / (w1 - w0)
    cont = (r0[None, :, :] + (r1 - r0)[None, :, :] * alpha[:, None, None]).astype(np.float32)
    cont[cont == 0] = 1e-6

    cr = (win / cont).astype(np.float32)
    cr_min = cr.min(axis=0)
    depth = (1.0 - cr_min).astype(np.float32)
    argm = cr.argmin(axis=0)
    min_wl = w[argm].astype(np.float32)
    one_minus = (1.0 - cr).astype(np.float32)
    R, C = depth.shape
    area = integrate_area(one_minus.reshape(w.size, -1), w).reshape(R, C).astype(np.float32)
    return min_wl, depth, area


def compute_ndvi_tile(subcube: np.ndarray, wavelengths_nm: np.ndarray, red_nm: float = 650.0, nir_nm: float = 860.0) -> np.ndarray:
    red_i = int(np.argmin(np.abs(wavelengths_nm - red_nm)))
    nir_i = int(np.argmin(np.abs(wavelengths_nm - nir_nm)))
    red = subcube[red_i]
    nir = subcube[nir_i]
    return ((nir - red) / (nir + red + 1e-6)).astype(np.float32)


def compute_depth_area_tiled(
    image_path: str,
    hdr_path: str,
    window: Window,
    feature_center_nm: float = 739.7,
    start_nm: float = 720.0,
    end_nm: float = 780.0,
    ndvi_thresh: float = 0.3,
    tile: int = 128,
    gdal_cache_mb: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Streamed continuum removal over a spatial window to reduce memory pressure.

    Returns:
      min_wl (H,W), depth (H,W), area (H,W), profile_subset, wavelengths_nm
    """
    wavelengths_nm = read_wavelengths_from_hdr(hdr_path).astype(np.float32)
    hdr_text = Path(hdr_path).read_text()
    scale_factor = _parse_hdr_scalar(hdr_text, "reflectance scale factor")
    nodata_hdr = _parse_hdr_scalar(hdr_text, "data ignore value")

    idx = np.where((wavelengths_nm >= start_nm) & (wavelengths_nm <= end_nm))[0]
    if idx.size < 3:
        raise ValueError(f"Not enough bands in {start_nm}-{end_nm} nm")

    # Use a modest GDAL cache to prevent ballooning memory
    with rasterio.Env(GDAL_CACHEMAX=gdal_cache_mb * 1024 * 1024), rasterio.open(image_path) as src:
        profile = src.profile
        subset_transform = rasterio.windows.transform(window, src.transform)
        H = int(window.height)
        W = int(window.width)
        prof_subset = profile.copy()
        prof_subset.update({"height": H, "width": W, "transform": subset_transform})

        # allocate outputs
        min_wl = np.full((H, W), np.nan, dtype=np.float32)
        depth = np.full((H, W), np.nan, dtype=np.float32)
        area = np.full((H, W), np.nan, dtype=np.float32)

        nodata_profile = src.nodata
        nodata_value = nodata_hdr if nodata_hdr is not None else nodata_profile

        for r0 in range(0, H, tile):
            r1 = min(H, r0 + tile)
            for c0 in range(0, W, tile):
                c1 = min(W, c0 + tile)
                win = Window(window.col_off + c0, window.row_off + r0, c1 - c0, r1 - r0)
                subcube = src.read(window=win, out_dtype="float32")  # (B,R,C)
                if nodata_value is not None:
                    subcube[subcube == nodata_value] = np.nan
                if scale_factor is not None:
                    subcube *= scale_factor
                # NDVI mask
                ndvi = compute_ndvi_tile(subcube, wavelengths_nm, red_nm=650.0, nir_nm=860.0)
                veg_mask = ndvi > ndvi_thresh

                mwl_tile, depth_tile, area_tile = continuum_remove_tile(subcube, wavelengths_nm, idx)
                mwl_tile[veg_mask] = np.nan
                depth_tile[veg_mask] = np.nan
                area_tile[veg_mask] = np.nan

                min_wl[r0:r1, c0:c1] = mwl_tile
                depth[r0:r1, c0:c1] = depth_tile
                area[r0:r1, c0:c1] = area_tile

    return min_wl, depth, area, prof_subset, wavelengths_nm


# -----------------------------------------------------------------------------
# Classification and NIR helpers
# -----------------------------------------------------------------------------


def classify_nd(
    depth_map: np.ndarray,
    depth_threshold_min: float,
    depth_threshold_max: Optional[float] = None,
) -> np.ndarray:
    """
    Create a binary Nd classification map from a band-depth map.
    """
    depth_map = np.asarray(depth_map)
    if depth_threshold_max is not None and depth_threshold_max < depth_threshold_min:
        raise ValueError("depth_threshold_max must be >= depth_threshold_min.")

    if depth_threshold_max is None:
        mask = depth_map >= depth_threshold_min
    else:
        mask = (depth_map >= depth_threshold_min) & (depth_map <= depth_threshold_max)

    classification = np.zeros(depth_map.shape, dtype=np.uint8)
    classification[mask] = 1
    classification[np.isnan(depth_map)] = 0
    return classification


def clean_classification(
    classification: np.ndarray,
    min_neighbors: int = 2,
    min_cluster_pixels: int = 0,
) -> np.ndarray:
    """
    Reduce salt-and-pepper noise in a binary classification.

    Steps:
      1) Majority filter: require at least `min_neighbors` active neighbors in the 8-neighborhood.
      2) Optional connected-component filter: keep only components with size >= min_cluster_pixels
         (requires scipy.ndimage; if unavailable, only the majority filter is applied).
    """
    mask = classification.astype(bool)

    # Majority filter via neighbor sum (8-neighborhood)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neighbor_sum = np.zeros_like(mask, dtype=np.int16)
    for dy, dx in shifts:
        neighbor_sum += np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    mask = mask & (neighbor_sum >= min_neighbors)

    if min_cluster_pixels > 1:
        try:
            from scipy import ndimage as ndi  # type: ignore
        except Exception:
            # If scipy is unavailable, return majority-filtered mask
            return mask.astype(np.uint8)

        labels, nlab = ndi.label(mask)
        if nlab == 0:
            return np.zeros_like(mask, dtype=np.uint8)
        sizes = ndi.sum(mask, labels, index=np.arange(1, nlab + 1))
        keep = sizes >= min_cluster_pixels
        # Map keep mask back to image
        keep_lut = np.zeros(nlab + 1, dtype=bool)
        keep_lut[1:] = keep
        mask = keep_lut[labels]

    return mask.astype(np.uint8)


def extract_nir_band(
    cube: np.ndarray, wavelengths_nm: np.ndarray, nir_display_nm: float = 860.0
) -> np.ndarray:
    """
    Extract a grayscale NIR band closest to a target wavelength.
    """
    wavelengths_nm = np.asarray(wavelengths_nm)
    idx = select_band_index_for_wavelength(wavelengths_nm, nir_display_nm)

    if cube.ndim != 3:
        raise ValueError("Cube must be 3D.")
    if cube.shape[0] == wavelengths_nm.size:
        nir = cube[idx, :, :]
    elif cube.shape[-1] == wavelengths_nm.size:
        nir = cube[:, :, idx]
    else:
        raise ValueError("Spectral dimension does not match wavelengths length.")
    return nir


def read_band_window(
    image_path: str,
    hdr_path: str,
    window: Window,
    target_nm: float,
) -> np.ndarray:
    """
    Read a single band (nearest to target_nm) over a spatial window, applying scale/nodata.
    """
    wavelengths_nm = read_wavelengths_from_hdr(hdr_path)
    idx = select_band_index_for_wavelength(wavelengths_nm, target_nm)

    hdr_text = Path(hdr_path).read_text()
    scale_factor = _parse_hdr_scalar(hdr_text, "reflectance scale factor")
    nodata_hdr = _parse_hdr_scalar(hdr_text, "data ignore value")

    with rasterio.open(image_path) as src:
        arr = src.read(idx + 1, window=window).astype(np.float32)
        nodata_profile = src.nodata

    nodata_value = nodata_hdr if nodata_hdr is not None else nodata_profile
    if nodata_value is not None:
        arr = np.where(arr == nodata_value, np.nan, arr)
    if scale_factor is not None:
        arr = arr * scale_factor
    return arr


def save_full_rgb_with_aoi(
    image_path: str,
    wavelengths_nm: np.ndarray,
    window: Window,
    out_path: str,
    rgb_targets_nm: tuple[float, float, float] = (660.0, 560.0, 490.0),
    max_display_px: int = 1500,
) -> None:
    """
    Save a downsampled full-scene RGB preview with an AOI box overlay.
    """
    with rasterio.open(image_path) as ds:
        H, W = ds.height, ds.width
        scale = int(np.ceil(max(H, W) / max_display_px)) if max(H, W) > max_display_px else 1
        out_h, out_w = int(np.ceil(H / scale)), int(np.ceil(W / scale))

        r_i = select_band_index_for_wavelength(wavelengths_nm, rgb_targets_nm[0])
        g_i = select_band_index_for_wavelength(wavelengths_nm, rgb_targets_nm[1])
        b_i = select_band_index_for_wavelength(wavelengths_nm, rgb_targets_nm[2])

        rgb = ds.read(
            [r_i + 1, g_i + 1, b_i + 1],
            out_shape=(3, out_h, out_w),
            resampling=Resampling.average,
        ).astype(np.float32)

    rgb_out = np.empty_like(rgb)
    for i in range(3):
        lo, hi = np.nanpercentile(rgb[i], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        rgb_out[i] = np.clip((rgb[i] - lo) / (hi - lo), 0, 1)
    rgb_img = np.moveaxis(rgb_out, 0, -1)

    # AOI box scaled to preview coordinates
    box = (
        window.col_off / scale,
        window.row_off / scale,
        (window.col_off + window.width) / scale,
        (window.row_off + window.height) / scale,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb_img)
    rect = plt.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Saving and visualization
# -----------------------------------------------------------------------------


def save_geotiff(out_path: str, array: np.ndarray, profile: Dict[str, Any], dtype: str = "float32") -> None:
    """
    Save a 2D array as a GeoTIFF. Retained for compatibility; PNG saving is preferred
    when georeferencing is not required.
    """
    arr = np.asarray(array)
    new_profile = profile.copy()
    new_profile.update({"count": 1, "dtype": dtype, "compress": "deflate"})
    with rasterio.open(out_path, "w", **new_profile) as dst:
        dst.write(arr.astype(dtype), 1)


def save_png(out_path: str, array: np.ndarray, cmap: str = "gray", vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    """
    Save a 2D array to a PNG using matplotlib for quick-look outputs.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_overlay_class_on_nir(out_path: str, nir_band: np.ndarray, classification: np.ndarray, alpha: float = 0.6) -> None:
    """Save classification mask over NIR grayscale."""
    fig, ax = plt.subplots(figsize=(8, 6))
    nir = nir_band.astype(float)
    lo, hi = np.nanpercentile(nir, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    nir_norm = np.clip((nir - lo) / (hi - lo), 0, 1)

    ax.imshow(nir_norm, cmap="gray", alpha=0.6)
    masked = np.ma.masked_where(classification == 0, classification)
    ax.imshow(masked, cmap="autumn", alpha=alpha)
    ax.axis("off")
    ax.set_title("Identified Nd Locations", fontsize=14, pad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_overlay_depth_on_nir(
    out_path: str,
    nir_band: np.ndarray,
    depth_map: np.ndarray,
    cmap_depth: str = "viridis",
    depth_vmin: Optional[float] = None,
    depth_vmax: Optional[float] = None,
    alpha: float = 0.6,
) -> None:
    """Save depth overlay on NIR grayscale."""
    fig, ax = plt.subplots(figsize=(8, 6))
    nir = nir_band.astype(float)
    lo, hi = np.nanpercentile(nir, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    nir_norm = np.clip((nir - lo) / (hi - lo), 0, 1)

    ax.imshow(nir_norm, cmap="gray", alpha=0.3)
    ax.imshow(depth_map, cmap=cmap_depth, vmin=depth_vmin, vmax=depth_vmax, alpha=alpha)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_depth_overlay_on_nir(
    nir_band: np.ndarray,
    depth_map: np.ndarray,
    cmap_depth: str = "viridis",
    depth_vmin: Optional[float] = None,
    depth_vmax: Optional[float] = None,
    alpha: float = 0.6,
) -> None:
    """
    Plot depth map overlaid on NIR grayscale.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(nir_band, cmap="gray")
    plt.imshow(depth_map, cmap=cmap_depth, vmin=depth_vmin, vmax=depth_vmax, alpha=alpha)
    plt.colorbar(label="Nd band depth")
    plt.title("Nd band depth over NIR")
    plt.axis("off")
    plt.tight_layout()


def plot_classification_overlay_on_nir(
    nir_band: np.ndarray, classification: np.ndarray, alpha: float = 0.6
) -> None:
    """
    Plot binary classification overlaid on NIR grayscale.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(nir_band, cmap="gray")
    masked = np.ma.masked_where(classification == 0, classification)
    plt.imshow(masked, cmap="autumn", alpha=alpha)
    plt.title("Nd classification over NIR")
    plt.axis("off")
    plt.tight_layout()


# -----------------------------------------------------------------------------
# High-level pipeline
# -----------------------------------------------------------------------------


@dataclass
class NdPipelineResult:
    cube_subset: np.ndarray
    profile_subset: Dict[str, Any]
    wavelengths_nm: np.ndarray
    depth_map: np.ndarray
    classification: np.ndarray
    nir_band: np.ndarray


def run_nd_pipeline(
    image_path: str,
    hdr_path: str,
    center_lat: float,
    center_lon: float,
    diameter_km: float,
    feature_center_nm: float = 739.7,
    left_continuum_nm: float = 733.6,
    right_continuum_nm: float = 755.3,
    depth_threshold_min: float = 0.05,
    depth_threshold_max: Optional[float] = None,
    nir_display_nm: float = 860.0,
) -> NdPipelineResult:
    """
    Convenience wrapper to run the full Nd detection pipeline for a ROI.
    """
    # Read wavelengths and basic profile without loading full data
    wavelengths_nm = read_wavelengths_from_hdr(hdr_path)
    with rasterio.open(image_path) as src:
        profile_full = src.profile
    window = compute_roi_window(profile_full, center_lat, center_lon, diameter_km)

    # Load only the ROI window
    cube_subset, profile_subset, _ = load_hyperspectral_cube(
        image_path, hdr_path, window=window
    )

    depth_map = compute_depth_map(
        cube_subset,
        wavelengths_nm,
        feature_center_nm=feature_center_nm,
        left_continuum_nm=left_continuum_nm,
        right_continuum_nm=right_continuum_nm,
    )
    classification = classify_nd(depth_map, depth_threshold_min, depth_threshold_max)
    nir_band = extract_nir_band(cube_subset, wavelengths_nm, nir_display_nm=nir_display_nm)

    return NdPipelineResult(
        cube_subset=cube_subset,
        profile_subset=profile_subset,
        wavelengths_nm=wavelengths_nm,
        depth_map=depth_map,
        classification=classification,
        nir_band=nir_band,
    )
