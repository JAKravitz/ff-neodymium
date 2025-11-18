"""
CLI entry point for Nd continuum-removal mapping without notebook overhead.

Example:
python run_nd_cli.py \
  --image /Users/jeremy/Desktop/ff1-2512-siwana/FF01_20251108_00501045_0000002512_L2A.tif \
  --hdr   /Users/jeremy/Desktop/ff1-2512-siwana/FF01_20251108_00501045_0000002512_L2A.tif.hdr \
  --center-lat 25.569517 --center-lon 72.358346 --diameter-km 10 \
  --feature-center 739.7 --left-shoulder 733.6 --right-shoulder 755.3 \
  --ndvi-thresh 0.3 --tile 128 --depth-threshold 0.05 \
  --min-neighbors 2 --min-cluster-pixels 20 --dilate-pixels 2 \
  --full-rgb --out-prefix nd_roi
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

import nd_continuum as nd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nd continuum removal CLI")
    p.add_argument("--image", required=True, help="Path to hyperspectral raster (.tif/.img)")
    p.add_argument("--hdr", required=True, help="Path to ENVI HDR with wavelengths/scale")
    p.add_argument("--center-lat", type=float, required=True, help="ROI center latitude (deg)")
    p.add_argument("--center-lon", type=float, required=True, help="ROI center longitude (deg)")
    p.add_argument("--diameter-km", type=float, default=10.0, help="ROI diameter in km (square window)")
    p.add_argument("--feature-center", type=float, default=739.7, help="Nd feature center nm")
    p.add_argument("--left-shoulder", type=float, default=733.6, help="Left continuum nm")
    p.add_argument("--right-shoulder", type=float, default=755.3, help="Right continuum nm")
    p.add_argument("--ndvi-thresh", type=float, default=0.3, help="NDVI mask threshold")
    p.add_argument("--tile", type=int, default=128, help="Tile size for streaming")
    p.add_argument("--depth-threshold", type=float, default=0.05, help="Band-depth threshold for mask")
    p.add_argument("--min-neighbors", type=int, default=2, help="Min active neighbors (8-connectivity) to keep a pixel")
    p.add_argument("--min-cluster-pixels", type=int, default=20, help="Minimum connected-component size to keep (requires scipy; 0 to disable)")
    p.add_argument("--dilate-pixels", type=int, default=1, help="Dilate classification for display only (pixels of radius in 4-neighborhood)")
    p.add_argument("--out-prefix", default=None, help="Prefix for PNG outputs (default: image name prefix in image folder)")
    p.add_argument("--full-rgb", action="store_true", help="Save a full-scene RGB preview with AOI box")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    hdr_path = Path(args.hdr)
    # Default output prefix: image stem in the same folder as the image
    if args.out_prefix is None:
        out_prefix = image_path.with_suffix("").name
    else:
        out_prefix = args.out_prefix
    out_dir = image_path.parent
    out_prefix_path = out_dir / out_prefix

    with rasterio.open(image_path) as src:
        profile = src.profile

    window = nd.compute_roi_window(
        profile, args.center_lat, args.center_lon, args.diameter_km
    )

    minwl_map, depth_map, area_map, profile_subset, wavelengths_nm = nd.compute_depth_area_tiled(
        str(image_path),
        str(hdr_path),
        window,
        feature_center_nm=args.feature_center,
        start_nm=args.left_shoulder,
        end_nm=args.right_shoulder,
        ndvi_thresh=args.ndvi_thresh,
        tile=args.tile,
    )

    print(
        f"Depth stats: min={np.nanmin(depth_map):.4f}, "
        f"max={np.nanmax(depth_map):.4f}, mean={np.nanmean(depth_map):.4f}"
    )

    class_map = nd.classify_nd(depth_map, args.depth_threshold)
    class_map = nd.clean_classification(
        class_map,
        min_neighbors=args.min_neighbors,
        min_cluster_pixels=args.min_cluster_pixels,
    )
    # Dilate for display if requested

    display_class = class_map.copy()

    if args.dilate_pixels > 0:

        try:

            from scipy import ndimage as ndi  # type: ignore

            struct = ndi.generate_binary_structure(2, 1)  # 4-neighborhood

            display_class = ndi.binary_dilation(display_class, structure=struct, iterations=args.dilate_pixels).astype(np.uint8)

        except Exception:

            pass



    nir_band = nd.read_band_window(
        str(image_path), str(hdr_path), window, target_nm=860.0
    )

    # Overlays: classification over NIR, depth over NIR
    nd.save_png(out_dir / f"{out_prefix}_depth.png", depth_map, cmap="viridis")
    nd.save_overlay_class_on_nir(out_dir / f"{out_prefix}_class_on_nir.png", nir_band, display_class)
    nd.save_overlay_depth_on_nir(out_dir / f"{out_prefix}_depth_on_nir.png", nir_band, depth_map)

    if args.full_rgb:
        # Save a downsampled full-scene RGB with AOI box
        full_rgb_path = out_dir / f"{out_prefix}_full_rgb_aoi.png"
        nd.save_full_rgb_with_aoi(
            str(image_path),
            wavelengths_nm,
            window,
            full_rgb_path,
            rgb_targets_nm=(660.0, 560.0, 490.0),
            max_display_px=1500,
        )

    print("Wrote:")
    print(f"  Depth: {out_prefix_path}_depth.png")
    print(f"  Class on NIR: {out_prefix_path}_class_on_nir.png")
    print(f"  Depth on NIR: {out_prefix_path}_depth_on_nir.png")
    if args.full_rgb:
        print(f"  Full RGB with AOI: {out_prefix_path}_full_rgb_aoi.png")


if __name__ == "__main__":
    main()
