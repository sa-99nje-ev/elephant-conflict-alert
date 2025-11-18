import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show

# ----------------------------
# CONFIG
# ----------------------------
TILES_DIR = "app/data/elevation_tiles"
OUTPUT_FILE = "app/data/sri_lanka_elevation.tif"
# ----------------------------

def load_tiles():
    tiles = []
    tile_files = sorted([f for f in os.listdir(TILES_DIR) if f.lower().endswith((".hgt", ".srtmgl1.2", ".hgt.zip"))])

    if not tile_files:
        raise FileNotFoundError("No HGT tiles found in app/data/elevation_tiles")

    print(f"Found {len(tile_files)} tiles:")
    for f in tile_files:
        print(" -", f)

    for filename in tile_files:
        path = os.path.join(TILES_DIR, filename)

        # Handle zipped tiles
        if filename.endswith(".zip"):
            raise RuntimeError("You must unzip files first. Found zipped tiles.")

        src = rasterio.open(path)
        tiles.append(src)

    return tiles

def merge_tiles(tiles):
    print("\nMerging tiles...")
    mosaic, out_transform = merge(tiles)
    print("Merge successful.")

    # Copy metadata from first tile
    out_meta = tiles[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw"
    })

    return mosaic, out_meta

def write_output(mosaic, metadata):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with rasterio.open(OUTPUT_FILE, "w", **metadata) as dest:
        dest.write(mosaic)

    print(f"\n✔ Saved merged elevation file: {OUTPUT_FILE}")

def main():
    print("=== SRTM Tile Merger (Rasterio) ===")
    tiles = load_tiles()
    mosaic, metadata = merge_tiles(tiles)
    write_output(mosaic, metadata)
    print("\nAll done!")

if __name__ == "__main__":
    main()
