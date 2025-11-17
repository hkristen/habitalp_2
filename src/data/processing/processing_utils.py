import os
import geopandas as gpd
from osgeo import gdal
from pathlib import Path
import rasterio
import shutil
from typing import List


def build_overviews(input_path: Path) -> None:
    """Build overviews for a raster file.

    Args:
        input_path (Path): File path to the raster file.

    Raises:
        RuntimeError: Failed to open the file
    """
    ds = gdal.Open(input_path, 1)
    if ds is None:
        raise RuntimeError(f"Failed to open {input_path} for building overviews")

    ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    ds = None


def create_vrt_from_tiles(tile_paths: List[str], vrt_path: str) -> str:
    """Create a Virtual Raster Tile (VRT) from a list of raster tile files.

    Args:
        tile_paths (List[str]): A list of file paths to the raster tile files to be combined.
        vrt_path (str): The file path where the resulting VRT file will be saved.

    Returns:
        str: The file path to the created VRT file.
    """
    # Enable GDAL exceptions
    gdal.UseExceptions()

    # Read source nodata value
    srs = gdal.Open(tile_paths[0])
    srcNodata = srs.GetRasterBand(1).GetNoDataValue()

    vrt_options = gdal.BuildVRTOptions(
        resampleAlg="nearest", srcNodata=srcNodata, VRTNodata="0"
    )
    vrt = gdal.BuildVRT(vrt_path, tile_paths, options=vrt_options)
    vrt = None
    return vrt_path


def create_cog(
    input_path: str,
    output_path: str,
    category: str = "orthophoto",
    outSRS: int = None,
    out_nodata_value: int | float = None,
    out_resolution: float = None,
) -> None:
    """Create a Cloud Optimized GeoTIFF (COG) from a Virtual Raster Tile (VRT) file.

    Args:
        input_path (str): The file path to the input VRT file.
        output_path (str): The file path where the COG will be created.
        category (str, optional): The type of dataset, either 'orthophoto' or 'elevation'. Defaults to 'orthophoto'.
        outSRS (int, optional): The desired output spatial reference system (EPSG code). If not provided, the input SRS will be used.
        out_nodata_value (int | float, optional): The defined nodata value for the output. If not provided, the input nodata value will be used.
        out_resolution (float, optional): The desired output resolution. If not provided, the input resolution will be used.

    Raises:
        ValueError: If the specified category is not recognized.
        RuntimeError: If the translation or warping process fails.

    Returns:
        str: The file path to the created COG.
    """
    # Enable GDAL exceptions
    gdal.UseExceptions()

    # Read source nodata value and original SRS
    vrt = gdal.Open(input_path)
    srcNodata = vrt.GetRasterBand(1).GetNoDataValue()
    original_srs = (
        vrt.GetProjection()
    )  # Get the original projection from the input file

    if not out_nodata_value:
        out_nodata_value = srcNodata
    # Set up creation options based on category
    if category == "orthophoto":
        compression = "JPEG"
        photometric = "YCBCR" if vrt.RasterCount > 1 else "MINISBLACK"
        predictor = "1"
    elif category == "elevation":
        compression = "LZW"
        photometric = "MINISBLACK"
        predictor = "2"
    else:
        raise ValueError(f"Category '{category}' not known.")

    # Determine the output SRS
    dstSRS = (
        f"EPSG:{outSRS}" if outSRS else original_srs
    )  # Use original SRS if outSRS is None

    # Determine the output resolution
    dstResolution = (
        out_resolution if out_resolution else vrt.GetGeoTransform()[1]
    )  # Use original resolution if out_resolution is None

    # TODO: Add mask layer for JPEG compressed output if needed to avoid artifacts
    warp_options = {
        #"srcSRS": "EPSG:XXXXX", # Set if not defined in the input
        "dstSRS": dstSRS,
        "format": "GTiff",
        "srcNodata": srcNodata,
        "dstNodata": out_nodata_value,
        "resampleAlg": "bilinear",
        "creationOptions": [
            f"COMPRESS={compression}",
            f"PHOTOMETRIC={photometric}",
            f"PREDICTOR={predictor}",
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "num_threads=20",
            "BIGTIFF=YES",
        ],
        "xRes": dstResolution,
        "yRes": dstResolution,
    }

    # Use gdal.Warp to handle both reprojection and COG creation
    result = gdal.Warp(output_path, input_path, **warp_options)

    if result is None:
        raise RuntimeError(f"Failed to create COG from {input_path} to {output_path}")

    # Close the dataset to ensure it's properly written
    result = None

    # Open for overview building
    build_overviews(output_path)

    # Verify output exists and can be opened
    if not os.path.exists(output_path):
        raise RuntimeError("COG file was not created")

    with rasterio.open(output_path) as src:
        print(f"Final raster shape: {src.shape}")
        print(f"Final raster CRS: {src.crs}")
        print(f"Final resolution: {src.res}")

    return output_path


def cleanup_temp_files(temp_files: List[str], temp_dir: str, vrt_path: str):
    """Remove temporary files and directories created during processing.

    Args:
        temp_files (List[str]): A list of file paths to temporary files to be deleted.
        temp_dir (str): The path to the temporary directory to be removed.
        vrt_path (str): The file path to the VRT file to be deleted.

    Returns:
        None
    """
    if temp_files:
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove {temp_file}: {e}")
    try:
        if os.path.exists(vrt_path):
            os.remove(vrt_path)
    except Exception as e:
        print(f"Warning: Could not remove VRT: {e}")
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(
                temp_dir
            )  # This will remove the directory and all its contents
    except Exception as e:
        print(f"Warning: Could not remove temp directory: {e}")


def dem_processing(
    input_dtm: Path,
    input_dsm: Path,
    output_folder: Path,
) -> None:
    """Generate slope, aspect, TPI, TRI, and roughness from elevation raster files.

    Args:
        input_dtm (Path): The file path to the input DTM file.
        input_dsm (Path): The file path to the input DSM file.
        output_folder (Path): The folder path where the processed files will be saved.
    """
    import richdem as rd

    # Enable GDAL exceptions
    gdal.UseExceptions()

    creationOptions = {
        "COMPRESS": "LZW",
        "PHOTOMETRIC": "MINISBLACK",
        "PREDICTOR": "2",
        "TILED": "YES",
        "BLOCKXSIZE": "256",
        "BLOCKYSIZE": "256",
        "num_threads": "20",
        "BIGTIFF": "YES",
    }

    input = gdal.Open(input_dtm)

    print("Calculate Slope ...")
    gdal.DEMProcessing(
        destName=output_folder / "slope.tif",
        srcDS=input,
        processing="slope",
        creationOptions=creationOptions,
    )
    build_overviews(output_folder / "slope.tif")

    print("Calculate Aspect ...")
    gdal.DEMProcessing(
        destName=output_folder / "aspect.tif",
        srcDS=input,
        processing="aspect",
        alg="Horn",
        creationOptions=creationOptions,
    )
    build_overviews(output_folder / "aspect.tif")

    print("Calculate TRI ...")
    gdal.DEMProcessing(
        destName=output_folder / "tri.tif",
        srcDS=input,
        processing="TRI",
        creationOptions=creationOptions,
    )
    build_overviews(output_folder / "tri.tif")

    print("Calculate TPI ...")
    gdal.DEMProcessing(
        destName=output_folder / "tpi.tif",
        srcDS=input,
        processing="TPI",
        creationOptions=creationOptions,
    )
    build_overviews(output_folder / "tpi.tif")

    print("Calculate terrain roughness ...")
    gdal.DEMProcessing(
        destName=output_folder / "roughness_terrain.tif",
        srcDS=input,
        processing="roughness",
        creationOptions=creationOptions,
    )
    build_overviews(output_folder / "roughness_terrain.tif")

    input = gdal.Open(input_dsm)

    print("Calculate canopy roughness ...")
    gdal.DEMProcessing(
        destName=output_folder / "roughness_canopy.tif",
        srcDS=input,
        processing="roughness",
        creationOptions=creationOptions,
    )
    build_overviews(output_folder / "roughness_canopy.tif")

    print("Calculate curvature ...")
    dtm = rd.LoadGDAL(input_dtm)

    for file in ["curvature", "profile_curvature", "planform_curvature"]:
        attribute = rd.TerrainAttribute(dtm, attrib=file)
        rd.SaveGDAL(output_folder / f"{file}_tmp.tif", attribute)

        result = gdal.Translate(
            destName=output_folder / f"{file}.tif",
            srcDS=output_folder / f"{file}_tmp.tif",
            creationOptions=creationOptions,
        )
        build_overviews(output_folder / f"{file}.tif")

        try:
            os.remove(output_folder / f"{file}_tmp.tif")
        except Exception as e:
            print(f"Warning: Could not remove {file}_tmp.tif: {e}")

    for file in ["curvature", "profile_curvature", "planform_curvature"]:
        attribute = rd.TerrainAttribute(dtm, attrib=file)
        rd.SaveGDAL(output_folder / f"{file}_tmp.tif", attribute)

        result = gdal.Translate(
            destName=output_folder / f"{file}.tif",
            srcDS=output_folder / f"{file}_tmp.tif",
            creationOptions=creationOptions,
        )
        build_overviews(output_folder / f"{file}.tif")

        try:
            os.remove(output_folder / f"{file}_tmp.tif")
        except Exception as e:
            print(f"Warning: Could not remove {file}_tmp.tif: {e}")


def read_bbox_from_gpkg(
    gpkg_path: str | Path, set_crs: int = None
) -> tuple[float, float, float, float]:
    """Reads bounding box coordinates from a GeoPackage file.

    Args:
        gpkg_path (Union[str, Path]): Path to the GeoPackage file.
        set_crs (int): EPSG code to change coordinate reference system.

    Returns:
        tuple[float, float, float, float]: A tuple containing the bounding box
            coordinates in the format (minx, miny, maxx, maxy).

    Raises:
        FileNotFoundError: If the specified GeoPackage file does not exist.
        ValueError: If the specified file is not a valid GeoPackage (.gpkg).
    """
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage file not found: {gpkg_path}")
    if gpkg_path.suffix.lower() != ".gpkg":
        raise ValueError(f"File must be a GeoPackage (.gpkg): {gpkg_path}")

    gdf = gpd.read_file(gpkg_path)

    if set_crs:
        if set_crs != gdf.crs.to_epsg():
            gdf = gdf.to_crs(set_crs)

    return gdf.total_bounds


def create_clipped_cog_with_bbox(
    input_path: str,
    gpkg_path: str | Path,
    output_path: str = None,
    category: str = "orthophoto",
    outSRS: int = None,
    out_nodata_value: int | float = None,
    out_resolution: float = None,
    temp_vrt_path: str = None,
) -> str:
    """Create a Cloud Optimized GeoTIFF (COG) clipped to a bounding box from a GeoPackage.

    Args:
        input_path (str): The file path to the input raster file.
        output_path (str): The file path where the clipped COG will be created.
        gpkg_path (str | Path): Path to the GeoPackage file containing the clipping geometry, should only contain one feature with a bounding box geometry.
        category (str, optional): The type of dataset, either 'orthophoto' or 'elevation'. Defaults to 'orthophoto'.
        outSRS (int, optional): The desired output spatial reference system (EPSG code).
        out_nodata_value (int | float, optional): The defined nodata value for the output.
        out_resolution (float, optional): The desired output resolution.
        temp_vrt_path (str, optional): Path for the temporary VRT file. If None, a path will be generated.

    Returns:
        str: The file path to the created clipped COG.
    """
    # Enable GDAL exceptions
    gdal.UseExceptions()

    # Get the bounding box from the GeoPackage,
    bbox = read_bbox_from_gpkg(gpkg_path, set_crs=outSRS)

    # If output_path is not defined, create one based on input_path
    if not output_path:
        input_path_obj = Path(input_path)
        output_path = input_path_obj.with_stem(input_path_obj.stem + "_clipped")

    # Create a temporary VRT path if not provided
    if temp_vrt_path is None:
        temp_dir = os.path.dirname(output_path)
        temp_vrt_path = os.path.join(temp_dir, "temp_clipped.vrt")
        print(f"Created temporary VRT path: {temp_vrt_path}")

    # Read source nodata value
    src_ds = gdal.Open(input_path)
    srcNodata = src_ds.GetRasterBand(1).GetNoDataValue()

    if out_nodata_value is None:
        out_nodata_value = srcNodata

    if out_resolution is None:
        out_resolution = src_ds.GetGeoTransform()[1]

    # Create a clipped VRT
    vrt_options = gdal.BuildVRTOptions(
        resampleAlg="bilinear",
        srcNodata=srcNodata,
        VRTNodata=out_nodata_value,
        outputBounds=bbox,  # Apply the bounding box for clipping
    )

    vrt = gdal.BuildVRT(temp_vrt_path, [input_path], options=vrt_options)
    vrt = None  # Close the dataset

    # Use the existing create_cog function to create the final COG
    result = create_cog(
        temp_vrt_path,
        output_path,
        category=category,
        outSRS=outSRS,
        out_nodata_value=out_nodata_value,
        out_resolution=out_resolution,
    )

    # Clean up the temporary VRT
    try:
        if os.path.exists(temp_vrt_path):
            os.remove(temp_vrt_path)
    except Exception as e:
        print(f"Warning: Could not remove temporary VRT: {e}")

    return result
