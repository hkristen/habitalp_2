import requests
import pandas as pd
import os
from pathlib import Path
import tempfile
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Get the project root by going up directories until we find the root
project_root = (
    Path(__file__).resolve().parents[3]
)  # Go up 3 levels from src/data/download to root
sys.path.append(str(project_root))
from src.data.processing.processing_utils import *


def get_arcgis_services_to_pd(base_url: str) -> pd.DataFrame:
    """Fetch services from ArcGIS REST endpoint and return as DataFrame.

    Args:
        base_url (str): The base URL of the ArcGIS REST endpoint.

    Returns:
        pd.DataFrame: A DataFrame containing the services with columns
        ['category', 'service_name', 'year', 'type', 'name'].
    """
    # Get the response
    response = requests.get(f"{base_url}?f=json")
    data = response.json()

    # Convert to DataFrame
    services_df = pd.DataFrame(data["services"])

    # Clean up the data
    if "name" in services_df.columns:
        # Split the name field if it contains '/'
        if services_df["name"].str.contains("/").any():
            services_df[["category", "service_name"]] = services_df["name"].str.split(
                "/", expand=True
            )
        else:
            services_df["category"] = ""
            services_df["service_name"] = services_df["name"]

    # Add year information if it exists in the name
    services_df["year"] = services_df["service_name"].str.extract(r"(\d{4})")

    # Reorder columns to a more logical sequence
    cols = ["category", "service_name", "year", "type", "name"]
    services_df = services_df[cols]

    # Sort by category and name
    services_df = services_df.sort_values(["category", "service_name"]).reset_index(
        drop=True
    )

    return services_df


def get_tile_metadata(args):
    """Get metadata for a single tile - designed for parallel execution."""
    tile_id, download_url = args

    download_params = {"rasterIds": str(tile_id), "f": "json"}

    try:
        response_download_one_tile = requests.get(download_url, download_params)
        response_download_one_tile.raise_for_status()
        data_one_tile = response_download_one_tile.json()

        # Skip tiles with errors or missing data
        if "error" in data_one_tile:
            return None
        if "rasterFiles" not in data_one_tile or not data_one_tile["rasterFiles"]:
            return None

        tile_filepath = data_one_tile["rasterFiles"][0]["id"]
        filename = tile_filepath.split("\\")[-1]

        # Skip overview tiles
        if filename.startswith("Ov_"):
            return None

        file_params = {
            "id": tile_filepath,
            "rasterId": str(tile_id),
        }
        return (filename, file_params)

    except:
        # Skip any tile that causes problems
        return None


def download_single_tile(args):
    """Download a single tile - designed for parallel execution."""
    (
        filename,
        file_params,
        file_endpoint_url,
        output_directory,
        max_retry,
        image_server_url,
    ) = args

    output_filepath = Path(os.path.join(output_directory, filename))

    # Check if tile is already downloaded
    if os.path.isfile(output_filepath):
        return output_filepath

    def get_request(retries=1):
        try:
            return requests.get(file_endpoint_url, file_params)
        except Exception as e:
            print(f"Request failed for {filename}: {e}")
            if retries >= max_retry:
                raise Exception from e
            retries += 1
            print(f"Retrying {filename} (attempt {retries})...")
            return get_request(retries=retries)

    try:
        response_file_endpoint_one_tile = get_request()

        # Download tile metadata
        metadata_params = {"f": "pjson"}
        metadata_url = f"{image_server_url}/{file_params['rasterId']}"
        metadata_response = requests.get(metadata_url, params=metadata_params)
        metadata_filename = f"{os.path.splitext(filename)[0]}.json"
        metadata_filepath = os.path.join(output_directory, metadata_filename)

        # Save tile data
        with open(output_filepath, "wb") as output_file:
            output_file.write(response_file_endpoint_one_tile.content)
        # Save tile metadata
        with open(metadata_filepath, "w") as metadata_file:
            json.dump(metadata_response.json(), metadata_file)

        return output_filepath

    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        raise


def download_raster_tiles_from_service_url(
    service_url: str,
    service_name: str,
    output_directory: str | Path,
    bbox_gpkg_path: str | Path = None,
    max_retry: int = 5,
    outSRS: int = 32633,
    parallel: bool = False,
    max_workers: int = 10,
):
    """Download all raster tiles from one service URL that intersect the bounding box.

    Args:
        service_url (str): The base URL of the service.
        service_name (str): The name of the service to download tiles from.
        output_directory (Union[str, Path]): Directory to save downloaded tiles.
        bbox_gpkg_path (Union[str, Path], optional): Path to the GeoPackage file for bounding box. Defaults to None.
        max_retry (int, optional): Maximum number of retries for failed requests. Defaults to 5.
        outSRS (int, optional): Output spatial reference system. Defaults to 32633.
        parallel (bool, optional): Whether to use parallel downloading. Defaults to False.
        max_workers (int, optional): Maximum number of parallel downloads (only used if parallel=True). Defaults to 10.

    Returns:
        list[Path]: A list of file paths for the downloaded raster tiles.
    """
    query_url = f"{service_url}/{service_name}/ImageServer/query"
    download_url = f"{service_url}/{service_name}/ImageServer/download"
    file_endpoint_url = f"{service_url}/{service_name}/ImageServer/file"
    image_server_url = f"{service_url}/{service_name}/ImageServer"

    if bbox_gpkg_path is not None:
        bbox = read_bbox_from_gpkg(bbox_gpkg_path)
        query_params = {
            "where": "1=1",
            "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",  # bbox returns tuple of (minx,miny,maxx,maxy)
            "geometryType": "esriGeometryEnvelope",
            "inSR": 32633,  # Specifying input coordinate system
            "spatialRel": "esriSpatialRelIntersects",
            "returnGeometry": "false",
            "returnIdsOnly": "true",
            "outSR": outSRS,  # Specifying output coordinate system
            "f": "json",
        }
    else:
        # if no bbox is provided, download all tiles
        query_params = {"where": "1=1", "f": "json", "returnIdsOnly": "true"}

    response_query_all_tiles_one_service = requests.get(query_url, params=query_params)
    data_all_tiles_one_service = response_query_all_tiles_one_service.json()
    if not data_all_tiles_one_service["objectIds"]:
        raise RuntimeError("No data available with these parameters.")

    # Download service metadata
    metadata_params = {"f": "pjson"}
    metadata_response = requests.get(image_server_url, params=metadata_params)
    metadata_filename = f"{service_name}.json"
    metadata_filepath = os.path.join(output_directory, metadata_filename)
    with open(metadata_filepath, "w") as metadata_file:
        json.dump(metadata_response.json(), metadata_file)

    # Get parameters for relevant files
    download_tasks = []
    total_tiles = len(data_all_tiles_one_service["objectIds"])

    if (
        parallel and total_tiles > 100
    ):  # Only parallelize metadata if there are many tiles
        print(f"Preparing metadata for {total_tiles} tiles in parallel...")

        # Prepare metadata tasks
        metadata_tasks = [
            (tile_id, download_url)
            for tile_id in data_all_tiles_one_service["objectIds"]
        ]

        # Execute metadata gathering in parallel
        with ThreadPoolExecutor(max_workers=min(max_workers, 20)) as executor:
            future_to_tile = {
                executor.submit(get_tile_metadata, task): task
                for task in metadata_tasks
            }

            for future in as_completed(future_to_tile):
                result = future.result()
                if result is not None:
                    filename, file_params = result
                    task_args = (
                        filename,
                        file_params,
                        file_endpoint_url,
                        output_directory,
                        max_retry,
                        image_server_url,
                    )
                    download_tasks.append(task_args)
    else:
        # Sequential metadata processing
        for tile_id in data_all_tiles_one_service["objectIds"]:
            try:
                download_params = {"rasterIds": str(tile_id), "f": "json"}

                response_download_one_tile = requests.get(download_url, download_params)
                response_download_one_tile.raise_for_status()
                data_one_tile = response_download_one_tile.json()

                # Skip tiles with errors or missing data
                if "error" in data_one_tile:
                    continue
                if (
                    "rasterFiles" not in data_one_tile
                    or not data_one_tile["rasterFiles"]
                ):
                    continue

                tile_filepath = data_one_tile["rasterFiles"][0]["id"]
                filename = tile_filepath.split("\\")[-1]

                # Skip overview tiles
                if filename.startswith("Ov_"):
                    continue

                file_params = {
                    "id": tile_filepath,
                    "rasterId": str(tile_id),
                }

                task_args = (
                    filename,
                    file_params,
                    file_endpoint_url,
                    output_directory,
                    max_retry,
                    image_server_url,
                )
                download_tasks.append(task_args)

            except:
                # Skip any tile that causes problems
                continue

    # Execute downloads - either parallel or sequential
    output_files = []
    total_tasks = len(download_tasks)

    if parallel and total_tasks > 1:
        # Parallel execution
        completed_count = 0
        print(
            f"Starting parallel download of {total_tasks} tiles with {max_workers} workers..."
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(download_single_tile, task): task
                for task in download_tasks
            }

            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    output_files.append(result)
                    completed_count += 1

                    # Progress update
                    if completed_count % 100 == 0 or completed_count == total_tasks:
                        print(
                            f"Downloaded {completed_count}/{total_tasks} tiles ({completed_count/total_tasks*100:.1f}%)"
                        )

                except Exception as exc:
                    filename = task[0]  # filename is first element in task tuple
                    print(f"Download failed for {filename}: {exc}")
                    # Optionally continue with other downloads rather than failing completely
                    # raise exc
    else:
        # Sequential execution (original behavior)
        print(f"Starting sequential download of {total_tasks} tiles...")
        for i, task_args in enumerate(download_tasks):
            try:
                result = download_single_tile(task_args)
                output_files.append(result)

                # Progress update
                if (i + 1) % 100 == 0 or (i + 1) == total_tasks:
                    print(
                        f"Downloaded {i + 1}/{total_tasks} tiles ({(i + 1)/total_tasks*100:.1f}%)"
                    )

            except Exception as exc:
                filename = task_args[0]  # filename is first element in task tuple
                print(f"Download failed for {filename}: {exc}")
                raise exc

    print(f"Completed downloading {len(output_files)} tiles")
    return output_files


# Keep the original function for backwards compatibility
# download_raster_tiles_from_service_url now handles both sequential and parallel


def download_and_create_cog(
    service_url,
    service_name,
    category,
    output_path,
    bbox_gpkg_path,
    raw_data_folder=None,
    max_retry=5,
    outSRS=32633,
    out_nodata_value: float | int = None,
    parallel=False,
    max_workers=10,
):
    """Download raster tiles, create a VRT, and convert it to a COG.

    Args:
        service_url (str): The base URL of the service.
        service_name (str): The name of the service to download tiles from.
        category (str): The category for the COG.
        output_path (str): Path to save the final COG file.
        bbox_gpkg_path (Union[str, Path]): Path to the GeoPackage file for bounding box.
        raw_data_folder (Union[str, Path], optional): Directory to save raw data. Defaults to None.
        max_retry (int, optional): Maximum number of retries for failed requests. Defaults to 5.
        outSRS (int, optional): Output spatial reference system. Defaults to 32633.
        out_nodata_value (float | int): Defined nodata value for the output. Pixel values will not be changed.
        parallel (bool, optional): Whether to use parallel downloading. Defaults to False.
        max_workers (int, optional): Maximum number of parallel downloads (only used if parallel=True). Defaults to 10.

    Raises:
        Exception: If an error occurs during processing.

    Returns:
        None
    """
    # Set tile and vrt file paths
    if not raw_data_folder:
        # Create temporary directory
        raw_data_folder = tempfile.mkdtemp()
        vrt_path = raw_data_folder / (service_name.lower() + ".vrt")
        cleanup = True
    else:
        Path(raw_data_folder).mkdir(parents=True, exist_ok=True)
        vrt_path = Path(raw_data_folder).parent / (service_name.lower() + ".vrt")
        cleanup = False

    # Create output file path
    Path(output_path).parent.mkdir(exist_ok=True)

    try:
        # Download tiles
        tiles = download_raster_tiles_from_service_url(
            service_url=service_url,
            service_name=service_name,
            output_directory=raw_data_folder,
            bbox_gpkg_path=bbox_gpkg_path,
            max_retry=max_retry,
            outSRS=outSRS,
            parallel=parallel,
            max_workers=max_workers,
        )

        # Create VRT
        print("Creating VRT...")
        create_vrt_from_tiles(tiles, vrt_path)

        # Convert to COG
        print("Converting to COG...")
        create_cog(
            vrt_path,
            output_path,
            category=category,
            outSRS=outSRS,
            out_nodata_value=out_nodata_value,
        )

        # Only cleanup after successful COG creation and verification if raw data is not to be saved
        if cleanup:
            cleanup_temp_files(tiles, raw_data_folder, vrt_path)

    except Exception as e:
        # In case of error, keep temporary files and re-raise the exception
        print(f"Error during processiing: {e}")
        print(f"Raster Tiles preserved in: {raw_data_folder}")
        raise


if __name__ == "__main__":
    # Example usage
    service_url = "https://gis.stmk.gv.at/image/rest/services/OGD_DOP"
    output_dir = Path(
        "/home/hkristen/habitalp2/data/processed/Falschfarben_2008_2011_script"
    )

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Test downloading tiles from a single service
    service_name = "Falschfarben_2008_2011"
    print("\nDownloading raster tiles...")
    download_raster_tiles_from_service_url(
        service_url=service_url,
        service_name=service_name,
        output_directory=output_dir,
        bbox_gpkg_path="/home/hkristen/habitalp2/data/raw/Verschneidung_NPG_Johnsbachtal_Europaschutzgebiet_BBOX.gpkg",
        parallel=True,  # Set to True for parallel download, False for sequential
        max_workers=10,  # Only used if parallel=True
    )
