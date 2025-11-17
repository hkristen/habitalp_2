import requests
import pandas as pd
import os
from pathlib import Path
import tempfile
import json
import sys

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


# Download all raster tiles from one service URL that intersect the bounding box
# service_url = 'https://gis.stmk.gv.at/image/rest/services/OGD_DOP'
def download_raster_tiles_from_service_url(
    service_url: str,
    service_name: str,
    output_directory: str | Path,
    bbox_gpkg_path: str | Path = None,
    max_retry: int = 5,
    outSRS: int = 32633,
):
    """Download all raster tiles from one service URL that intersect the bounding box.

    Args:
        service_url (str): The base URL of the service.
        service_name (str): The name of the service to download tiles from.
        output_directory (Union[str, Path]): Directory to save downloaded tiles.
        bbox_gpkg_path (Union[str, Path], optional): Path to the GeoPackage file for bounding box. Defaults to None.
        max_retry (int, optional): Maximum number of retries for failed requests. Defaults to 5.
        outSRS (int, optional): Output spatial reference system. Defaults to 32633.

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

    file_param_list = {}

    # Get parameters for relevant files
    for tile_id in data_all_tiles_one_service["objectIds"]:
        download_params = {"rasterIds": str(tile_id), "f": "json"}

        try:
            response_download_one_tile = requests.get(
                download_url, download_params, timeout=10
            )
        except requests.exceptions.Timeout:
            print(f"Timeout for tile {tile_id}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"Error for tile {tile_id}: {e}")
            continue

        response_download_one_tile.raise_for_status()
        data_one_tile = response_download_one_tile.json()
        if "error" in data_one_tile:
            err = data_one_tile["error"]
            print(f'Error details: {data_one_tile["error"]["details"][0]}')
            raise RuntimeError(f'{err["code"]}: {err["message"]}')

        tile_filepath = data_one_tile["rasterFiles"][0]["id"]

        # Skip files that start with "Ov_" -> these are overview tiles that we don't need
        filename = tile_filepath.split("\\")[-1]
        if not filename.startswith("Ov_"):
            file_params = {
                "id": tile_filepath,
                "rasterId": str(tile_id),
            }

            file_param_list[filename] = file_params

    output_files = []
    print(
        f"Downloading {len(file_param_list)} tiles from {service_name} to {output_directory}"
    )

    # download each tile and write to output directory
    for it, (filename, file_params) in enumerate(file_param_list.items()):
        output_filepath = Path(os.path.join(output_directory, filename))

        # Check if tile is already downloaded
        if os.path.isfile(output_filepath):
            output_files.append(output_filepath)
            continue

        def get_request(retries=1):
            try:
                return requests.get(file_endpoint_url, file_params)
            except Exception as e:
                print(f"Request failed: {e}")
                if retries >= max_retry:
                    raise Exception from e
                retries += 1

                print("Repeating request ...")
                return get_request(retries=retries)

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
        output_files.append(output_filepath)

        print(
            f"Downloaded tile {file_params['rasterId']} ({it + 1} of {len(file_param_list)}) to {output_filepath}",
            end="\r",
        )
    return output_files


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
    )

    # # Test getting services
    # print("Fetching available services...")
    # services_df = get_arcgis_services_to_pd(service_url)
    # print(services_df.head(50))

    # # loop through services_df and download all tiles from each service using the service_url and service_name
    # for index, row in services_df.iterrows():
    #     print(f"Downloading tiles from service: {row['service_name']}")
    #     #create output directory for each service
    #     output_dir_service = output_dir / row['service_name']
    #     output_dir_service.mkdir(exist_ok=True)

    #     download_raster_tiles_from_service_url(
    #         service_url=service_url,
    #         service_name=row['service_name'],
    #         output_directory=output_dir_service,
    #         bbox_gpkg_path="/home/hkristen/habitalp2/data/raw/Verschneidung_NPG_Johnsbachtal_Europaschutzgebiet_BBOX_SMALL.gpkg"
    #     )
