import pandas as pd
import ee
from datetime import datetime
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import os
import logging
import glob
import json
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union

ee.Initialize()

input_geojson = "/Users/milindsoni/Documents/projects/rice/bamboo/mygeojsonag.geojson"
output_csv = "polygon_data_with_weather.csv"
temp_output_folder = "temp_results_polygons"
log_file = "processing_log_polygons.txt"

NUM_WORKERS = mp.cpu_count() - 1
SAVE_INTERVAL = 100

start_date = ee.Date("2022-05-01")
end_date = ee.Date("2022-11-30")

s2_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
s1_bands = ["VV", "VH"]
weather_bands = [
    "u_component_of_wind_10m_max",
    "v_component_of_wind_10m_max",
    "temperature_2m_max",
    "temperature_2m_min",
    "total_precipitation_sum",
    "potential_evaporation_sum",
    "surface_net_solar_radiation_sum",
    "volumetric_soil_water_layer_1",
]

logging.basicConfig(
    filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s"
)


def create_15day_mosaics(
    collection, start_date, end_date, bands, reducer=ee.Reducer.median()
):
    def create_mosaic(d):
        date = ee.Date(d)
        end = date.advance(15, "day")
        mosaic = collection.filterDate(date, end).reduce(reducer)
        return mosaic.set("system:time_start", date.millis())

    dates = ee.List.sequence(start_date.millis(), end_date.millis(), 15 * 86400 * 1000)
    mosaics = ee.ImageCollection(dates.map(lambda d: create_mosaic(d)))
    return mosaics.select([f"{b}_.*" for b in bands], bands)


def process_polygon(feature):
    geometry = shape(feature["geometry"])

    # Ensure the geometry is valid
    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    # Convert to MultiPolygon if it's a Polygon
    if isinstance(geometry, Polygon):
        geometry = MultiPolygon([geometry])

    # Create a list of EE Polygons
    ee_polygons = [
        ee.Geometry.Polygon(list(poly.exterior.coords)) for poly in geometry.geoms
    ]

    # Create a EE FeatureCollection from the list of polygons
    ee_feature_collection = ee.FeatureCollection(ee_polygons)

    # Sentinel-2 collection
    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee_feature_collection)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    # Sentinel-1 collection
    s1_collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(ee_feature_collection)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    # Weather collection
    weather_collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(ee_feature_collection)
        .filterDate(start_date, end_date)
        .select(weather_bands)
    )

    # Soil type
    soil_image = ee.Image("users/2019ht12445/gldas_soil")

    s2_mosaics = create_15day_mosaics(s2_collection, start_date, end_date, s2_bands)
    s1_mosaics = create_15day_mosaics(s1_collection, start_date, end_date, s1_bands)
    weather_mosaics = create_15day_mosaics(
        weather_collection, start_date, end_date, weather_bands, ee.Reducer.mean()
    )

    results = {
        "Name": feature["properties"].get("Name", ""),
        "geometry": feature["geometry"],
    }

    all_dates = set()
    temp_results = {}

    for prefix, mosaics, bands in [
        ("Sentinel_", s2_mosaics, s2_bands),
        ("SAR_", s1_mosaics, s1_bands),
        ("ECMWF_", weather_mosaics, weather_bands),
    ]:
        try:
            # Convert ImageCollection to a list of Images
            image_list = mosaics.toList(mosaics.size())
            size = image_list.size().getInfo()

            for i in range(size):
                image = ee.Image(image_list.get(i))
                data = image.reduceRegions(
                    collection=ee_feature_collection,
                    reducer=ee.Reducer.mean(),
                    scale=10,
                ).getInfo()["features"]

                for feature in data:
                    properties = feature["properties"]
                    date_str = datetime.utcfromtimestamp(
                        image.get("system:time_start").getInfo() / 1000
                    ).strftime("%Y-%m-%d")
                    all_dates.add(date_str)

                    for band in bands:
                        value = properties.get(band)
                        if prefix == "ECMWF_":
                            if "temperature" in band:
                                value -= 273.15  # Convert from Kelvin to Celsius
                            elif band == "total_precipitation_sum":
                                value *= 1000  # Convert from meters to millimeters

                        # Handle null or empty values
                        if value is None or pd.isna(value):
                            value = -9999  # Use a sentinel value for missing data
                        else:
                            value = f"{value:.4f}"

                        band_key = f"{prefix}{band}"
                        if band_key not in temp_results:
                            temp_results[band_key] = {}
                        temp_results[band_key][date_str] = value

        except ee.ee_exception.EEException as e:
            logging.error(
                f"Error retrieving {prefix} data for polygon {feature['properties'].get('Name', '')}: {e}"
            )

    # Extract soil type
    try:
        soil_value = (
            soil_image.reduceRegion(
                ee.Reducer.mean(), ee_feature_collection.geometry(), 30
            )
            .get("b1")
            .getInfo()
        )
        results["soil_type"] = soil_value if soil_value is not None else -9999
    except ee.ee_exception.EEException as e:
        logging.error(
            f"Error retrieving soil type for polygon {feature['properties'].get('Name', '')}: {e}"
        )
        results["soil_type"] = -9999

    # Reorganize results
    all_dates = sorted(list(all_dates))
    for band in temp_results:
        for date in all_dates:
            results[f"{band}_{date}"] = temp_results[band].get(date, -9999)

    return results


def process_geojson():
    with open(input_geojson, "r") as f:
        geojson_data = json.load(f)

    features = geojson_data["features"]

    if not os.path.exists(temp_output_folder):
        os.makedirs(temp_output_folder)

    pool = mp.Pool(processes=NUM_WORKERS)

    results = []
    with tqdm(total=len(features), desc="Processing polygons") as pbar:
        for i, result in enumerate(
            pool.imap(process_polygon, features),
            start=0,
        ):
            if result:
                results.append(result)
            pbar.update()

            if (i + 1) % SAVE_INTERVAL == 0 or i == len(features) - 1:
                temp_df = pd.DataFrame(results)
                temp_output_file = os.path.join(
                    temp_output_folder, f"temp_results_{i+1}.csv"
                )
                temp_df.to_csv(temp_output_file, index=False)
                logging.info(f"Intermediate results saved to {temp_output_file}")

                # Clear the results list to free up memory
                results = []

    pool.close()
    pool.join()

    logging.info("Processing complete. Concatenating all temporary files...")
    final_df = combine_temp_files()

    if final_df is not None:
        # Reorder columns
        static_columns = [
            "Name",
            "geometry",
            "soil_type",
        ]
        dynamic_columns = [col for col in final_df.columns if col not in static_columns]

        # Sort dynamic columns by band and then by date
        dynamic_columns.sort(key=lambda x: (x.split("_")[0], x.split("_")[-1]))

        final_df = final_df[static_columns + dynamic_columns]

        final_df.to_csv(output_csv, index=False)
        logging.info(
            f"Full data with weather information and soil type saved to {output_csv}"
        )

        # Print summary statistics
        logging.info("\nSummary:")
        logging.info(f"Total number of polygons in input: {len(features)}")
        logging.info(f"Total number of polygons processed: {len(final_df)}")
        logging.info(f"Number of polygons skipped: {len(features) - len(final_df)}")
        logging.info(f"Total number of features: {len(final_df.columns)}")

        # Print a few sample values from the first row
        logging.info("\nSample values from the first row:")
        for col in final_df.columns[:10]:  # Print first 10 columns as an example
            logging.info(f"{col}: {final_df[col].iloc[0]}")
    else:
        logging.error(
            "Failed to combine temporary files. Please check the temporary files in the output folder."
        )


def combine_temp_files():
    temp_files = sorted(
        glob.glob(os.path.join(temp_output_folder, "temp_results_*.csv")),
        key=lambda f: int(f.split("_")[-1].split(".")[0]),
    )

    if not temp_files:
        logging.warning("No temporary files found to combine.")
        return None

    combined_df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)

    logging.info(f"Combined {len(temp_files)} temporary files.")
    logging.info(f"Total rows in combined dataframe: {len(combined_df)}")

    return combined_df


if __name__ == "__main__":
    process_geojson()
    print(
        f"Processing complete. Check {log_file} for details and {output_csv} for results."
    )
