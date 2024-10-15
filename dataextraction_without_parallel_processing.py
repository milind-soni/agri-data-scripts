import ee
import geopandas as gpd
import pandas as pd
from datetime import datetime, timedelta
import time

# Initialize Earth Engine
ee.Initialize()


def get_all_bands_data(geometry, start_date, end_date):
    collections = {
        "COPERNICUS/S2_SR_HARMONIZED": None,
        "COPERNICUS/S1_GRD": None,
        "ECMWF/ERA5_LAND/HOURLY": None,
    }

    data = {}
    for name, bands in collections.items():
        try:
            print(f"Processing {name} from {start_date} to {end_date}...")
            ic = (
                ee.ImageCollection(name)
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
            )

            if bands:
                ic = ic.select(bands)

            if ic.size().getInfo() > 0:
                mosaic = ic.mosaic()
                values = mosaic.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=10,
                    maxPixels=1e9,
                ).getInfo()

                for band, value in values.items():
                    data[f"{name.split('/')[-1]}_{band}"] = (
                        value if value is not None else -9999
                    )
            else:
                print(f"No data available for {name} from {start_date} to {end_date}")
        except Exception as e:
            print(f"Error with {name}: {str(e)}")

    return data


def geometry_to_ee(geom):
    if geom.type == "Polygon":
        return ee.Geometry.Polygon(list(geom.exterior.coords))
    elif geom.type == "MultiPolygon":
        return ee.Geometry.MultiPolygon(
            [list(poly.exterior.coords) for poly in geom.geoms]
        )
    else:
        raise ValueError(f"Unsupported geometry type: {geom.type}")


def process_polygon(polygon):
    geometry = geometry_to_ee(polygon.geometry)

    start_date = datetime(2022, 5, 1)
    end_date = start_date + timedelta(days=150)

    all_data = {}
    current_date = start_date
    while current_date < end_date:
        period_end = current_date + timedelta(days=15)
        data = get_all_bands_data(
            geometry, current_date.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d")
        )
        for key, value in data.items():
            if key not in all_data:
                all_data[key] = {}
            all_data[key][current_date.strftime("%Y-%m-%d")] = value
        current_date = period_end

    formatted_data = {
        f"{band}_{date}": value
        for band, dates in all_data.items()
        for date, value in dates.items()
    }

    formatted_data["Shape_Name"] = polygon.get("Name", "Unknown")

    return formatted_data


def main():
    start_time = time.time()

    gdf = gpd.read_file(
        "/Users/milindsoni/Documents/projects/rice/bamboo/mygeojsonag.geojson"
    )

    print(f"Processing {len(gdf)} polygons...")

    all_polygon_data = []
    for index, polygon in gdf.iterrows():
        print(
            f"Processing polygon {index + 1}/{len(gdf)}: {polygon.get('Name', 'Unknown')}"
        )
        polygon_data = process_polygon(polygon)
        all_polygon_data.append(polygon_data)

    df = pd.DataFrame(all_polygon_data)

    df.columns = [col.replace("SR_HARMONIZED", "Sentinel") for col in df.columns]
    df.columns = [col.replace("GRD", "SAR") for col in df.columns]
    df.columns = [col.replace("ERA5_LAND", "ECMWF") for col in df.columns]

    # Move Shape_Name to the first column
    cols = df.columns.tolist()
    cols = ["Shape_Name"] + [col for col in cols if col != "Shape_Name"]
    df = df[cols]

    df.to_csv("extracted_data_all_polygons_5months.csv", index=False)
    print("Data saved to extracted_data_all_polygons_5months.csv")

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
