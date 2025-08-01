from pyproj import Transformer
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box, Polygon

def make_5km_square(lat, lon):
    # Step 1: Create a GeoDataFrame with the point
    gdf = gpd.GeoDataFrame(
        {"name": ["center_point"], "geometry": [Point(lon, lat)]},
        crs="EPSG:4326"
    )

    # Step 2: Estimate UTM CRS from point
    utm_crs = gdf.estimate_utm_crs()

    # Step 3: Project to UTM
    gdf_utm = gdf.to_crs(utm_crs)
    x, y = gdf_utm.geometry.iloc[0].x, gdf_utm.geometry.iloc[0].y

    # Step 4: Create 5km x 5km square (centered)
    half_side = 2500  # meters
    square_utm = box(x - half_side, y - half_side, x + half_side, y + half_side)

    # Step 5: Convert square back to EPSG:4326
    square_geo = gpd.GeoSeries([square_utm], crs=utm_crs).to_crs("EPSG:4326").iloc[0]

    return square_geo

# Example usage
df_points = pd.read_csv("test_filtered.csv")
wkts = []
for i in range(len(df_points)):
    row = df_points.iloc[i]
    lat = row['lat']
    lon = row['lon']
    center_point = Point(lon, lat)
    square_polygon = make_5km_square(lat, lon)
    wkts.append(square_polygon.wkt)

df_points["geometry"] = wkts
df_points.to_csv("dataset_with_aoi.csv", index=False)