# Databricks notebook source
# MAGIC %md # NB-03: Analyze + Render LiDAR Data
# MAGIC
# MAGIC > Analysis of point cloud data from a Delta table and example of rendering with Plotly.
# MAGIC
# MAGIC #### Databricks Author(s)
# MAGIC * __Original: Stuart Lynn (On Sabbatical)__
# MAGIC * __Maintainer: Michael Johns | <mjohns@databricks.com>__
# MAGIC ---
# MAGIC _Last Modified: 26 JUL 2022_

# COMMAND ----------

# MAGIC %md ## Install Libaries
# MAGIC
# MAGIC > Using [Notebook-scoped Python libraries](https://docs.databricks.com/libraries/notebooks-python-libraries.html#notebook-scoped-python-libraries) via `%pip` to install `shapely`, `pyproj`, `geopandas`, and `rtree` across the cluster. __Note: needs to be at the top of the notebook as it restarts the python interpreter.__

# COMMAND ----------

# MAGIC %pip install shapely pyproj geopandas rtree rasterio

# COMMAND ----------

spark.conf.set("spark.databricks.geo.st.enabled", "true")

# COMMAND ----------

import io

import numpy as np

from PIL import Image
from base64 import b64encode
from pyspark import Row
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# COMMAND ----------

import rasterio
from matplotlib import pyplot
from rasterio.plot import show

def plot_file(file_path):
  fig, ax = pyplot.subplots(1, figsize=(4, 4))

  with rasterio.open(file_path) as src:
    show(src, ax=ax)
    pyplot.show()

# COMMAND ----------

# MAGIC %md ## Content + Scale
# MAGIC
# MAGIC > A closer look at the data reveals __220M Points (rows)__.

# COMMAND ----------

lidar_raw_sdf = spark.table("stuart.lidar.raw")
lidar_raw_sdf.display()

# COMMAND ----------

lidar_raw_sdf.createOrReplaceTempView("lidar")

# COMMAND ----------

# MAGIC %sql
# MAGIC select min(x), max(x), min(y), max(y) from lidar group by true

# COMMAND ----------

print(f"""count: {lidar_raw_sdf.count():,}""")

# COMMAND ----------

# MAGIC %md ## Basic task 1: Query for a Subset of the Points
# MAGIC
# MAGIC > Want to work with a subset of points (~50K in this example) for visualizing within the notebook.

# COMMAND ----------

X_min = 578000
Y_min = 185400
h = 200
w = 200

lidar_3d_sdf = (
  lidar_raw_sdf
  .where(col("ReturnNumber") == 1)
  .where(col("X").between(X_min, X_min + w - 1))
  .where(col("Y").between(Y_min, Y_min + h - 1))
)

print(f"""count: {lidar_3d_sdf.count():,}""")

# COMMAND ----------

# MAGIC %md ## Basic task 2: Visualize Point Cloud

# COMMAND ----------

# MAGIC %md __Convert the subset of points to Pandas DataFrame using Spark's `toPandas()` built-in function.__

# COMMAND ----------

lidar_3d_pdf = lidar_3d_sdf.select("x", "y", "z").toPandas()

# COMMAND ----------

# MAGIC %md ### Visualize with Plotly 
# MAGIC
# MAGIC > Plotly [graph_objects](https://plotly.com/python-api-reference/generated/plotly.graph_objects.pointcloud.html) imported as `go`, supporting e.g. `scatter3d` and `pointcloud`

# COMMAND ----------

from plotly.offline import init_notebook_mode, plot
import plotly.graph_objs as go

# COMMAND ----------

# MAGIC %md __Provide `X`, `Y`, and `Z` series data to the scatter3d plot.__

# COMMAND ----------

init_notebook_mode(connected=True)

trace1 = go.Scatter3d(
  x=lidar_3d_pdf.x, y=lidar_3d_pdf.y, z=lidar_3d_pdf.z, mode='markers',  
  marker=dict(size=3, color=lidar_3d_pdf.z, colorscale='Viridis', opacity=1)
)

data = [trace1]
layout = go.Layout(
  autosize=True, width=1100, height=600,
  margin=dict(l=0, r=0, b=0, t=0),
  scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z"), aspectmode="data")
)

fig = go.Figure(data=data, layout=layout)
displayHTML(plot(fig, filename='3d-scatter-colorscale', output_type='div'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic task 3: Generate DEM
# MAGIC
# MAGIC > Using linear interpolation / triangulation
# MAGIC
# MAGIC __Using the following limits, results in ~1.7M points (rows):__
# MAGIC
# MAGIC ```
# MAGIC X_min = 578000
# MAGIC Y_min = 185400
# MAGIC h = 1000
# MAGIC w = 1000
# MAGIC ```

# COMMAND ----------

X_min = 578000
Y_min = 185400
h = 1000
w = 1000

points_sdf = (
  lidar_raw_sdf
  .where(col("X").between(X_min, X_min + w - 1))
  .where(col("Y").between(Y_min, Y_min + h - 1))
  .where(col("ReturnNumber") == 1)
  .select(col("Z").cast("double"), expr("st_aswkb(st_point(X, Y))").alias("geom"))
  )

# COMMAND ----------

print(f"""count: {points_sdf.count():,}""")

# COMMAND ----------

# MAGIC %sh rm /tmp/lidar_agg*

# COMMAND ----------

import osgeo.gdal as gdal
import osgeo.ogr as ogr
import osgeo.osr as osr
import osgeo.gdalconst as gdalconst

gdal.UseExceptions()
ogr.UseExceptions()

bng = osr.SpatialReference()
bng.ImportFromEPSG(27700)

source_ds = (
  gdal.GetDriverByName('Memory')
  .Create('mem', 0, 0, 0, gdal.GDT_Unknown)
  )

layer = source_ds.CreateLayer('points', bng, ogr.wkbPoint)
value_field_name = "Z"
layer.CreateField(ogr.FieldDefn(value_field_name, ogr.OFTReal))

point_geoms = points_sdf.toLocalIterator()  
for rw in point_geoms:
  feat = ogr.Feature(layer.GetLayerDefn())
  geom = ogr.CreateGeometryFromWkb(rw.geom)
  feat.SetGeometry(geom)
  feat.SetField(value_field_name, rw.Z)
  layer.CreateFeature(feat)

layer.SyncToDisk()
source_ds.FlushCache()

options = gdal.GridOptions(
  outputType=gdalconst.GDT_Float32,
  height=1000,
  width=1000,
  format="GTiff",
  algorithm="linear:radius=-1",
  zfield="Z"
  )

grid = gdal.Grid(
  destName="/tmp/lidar_agg.tif",
  srcDS=source_ds,
  options=options
)

grid.FlushCache()
del grid

# COMMAND ----------

# MAGIC %sh gdalinfo -stats /tmp/lidar_agg.tif

# COMMAND ----------

plot_file("/tmp/lidar_agg.tif")

# COMMAND ----------


