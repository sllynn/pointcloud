# Databricks notebook source
# MAGIC %md # NB-01: Create the LiDAR Dataset
# MAGIC
# MAGIC > Download a grid reference chips of LiDAR Point Cloud LAZ data from Ordnance Survey in UK onto DBFS and then split the files into smaller files.
# MAGIC
# MAGIC #### Databricks Author(s)
# MAGIC * __Original: Stuart Lynn (On Sabbatical)__
# MAGIC * __Maintainer: Michael Johns | <mjohns@databricks.com>__
# MAGIC ---
# MAGIC _Last Modified: 26 JUL 2022_

# COMMAND ----------

# MAGIC %pip install "pdal==3.0.2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import shutil

import pandas as pd

from subprocess import run
from pdal import Pipeline
from glob import glob

from pyspark import Row
from pyspark.sql.types import *
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md ## Data Sourcing

# COMMAND ----------

# MAGIC %md
# MAGIC Download a set of OS grid reference chips of LiDAR Point Cloud data.

# COMMAND ----------

grid_refs = ["TQ7080", "TQ7085", "TQ7580", "TQ7585", "TQ8080", "TQ8085"]

urls = [
  f"https://api.agrimetrics.co.uk/tiles/collections/survey/lidar_point_cloud/2021/NaN/{grid_ref}?subscription-key=public" 
  for grid_ref in grid_refs
  ]
  
local_uris = [f"/tmp/lidar_point_cloud-2021-{grid_ref}.zip" for grid_ref in grid_refs]

for url, local_uri in zip(urls, local_uris):
  if not os.path.exists(local_uri):
    output_wget = run(['wget', '-O', local_uri, url], capture_output=False)
    print(output_wget)
  else:
    print(f"skipping download for {url}")
  output_unzip = run(['unzip', '-o', '-d', '/tmp', local_uri])
  print(output_unzip)

# COMMAND ----------

# MAGIC %sh pdal info /tmp/TQ8284_P_12457_20211105_20211108.laz

# COMMAND ----------

# MAGIC %md __Move these downloaded LAZ files to DBFS__

# COMMAND ----------

dst_root = "/home/stuart@databricks.com/datasets/lidar/raw"

local_files_df = (
  spark.createDataFrame(dbutils.fs.ls("file:/tmp/"))
  .where(F.col("name").endswith(".laz"))
)
local_files = local_files_df.select("path").toLocalIterator()
for rw in local_files:
  src = rw.path
  dst = rw.path.replace("file:/tmp", dst_root)
  dbutils.fs.cp(src, dst, True)

# COMMAND ----------

display(dbutils.fs.ls(dst_root))

# COMMAND ----------

# MAGIC %md ## Pre-processing
# MAGIC For demo purposes, we can divide these files into many smaller parts to facilitate parallel processing.

# COMMAND ----------

def dbfs_to_local(path: str) -> str:
  return f"dbfs{path}" if path[0] == "/" else f"dbfs/{path}"

def local_to_dbfs(path: str) -> str:
  return path.split("/dbfs")[-1]

# COMMAND ----------

# MAGIC %md __A function that will split an input LAZ file into multiple smaller files (based on a path supplied in a Pandas DataFrame).__

# COMMAND ----------

def split_laz(pdf: pd.DataFrame) -> pd.DataFrame:
  in_path = pdf.loc[0, "input_uri"]
  output_path_local = pdf.loc[0, "output_path"]
  output_filename_stem = os.path.splitext(os.path.basename(in_path))[0]
  
  os.makedirs(output_path_local, exist_ok=True)
  os.makedirs(f"/tmp/{output_filename_stem}", exist_ok=True)
  
#{"type":"filters.divider", "count": 10},

  params = {
    "pipeline": [
      {"type": "readers.las", "filename": in_path},
      {"type":"filters.chipper", "capacity": "10000"},
      {
        "type":"writers.las", 
        "compression": "zip",
        "filename": f"/tmp/{output_filename_stem}/{output_filename_stem}_#.laz"
      }
    ]}
  pipeline = Pipeline(json.dumps(params))
  pipeline.execute()
  for filename in glob(os.path.join(f"/tmp/{output_filename_stem}", '*.laz')):
    shutil.copy(filename, output_path_local)
  return pd.DataFrame([pdf["input_uri"], pd.Series(["OK"])])

# COMMAND ----------

# MAGIC %md __Create my 'target' dataframe, with paths to files to original LAZ files.__

# COMMAND ----------

lidar_inputs = glob("/dbfs/home/stuart@databricks.com/datasets/lidar/raw/*.laz")
output_dir = "/dbfs/home/stuart@databricks.com/datasets/lidar/"

lidar_inputs_sdf = (
  spark.createDataFrame(
    [Row(pth, output_dir) for pth in lidar_inputs], 
    schema=StructType([
      StructField("input_uri", StringType()),
      StructField("output_path", StringType())
    ])
  )
)

lidar_inputs_sdf.display()

# COMMAND ----------

# MAGIC %md __GroupBy + Apply execution of the function, in parallel across the cluster.__
# MAGIC
# MAGIC > Uses UDF `split_laz`

# COMMAND ----------

spark.conf.set("spark.sql.adaptive.enabled", "false")

# COMMAND ----------

split_results = (
  lidar_inputs_sdf
  .groupBy("input_uri", "output_path")
  .applyInPandas(split_laz, schema="result string")
  )

split_results.cache()

split_results.write.format("noop").mode("overwrite").save()
