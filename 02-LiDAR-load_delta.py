# Databricks notebook source
# MAGIC %md # NB-02: Load LiDAR LAZ files into Delta Lake
# MAGIC
# MAGIC > Use PDAL within vectorized Pandas UDF (via [Spark 3.x applyInPandas](https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html)) to process the LAZ files containing point cloud information into Delta, __resulting in 220M points (rows).__
# MAGIC
# MAGIC #### Databricks Author(s)
# MAGIC * __Original: Stuart Lynn (On Sabbatical)__
# MAGIC * __Maintainer: Michael Johns | <mjohns@databricks.com>__
# MAGIC ---
# MAGIC _Last Modified: 26 JUL 2022_

# COMMAND ----------

# MAGIC %pip install "pdal==3.0.2"

# COMMAND ----------

import json

import pandas as pd

from pdal import Pipeline
from glob import glob

from pyspark import Row
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md ### Start from DataFrame with LAZ paths only
# MAGIC
# MAGIC > Create Spark DataFrame `lidar_inputs_sdf`, with paths to files to the already split LAZ files

# COMMAND ----------

lidar_inputs = glob("/dbfs/home/stuart@databricks.com/datasets/lidar/*.laz")

lidar_inputs_sdf = (
  spark.createDataFrame(
    [Row(pth) for pth in lidar_inputs], 
    schema=StructType([StructField("path", StringType())])
  )
)

lidar_inputs_sdf.display()

# COMMAND ----------

# MAGIC %md ### Parse LAZ Logic
# MAGIC
# MAGIC > Function to extract raw data as a numpy array and return in a Pandas DF (for use in the following `applyInPandas()` grouped execution).

# COMMAND ----------

def read_laz(pdf: pd.DataFrame) -> pd.DataFrame:
  
  # extract filename for PDAL
  fl = pdf.loc[0, "path"]
  
  # PDAL pipeline, simplest case
  params = {"pipeline": [{"type": "readers.las", "filename": fl}]}
  
  pipeline = Pipeline(json.dumps(params))
  
  # get points + sensor parameters from file and generate Pandas dataframe
  arr_size = pipeline.execute()
  arr = pipeline.arrays[0]
  output_pdf = pd.DataFrame(arr)
  
  # grab the file metadata
  metadata = json.loads(pipeline.metadata)["metadata"]["readers.las"]
  
  # parse out useful metadata and store inline with points
  output_pdf["_minXYZ"] = json.dumps({"minX": metadata["minx"], "minY": metadata["miny"], "minZ":  metadata["minx"]})
  output_pdf["_maxXYZ"] = json.dumps({"maxX": metadata["maxx"], "maxY": metadata["maxy"], "maxZ":  metadata["maxx"]})
  output_pdf["_horizontalCRS"] = json.dumps(metadata["srs"]["horizontal"])
  output_pdf["_verticalCRS"] = json.dumps(metadata["srs"]["vertical"])
  output_pdf["_file_path"] = fl
  
  # return dataframe
  return output_pdf

# COMMAND ----------

# MAGIC %md ### Parallel Processing of LAZ
# MAGIC
# MAGIC > GroupBy `path` + Apply execution of the `read_laz` function in parallel across the cluster.

# COMMAND ----------

lidar_schema = StructType([
  StructField("X", FloatType()),
  StructField("Y", FloatType()),
  StructField("Z", FloatType()),
  StructField("Intensity", IntegerType()),
  StructField("ReturnNumber", IntegerType()),
  StructField("NumberOfReturns", IntegerType()),
  StructField("ScanDirectionFlag", IntegerType()),
  StructField("EdgeOfFlightLine", IntegerType()),
  StructField("Classification", IntegerType()),
  StructField("ScanAngleRank", FloatType()),
  StructField("UserData", IntegerType()),
  StructField("PointSourceId", IntegerType()),
  StructField("GpsTime", DoubleType()),
  StructField("Red", IntegerType()),
  StructField("Green", IntegerType()),
  StructField("Blue", IntegerType()),
  StructField("_minXYZ", StringType()),
  StructField("_maxXYZ", StringType()),
  StructField("_horizontalCRS", StringType()),
  StructField("_verticalCRS", StringType()),
  StructField("_file_path", StringType())
])

# Apply my python code to a Spark dataframe
lidar_data_sdf = (
  lidar_inputs_sdf
  .groupBy("path")
  .applyInPandas(read_laz, schema=lidar_schema)
)

# COMMAND ----------

# MAGIC %md ### Summary (So Far)
# MAGIC
# MAGIC > What's going on here?
# MAGIC
# MAGIC - Task and data are serialized and sent to worker nodes
# MAGIC - Workers allocate the task to a free executor
# MAGIC - Executor runs commands in Python interpreter, with data sent / returned between JVM and Python using the Apache Arrow standard.
# MAGIC - Spark's execution's model is lazy: nothing happens unless we call an action like 'write' on the DataFrame.

# COMMAND ----------

# MAGIC %md ### Write to Delta Lake
# MAGIC
# MAGIC > This will materialize the execution plan that we have built up thus far onto (cheap) Cloud Object Storage -- [S3](https://aws.amazon.com/s3/) (AWS), [ADLS (Gen2)](https://docs.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction) (Azure), [GCS](https://cloud.google.com/storage) (Google) with all the performance benefits of [Delta Lake](https://databricks.com/product/delta-lake-on-databricks) on Databricks.

# COMMAND ----------

# MAGIC %sql
# MAGIC create schema if not exists stuart.lidar;
# MAGIC drop table if exists stuart.lidar.raw

# COMMAND ----------

spark.conf.set("spark.sql.adaptive.enabled", "false")

# COMMAND ----------

lidar_data_sdf.write.format("delta").mode("overwrite").saveAsTable("stuart.lidar.raw")

# COMMAND ----------

# MAGIC %md __167M Points (Rows)__

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) as points from stuart.lidar.raw

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from stuart.lidar.raw

# COMMAND ----------

# MAGIC %md ### Optimize Layout with Z-Ordering
# MAGIC
# MAGIC > Multi-Dimensional clustering for columns (`X`,`Y`,`Z`) -- [Z-Ordering](https://docs.databricks.com/delta/optimizations/file-mgmt.html#z-ordering-multi-dimensional-clustering) is a technique to colocate related information in the same set of files. This co-locality is automatically used by Delta Lake on Databricks data-skipping algorithms. This behavior dramatically reduces the amount of data that Delta Lake on Databricks needs to read. 

# COMMAND ----------

# MAGIC %sql
# MAGIC optimize stuart.lidar.raw zorder by (X, Y, Z)
