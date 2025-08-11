# Define catalog, schema, volume, file and table names
catalog = "workspace"
schema = "schema_iris"
volume = "volume_iris"
file_name = "Iris.csv"
table_name = "table_iris"

# Define paths
path_volume = f"/Volumes/{catalog}/{schema}/{volume}"
path_table = f"{catalog}.{schema}.{table_name}"

# Read CSV from catalog volume
df = spark.read.csv(f"{path_volume}/{file_name}", header=True, inferSchema=True, sep=",")

# Save DataFrame to Unity Catalog table as Delta format (overwrites if table exists)
df.write.format("delta").mode("overwrite").saveAsTable(path_table)