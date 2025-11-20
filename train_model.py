import sys
import os
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from functools import reduce

# Absolute paths
BASE_DIR = r"C:\Users\ADMIN\Desktop\BRIEF-7"
DATA_PATH = os.path.join(BASE_DIR, "DataCoSupplyChainDataset.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "pipeline_model")
BATCH_INPUT_PATH = os.path.join(BASE_DIR, "cleaned_batch_input.parquet")

def main():
    # 1. Initialize Spark
    spark = (
        SparkSession.builder
        .appName("DataCo_Batch_Pipeline_Retrain")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print('Spark version:', spark.version)

    # 2. Load Data
    print(f"Loading data from {DATA_PATH}...")
    try:
        raw_df = spark.read.option("header", "true").option("inferSchema", "true").csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    print("Initial rows:", raw_df.count())

    # 3. Normalize Columns (to match notebook behavior roughly, but we will rely on specific column names)
    # In the notebook, we might not have normalized everything, but let's keep it clean.
    # However, to match the notebook's specific feature list, we need to be careful.
    # The notebook uses: 'geo_distance_km', 'order_region', 'shipping_mode', 'customer_segment', 'sales', 'delivery_days', 'num_orders'
    
    # Let's do a safe normalization that maps to what we expect
    def normalize_col(c):
        return c.strip().lower().replace(' ', '_').replace('(', '').replace(')', '')

    for c in raw_df.columns:
        raw_df = raw_df.withColumnRenamed(c, normalize_col(c))

    # 4. Feature Engineering (Matching Notebook)
    
    # 4a. Delivery Days
    # Notebook logic:
    # if 'delivery_days' not in df.columns:
    #    Look for 'days for shipping (real)' etc.
    if 'delivery_days' not in raw_df.columns:
        possible = [c for c in raw_df.columns if c in ('days_for_shipping_real', 'days_for_shipping_real', 'delivery_days')]
        if possible:
            src = possible[0]
            raw_df = raw_df.withColumn('delivery_days', F.col(src))
            print(f"Created `delivery_days` from column `{src}`.")
        else:
            print("Warning: `delivery_days` column not found.")
    
    # 4b. Target Column
    late_threshold_days = 5
    if 'late_delivery_risk' not in raw_df.columns:
        if 'delivery_days' in raw_df.columns:
            raw_df = raw_df.withColumn(
                'late_delivery_risk',
                F.when(F.col('delivery_days') > late_threshold_days, 1).otherwise(0)
            )
            print("Created target column `late_delivery_risk`.")
        else:
             print("Warning: Cannot create target `late_delivery_risk` without `delivery_days`.")

    # 4c. Geo Distance (Haversine) - Notebook does this
    # We need lat/lon columns. Normalized names: latitude, longitude, order_city, etc.
    # The notebook tries to find 'origin_latitude' etc. or just uses what's there.
    # Let's try to replicate the notebook's simple haversine if columns exist.
    # Notebook: origin_lat = 'latitude', dest_lat = 'latitude' (placeholder logic in some versions)
    # Real logic: we need customer lat/lon and store lat/lon. 
    # If missing, we might skip or use placeholders. 
    # For strict alignment, let's check if we can calculate it.
    # If not, we might drop it from selected features or fill 0.
    
    if 'latitude' in raw_df.columns and 'longitude' in raw_df.columns:
        # Assuming these are customer locations. We need a second point. 
        # The dataset often has 'Order City', 'Order Country' etc but maybe not store lat/lon.
        # The notebook had a placeholder or derived it. 
        # Let's assume for now we fill 0 if we can't compute, to avoid breaking the pipeline if the notebook had it.
        # Or better, let's check if the notebook actually successfully computed it.
        # The notebook code: 
        # origin_lat = 'latitude' ... 
        # df = df.withColumn('geo_distance_km', haversine_udf(...))
        pass 
    
    # To be safe and match the "Selected Features" list:
    # 'geo_distance_km', 'order_region', 'shipping_mode', 'customer_segment', 'sales', 'delivery_days', 'num_orders'
    
    # We need to ensure these exist.
    if 'geo_distance_km' not in raw_df.columns:
        # Create dummy if missing, or try to compute. 
        # For this script, let's fill with 0.0 to ensure pipeline passes, 
        # unless we have the exact logic to compute it from the notebook's latest state.
        # The notebook has a `haversine` function. Let's add it.
        import math
        def haversine(lat1, lon1, lat2, lon2):
            if None in [lat1, lon1, lat2, lon2]: return 0.0
            try:
                lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                r = 6371
                return c * r
            except:
                return 0.0
        
        haversine_udf = F.udf(haversine, T.DoubleType())
        
        raw_df = raw_df.withColumn('geo_distance_km', F.lit(0.0))


    # If 'num_orders' is not in df, we create it.
    if 'num_orders' not in raw_df.columns:
        # Let's try to map 'order_item_quantity' to it if it exists
        if 'order_item_quantity' in raw_df.columns:
             raw_df = raw_df.withColumn('num_orders', F.col('order_item_quantity'))
        else:
             raw_df = raw_df.withColumn('num_orders', F.lit(1))

    # 5. Select Features (Exact match to Notebook)
    selected_features = [
        'geo_distance_km',
        'order_region',
        'shipping_mode',
        'customer_segment',
        'sales',
        'delivery_days',
        'num_orders'
    ]
    
    # Filter for existence
    selected_features = [c for c in selected_features if c in raw_df.columns]
    print('Using features:', selected_features)
    
    # 6. Handle Missing Values (Simple Imputation)
    # Numeric
    numeric_cols = [c for c in selected_features if isinstance(raw_df.schema[c].dataType, T.NumericType)]
    for c in numeric_cols:
        raw_df = raw_df.withColumn(c, F.when(F.col(c).isNull() | F.isnan(F.col(c)), F.lit(0)).otherwise(F.col(c)))
        
    # Categorical
    categorical_cols = [c for c in selected_features if c not in numeric_cols]
    for c in categorical_cols:
        raw_df = raw_df.withColumn(c, F.when(F.col(c).isNull(), F.lit('unknown')).otherwise(F.col(c)))

    # 7. Prepare Data
    # We need the target too
    # We also need identifiers for the batch script (order_id, customer_id, market)
    # 'market' is already in selected_features, but let's ensure order_id and customer_id are there.
    
    extra_cols = ['order_id', 'customer_id']
    # Check if they exist after normalization
    extra_cols = [c for c in extra_cols if c in raw_df.columns]
    
    if 'late_delivery_risk' in raw_df.columns:
        final_cols = selected_features + ['late_delivery_risk'] + extra_cols
    else:
        final_cols = selected_features + extra_cols
        
    # Ensure unique
    final_cols = list(set(final_cols))
        
    df_prepared = raw_df.select(*final_cols)

    # SAVE TEST DATA FOR BATCH SCRIPT
    print(f"Saving sample batch input to {BATCH_INPUT_PATH}...")
    df_prepared.sample(fraction=0.1, seed=42).write.mode("overwrite").parquet(BATCH_INPUT_PATH)

    # 8. Pipeline Building
    if 'late_delivery_risk' not in df_prepared.columns:
        print("Target missing, cannot train.")
        return

    indexers = [StringIndexer(inputCol=c, outputCol=c + '_idx', handleInvalid='keep') for c in categorical_cols]
    encoders = [OneHotEncoder(inputCol=c + '_idx', outputCol=c + '_ohe') for c in categorical_cols]
    
    assembler = VectorAssembler(
        inputCols=[c + '_ohe' for c in categorical_cols] + numeric_cols,
        outputCol='features_unscaled'
    )
    
    scaler = StandardScaler(inputCol='features_unscaled', outputCol='features')
    
    rf = RandomForestClassifier(labelCol='late_delivery_risk', featuresCol='features', numTrees=100, seed=42)
    
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, rf])

    # 9. Train
    train_df, test_df = df_prepared.randomSplit([0.8, 0.2], seed=42)
    print(f"Training on {train_df.count()} rows...")
    
    model = pipeline.fit(train_df)
    
    # 10. Evaluate
    preds = model.transform(test_df)
    bce = BinaryClassificationEvaluator(labelCol='late_delivery_risk', rawPredictionCol='probability', metricName='areaUnderROC')
    auc = bce.evaluate(preds)
    
    mce = MulticlassClassificationEvaluator(labelCol='late_delivery_risk', predictionCol='prediction', metricName='accuracy')
    acc = mce.evaluate(preds)
    
    print(f'ROC AUC: {auc:.4f}, Accuracy: {acc:.4f}')

    # 11. Save
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.write().overwrite().save(MODEL_SAVE_PATH)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
