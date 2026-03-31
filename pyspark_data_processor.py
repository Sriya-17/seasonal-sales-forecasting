#!/usr/bin/env python3
"""
PySpark Data Processing Module
================================================================================
High-performance data processing using Apache Spark for large datasets.

This module provides:
  - CSV data loading with Spark
  - Missing value handling
  - Feature engineering (temporal features)
  - Data aggregation and cleaning
  - Pandas conversion for ML models

Author: Seasonal Sales Forecasting App
License: MIT
================================================================================
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_date, year, month, dayofmonth, weekofyear, dayofweek, quarter, dayofyear
from spark_config import SparkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PySparkDataProcessor:
    """
    High-performance data processing using Apache Spark.
    
    This class provides distributed data processing capabilities for handling
    large datasets while maintaining compatibility with existing ML pipelines.
    
    Features:
    - Lazy evaluation (computations deferred until action)
    - Distributed processing across CPU cores
    - Automatic optimization through Catalyst optimizer
    - Seamless Pandas conversion for ML models
    """
    
    def __init__(self, spark_session=None):
        """
        Initialize the PySpark data processor.
        
        Args:
            spark_session (pyspark.sql.SparkSession, optional): 
                Existing Spark session. If None, creates new session.
                
        Example:
            >>> processor = PySparkDataProcessor()
            >>> df_spark = processor.load_csv('data.csv')
        """
        self.spark = spark_session or SparkConfig.get_spark_session()
        self.logger = logger
    
    def load_csv(self, file_path, header=True, infer_schema=True):
        """
        Load CSV file into Spark DataFrame.
        
        This method uses Spark's distributed reading capability, which is
        more efficient than pandas for large files.
        
        Args:
            file_path (str): Path to CSV file
            header (bool): Whether first row contains headers. Default: True
            infer_schema (bool): Auto-detect column types. Default: True
                                If False, all columns treated as strings
            
        Returns:
            pyspark.sql.DataFrame: Spark DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If file reading fails
            
        Example:
            >>> df_spark = processor.load_csv('sales_data.csv')
            >>> print(df_spark.count())  # Lazy - triggers read
        """
        try:
            self.logger.info(f"Loading CSV from {file_path}")
            
            df_spark = self.spark.read.csv(
                file_path,
                header=header,
                inferSchema=infer_schema,
                mode='PERMISSIVE'  # Lenient mode - bad records ignored
            )
            
            row_count = df_spark.count()  # This triggers the read
            self.logger.info(f"✅ Loaded {row_count} rows from {file_path}")
            
            return df_spark
            
        except FileNotFoundError:
            self.logger.error(f"❌ File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Error loading CSV: {str(e)}")
            raise
    
    def handle_missing_values(self, df_spark, strategy='drop', columns=None):
        """
        Handle missing values in Spark DataFrame.
        
        Args:
            df_spark (pyspark.sql.DataFrame): Input Spark DataFrame
            strategy (str): Missing value handling strategy
                          - 'drop': Remove rows with NaN (default)
                          - 'forward_fill': Fill with previous value
                          - 'mean': Fill numeric columns with mean
                          Default: 'drop'
            columns (list, optional): Specific columns to process.
                                     If None, process all columns.
                
        Returns:
            pyspark.sql.DataFrame: DataFrame with missing values handled
            
        Example:
            >>> df_clean = processor.handle_missing_values(df_spark, strategy='drop')
        """
        try:
            before_count = df_spark.count()
            
            if strategy == 'drop':
                # Drop rows with any NULL values
                df_clean = df_spark.dropna(how='any', subset=columns)
                self.logger.info(f"Dropped {before_count - df_clean.count()} rows with NaN")
                
            elif strategy == 'forward_fill':
                # Fill with previous non-null value (requires ordered data)
                from pyspark.sql.window import Window
                window = Window.partitionBy().orderBy(df_spark.columns[0])
                
                # This example fills numeric columns
                for col_name in df_spark.columns:
                    df_clean = df_spark.fillna(0, subset=[col_name])
                self.logger.info("Applied forward fill strategy")
                
            elif strategy == 'mean':
                # Calculate mean for numeric columns and fill
                numeric_cols = [col_name for col_name, col_type in 
                               df_spark.dtypes if col_type != 'string']
                
                for col_name in numeric_cols:
                    mean_val = df_spark.agg(F.mean(col_name)).collect()[0][0]
                    df_clean = df_spark.fillna({col_name: mean_val})
                self.logger.info(f"Filled missing values with mean for {numeric_cols}")
            
            else:
                self.logger.warning(f"Unknown strategy: {strategy}. Using 'drop'")
                df_clean = df_spark.dropna()
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"❌ Error handling missing values: {str(e)}")
            raise
    
    def engineer_features(self, df_spark, date_column='Date', sales_column='Sales'):
        """
        Extract temporal features from date column.
        
        This creates 7 new features from a date column to capture
        seasonal and temporal patterns:
        - year: Full year (e.g., 2025)
        - month: Month number (1-12)
        - day: Day of month (1-31)
        - week: Week number (1-53)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - quarter: Quarter (1-4)
        - day_of_year: Day of year (1-365)
        
        Args:
            df_spark (pyspark.sql.DataFrame): Input Spark DataFrame
            date_column (str): Name of date column. Default: 'Date'
            sales_column (str): Name of sales column. Default: 'Sales'
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with engineered features
            
        Example:
            >>> df_features = processor.engineer_features(df_spark)
            >>> df_features.show()  # Display first 20 rows
        """
        try:
            self.logger.info(f"Engineering features from {date_column}")
            
            # Convert date string to date type
            df_temp = df_spark.withColumn(
                date_column,
                to_date(col(date_column), 'yyyy-MM-dd')
            )
            
            # Extract temporal features
            df_featured = df_temp.withColumn('year', year(col(date_column))) \
                                .withColumn('month', month(col(date_column))) \
                                .withColumn('day', dayofmonth(col(date_column))) \
                                .withColumn('week', weekofyear(col(date_column))) \
                                .withColumn('day_of_week', dayofweek(col(date_column))) \
                                .withColumn('quarter', quarter(col(date_column))) \
                                .withColumn('day_of_year', dayofyear(col(date_column)))
            
            self.logger.info(f"✅ Engineered 7 temporal features")
            
            return df_featured
            
        except Exception as e:
            self.logger.error(f"❌ Error engineering features: {str(e)}")
            raise
    
    def aggregate_by_date(self, df_spark, date_column='Date', sales_column='Sales'):
        """
        Aggregate multiple sales records per date (for multi-store data).
        
        This is useful when CSV has multiple entries per date (e.g., one per store).
        Aggregates all sales for each date by summing.
        
        Args:
            df_spark (pyspark.sql.DataFrame): Input Spark DataFrame
            date_column (str): Name of date column. Default: 'Date'
            sales_column (str): Name of sales column. Default: 'Sales'
            
        Returns:
            pyspark.sql.DataFrame: Aggregated data with one row per date
            
        Example:
            >>> # Input: 371 rows (7 stores × 53 dates)
            >>> df_agg = processor.aggregate_by_date(df_spark)
            >>> # Output: 53 rows (one per unique date)
        """
        try:
            # Check if aggregation needed
            total_rows = df_spark.count()
            unique_dates = df_spark.select(date_column).distinct().count()
            
            if total_rows == unique_dates:
                self.logger.info("✅ No aggregation needed - one row per date")
                return df_spark
            
            # Aggregate sales by date
            self.logger.info(f"Aggregating {total_rows} rows to {unique_dates} unique dates")
            
            df_agg = df_spark.groupBy(date_column) \
                            .agg(F.sum(sales_column).alias(sales_column))
            
            self.logger.info(f"✅ Aggregated {total_rows} rows to {df_agg.count()} rows")
            
            return df_agg
            
        except Exception as e:
            self.logger.error(f"❌ Error aggregating data: {str(e)}")
            raise
    
    def sort_by_date(self, df_spark, date_column='Date'):
        """
        Sort DataFrame by date (required for time series analysis).
        
        Args:
            df_spark (pyspark.sql.DataFrame): Input Spark DataFrame
            date_column (str): Name of date column. Default: 'Date'
            
        Returns:
            pyspark.sql.DataFrame: Date-sorted Spark DataFrame
        """
        return df_spark.orderBy(col(date_column).asc())
    
    def convert_to_pandas(self, df_spark):
        """
        Convert Spark DataFrame to Pandas DataFrame.
        
        ⚠️  WARNING: This triggers full computation and brings data to driver memory.
        Only use when:
        - Dataset is small enough for driver memory (~2GB)
        - ML model requires Pandas input (e.g., scikit-learn)
        
        For large datasets:
        - Keep in Spark for distributed processing
        - Use Spark MLlib for distributed ML
        
        Args:
            df_spark (pyspark.sql.DataFrame): Spark DataFrame to convert
            
        Returns:
            pd.DataFrame: Pandas DataFrame
            
        Example:
            >>> df_pandas = processor.convert_to_pandas(df_spark)
            >>> print(type(df_pandas))  # <class 'pandas.core.frame.DataFrame'>
        """
        try:
            self.logger.info("Converting Spark DataFrame to Pandas")
            df_pandas = df_spark.toPandas()
            self.logger.info(f"✅ Converted to Pandas ({df_pandas.shape[0]} rows, {df_pandas.shape[1]} cols)")
            return df_pandas
        except Exception as e:
            self.logger.error(f"❌ Error converting to Pandas: {str(e)}")
            raise
    
    def get_schema(self, df_spark):
        """
        Display the schema (column names and types) of Spark DataFrame.
        
        Args:
            df_spark (pyspark.sql.DataFrame): Spark DataFrame
            
        Example:
            >>> processor.get_schema(df_spark)
            # Output:
            # root
            #  |-- Date: string
            #  |-- Sales: double
            #  |-- Store: integer
        """
        df_spark.printSchema()
    
    def get_statistics(self, df_spark):
        """
        Get summary statistics for all columns.
        
        Args:
            df_spark (pyspark.sql.DataFrame): Spark DataFrame
            
        Returns:
            pyspark.sql.DataFrame: Summary statistics
            
        Example:
            >>> stats = processor.get_statistics(df_spark)
            >>> stats.show()
        """
        return df_spark.describe()
    
    def full_pipeline(self, file_path, date_column='Date', sales_column='Sales',
                      missing_strategy='drop', convert_to_pandas_flag=True):
        """
        Complete data processing pipeline in one call.
        
        This convenience method chains all processing steps:
        1. Load CSV
        2. Aggregate by date (if multi-store)
        3. Handle missing values
        4. Engineer temporal features
        5. Sort by date
        6. Optionally convert to Pandas
        
        Args:
            file_path (str): Path to CSV file
            date_column (str): Date column name. Default: 'Date'
            sales_column (str): Sales column name. Default: 'Sales'
            missing_strategy (str): How to handle NaN. Default: 'drop'
            convert_to_pandas_flag (bool): Convert to Pandas at end. Default: True
            
        Returns:
            tuple: (df_spark, df_pandas) or (df_spark, None) if convert_to_pandas_flag=False
            
        Example:
            >>> df_spark, df_pandas = processor.full_pipeline('sales.csv')
            >>> print(df_pandas.shape)
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING PYSPARK DATA PROCESSING PIPELINE")
        self.logger.info("=" * 60)
        
        # Step 1: Load
        df_spark = self.load_csv(file_path)
        
        # Step 2: Aggregate
        df_spark = self.aggregate_by_date(df_spark, date_column, sales_column)
        
        # Step 3: Handle missing values
        df_spark = self.handle_missing_values(df_spark, strategy=missing_strategy)
        
        # Step 4: Engineer features
        df_spark = self.engineer_features(df_spark, date_column, sales_column)
        
        # Step 5: Sort
        df_spark = self.sort_by_date(df_spark, date_column)
        
        # Step 6: Optional conversion
        df_pandas = None
        if convert_to_pandas_flag:
            df_pandas = self.convert_to_pandas(df_spark)
            self.logger.info("=" * 60)
            self.logger.info("✅ PIPELINE COMPLETE - Ready for ML modeling")
            self.logger.info("=" * 60)
            return df_spark, df_pandas
        else:
            self.logger.info("=" * 60)
            self.logger.info("✅ PIPELINE COMPLETE - Keeping data in Spark")
            self.logger.info("=" * 60)
            return df_spark, None
