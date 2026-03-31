#!/usr/bin/env python3
"""
Apache Spark Configuration Module
================================================================================
Centralized configuration for PySpark session initialization and management.

This module handles:
  - Spark session creation
  - Configuration settings
  - Resource management
  - Error handling

Author: Seasonal Sales Forecasting App
License: MIT
================================================================================
"""

import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SparkConfig:
    """
    Manages Spark configuration and session lifecycle.
    
    This class encapsulates best practices for:
    - Creating and reusing Spark sessions
    - Configuring memory and compute resources
    - Managing data formats
    - Error handling and cleanup
    """
    
    _spark_session = None
    
    @staticmethod
    def get_spark_session(app_name="SalesForecasting", master="local[*]", 
                          executor_memory="4g", driver_memory="2g"):
        """
        Get or create a Spark session.
        
        This method implements a singleton pattern to reuse the same Spark
        session across the application, improving performance.
        
        Args:
            app_name (str): Application name for Spark UI. Default: "SalesForecasting"
            master (str): Spark cluster master URL. 
                         - "local[*]" = use all CPU cores locally (default)
                         - "local[4]" = use 4 cores
                         - "yarn" = YARN cluster
                         - "spark://host:port" = Standalone cluster
            executor_memory (str): Memory per executor. Default: "4g" (4 GB)
            driver_memory (str): Memory for driver. Default: "2g" (2 GB)
            
        Returns:
            pyspark.sql.SparkSession: Active Spark session
            
        Example:
            >>> spark = SparkConfig.get_spark_session()
            >>> df = spark.read.csv("data.csv", header=True, inferSchema=True)
        """
        if SparkConfig._spark_session is None:
            try:
                logger.info("Creating new Spark session...")
                SparkConfig._spark_session = (
                    SparkSession.builder
                    .appName(app_name)
                    .master(master)
                    .config("spark.driver.memory", driver_memory)
                    .config("spark.executor.memory", executor_memory)
                    .config("spark.sql.shuffle.partitions", "200")
                    .config("spark.sql.adaptive.enabled", "true")
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                    .getOrCreate()
                )
                
                # Suppress verbose logging
                SparkConfig._spark_session.sparkContext.setLogLevel("ERROR")
                logger.info("✅ Spark session created successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to create Spark session: {str(e)}")
                raise RuntimeError(f"Spark session initialization failed: {str(e)}")
        
        return SparkConfig._spark_session
    
    @staticmethod
    def stop_spark_session():
        """
        Stop the active Spark session and cleanup resources.
        
        Call this method during application shutdown to release resources.
        """
        if SparkConfig._spark_session is not None:
            try:
                SparkConfig._spark_session.stop()
                SparkConfig._spark_session = None
                logger.info("✅ Spark session stopped successfully")
            except Exception as e:
                logger.error(f"❌ Error stopping Spark session: {str(e)}")
    
    @staticmethod
    def get_schema_for_csv():
        """
        Define the expected schema for sales CSV files.
        
        This schema is used for strict type validation when loading CSVs.
        
        Returns:
            pyspark.sql.types.StructType: Schema definition
            
        Schema Columns:
            - Date (StringType): Date in YYYY-MM-DD format
            - Sales (DoubleType): Sales amount in dollars
            - Store (IntegerType, optional): Store ID for multi-store data
            - Weekly_Sales (DoubleType, optional): Alternative sales column
        """
        return StructType([
            StructField("Date", StringType(), True),
            StructField("Sales", DoubleType(), True),
            StructField("Store", IntegerType(), True),
            StructField("Weekly_Sales", DoubleType(), True),
        ])
    
    @staticmethod
    def get_system_info():
        """
        Get information about Spark cluster and system resources.
        
        Returns:
            dict: System information including:
                - master: Cluster master URL
                - app_name: Application name
                - spark_version: Spark version
                - executor_cores: Cores per executor
                - executor_memory: Memory per executor
        """
        spark = SparkConfig.get_spark_session()
        return {
            'master': spark.sparkContext.master,
            'app_name': spark.sparkContext.appName,
            'spark_version': spark.version,
            'executor_cores': spark.sparkContext.getConf().get('spark.executor.cores', 'Default'),
            'executor_memory': spark.sparkContext.getConf().get('spark.executor.memory', 'Default'),
            'driver_memory': spark.sparkContext.getConf().get('spark.driver.memory', 'Default'),
        }


# Configuration presets for different scenarios
SPARK_CONFIGS = {
    'local_development': {
        'master': 'local[*]',
        'driver_memory': '2g',
        'executor_memory': '4g',
    },
    'local_heavy': {
        'master': 'local[*]',
        'driver_memory': '4g',
        'executor_memory': '8g',
    },
    'yarn_cluster': {
        'master': 'yarn',
        'driver_memory': '4g',
        'executor_memory': '8g',
    },
    'standalone_cluster': {
        'master': 'spark://sparkmaster:7077',
        'driver_memory': '4g',
        'executor_memory': '8g',
    },
}


def initialize_spark(config_type='local_development'):
    """
    Initialize Spark with a predefined configuration.
    
    Args:
        config_type (str): Configuration preset to use.
                          Options: 'local_development', 'local_heavy', 'yarn_cluster', 'standalone_cluster'
                          Default: 'local_development'
    
    Returns:
        pyspark.sql.SparkSession: Initialized Spark session
        
    Example:
        >>> spark = initialize_spark('local_development')
    """
    if config_type not in SPARK_CONFIGS:
        logger.warning(f"Unknown config type: {config_type}. Using 'local_development'")
        config_type = 'local_development'
    
    config = SPARK_CONFIGS[config_type]
    return SparkConfig.get_spark_session(
        master=config['master'],
        driver_memory=config['driver_memory'],
        executor_memory=config['executor_memory']
    )
