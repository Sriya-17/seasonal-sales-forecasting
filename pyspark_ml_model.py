#!/usr/bin/env python3
"""
PySpark ML Model Module
================================================================================
Machine Learning models using PySpark MLlib and scikit-learn integration.

Provides two options:
  Option 1: Spark MLlib Random Forest (distributed training)
  Option 2: scikit-learn Random Forest (from pandas data)

Author: Seasonal Sales Forecasting App
License: MIT
================================================================================
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

# PySpark imports
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor as SparkRandomForest
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# scikit-learn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PySparkMLModel:
    """
    Machine Learning using PySpark MLlib for distributed training.
    
    This class implements Random Forest using Spark's distributed ML library,
    which is beneficial for very large datasets that don't fit in memory.
    
    Features:
    - Distributed training across cluster
    - Automatic parallelization
    - Optimal for datasets > 1GB
    """
    
    def __init__(self, num_trees=100, max_depth=20, seed=42):
        """
        Initialize Spark ML Random Forest model.
        
        Args:
            num_trees (int): Number of trees in forest. Default: 100
            max_depth (int): Maximum tree depth. Default: 20
            seed (int): Random seed for reproducibility. Default: 42
        """
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.seed = seed
        self.model = None
        self.pipeline = None
        self.metrics = {}
        
    def prepare_features(self, df_spark, feature_columns, label_column='Sales'):
        """
        Prepare features for Spark ML using VectorAssembler.
        
        Spark ML requires features to be in a single vector column.
        This method creates that vector.
        
        Args:
            df_spark (pyspark.sql.DataFrame): Input Spark DataFrame
            feature_columns (list): Column names to use as features
            label_column (str): Target column name. Default: 'Sales'
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with 'features' vector column
            
        Example:
            >>> features = ['year', 'month', 'day', 'week']
            >>> df_prepared = model.prepare_features(df_spark, features)
        """
        try:
            # Remove any null rows
            df_clean = df_spark.dropna(subset=feature_columns + [label_column])
            
            # Create feature vector
            assembler = VectorAssembler(
                inputCols=feature_columns,
                outputCol='features'
            )
            df_assembled = assembler.transform(df_clean)
            
            logger.info(f"✅ Assembled {len(feature_columns)} features")
            
            return df_assembled.select('features', label_column)
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {str(e)}")
            raise
    
    def train(self, df_assembled, label_column='Sales', test_fraction=0.2):
        """
        Train Spark Random Forest model on distributed data.
        
        Args:
            df_assembled (pyspark.sql.DataFrame): DataFrame with 'features' column
            label_column (str): Target column name. Default: 'Sales'
            test_fraction (float): Fraction for test set. Default: 0.2 (80/20 split)
            
        Returns:
            dict: Training results with metrics
            
        Example:
            >>> results = model.train(df_assembled)
            >>> print(f"R² = {results['r2_score']}")
        """
        try:
            logger.info("Starting Spark Random Forest training...")
            
            # Split data
            train_df, test_df = df_assembled.randomSplit([1-test_fraction, test_fraction], 
                                                         seed=self.seed)
            
            logger.info(f"Training set: {train_df.count()} rows")
            logger.info(f"Test set: {test_df.count()} rows")
            
            # Create and train model
            rf_model = SparkRandomForest(
                numTrees=self.num_trees,
                maxDepth=self.max_depth,
                labelCol=label_column,
                featuresCol='features',
                seed=self.seed,
                subsamplingStrategy='auto'
            )
            
            self.model = rf_model.fit(train_df)
            logger.info("✅ Model training complete")
            
            # Evaluate
            evaluator = RegressionEvaluator(
                labelCol=label_column,
                predictionCol='prediction',
                metricName='r2'
            )
            
            # Make predictions
            train_predictions = self.model.transform(train_df)
            test_predictions = self.model.transform(test_df)
            
            # Calculate metrics
            train_r2 = evaluator.evaluate(train_predictions)
            test_r2 = evaluator.evaluate(test_predictions)
            
            # Calculate MAE and RMSE
            evaluator_mae = RegressionEvaluator(
                labelCol=label_column,
                predictionCol='prediction',
                metricName='mae'
            )
            evaluator_rmse = RegressionEvaluator(
                labelCol=label_column,
                predictionCol='prediction',
                metricName='rmse'
            )
            
            test_mae = evaluator_mae.evaluate(test_predictions)
            test_rmse = evaluator_rmse.evaluate(test_predictions)
            
            # Store metrics
            self.metrics = {
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'num_trees': self.num_trees,
                'max_depth': self.max_depth,
            }
            
            logger.info(f"📊 Train R²: {train_r2:.4f}")
            logger.info(f"📊 Test R²: {test_r2:.4f}")
            logger.info(f"💰 Test MAE: ${test_mae:.2f}")
            logger.info(f"📈 Test RMSE: ${test_rmse:.2f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"❌ Error training model: {str(e)}")
            raise
    
    def predict(self, df_assembled, label_column='Sales'):
        """
        Make predictions on new data.
        
        Args:
            df_assembled (pyspark.sql.DataFrame): DataFrame with 'features' column
            label_column (str): Actual values column (for comparison). Default: 'Sales'
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained first. Call train() method.")
        
        return self.model.transform(df_assembled)
    
    def get_feature_importance(self):
        """
        Get feature importance from trained model.
        
        Returns:
            dict: Feature importances (requires manual feature tracking)
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Note: Spark MLlib doesn't provide direct feature importance access
        # You need to track feature names separately
        return {"note": "Track feature names during prepare_features()"}
    
    def save_model(self, path):
        """
        Save trained model to disk.
        
        Args:
            path (str): Directory path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        self.model.save(path)
        logger.info(f"✅ Model saved to {path}")


class SklearnMLModel:
    """
    Machine Learning using scikit-learn Random Forest.
    
    This class wraps scikit-learn's Random Forest for use with Pandas data.
    Best for datasets that fit in memory (< 100GB).
    
    Features:
    - Lightning-fast training on single machine
    - Feature importance analysis
    - Model persistence
    - Extensive scikit-learn ecosystem compatibility
    """
    
    def __init__(self, n_estimators=100, max_depth=20, test_size=0.2, random_state=42):
        """
        Initialize scikit-learn Random Forest model.
        
        Args:
            n_estimators (int): Number of trees. Default: 100
            max_depth (int): Maximum tree depth. Default: 20
            test_size (float): Test set fraction. Default: 0.2
            random_state (int): Random seed. Default: 42
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.metrics = {}
        self.feature_importance_dict = {}
        
    def train(self, df_pandas, feature_columns, label_column='Sales'):
        """
        Train Random Forest on Pandas DataFrame.
        
        Args:
            df_pandas (pd.DataFrame): Pandas DataFrame with all data
            feature_columns (list): Column names for features
            label_column (str): Target column name. Default: 'Sales'
            
        Returns:
            dict: Training metrics
            
        Example:
            >>> features = ['year', 'month', 'day', 'week', 'day_of_week', 'quarter']
            >>> metrics = model.train(df_pandas, features)
        """
        try:
            logger.info("Starting scikit-learn Random Forest training...")
            
            self.feature_columns = feature_columns
            
            # Prepare data
            X = df_pandas[feature_columns].values
            y = df_pandas[label_column].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            logger.info(f"Training set: {X_train.shape[0]} rows")
            logger.info(f"Test set: {X_test.shape[0]} rows")
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,  # Use all CPU cores
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            self.model.fit(X_train, y_train)
            logger.info("✅ Model training complete")
            
            # Evaluate
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Store metrics
            self.metrics = {
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
            }
            
            # Feature importance
            for feat, importance in zip(feature_columns, self.model.feature_importances_):
                self.feature_importance_dict[feat] = float(importance)
            
            logger.info(f"📊 Train R²: {train_r2:.4f}")
            logger.info(f"📊 Test R²: {test_r2:.4f}")
            logger.info(f"💰 Test MAE: ${test_mae:.2f}")
            logger.info(f"📈 Test RMSE: ${test_rmse:.2f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"❌ Error training model: {str(e)}")
            raise
    
    def predict_future(self, future_dates_df):
        """
        Generate predictions for future dates.
        
        Args:
            future_dates_df (pd.DataFrame): DataFrame with future dates and engineered features
            
        Returns:
            pd.DataFrame: Original data + predictions
            
        Example:
            >>> future_preds = model.predict_future(future_df)
            >>> print(future_preds[['Date', 'Predicted_Sales']])
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        X_future = future_dates_df[self.feature_columns].values
        predictions = self.model.predict(X_future)
        
        result_df = future_dates_df.copy()
        result_df['Predicted_Sales'] = predictions
        
        return result_df
    
    def get_feature_importance(self):
        """
        Get feature importance from trained model.
        
        Returns:
            dict: Feature names with importance scores (0-1)
            
        Example:
            >>> importance = model.get_feature_importance()
            >>> print(importance)
            # {'month': 0.35, 'day_of_week': 0.28, ...}
        """
        return self.feature_importance_dict
    
    def save_model(self, file_path):
        """
        Save trained model to disk using joblib.
        
        Args:
            file_path (str): Path to save model file
            
        Example:
            >>> model.save_model('models/sales_predictor.joblib')
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.model, file_path)
        logger.info(f"✅ Model saved to {file_path}")
    
    def load_model(self, file_path):
        """
        Load previously trained model from disk.
        
        Args:
            file_path (str): Path to model file
            
        Example:
            >>> model = SklearnMLModel()
            >>> model.load_model('models/sales_predictor.joblib')
        """
        if os.path.exists(file_path):
            self.model = joblib.load(file_path)
            logger.info(f"✅ Model loaded from {file_path}")
        else:
            logger.error(f"❌ Model file not found: {file_path}")
            raise FileNotFoundError(f"Model file not found: {file_path}")
    
    def get_metrics(self):
        """Get training metrics."""
        return self.metrics


class HybridMLPipeline:
    """
    Unified interface for both Spark MLlib and sklearn models.
    
    This class allows seamless switching between Spark (for big data)
    and sklearn (for standard datasets).
    
    Features:
    - Single interface for both algorithms
    - Automatic selection based on data size
    - Easy comparison between approaches
    """
    
    def __init__(self, use_spark=True):
        """
        Initialize hybrid pipeline.
        
        Args:
            use_spark (bool): Use Spark MLlib if True, sklearn if False. Default: True
        """
        self.use_spark = use_spark
        self.spark_model = None
        self.sklearn_model = None
        
        if use_spark:
            self.spark_model = PySparkMLModel()
            logger.info("Using Spark MLlib for training")
        else:
            self.sklearn_model = SklearnMLModel()
            logger.info("Using scikit-learn for training")
    
    def train(self, data, feature_columns, label_column='Sales', **kwargs):
        """
        Train using appropriate backend.
        
        Args:
            data: Spark DataFrame if use_spark=True, Pandas DataFrame otherwise
            feature_columns (list): Feature column names
            label_column (str): Target column name. Default: 'Sales'
            **kwargs: Additional arguments for model
            
        Returns:
            dict: Training metrics
        """
        if self.use_spark:
            df_prepared = self.spark_model.prepare_features(data, feature_columns, label_column)
            return self.spark_model.train(df_prepared, label_column)
        else:
            return self.sklearn_model.train(data, feature_columns, label_column)
    
    def predict(self, data, **kwargs):
        """Make predictions using appropriate backend."""
        if self.use_spark:
            return self.spark_model.predict(data)
        else:
            return self.sklearn_model.predict_future(data)
    
    def get_metrics(self):
        """Get training metrics."""
        if self.use_spark:
            return self.spark_model.metrics
        else:
            return self.sklearn_model.metrics
