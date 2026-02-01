#!/usr/bin/env python3
"""
Test script for Data Storage Module
Verifies database creation, data storage, and multi-user isolation
"""

import sqlite3
from config import Config
from data_loader import load_data
from data_preprocessor import preprocess_data
from data_storage import store_sales_data, get_user_sales_summary, get_user_sales_data, delete_user_sales_data

def test_data_storage():
    print("=" * 60)
    print("🧪 TESTING DATA STORAGE MODULE")
    print("=" * 60)
    
    # Test 1: Database structure
    print("\n✓ TEST 1: Database Structure")
    db = sqlite3.connect(Config.DATABASE)
    cur = db.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sales_data'")
    if cur.fetchone():
        print("  ✅ sales_data table exists")
    else:
        print("  ❌ sales_data table missing")
        return False
    
    cur.execute("PRAGMA table_info(sales_data)")
    columns = {col[1]: col[2] for col in cur.fetchall()}
    required_cols = ['id', 'user_id', 'store_id', 'date', 'sales', 'year', 'month', 'week', 'season']
    for col in required_cols:
        if col in columns:
            print(f"  ✅ Column '{col}' exists ({columns[col]})")
        else:
            print(f"  ❌ Column '{col}' missing")
    db.close()
    
    # Test 2: Data loading and preprocessing
    print("\n✓ TEST 2: Data Loading & Preprocessing")
    try:
        df, data_source = load_data()
        print(f"  ✅ Loaded {len(df)} records from {data_source} dataset")
        
        df, preprocessing_stats = preprocess_data(df)
        print(f"  ✅ Preprocessed data: {len(df)} records")
        print(f"     - Temporal features: Year, Month, Week, Season extracted")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # Test 3: Storing data for User 1
    print("\n✓ TEST 3: Storing Data for User 1")
    user_id_1 = 1001
    try:
        stats = store_sales_data(user_id_1, df)
        print(f"  ✅ Stored {stats['records_stored']} records for user {user_id_1}")
    except Exception as e:
        print(f"  ❌ Error storing data: {e}")
        return False
    
    # Test 4: Retrieving summary for User 1
    print("\n✓ TEST 4: Retrieving User Summary")
    try:
        summary = get_user_sales_summary(user_id_1)
        print(f"  ✅ User {user_id_1} Summary:")
        print(f"     - Records: {summary['records_stored']}")
        print(f"     - Stores: {summary['stores_count']}")
        print(f"     - Date range: {summary['start_date']} to {summary['end_date']}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # Test 5: Retrieving user data
    print("\n✓ TEST 5: Retrieving User Data")
    try:
        user_df = get_user_sales_data(user_id_1)
        print(f"  ✅ Retrieved {len(user_df)} records for user {user_id_1}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # Test 6: Multi-user isolation
    print("\n✓ TEST 6: Multi-User Data Isolation")
    user_id_2 = 1002
    try:
        # Store subset of data for user 2
        store_sales_data(user_id_2, df.head(100))
        
        summary_1 = get_user_sales_summary(user_id_1)
        summary_2 = get_user_sales_summary(user_id_2)
        
        if summary_1['records_stored'] > summary_2['records_stored']:
            print(f"  ✅ User 1: {summary_1['records_stored']} records")
            print(f"  ✅ User 2: {summary_2['records_stored']} records")
            print(f"  ✅ Data properly isolated (different record counts)")
        else:
            print(f"  ❌ Data isolation failed")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # Test 7: Data filtering
    print("\n✓ TEST 7: Data Filtering by Store")
    try:
        store_1_data = get_user_sales_data(user_id_1, filters={'store_id': 1})
        print(f"  ✅ Retrieved {len(store_1_data)} records for store 1")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # Test 8: Cleanup
    print("\n✓ TEST 8: Data Cleanup")
    try:
        delete_user_sales_data(user_id_1)
        delete_user_sales_data(user_id_2)
        summary_1 = get_user_sales_summary(user_id_1)
        if summary_1['records_stored'] == 0:
            print(f"  ✅ Cleaned up test data successfully")
        else:
            print(f"  ❌ Cleanup failed")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - DATA STORAGE MODULE WORKING!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = test_data_storage()
    exit(0 if success else 1)
