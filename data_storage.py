"""
Data Storage Module
Handles persistent storage of preprocessed sales data in SQLite database.
Enables multi-user support by linking data to logged-in users.
"""

import sqlite3
import pandas as pd
from config import Config
import os


def init_sales_db():
    """Initialize sales database with proper schema."""
    db_path = Config.DATABASE
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    db = sqlite3.connect(db_path)
    cur = db.cursor()
    
    # Create sales_data table with all temporal features and sales metrics
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sales_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            store_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            sales REAL,
            year INTEGER,
            month INTEGER,
            week INTEGER,
            quarter INTEGER,
            season TEXT,
            day_of_week INTEGER,
            day_of_year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, store_id, date)
        )
    ''')
    
    # Create index for faster queries
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_id ON sales_data(user_id)
    ''')
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_store_date ON sales_data(user_id, store_id, date)
    ''')
    
    db.commit()
    db.close()
    print("✅ Sales database initialized")


def store_sales_data(user_id, df):
    """
    Store preprocessed sales data to database for a specific user.
    
    Args:
        user_id (int): ID of the user storing the data
        df (pd.DataFrame): Preprocessed dataframe with columns:
                          Store, Date, Weekly_Sales, Year, Month, Week, Quarter, 
                          Season, DayOfWeek, DayOfYear
    
    Returns:
        dict: Status information with records stored/updated/failed counts
    """
    db = sqlite3.connect(Config.DATABASE)
    cur = db.cursor()
    
    stats = {
        'records_stored': 0,
        'records_updated': 0,
        'records_failed': 0,
        'total_attempted': len(df)
    }
    
    for _, row in df.iterrows():
        try:
            # Handle different column name variations
            store_id = int(row.get('Store', row.get('store', 0)))
            date = str(row.get('Date', row.get('date', '')))
            sales = float(row.get('Weekly_Sales', row.get('sales', 0)))
            
            # Get temporal features (with fallbacks if not present)
            year = int(row.get('Year', 0)) if 'Year' in row else 0
            month = int(row.get('Month', 0)) if 'Month' in row else 0
            week = int(row.get('Week', 0)) if 'Week' in row else 0
            quarter = int(row.get('Quarter', 0)) if 'Quarter' in row else 0
            season = str(row.get('Season', '')) if 'Season' in row else ''
            day_of_week = int(row.get('DayOfWeek', 0)) if 'DayOfWeek' in row else 0
            day_of_year = int(row.get('DayOfYear', 0)) if 'DayOfYear' in row else 0
            
            # Insert or replace record
            cur.execute('''
                INSERT OR REPLACE INTO sales_data 
                (user_id, store_id, date, sales, year, month, week, quarter, season, day_of_week, day_of_year)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, store_id, date, sales, year, month, week, quarter, season, day_of_week, day_of_year))
            
            if cur.lastrowid:
                stats['records_stored'] += 1
            else:
                stats['records_updated'] += 1
                
        except Exception as e:
            stats['records_failed'] += 1
            print(f"⚠️ Error storing record: {e}")
    
    db.commit()
    db.close()
    
    print(f"✅ Data stored: {stats['records_stored']} new, {stats['records_updated']} updated, {stats['records_failed']} failed")
    return stats


def get_user_sales_data(user_id, filters=None):
    """
    Retrieve preprocessed sales data for a specific user.
    
    Args:
        user_id (int): ID of the user
        filters (dict): Optional filters like store_id, start_date, end_date
        
    Returns:
        pd.DataFrame: Sales data for the user
    """
    db = sqlite3.connect(Config.DATABASE)
    
    query = 'SELECT * FROM sales_data WHERE user_id = ?'
    params = [user_id]
    
    # Apply optional filters
    if filters:
        if 'store_id' in filters:
            query += ' AND store_id = ?'
            params.append(filters['store_id'])
        if 'start_date' in filters:
            query += ' AND date >= ?'
            params.append(filters['start_date'])
        if 'end_date' in filters:
            query += ' AND date <= ?'
            params.append(filters['end_date'])
    
    query += ' ORDER BY store_id, date'
    
    try:
        df = pd.read_sql_query(query, db, params=params)
        db.close()
        return df
    except Exception as e:
        print(f"⚠️ Error retrieving user data: {e}")
        db.close()
        return pd.DataFrame()


def get_user_sales_summary(user_id):
    """
    Get summary statistics for user's sales data.
    
    Args:
        user_id (int): ID of the user
        
    Returns:
        dict: Summary with records_stored, stores_count, date_range
    """
    db = sqlite3.connect(Config.DATABASE)
    cur = db.cursor()
    
    try:
        # Count total records
        cur.execute('SELECT COUNT(*) FROM sales_data WHERE user_id = ?', (user_id,))
        total_records = cur.fetchone()[0]
        
        # Count distinct stores
        cur.execute('SELECT COUNT(DISTINCT store_id) FROM sales_data WHERE user_id = ?', (user_id,))
        stores_count = cur.fetchone()[0]
        
        # Get date range
        cur.execute('''
            SELECT MIN(date) as start_date, MAX(date) as end_date 
            FROM sales_data WHERE user_id = ?
        ''', (user_id,))
        date_info = cur.fetchone()
        
        db.close()
        
        return {
            'records_stored': total_records,
            'stores_count': stores_count,
            'start_date': date_info[0] if date_info[0] else 'N/A',
            'end_date': date_info[1] if date_info[1] else 'N/A'
        }
    except Exception as e:
        print(f"⚠️ Error getting summary: {e}")
        db.close()
        return {
            'records_stored': 0,
            'stores_count': 0,
            'start_date': 'N/A',
            'end_date': 'N/A'
        }


def delete_user_sales_data(user_id):
    """
    Delete all sales data for a specific user (e.g., for re-upload).
    
    Args:
        user_id (int): ID of the user
        
    Returns:
        dict: Status with records_deleted count
    """
    db = sqlite3.connect(Config.DATABASE)
    cur = db.cursor()
    
    try:
        cur.execute('SELECT COUNT(*) FROM sales_data WHERE user_id = ?', (user_id,))
        records_deleted = cur.fetchone()[0]
        
        cur.execute('DELETE FROM sales_data WHERE user_id = ?', (user_id,))
        db.commit()
        db.close()
        
        print(f"✅ Deleted {records_deleted} sales records for user {user_id}")
        return {'records_deleted': records_deleted}
    except Exception as e:
        print(f"⚠️ Error deleting user data: {e}")
        db.close()
        return {'records_deleted': 0}


def export_user_data(user_id, format='csv'):
    """
    Export user's sales data to file (CSV or Excel).
    
    Args:
        user_id (int): ID of the user
        format (str): 'csv' or 'excel'
        
    Returns:
        pd.DataFrame: User's data or None if error
    """
    df = get_user_sales_data(user_id)
    
    if df.empty:
        print(f"⚠️ No data to export for user {user_id}")
        return None
    
    return df


def get_database_stats():
    """Get overall database statistics."""
    db = sqlite3.connect(Config.DATABASE)
    cur = db.cursor()
    
    try:
        cur.execute('SELECT COUNT(DISTINCT user_id) FROM sales_data')
        users_with_data = cur.fetchone()[0]
        
        cur.execute('SELECT COUNT(*) FROM sales_data')
        total_records = cur.fetchone()[0]
        
        cur.execute('SELECT COUNT(DISTINCT store_id) FROM sales_data')
        total_stores = cur.fetchone()[0]
        
        db.close()
        
        return {
            'users_with_data': users_with_data,
            'total_records': total_records,
            'total_stores': total_stores
        }
    except Exception as e:
        print(f"⚠️ Error getting database stats: {e}")
        db.close()
        return {
            'users_with_data': 0,
            'total_records': 0,
            'total_stores': 0
        }
