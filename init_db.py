"""Database initialization script for seasonal_sales_forecasting."""
import sqlite3
from config import Config

def init_db():
    db = sqlite3.connect(Config.DATABASE)
    cur = db.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    db.commit()
    db.close()

if __name__ == '__main__':
    init_db()
    print('Database initialized at', Config.DATABASE)
