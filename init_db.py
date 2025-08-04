#!/usr/bin/env python3
"""
Database initialization script for Scan & Know
This script creates the SQLite database and all required tables.
"""

from app import app, db
from models import User, ScanHistory

def init_database():
    """Initialize the database with all tables"""
    with app.app_context():
        print("🗄️  Creating database tables...")
        
        # Create all tables
        db.create_all()
        
        print("✅ Database tables created successfully!")
        print("📁 Database file: scan_know.db")
        
        # Check if tables were created
        try:
            tables = db.engine.table_names()
            print(f"📋 Created tables: {', '.join(tables)}")
        except:
            print("📋 Tables created successfully!")

if __name__ == "__main__":
    init_database() 