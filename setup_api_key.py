#!/usr/bin/env python3
"""
Setup script for Gemini API key configuration
"""

import os
import sys

def main():
    print("🤖 Gemini API Key Setup for Scan & Know")
    print("=" * 50)
    
    # Check if .env file already exists
    if os.path.exists('.env'):
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    print("\n📋 Instructions:")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated API key")
    print("\n")
    
    api_key = input("🔑 Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return
    
    # Create .env file
    try:
        with open('.env', 'w') as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")
        
        print("✅ API key saved successfully!")
        print("📁 Created .env file with your API key")
        print("\n🚀 You can now run the application:")
        print("   python app.py")
        
    except Exception as e:
        print(f"❌ Error saving API key: {e}")
        return

if __name__ == "__main__":
    main() 