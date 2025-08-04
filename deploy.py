#!/usr/bin/env python3
"""
Deployment readiness checker for Scan & Know
This script checks if your app is ready for deployment.
"""

import os
import sys
import importlib

def check_environment():
    """Check if environment variables are set"""
    print("🔍 Checking environment variables...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("   Set it in your hosting platform's environment variables")
        return False
    else:
        print("✅ GEMINI_API_KEY is set")
        return True

def check_dependencies():
    """Check if all required packages are available"""
    print("\n🔍 Checking Python dependencies...")
    
    required_packages = [
        'flask', 'requests', 'python-dotenv', 'flask-cors',
        'Pillow', 'pyzbar', 'opencv-python', 'numpy',
        'google-generativeai', 'pytesseract', 'easyocr'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is available"""
    print("\n🔍 Checking Tesseract OCR...")
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract OCR is available (version: {version})")
        return True
    except Exception as e:
        print(f"❌ Tesseract OCR not available: {e}")
        print("   This might be okay for deployment - the server will install it")
        return True  # Don't fail deployment for this

def check_files():
    """Check if required deployment files exist"""
    print("\n🔍 Checking deployment files...")
    
    required_files = [
        'Procfile',
        'runtime.txt',
        'Aptfile',
        'requirements.txt'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n📁 Create missing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_security():
    """Check for security issues"""
    print("\n🔍 Checking security...")
    
    # Check if .env file exists and warn
    if os.path.exists('.env'):
        print("⚠️  .env file found - make sure it's in .gitignore")
        print("   Never commit API keys to version control!")
    
    # Check if .gitignore exists
    if os.path.exists('.gitignore'):
        print("✅ .gitignore file exists")
    else:
        print("❌ .gitignore file missing")
        return False
    
    return True

def main():
    """Main deployment checker"""
    print("🚀 Scan & Know - Deployment Readiness Check")
    print("=" * 50)
    
    checks = [
        check_environment,
        check_dependencies,
        check_tesseract,
        check_files,
        check_security
    ]
    
    all_passed = True
    
    for check in checks:
        if not check():
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 Your app is ready for deployment!")
        print("\n📋 Next steps:")
        print("1. Choose a hosting platform (Heroku, Railway, etc.)")
        print("2. Set GEMINI_API_KEY in your hosting platform's environment variables")
        print("3. Deploy your repository")
        print("4. Test the deployed application")
    else:
        print("❌ Some issues need to be fixed before deployment")
        print("\n🔧 Fix the issues above and run this script again")
        sys.exit(1)

if __name__ == "__main__":
    main() 