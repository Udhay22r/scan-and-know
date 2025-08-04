#!/usr/bin/env python3
"""
Setup script for Tesseract OCR
This script checks if Tesseract OCR is properly installed and provides installation instructions if needed.
"""

import os
import sys
import subprocess
import platform

def check_tesseract_installation():
    """Check if Tesseract OCR is properly installed and accessible"""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract OCR is properly installed (version: {version})")
        return True
    except ImportError:
        print("❌ pytesseract Python package is not installed")
        print("   Run: pip install pytesseract")
        return False
    except Exception as e:
        print(f"❌ Tesseract OCR is not properly installed: {e}")
        return False

def get_installation_instructions():
    """Get installation instructions based on the operating system"""
    system = platform.system().lower()
    
    if system == "windows":
        return """
📋 Tesseract OCR Installation Instructions for Windows:

Option 1 - Using winget (Recommended):
   winget install UB-Mannheim.TesseractOCR

Option 2 - Manual Installation:
   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
   2. Run the installer
   3. Make sure to check "Add to PATH" during installation
   4. Restart your terminal/command prompt

After installation, restart your terminal and run this script again.
"""
    elif system == "darwin":  # macOS
        return """
📋 Tesseract OCR Installation Instructions for macOS:

Using Homebrew:
   brew install tesseract

After installation, restart your terminal and run this script again.
"""
    elif system == "linux":
        return """
📋 Tesseract OCR Installation Instructions for Linux:

Ubuntu/Debian:
   sudo apt update
   sudo apt install tesseract-ocr

CentOS/RHEL/Fedora:
   sudo yum install tesseract
   # or
   sudo dnf install tesseract

After installation, restart your terminal and run this script again.
"""
    else:
        return """
📋 Tesseract OCR Installation Instructions:

Please visit: https://github.com/tesseract-ocr/tesseract
Download and install Tesseract OCR for your operating system.
Make sure to add it to your system PATH.

After installation, restart your terminal and run this script again.
"""

def main():
    """Main setup function"""
    print("🔍 Checking Tesseract OCR installation...")
    print("=" * 50)
    
    if check_tesseract_installation():
        print("\n🎉 Setup complete! Tesseract OCR is ready to use.")
        print("You can now run: python app.py")
        return True
    else:
        print("\n" + get_installation_instructions())
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 