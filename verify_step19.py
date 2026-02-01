#!/usr/bin/env python3
"""
STEP 19: Verification Script
Verifies all STEP 19 files are in place and system is ready
"""

import os
import sys
from pathlib import Path

def check_files():
    """Verify all STEP 19 files exist"""
    print("\n" + "="*60)
    print("🔍 STEP 19: File Verification")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    seasonal_folder = project_root / "seasonal_sales_forecasting"
    
    # Check code files
    code_files = {
        seasonal_folder / "validate_system.py": "Validation Module",
        seasonal_folder / "test_validation.py": "Test Suite",
        seasonal_folder / "run_validation.py": "Validation Runner",
    }
    
    # Check documentation files
    doc_files = {
        project_root / "STEP_19_DOCUMENTATION.md": "Technical Documentation",
        project_root / "STEP_19_QUICK_REFERENCE.md": "Quick Reference",
        project_root / "STEP_19_COMPLETION.md": "Completion Report",
        project_root / "STEP_19_EXECUTIVE_SUMMARY.md": "Executive Summary",
        project_root / "STEP_19_FILE_INVENTORY.md": "File Inventory",
    }
    
    all_exist = True
    
    # Check code files
    print("\n📝 CODE FILES")
    print("-" * 60)
    for file_path, description in code_files.items():
        if file_path.exists():
            size = file_path.stat().st_size
            lines = len(file_path.read_text().splitlines())
            print(f"✅ {description}")
            print(f"   📁 {file_path.name} ({lines} lines, {size:,} bytes)")
        else:
            print(f"❌ {description}")
            print(f"   Missing: {file_path}")
            all_exist = False
    
    # Check documentation files
    print("\n📚 DOCUMENTATION FILES")
    print("-" * 60)
    for file_path, description in doc_files.items():
        if file_path.exists():
            size = file_path.stat().st_size
            lines = len(file_path.read_text().splitlines())
            print(f"✅ {description}")
            print(f"   📄 {file_path.name} ({lines} lines, {size:,} bytes)")
        else:
            print(f"❌ {description}")
            print(f"   Missing: {file_path}")
            all_exist = False
    
    return all_exist


def check_database():
    """Verify database exists"""
    print("\n🗄️  DATABASE")
    print("-" * 60)
    
    project_root = Path(__file__).parent.parent
    db_path = project_root / "seasonal_sales_forecasting" / "database" / "sales.db"
    
    if db_path.exists():
        size = db_path.stat().st_size
        print(f"✅ Database found")
        print(f"   📁 {db_path.name} ({size:,} bytes)")
        return True
    else:
        print(f"⚠️  Database not found")
        print(f"   This is optional for running validations on test data")
        return False


def check_imports():
    """Verify all imports work"""
    print("\n🔧 IMPORT VERIFICATION")
    print("-" * 60)
    
    try:
        # Add seasonal_sales_forecasting to path
        project_root = Path(__file__).parent.parent
        seasonal_folder = project_root / "seasonal_sales_forecasting"
        sys.path.insert(0, str(seasonal_folder))
        
        # Try importing validation module
        from validate_system import (
            DataIsolationValidator,
            ForecastAccuracyValidator,
            GraphDisplayValidator,
            SystemValidator
        )
        print("✅ validate_system imports successfully")
        print("   - DataIsolationValidator")
        print("   - ForecastAccuracyValidator")
        print("   - GraphDisplayValidator")
        print("   - SystemValidator")
        
        # Try importing test module
        import test_validation
        print("✅ test_validation imports successfully")
        
        # Try importing runner
        import run_validation
        print("✅ run_validation imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import verification failed: {str(e)}")
        return False


def print_summary():
    """Print summary and next steps"""
    print("\n" + "="*60)
    print("📋 NEXT STEPS")
    print("="*60)
    
    print("\n1. Run Full Validation:")
    print("   cd seasonal_sales_forecasting")
    print("   python3 validate_system.py")
    
    print("\n2. Run Test Suite:")
    print("   python3 -m unittest test_validation.py -v")
    
    print("\n3. Generate Reports:")
    print("   python3 run_validation.py --export-html report.html")
    
    print("\n4. Review Documentation:")
    print("   - STEP_19_QUICK_REFERENCE.md (quick start)")
    print("   - STEP_19_DOCUMENTATION.md (technical details)")
    print("   - STEP_19_EXECUTIVE_SUMMARY.md (overview)")
    
    print("\n" + "="*60)


def main():
    """Run all verification checks"""
    print("\n" + "="*70)
    print("STEP 19: SYSTEM VALIDATION MODULE")
    print("File Verification & Readiness Check")
    print("="*70)
    
    # Run checks
    files_ok = check_files()
    db_ok = check_database()
    imports_ok = check_imports()
    
    # Summary
    print("\n" + "="*60)
    print("✅ VERIFICATION RESULTS")
    print("="*60)
    
    if files_ok and imports_ok:
        print("✅ All code files present and importable")
        print("✅ All documentation files present")
        if db_ok:
            print("✅ Database available")
        else:
            print("⚠️  Database not found (will test with generated data)")
        
        print("\n" + "="*60)
        print("🎉 STEP 19 IS READY TO RUN")
        print("="*60)
        
        print_summary()
        return 0
    else:
        print("❌ Some files are missing or imports failed")
        print("Please check the above errors")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
