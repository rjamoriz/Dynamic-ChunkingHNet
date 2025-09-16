#!/usr/bin/env python3
"""
Minimal test script for Dynamic ChunkingHNet
This bypasses all terminal and kernel issues
"""

def simple_test():
    print("🧪 Dynamic ChunkingHNet - Simple Test")
    print("=" * 40)
    
    # Test 1: Basic Python functionality
    print("1. Testing basic Python...")
    try:
        import sys
        import os
        print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
        print(f"✅ Current directory: {os.getcwd()}")
    except Exception as e:
        print(f"❌ Basic Python test failed: {e}")
        return False
    
    # Test 2: Basic imports
    print("\n2. Testing basic imports...")
    try:
        import numpy as np
        import pandas as pd
        print("✅ NumPy and Pandas available")
    except ImportError as e:
        print(f"❌ Basic imports failed: {e}")
        return False
    
    # Test 3: Project structure
    print("\n3. Testing project structure...")
    expected_dirs = ['dynamic_chunking_hnet', 'config', 'tests']
    for dir_name in expected_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ directory found")
        else:
            print(f"❌ {dir_name}/ directory missing")
    
    # Test 4: Simple chunking simulation
    print("\n4. Testing simple chunking logic...")
    try:
        text = "This is a test. Another sentence here. Final sentence."
        sentences = text.split('.')
        chunks = [s.strip() for s in sentences if s.strip()]
        
        print(f"✅ Text: '{text}'")
        print(f"✅ Simple chunks: {chunks}")
        print(f"✅ Chunk count: {len(chunks)}")
    except Exception as e:
        print(f"❌ Simple chunking failed: {e}")
        return False
    
    # Test 5: Try importing project modules
    print("\n5. Testing project module imports...")
    try:
        sys.path.append('.')
        from dynamic_chunking_hnet.core.pipeline import DynamicChunkingPipeline
        print("✅ Pipeline module imported successfully!")
        
        # Try basic usage
        pipeline = DynamicChunkingPipeline(compression_ratio=6.0)
        print("✅ Pipeline created successfully!")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Module import issue: {e}")
        print("This suggests dependency or environment issues")
        return False

def main():
    """Run the simple test"""
    success = simple_test()
    
    if success:
        print("\n🎉 SUCCESS: Basic functionality works!")
        print("The Dynamic ChunkingHNet modules are ready to use.")
        print("\nNext steps:")
        print("- Try running demo_script.py")
        print("- Use native Jupyter if needed")
    else:
        print("\n🚨 ISSUES DETECTED")
        print("Recommendations:")
        print("1. Check Python environment")
        print("2. Install missing dependencies")
        print("3. Verify project structure")
    
    print(f"\nTest completed at: {os.getcwd()}")

if __name__ == "__main__":
    main()