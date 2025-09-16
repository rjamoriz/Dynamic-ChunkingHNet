#!/usr/bin/env python3
"""
Diagnostic script to identify kernel detection issues in Qoder IDE
"""

import sys
import os
import subprocess
import importlib
import traceback

def check_python_environment():
    """Check Python environment and installations"""
    print("🔍 Python Environment Diagnostic")
    print("=" * 50)
    
    print(f"Current Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[:3]}...")  # First 3 paths
    print(f"Current working directory: {os.getcwd()}")
    
    # Check for multiple Python installations
    try:
        result = subprocess.run(['which', 'python3'], capture_output=True, text=True)
        print(f"System Python3 location: {result.stdout.strip()}")
    except:
        print("Could not determine system Python3 location")
    
    try:
        result = subprocess.run(['python3', '--version'], capture_output=True, text=True)
        print(f"System Python3 version: {result.stdout.strip()}")
    except:
        print("Could not get system Python3 version")

def check_required_packages():
    """Check if all required packages are available"""
    print("\n📦 Package Availability Check")
    print("=" * 50)
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 
        'nltk', 'jupyter', 'notebook', 'ipykernel'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}: Available")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
    else:
        print("✅ All required packages are available!")

def check_project_modules():
    """Check if project modules can be imported"""
    print("\n🏗️  Project Module Check")
    print("=" * 50)
    
    # Add current directory to path
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    modules_to_check = [
        'dynamic_chunking_hnet.core.pipeline',
        'dynamic_chunking_hnet.core.boundary_detector',
        'dynamic_chunking_hnet.core.routing_module',
        'dynamic_chunking_hnet.utils.config',
        'dynamic_chunking_hnet.evaluation.metrics'
    ]
    
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"✅ {module}: Available")
        except ImportError as e:
            print(f"❌ {module}: Failed - {e}")

def check_jupyter_kernel():
    """Check Jupyter kernel configuration"""
    print("\n⚙️  Jupyter Kernel Check")
    print("=" * 50)
    
    try:
        # Check if ipykernel is available
        import ipykernel
        print(f"✅ ipykernel version: {ipykernel.__version__}")
        
        # Check available kernels
        result = subprocess.run(['jupyter', 'kernelspec', 'list'], 
                              capture_output=True, text=True, timeout=10)
        print("Available Jupyter kernels:")
        print(result.stdout)
        
    except subprocess.TimeoutExpired:
        print("❌ Jupyter kernel list command timed out")
    except ImportError:
        print("❌ ipykernel not available")
    except Exception as e:
        print(f"❌ Error checking Jupyter kernels: {e}")

def check_memory_usage():
    """Check system memory usage"""
    print("\n💾 System Resource Check")
    print("=" * 50)
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total memory: {memory.total / (1024**3):.1f} GB")
        print(f"Available memory: {memory.available / (1024**3):.1f} GB")
        print(f"Memory usage: {memory.percent}%")
        
        if memory.percent > 85:
            print("⚠️  High memory usage may cause kernel issues")
        else:
            print("✅ Memory usage is acceptable")
            
    except ImportError:
        print("psutil not available - cannot check memory usage")

def recommend_solutions():
    """Provide recommendations based on findings"""
    print("\n💡 Recommendations")
    print("=" * 50)
    
    print("1. 🚀 Quick Fix - Use Python Script:")
    print("   Run: python3 demo_script.py")
    
    print("\n2. 🔧 Kernel Issues - Reset Environment:")
    print("   - Restart Qoder IDE completely")
    print("   - Clear any cached kernels")
    print("   - Use: pip install --upgrade jupyter ipykernel")
    
    print("\n3. 📱 Alternative - Native Jupyter:")
    print("   cd /Users/Ruben_MACPRO/Desktop/IA\\ DevOps/Dynamic_ChunkingHNet")
    print("   jupyter notebook dynamic_chunking_improved_demo.ipynb")
    
    print("\n4. 🛠️  If All Else Fails:")
    print("   - Create new Python virtual environment")
    print("   - Reinstall dependencies in clean environment")

def main():
    """Run complete diagnostic"""
    print("🔬 Qoder IDE Kernel Detection Diagnostic")
    print("=" * 60)
    
    try:
        check_python_environment()
        check_required_packages()
        check_project_modules()
        check_jupyter_kernel()
        check_memory_usage()
        recommend_solutions()
        
        print("\n✅ Diagnostic completed!")
        print("Use the recommendations above to resolve kernel issues.")
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()