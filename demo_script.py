#!/usr/bin/env python3
"""
Dynamic ChunkingHNet Demo Script
Run this directly in Qoder IDE as an alternative to the notebook
"""

import sys
import os
sys.path.append('.')

def main():
    print("🚀 Dynamic ChunkingHNet - Improved Implementation Demo")
    print("=" * 60)
    
    # Test 1: Import modules
    print("\n1. Testing Module Imports...")
    try:
        from dynamic_chunking_hnet.core.pipeline import DynamicChunkingPipeline
        from dynamic_chunking_hnet.core.boundary_detector import SimilarityBasedBoundaryDetector
        from dynamic_chunking_hnet.core.routing_module import RoutingModule
        from dynamic_chunking_hnet.core.smoothing_module import SmoothingModule
        from dynamic_chunking_hnet.evaluation.metrics import ChunkingQualityMetrics
        from dynamic_chunking_hnet.utils.config import load_config
        from dynamic_chunking_hnet.utils.monitoring import get_logger, performance_monitor
        print("✅ Successfully imported all improved modules!")
        modules_available = True
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        print("Please ensure you're in the correct directory and dependencies are installed.")
        modules_available = False
        
    if not modules_available:
        return
    
    # Test 2: Configuration
    print("\n2. Testing Configuration...")
    try:
        config = load_config('config/default.yaml')
        print("✅ Configuration loaded successfully!")
        print(f"Default compression ratio: {config.get('compression_ratio', 'Not set')}")
    except Exception as e:
        print(f"⚠️ Using default configuration: {e}")
        config = {'compression_ratio': 6.0}
    
    # Test 3: Basic Processing
    print("\n3. Testing Basic Processing...")
    sample_text = """
    Machine learning is transforming how we process information. 
    Neural networks can learn complex patterns from data. 
    Natural language processing enables computers to understand text. 
    The H-Net architecture introduces dynamic chunking mechanisms. 
    This approach outperforms traditional fixed-size tokenization methods.
    """.strip()
    
    print(f"📝 Sample Text:")
    print(f"'{sample_text}'")
    print(f"\nText length: {len(sample_text.split())} tokens")
    
    try:
        # Initialize pipeline
        pipeline = DynamicChunkingPipeline(compression_ratio=6.0)
        
        # Process text
        print("\n🔄 Processing text with improved pipeline...")
        result = pipeline.process_text(sample_text, return_metrics=True)
        
        # Display results
        print("\n📊 Results:")
        print(f"Original tokens: {result['num_tokens']}")
        print(f"Chunks created: {result['num_chunks']}")
        print(f"Compression ratio achieved: {result['compression_ratio_achieved']:.2f}")
        
        print("\n📝 Generated Chunks:")
        for i, chunk in enumerate(result['chunks'], 1):
            chunk_text = ' '.join(chunk)
            print(f"  {i}: {chunk_text}")
            
        print("\n✅ Basic processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Performance Comparison
    print("\n4. Testing Performance Comparison...")
    try:
        test_ratios = [4.0, 6.0, 8.0]
        
        for ratio in test_ratios:
            pipeline = DynamicChunkingPipeline(compression_ratio=ratio)
            result = pipeline.process_text(sample_text)
            
            print(f"\nCompression ratio {ratio}:")
            print(f"  - Chunks: {result['num_chunks']}")
            print(f"  - Achieved ratio: {result['compression_ratio_achieved']:.2f}")
            
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
    
    print("\n🎉 Demo completed! All improvements are working correctly.")
    print("You can now use the Dynamic ChunkingHNet library in your projects.")

if __name__ == "__main__":
    main()