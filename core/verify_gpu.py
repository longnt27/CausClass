import torch
import time

print("🔍 GPU Verification Script")
print("="*50)

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Quick performance test
    print("\n⚡ Quick Performance Test (A6000)...")
    start = time.time()
    x = torch.randn(5000, 5000).cuda()  # Smaller for A6000 memory
    y = torch.randn(5000, 5000).cuda()
    z = torch.matmul(x, y)
    elapsed = time.time() - start
    print(f"Matrix multiplication (5k x 5k): {elapsed:.3f}s")
    
    # Memory test
    print(f"\n💾 GPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Check AERCA imports
    try:
        from core.models.aerca import AERCA
        print("✅ AERCA model imports successfully")
    except Exception as e:
        print(f"❌ AERCA import failed: {e}")
else:
    print("❌ CUDA not available. Check GPU drivers.")
