import time
import argparse
import sys
import os

from core.legacy.main import main as run_training

datasets = [
    ('linear', 200, 64),
    ('nonlinear', 200, 64),
    ('lorenz96', 200, 32),  # Reduced epochs for speed
    ('lotka_volterra', 200, 32),  # Very slow, minimal epochs
]

def test_dataset(name, epochs, batch_size):
    print(f"\n{'='*60}")
    print(f"🧪 Testing dataset: {name}")
    print(f"📊 Config: epochs={epochs}, batch_size={batch_size}")
    print(f"⏰ Start time: {time.strftime('%H:%M:%S')}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        # Run main
        import sys
        original_argv = sys.argv
        sys.argv = ['test.py', '--dataset', name, '--epochs', str(epochs), '--device', 'cuda', '--batch_size', str(batch_size)]
        run_training(sys.argv)
        sys.argv = original_argv
        
        elapsed = time.time() - start_time
        print(f"✅ {name}: Completed in {elapsed:.2f}s ({elapsed/60:.2f}min)")
        return {"dataset": name, "status": "success", "time": elapsed}
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {name}: FAILED after {elapsed:.2f}s")
        print(f"   Error: {e}")
        return {"dataset": name, "status": "failed", "time": elapsed, "error": str(e)}

def main():
    print("🚀 AERCA Comprehensive GPU Test Suite")
    print(f"💻 Device: CUDA (A6000)")
    print(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📦 Total datasets: {len(datasets)}")
    
    results = []
    total_start = time.time()
    
    for dataset_config in datasets:
        name, epochs, batch_size = dataset_config
        result = test_dataset(name, epochs, batch_size)
        results.append(result)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("📊 FINAL REPORT")
    print('='*60)
    
    success = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - success
    
    print(f"✅ Success: {success}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
    
    print("\n📈 Detailed results:")
    for r in results:
        status_icon = "✅" if r['status'] == 'success' else "❌"
        print(f"  {status_icon} {r['dataset']}: {r['time']:.2f}s")
        if r['status'] == 'failed':
            print(f"     Error: {r['error'][:100]}...")
    
    # Save results to file
    import json
    with open('aerca_test_results.json', 'w') as f:
        json.dump({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "device": "cuda",
            "total_time": total_time,
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: aerca_test_results.json")
    
    # Quick performance analysis
    if success > 0:
        avg_time = sum(r['time'] for r in results if r['status'] == 'success') / success
        print(f"\n📊 Performance stats:")
        print(f"  Average time per dataset: {avg_time:.2f}s")
        print(f"  Estimated full 100-epoch runs: {avg_time * 5:.2f}s per dataset")
        
        # Project to full epochs
        print(f"\n🎯 Projection for full 100-epoch runs:")
        for r in results:
            if r['status'] == 'success':
                projected = r['time'] * (100 / [e for n,e,b in datasets if n == r['dataset']][0])
                print(f"  {r['dataset']}: ~{projected:.2f}s")

if __name__ == '__main__':
    main()
