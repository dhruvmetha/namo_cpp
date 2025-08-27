#!/usr/bin/env python3
"""Test script to validate cross-host data collection functionality."""

import subprocess
import sys
from pathlib import Path

def test_multihost_simulation():
    """Test cross-host data collection by simulating different hostnames."""
    
    output_dir = "./test_multihost_data"
    
    # Test configurations for different "hosts"
    host_configs = [
        {
            "hostname": "host_a", 
            "start_idx": 0, 
            "end_idx": 2,
            "workers": 1,
            "episodes": 1
        },
        {
            "hostname": "host_b", 
            "start_idx": 2, 
            "end_idx": 4,
            "workers": 1, 
            "episodes": 1
        }
    ]
    
    results = []
    
    for config in host_configs:
        print(f"\n🖥️  Simulating data collection on {config['hostname']}")
        
        cmd = [
            "/common/users/dm1487/envs/mjxrl/bin/python",
            "python/parallel_data_collection.py",
            "--output-dir", output_dir,
            "--start-idx", str(config["start_idx"]),
            "--end-idx", str(config["end_idx"]),
            "--workers", str(config["workers"]),
            "--episodes-per-env", str(config["episodes"]),
            "--mcts-budget", "5",  # Very small for testing
            "--hostname", config["hostname"]
        ]
        
        env = {"PYTHONPATH": "/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl"}
        
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
            
            success = result.returncode == 0
            results.append({
                "hostname": config["hostname"],
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
            
            print(f"✅ {config['hostname']}: {'SUCCESS' if success else 'FAILED'}")
            if not success:
                print(f"   Error: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {config['hostname']}: TIMEOUT")
            results.append({"hostname": config["hostname"], "success": False, "error": "timeout"})
        except Exception as e:
            print(f"💥 {config['hostname']}: EXCEPTION - {e}")
            results.append({"hostname": config["hostname"], "success": False, "error": str(e)})
    
    # Check output structure
    print(f"\n📁 Checking output directory structure:")
    base_path = Path(output_dir)
    
    for config in host_configs:
        host_dir = base_path / f"data_{config['hostname']}"
        exists = host_dir.exists()
        print(f"   data_{config['hostname']}: {'✅ EXISTS' if exists else '❌ MISSING'}")
        
        if exists:
            files = list(host_dir.glob("*"))
            print(f"      Files: {len(files)} ({[f.name for f in files[:3]]})")
    
    # Summary
    successful_hosts = sum(1 for r in results if r.get("success", False))
    print(f"\n🏁 Multi-host test summary:")
    print(f"   Successful hosts: {successful_hosts}/{len(host_configs)}")
    print(f"   Output directory: {output_dir}")
    
    if successful_hosts == len(host_configs):
        print("✅ Multi-host data collection PASSED")
        return True
    else:
        print("❌ Multi-host data collection FAILED")
        return False

if __name__ == "__main__":
    success = test_multihost_simulation()
    sys.exit(0 if success else 1)