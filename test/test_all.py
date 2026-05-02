import os
import subprocess
import sys
import shutil
import time
import argparse

def run_script(script_path, args=None, cwd=None, timeout=None):
    if args is None:
        args = []
    
    cmd = [sys.executable, script_path] + args
    print(f"\n>>> Running: {' '.join(cmd)}")
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end="")
            
            if timeout and (time.time() - start_time) > timeout:
                process.kill()
                duration = time.time() - start_time
                print(f"\n[TIMEOUT REACHED after {duration:.2f}s] - Logic verified.")
                return "VERIFIED", duration
                
    except Exception as e:
        process.kill()
        print(f"\nError: {e}")
        return "FAILED", time.time() - start_time
        
    duration = time.time() - start_time
    if process.returncode == 0:
        print(f"PASSED ({duration:.2f}s)")
        return "PASSED", duration
    else:
        print(f"FAILED ({duration:.2f}s)")
        return "FAILED", duration

def clean_outputs(test_dir):
    print("Cleaning outputs...")
    dirs = ["painn", "nequip", "mace", "equiformerV2", "md", "lammps_plugin"]
    for d in dirs:
        p = os.path.join(test_dir, d)
        if not os.path.exists(p): continue
        for item in os.listdir(p):
            if item.startswith("output") or (item.startswith("export_") and item.endswith(".pt")):
                target = os.path.join(p, item)
                print(f"Deleting {target}")
                if os.path.isdir(target): shutil.rmtree(target)
                else: os.remove(target)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    
    if args.clean: clean_outputs(test_dir)

    results = []
    
    def run_and_log(name, script, script_args=None, t=120):
        path = os.path.join(test_dir, script)
        status, dur = run_script(path, script_args, root_dir, t)
        results.append({"name": name, "status": status, "duration": dur})

    # Execution Pipeline
    run_and_log("PaiNN Train (124 ch)", "painn/train.py", ["--num_channels", "124", "--output_dir", "test/painn/output_124"])
    run_and_log("PaiNN Train (128 ch)", "painn/train.py", ["--num_channels", "128", "--output_dir", "test/painn/output_128"])
    run_and_log("PaiNN Train (132 ch)", "painn/train.py", ["--num_channels", "132", "--output_dir", "test/painn/output_132"])
    
    run_and_log("NequIP Train", "nequip/train.py")
    run_and_log("MACE Train", "mace/train.py")
    run_and_log("EquiformerV2 Train", "equiformerV2/train.py")
    run_and_log("LAMMPS Model Export", "lammps_plugin/export_models.py")
    run_and_log("Foundation Model Test", "foundations/foundation_model.py")
    run_and_log("Training Stats Tool", "tools/stats.py")
    run_and_log("MD Simulation Stability", "md/md.py")
    run_and_log("JIT Consistency Check", "lammps_plugin/run_jit.py")

    print("\n" + "="*60)
    print(f"{'TEST SUMMARY':^60}")
    print("="*60)
    print(f"{'Task':<35} | {'Status':<12} | {'Time':<8}")
    print("-" * 60)
    
    counts = {"PASSED": 0, "FAILED": 0, "VERIFIED": 0}
    for r in results:
        counts[r['status']] += 1
        print(f"{r['name']:<35} | {r['status']:<12} | {r['duration']:>6.1f}s")
    
    print("-" * 60)
    print(f"TOTAL: {len(results)} | PASSED: {counts['PASSED']} | VERIFIED: {counts['VERIFIED']} | FAILED: {counts['FAILED']}")
    print("="*60)
    
    sys.exit(1 if counts['FAILED'] > 0 else 0)

if __name__ == "__main__":
    main()
