import os
import subprocess
import sys
import time
import argparse

def run_tests():
    parser = argparse.ArgumentParser(description="IANN Test Workflow")
    parser.add_argument("--folders", nargs="+", help="Specific test folders to run (e.g., demo mace)")
    parser.add_argument("--skip", nargs="+", help="Folders to skip")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per test in seconds")
    args = parser.parse_args()

    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    results = []

    # Get all subdirectories in test/
    subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    subdirs.sort()

    if args.folders:
        subdirs = [d for d in subdirs if d in args.folders]
    
    if args.skip:
        subdirs = [d for d in subdirs if d not in args.skip]

    print(f"Starting test workflow in {root_dir}")
    print(f"Folders: {', '.join(subdirs)}")
    print("-" * 50)

    for subdir in subdirs:
        if subdir in ['__pycache__', 'output', 'tools']:
            continue
            
        subdir_path = os.path.join(test_dir, subdir)
        py_files = [f for f in os.listdir(subdir_path) if f.endswith('.py') and f != '__init__.py']
        # Prioritize training scripts so prediction scripts have models to work with
        py_files.sort(key=lambda x: (0 if 'train' in x.lower() else 1, x))

        if not py_files:
            continue

        print(f"\nEntering folder: {subdir}")
        
        for py_file in py_files:
            rel_script_path = os.path.join("test", subdir, py_file)
            print(f"  Running {rel_script_path}...", end=" ", flush=True)
            
            start_time = time.time()
            try:
                # Run the python file from the root directory
                process = subprocess.run(
                    [sys.executable, rel_script_path],
                    cwd=root_dir,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout
                )
                
                duration = time.time() - start_time
                
                if process.returncode == 0:
                    print(f"PASSED ({duration:.2f}s)")
                    results.append((subdir, py_file, "PASSED", duration))
                else:
                    print(f"FAILED ({duration:.2f}s)")
                    # print(f"\nError in {rel_script_path}:")
                    # print(process.stderr)
                    results.append((subdir, py_file, "FAILED", duration))
                    
            except subprocess.TimeoutExpired:
                duration = time.time() - start_time
                print(f"TIMEOUT ({duration:.2f}s)")
                results.append((subdir, py_file, "TIMEOUT", duration))
            except Exception as e:
                duration = time.time() - start_time
                print(f"ERROR ({duration:.2f}s)")
                print(f"Exception: {str(e)}")
                results.append((subdir, py_file, "ERROR", duration))

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results if r[2] == "PASSED")
    failed = sum(1 for r in results if r[2] != "PASSED")
    
    for subdir, py_file, status, duration in results:
        print(f"{status:8} | {subdir}/{py_file:25} | {duration:6.2f}s")
        
    print("-" * 50)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
