import os
import subprocess
import sys
import shutil
import time

def run_script(script_path, args=None, cwd=None):
    if args is None:
        args = []
    
    cmd = [sys.executable, script_path] + args
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    process = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    duration = time.time() - start_time
    
    if process.returncode == 0:
        print(f"PASSED ({duration:.2f}s)")
        return True
    else:
        print(f"FAILED ({duration:.2f}s)")
        print(process.stdout)
        print(process.stderr)
        return False

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(root_dir, "test")
    
    print(f"Starting test workflow in {root_dir}")
    print("-" * 50)

    # 1. Run train for painn 124, 128, 132
    painn_train = os.path.join(test_dir, "painn", "train.py")
    for channels in [124, 128, 132]:
        output_dir = f"test/painn/output_{channels}"
        if not run_script(painn_train, ["--num_channels", str(channels), "--output_dir", output_dir], cwd=root_dir):
            sys.exit(1)

    # 2. Copy output_128 to output
    src_dir = os.path.join(test_dir, "painn", "output_128")
    dst_dir = os.path.join(test_dir, "painn", "output")
    print(f"Copying {src_dir} to {dst_dir}")
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    # 3. Run nequip
    nequip_train = os.path.join(test_dir, "nequip", "train.py")
    if not run_script(nequip_train, cwd=root_dir):
        sys.exit(1)

    # 4. Run mace
    mace_train = os.path.join(test_dir, "mace", "train.py")
    if not run_script(mace_train, cwd=root_dir):
        sys.exit(1)

    # 5. Run equiformerV2
    equiformer_train = os.path.join(test_dir, "equiformerV2", "train.py")
    if not run_script(equiformer_train, cwd=root_dir):
        sys.exit(1)

    # 6. Run lammps_plugin/export_models.py
    export_models = os.path.join(test_dir, "lammps_plugin", "export_models.py")
    if not run_script(export_models, cwd=root_dir):
        sys.exit(1)

    # 7. Run foundation_model.py
    foundation_model = os.path.join(test_dir, "foundations", "foundation_model.py")
    if not run_script(foundation_model, cwd=root_dir):
        sys.exit(1)

    # 8. Run stats.py
    stats_script = os.path.join(test_dir, "tools", "stats.py")
    if not run_script(stats_script, cwd=root_dir):
        sys.exit(1)

    print("-" * 50)
    print("ALL TESTS COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
