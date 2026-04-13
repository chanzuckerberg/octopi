import subprocess, sys, yaml

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def _run(cmd: list[str], env: dict):

    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with return code {result.returncode}")
        sys.exit(result.returncode)