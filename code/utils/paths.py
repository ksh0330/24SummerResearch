from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]  # repo root
with open(ROOT / "configs" / "paths.yaml", "r", encoding="utf-8") as f:
    C = yaml.safe_load(f)

DATA_ROOT = ROOT / C["data_root"]
SAMPLES   = ROOT / C["samples_dir"]
WEIGHTS   = {k: ROOT / v for k, v in C["weights"].items()}
