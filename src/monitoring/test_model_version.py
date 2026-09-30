import sys
from pathlib import Path


# ==================================================
# Project Path
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# Import
# ==================================================

from src.monitoring.model_version import ModelVersionManager


# ==================================================
# Main
# ==================================================

def main():

    manager = ModelVersionManager()

    metadata = manager.initialize()

    print()
    print("=" * 60)
    print("MODEL VERSION")
    print("=" * 60)

    for key, value in metadata.items():
        print(f"{key}: {value}")


# ==================================================
# Run
# ==================================================

if __name__ == "__main__":
    main()