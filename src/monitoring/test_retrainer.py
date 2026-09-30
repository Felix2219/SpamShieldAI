import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.retrainer import ModelRetrainer


def main():

    retrainer = ModelRetrainer()

    # --------------------------------------------------
    # TEMPORARY TEST SETTING
    # --------------------------------------------------
    # We currently have 3 real feedback records.
    # Lower the threshold temporarily so we can
    # test the complete retraining pipeline.
    retrainer.RETRAIN_THRESHOLD = 3

    print()
    print("=" * 60)
    print("RETRAINING PIPELINE TEST")
    print("=" * 60)

    print(
        f"Current feedback count: "
        f"{retrainer.get_feedback_count()}"
    )

    print(
        f"Temporary threshold: "
        f"{retrainer.RETRAIN_THRESHOLD}"
    )

    print()

    result = retrainer.check_and_retrain()

    print()
    print("=" * 60)
    print("RETRAINING RESULT")
    print("=" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()