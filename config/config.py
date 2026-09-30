from pathlib import Path

# =====================================================
# Project Root
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =====================================================
# Data Paths
# =====================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

FEEDBACK_DATA_DIR = DATA_DIR / "feedback"

# =====================================================
# Model Paths
# =====================================================

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "spam_classifier.pkl"

VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"

# =====================================================
# Output Paths
# =====================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# =====================================================
# Dataset Files
# =====================================================

SMS_DATASET = RAW_DATA_DIR / "spam_sms.csv"

EMAIL_DATASET = RAW_DATA_DIR / "spam_email.csv"

CLEAN_DATASET = PROCESSED_DATA_DIR / "clean_dataset.csv"

# ==================================================
# Model Versioning
# ==================================================

MODEL_METADATA_FILE = MODEL_DIR / "model_metadata.json"