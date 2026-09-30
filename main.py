import mlflow

from sklearn.model_selection import train_test_split

from src.data.loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.features.vectorizer import TfidfFeatureExtractor
from src.training.trainer import ModelTrainer
from config.config import MODEL_FILE


def main():

    # ==================================================
    # 0. MLflow Experiment
    # ==================================================

    mlflow.set_experiment("SpamShieldAI")

    # ==================================================
    # 1. LOAD DATASETS
    # ==================================================

    loader = DatasetLoader()

    sms, email = loader.load_all()

    # ==================================================
    # 2. PREPROCESS DATA
    # ==================================================

    preprocessor = DataPreprocessor()

    sms = preprocessor.prepare_sms_dataset(sms)

    email = preprocessor.prepare_email_dataset(email)

    merged = preprocessor.merge_datasets(sms, email)

    final_dataset = preprocessor.process_dataset(merged)

    print("=" * 60)
    print("Final Dataset")
    print("=" * 60)

    print(final_dataset.head())

    print()

    print("Shape:", final_dataset.shape)

    # ==================================================
    # 3. TRAIN / TEST SPLIT
    # ==================================================

    X = final_dataset["clean_text"]
    y = final_dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print()
    print("=" * 60)
    print("Train / Test Split")
    print("=" * 60)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ==================================================
    # 4. TF-IDF FEATURE ENGINEERING
    # ==================================================

    tfidf = TfidfFeatureExtractor()

    X_train_tfidf = tfidf.fit_transform(X_train)

    X_test_tfidf = tfidf.transform(X_test)

    # Save TF-IDF vectorizer
    tfidf.save()

    print()
    print("=" * 60)
    print("TF-IDF Feature Engineering")
    print("=" * 60)

    print("Training feature shape:", X_train_tfidf.shape)
    print("Testing feature shape:", X_test_tfidf.shape)
    print("Vectorizer saved successfully.")

    # ==================================================
    # 5. MODEL TRAINING
    # ==================================================

    print()
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    trainer = ModelTrainer()

    trainer.train_models(
        X_train_tfidf,
        y_train
    )

    # ==================================================
    # 6. MODEL EVALUATION
    # ==================================================

    trainer.evaluate_models(
        X_test_tfidf,
        y_test
    )

    # Display comparison
    trainer.display_results()

    # ==================================================
    # 7. SELECT BEST MODEL
    # ==================================================

    best_name, best_model = trainer.get_best_model()

    print()
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print("Selected model:", best_name)

    # ==================================================
    # 8. SAVE BEST MODEL
    # ==================================================

    trainer.save_model(
        best_model,
        MODEL_FILE
    )


if __name__ == "__main__":
    main()