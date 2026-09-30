import json
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score,
    accuracy_score
)

from config.config import (
    CLEAN_DATASET,
    FEEDBACK_DATA_DIR,
    MODEL_FILE,
    VECTORIZER_FILE
)

from src.data.preprocessor import DataPreprocessor
from src.features.vectorizer import TfidfFeatureExtractor
from src.monitoring.model_version import ModelVersionManager


class ModelRetrainer:
    """
    Handles automatic model retraining using user feedback.

    The retrainer:
    1. Checks the feedback threshold.
    2. Combines the original dataset with corrected feedback.
    3. Trains a candidate Linear SVM.
    4. Evaluates the current production model.
    5. Evaluates the candidate model on the same test set.
    6. Promotes the candidate only if its F1-score is better.
    7. Updates model version metadata after promotion.
    """

    # ==================================================
    # Configuration
    # ==================================================

    RETRAIN_THRESHOLD = 100

    def __init__(self):

        self.feedback_file = (
            FEEDBACK_DATA_DIR / "feedback.csv"
        )

        self.state_file = (
            FEEDBACK_DATA_DIR / "retraining_state.json"
        )

        self.preprocessor = DataPreprocessor()

        self.version_manager = (
            ModelVersionManager()
        )


    # ==================================================
    # Feedback Count
    # ==================================================

    def get_feedback_count(self):

        if not self.feedback_file.exists():
            return 0

        feedback = pd.read_csv(
            self.feedback_file
        )

        return len(feedback)


    # ==================================================
    # Retraining State
    # ==================================================

    def get_last_retrained_count(self):

        if not self.state_file.exists():
            return 0

        try:

            with open(
                self.state_file,
                "r"
            ) as file:

                state = json.load(file)

            return state.get(
                "last_retrained_count",
                0
            )

        except (
            json.JSONDecodeError,
            OSError
        ):

            return 0


    def save_retraining_state(
        self,
        count
    ):

        self.state_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        state = {
            "last_retrained_count": count,
            "last_retrained_at": (
                datetime.now().isoformat()
            )
        }

        with open(
            self.state_file,
            "w"
        ) as file:

            json.dump(
                state,
                file,
                indent=4
            )


    # ==================================================
    # Check Retraining Requirement
    # ==================================================

    def should_retrain(self):

        feedback_count = (
            self.get_feedback_count()
        )

        last_retrained_count = (
            self.get_last_retrained_count()
        )

        new_feedback = (
            feedback_count
            - last_retrained_count
        )

        return (
            new_feedback
            >= self.RETRAIN_THRESHOLD
        )


    # ==================================================
    # Prepare Training Data
    # ==================================================

    def prepare_training_data(self):

        # ----------------------------------------------
        # Load original dataset
        # ----------------------------------------------

        base_data = pd.read_csv(
            CLEAN_DATASET
        )

        base_data = base_data[
            [
                "label",
                "text",
                "clean_text"
            ]
        ].copy()


        # ----------------------------------------------
        # Load feedback
        # ----------------------------------------------

        if not self.feedback_file.exists():

            return base_data

        feedback = pd.read_csv(
            self.feedback_file
        )

        if feedback.empty:

            return base_data


        # ----------------------------------------------
        # Convert feedback to training format
        # ----------------------------------------------

        feedback_data = pd.DataFrame({
            "label": feedback[
                "correct_label"
            ],
            "text": feedback[
                "text"
            ]
        })

        feedback_data[
            "clean_text"
        ] = feedback_data[
            "text"
        ].apply(
            self.preprocessor.clean_text
        )


        # ----------------------------------------------
        # Remove empty feedback
        # ----------------------------------------------

        feedback_data = feedback_data[
            feedback_data[
                "clean_text"
            ] != ""
        ].copy()


        if feedback_data.empty:

            return base_data


        # ----------------------------------------------
        # Feedback corrections override
        # original dataset labels
        # ----------------------------------------------

        corrected_texts = set(
            feedback_data[
                "clean_text"
            ]
        )

        base_data = base_data[
            ~base_data[
                "clean_text"
            ].isin(corrected_texts)
        ]


        # ----------------------------------------------
        # Combine original + feedback
        # ----------------------------------------------

        combined = pd.concat(
            [
                base_data,
                feedback_data
            ],
            ignore_index=True
        )


        # ----------------------------------------------
        # Remove duplicate text
        # ----------------------------------------------

        combined.drop_duplicates(
            subset="clean_text",
            keep="last",
            inplace=True
        )

        combined.reset_index(
            drop=True,
            inplace=True
        )


        return combined


    # ==================================================
    # Retrain Model
    # ==================================================

    def retrain(self):

        print()
        print("=" * 60)
        print("AUTOMATIC MODEL RETRAINING")
        print("=" * 60)


        # ----------------------------------------------
        # Prepare data
        # ----------------------------------------------

        dataset = (
            self.prepare_training_data()
        )

        print(
            "Training samples:",
            len(dataset)
        )


        X = dataset[
            "clean_text"
        ]

        y = dataset[
            "label"
        ]


        # ----------------------------------------------
        # Train/Test Split
        # ----------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        )


        # ==================================================
        # Candidate TF-IDF
        # ==================================================

        candidate_vectorizer = (
            TfidfFeatureExtractor()
        )

        X_train_tfidf = (
            candidate_vectorizer.fit_transform(
                X_train
            )
        )

        X_test_tfidf = (
            candidate_vectorizer.transform(
                X_test
            )
        )


        # ==================================================
        # Candidate Model
        # ==================================================

        candidate_model = LinearSVC(
            random_state=42
        )

        candidate_model.fit(
            X_train_tfidf,
            y_train
        )


        # ----------------------------------------------
        # Candidate predictions
        # ----------------------------------------------

        candidate_predictions = (
            candidate_model.predict(
                X_test_tfidf
            )
        )


        # ----------------------------------------------
        # Candidate metrics
        # ----------------------------------------------

        candidate_f1 = f1_score(
            y_test,
            candidate_predictions,
            zero_division=0
        )

        candidate_accuracy = (
            accuracy_score(
                y_test,
                candidate_predictions
            )
        )


        # ==================================================
        # Current Production Model
        # ==================================================

        current_model = joblib.load(
            MODEL_FILE
        )

        current_vectorizer = joblib.load(
            VECTORIZER_FILE
        )


        # ----------------------------------------------
        # Current model features
        # ----------------------------------------------

        current_features = (
            current_vectorizer.transform(
                X_test
            )
        )


        # ----------------------------------------------
        # Current predictions
        # ----------------------------------------------

        current_predictions = (
            current_model.predict(
                current_features
            )
        )


        # ----------------------------------------------
        # Current metrics
        # ----------------------------------------------

        current_f1 = f1_score(
            y_test,
            current_predictions,
            zero_division=0
        )

        current_accuracy = (
            accuracy_score(
                y_test,
                current_predictions
            )
        )


        # ==================================================
        # Display Results
        # ==================================================

        print()
        print("Current model:")
        print(
            f"Accuracy: "
            f"{current_accuracy:.4f}"
        )
        print(
            f"F1 Score: "
            f"{current_f1:.4f}"
        )

        print()
        print("Candidate model:")
        print(
            f"Accuracy: "
            f"{candidate_accuracy:.4f}"
        )
        print(
            f"F1 Score: "
            f"{candidate_f1:.4f}"
        )


        # ==================================================
        # MLflow Tracking
        # ==================================================

        with mlflow.start_run(
            run_name="Automatic_Retraining"
        ):

            mlflow.log_param(
                "training_type",
                "automatic_feedback_retraining"
            )

            mlflow.log_param(
                "feedback_count",
                self.get_feedback_count()
            )

            mlflow.log_param(
                "training_samples",
                len(dataset)
            )

            mlflow.log_metric(
                "candidate_accuracy",
                candidate_accuracy
            )

            mlflow.log_metric(
                "candidate_f1",
                candidate_f1
            )

            mlflow.log_metric(
                "current_accuracy",
                current_accuracy
            )

            mlflow.log_metric(
                "current_f1",
                current_f1
            )

            mlflow.sklearn.log_model(
                candidate_model,
                name="candidate_model"
            )


        # ==================================================
        # Model Promotion
        # ==================================================

        if candidate_f1 > current_f1:

            print()
            print(
                "Candidate model is better."
            )

            print(
                "Promoting candidate model..."
            )


            # ----------------------------------------------
            # Save candidate model
            # ----------------------------------------------

            joblib.dump(
                candidate_model,
                MODEL_FILE
            )


            # ----------------------------------------------
            # Save candidate vectorizer
            # ----------------------------------------------

            candidate_vectorizer.save()


            # ----------------------------------------------
            # Create new model version
            # ----------------------------------------------

            metadata = (
                self.version_manager
                .create_new_version(
                    f1_score=candidate_f1,
                    accuracy=candidate_accuracy,
                    training_samples=len(dataset),
                    feedback_used=(
                        self.get_feedback_count()
                    )
                )
            )


            # ----------------------------------------------
            # Update retraining state
            # ----------------------------------------------

            self.save_retraining_state(
                self.get_feedback_count()
            )


            print(
                "New model promoted successfully."
            )

            print(
                f"New model version: "
                f"{metadata['version']}"
            )


            return {
                "status": "promoted",
                "version": metadata[
                    "version"
                ],
                "current_f1": current_f1,
                "candidate_f1": candidate_f1
            }


        # ==================================================
        # Candidate Rejected
        # ==================================================

        else:

            print()
            print(
                "Candidate model did not improve."
            )

            print(
                "Keeping current production model."
            )


            # ----------------------------------------------
            # Update current model metrics
            # ----------------------------------------------

            self.version_manager.update_current_metrics(
                f1_score=current_f1,
                accuracy=current_accuracy,
                training_samples=len(dataset)
            )


            # ----------------------------------------------
            # Update retraining state
            # ----------------------------------------------

            self.save_retraining_state(
                self.get_feedback_count()
            )


            return {
                "status": "rejected",
                "current_f1": current_f1,
                "candidate_f1": candidate_f1
            }


    # ==================================================
    # Main Retraining Check
    # ==================================================

    def check_and_retrain(self):

        feedback_count = (
            self.get_feedback_count()
        )

        last_retrained_count = (
            self.get_last_retrained_count()
        )

        new_feedback = (
            feedback_count
            - last_retrained_count
        )


        print()
        print(
            f"Feedback records: "
            f"{feedback_count}/"
            f"{self.RETRAIN_THRESHOLD}"
        )


        print(
            f"New feedback since last retraining: "
            f"{new_feedback}"
        )


        # ----------------------------------------------
        # Not enough feedback
        # ----------------------------------------------

        if not self.should_retrain():

            return {
                "status": "not_ready",
                "feedback_count": feedback_count,
                "new_feedback": new_feedback,
                "threshold": (
                    self.RETRAIN_THRESHOLD
                )
            }


        # ----------------------------------------------
        # Start retraining
        # ----------------------------------------------

        return self.retrain()