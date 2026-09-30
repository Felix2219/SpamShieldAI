import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelTrainer:
    """
    Trains and evaluates multiple machine learning models
    for spam email classification.
    """

    def __init__(self):

        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

            "Naive Bayes": MultinomialNB(),

            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),

            "Linear SVM": LinearSVC(
                random_state=42
            )
        }

        self.results = {}

    # --------------------------------------------------
    # Train Models
    # --------------------------------------------------

    def train_models(self, X_train, y_train):
        """
        Train all machine learning models.
        """

        for name, model in self.models.items():

            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)

            print(f"{name} training completed.")

    # --------------------------------------------------
    # Evaluate Models
    # --------------------------------------------------

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models and log their
        metrics and parameters to MLflow.
        """

        for name, model in self.models.items():

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            # Store results
            self.results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

            # ------------------------------------------
            # MLflow Tracking
            # ------------------------------------------

            with mlflow.start_run(run_name=name):

                mlflow.log_param(
                    "model",
                    name
                )

                mlflow.log_params(
                    model.get_params()
                )

                mlflow.log_metric(
                    "accuracy",
                    accuracy
                )

                mlflow.log_metric(
                    "precision",
                    precision
                )

                mlflow.log_metric(
                    "recall",
                    recall
                )

                mlflow.log_metric(
                    "f1_score",
                    f1
                )

                mlflow.sklearn.log_model(
                    model,
                    name="model"
                )

    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------

    def display_results(self):
        """
        Display the performance of all models.
        """

        print()
        print("=" * 85)
        print("MODEL COMPARISON")
        print("=" * 85)

        print(
            f"{'Model':<25}"
            f"{'Accuracy':<15}"
            f"{'Precision':<15}"
            f"{'Recall':<15}"
            f"{'F1 Score':<15}"
        )

        print("-" * 85)

        for name, metrics in self.results.items():

            print(
                f"{name:<25}"
                f"{metrics['accuracy']:<15.4f}"
                f"{metrics['precision']:<15.4f}"
                f"{metrics['recall']:<15.4f}"
                f"{metrics['f1_score']:<15.4f}"
            )

        print("=" * 85)

    # --------------------------------------------------
    # Select Best Model
    # --------------------------------------------------

    def get_best_model(self):
        """
        Select the model with the highest F1-score.
        """

        best_model_name = max(
            self.results,
            key=lambda name: self.results[name]["f1_score"]
        )

        best_model = self.models[best_model_name]

        return best_model_name, best_model

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    def save_model(self, model, path):
        """
        Save the selected model to disk.
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model,
            path
        )

        print()
        print(f"Best model saved to: {path}")