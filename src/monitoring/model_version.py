import json
from datetime import datetime

from config.config import MODEL_METADATA_FILE


class ModelVersionManager:
    """
    Manages model version information.
    """

    def __init__(self):
        self.metadata_file = MODEL_METADATA_FILE

        self.metadata_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def initialize(
        self,
        f1_score=0.9488,
        accuracy=0.9816,
        training_samples=10575
    ):
        """
        Create initial model metadata if it does not exist.
        """

        if self.metadata_file.exists():
            return self.load_metadata()

        metadata = {
            "version": "v1",
            "model": "Linear SVM",
            "feature_engineering": "TF-IDF",
            "f1_score": f1_score,
            "accuracy": accuracy,
            "training_samples": training_samples,
            "feedback_used": 0,
            "last_retrained": None,
            "created_at": datetime.now().isoformat()
        }

        self.save_metadata(metadata)

        return metadata

    def load_metadata(self):
        """
        Load current model metadata.
        """

        if not self.metadata_file.exists():
            return self.initialize()

        with open(
            self.metadata_file,
            "r"
        ) as file:

            return json.load(file)

    def save_metadata(self, metadata):
        """
        Save model metadata.
        """

        with open(
            self.metadata_file,
            "w"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

    def update_current_metrics(
        self,
        f1_score,
        accuracy,
        training_samples
    ):
        """
        Update metrics for the current production model
        without changing its version.
        """

        metadata = self.load_metadata()

        metadata["f1_score"] = f1_score
        metadata["accuracy"] = accuracy
        metadata["training_samples"] = training_samples

        self.save_metadata(metadata)

        return metadata

    def create_new_version(
        self,
        f1_score,
        accuracy,
        training_samples,
        feedback_used
    ):
        """
        Create metadata for a newly promoted model.
        """

        current = self.load_metadata()

        current_version = current.get(
            "version",
            "v1"
        )

        try:
            version_number = int(
                current_version.replace("v", "")
            )
        except ValueError:
            version_number = 1

        new_version = f"v{version_number + 1}"

        metadata = {
            "version": new_version,
            "model": "Linear SVM",
            "feature_engineering": "TF-IDF",
            "f1_score": f1_score,
            "accuracy": accuracy,
            "training_samples": training_samples,
            "feedback_used": feedback_used,
            "last_retrained": datetime.now().isoformat(),
            "created_at": current.get(
                "created_at",
                datetime.now().isoformat()
            )
        }

        self.save_metadata(metadata)

        return metadata