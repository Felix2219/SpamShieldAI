import pandas as pd
from datetime import datetime
from pathlib import Path

from config.config import FEEDBACK_DATA_DIR


class FeedbackManager:
    """
    Stores user feedback from spam/ham predictions.
    """

    def __init__(self):

        self.feedback_dir = FEEDBACK_DATA_DIR

        self.feedback_file = (
            self.feedback_dir / "feedback.csv"
        )

        # Create feedback directory if needed
        self.feedback_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Create CSV if it doesn't exist
        if not self.feedback_file.exists():

            dataframe = pd.DataFrame(
                columns=[
                    "text",
                    "predicted_label",
                    "correct_label",
                    "timestamp"
                ]
            )

            dataframe.to_csv(
                self.feedback_file,
                index=False
            )

    # --------------------------------------------------
    # Save Feedback
    # --------------------------------------------------

    def save_feedback(
        self,
        text,
        predicted_label,
        correct_label
    ):
        """
        Save one user feedback entry.
        """

        feedback = pd.DataFrame([
            {
                "text": text,
                "predicted_label": predicted_label,
                "correct_label": correct_label,
                "timestamp": datetime.now().isoformat()
            }
        ])

        feedback.to_csv(
            self.feedback_file,
            mode="a",
            header=False,
            index=False
        )

    # --------------------------------------------------
    # Load Feedback
    # --------------------------------------------------

    def load_feedback(self):
        """
        Load all collected feedback.
        """

        if not self.feedback_file.exists():

            return pd.DataFrame(
                columns=[
                    "text",
                    "predicted_label",
                    "correct_label",
                    "timestamp"
                ]
            )

        return pd.read_csv(
            self.feedback_file
        )

    # --------------------------------------------------
    # Feedback Count
    # --------------------------------------------------

    def get_feedback_count(self):
        """
        Return the number of feedback records.
        """

        dataframe = self.load_feedback()

        return len(dataframe)

    # --------------------------------------------------
    # Correct Prediction Count
    # --------------------------------------------------

    def get_correct_count(self):
        """
        Return the number of correctly predicted messages
        according to user feedback.
        """

        dataframe = self.load_feedback()

        if dataframe.empty:
            return 0

        return (
            dataframe["predicted_label"]
            == dataframe["correct_label"]
        ).sum()

    # --------------------------------------------------
    # Incorrect Prediction Count
    # --------------------------------------------------

    def get_incorrect_count(self):
        """
        Return the number of incorrectly predicted messages.
        """

        dataframe = self.load_feedback()

        if dataframe.empty:
            return 0

        return (
            dataframe["predicted_label"]
            != dataframe["correct_label"]
        ).sum()