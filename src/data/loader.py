import pandas as pd
from config.config import SMS_DATASET, EMAIL_DATASET


class DatasetLoader:
    """
    Loads all datasets used in the project.
    """

    def __init__(self):
        self.sms_path = SMS_DATASET
        self.email_path = EMAIL_DATASET

    def load_sms_dataset(self):
        """Load the SMS spam dataset."""
        return pd.read_csv(
            self.sms_path,
            encoding="latin-1"
        )

    def load_email_dataset(self):
        """Load the Email spam dataset."""
        return pd.read_csv(
            self.email_path
        )

    def load_all(self):
        """Load both datasets."""
        sms = self.load_sms_dataset()
        email = self.load_email_dataset()

        return sms, email