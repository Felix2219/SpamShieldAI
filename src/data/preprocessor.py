import re
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords


class DataPreprocessor:

    def __init__(self):

        nltk.download("stopwords", quiet=True)

        self.stop_words = set(
            stopwords.words("english")
        )

    # -------------------------
    # Clean SMS Dataset
    # -------------------------

    def prepare_sms_dataset(self, sms_df):

        sms_df = sms_df[
            ["v1", "v2"]
        ].copy()

        sms_df.columns = [
            "label",
            "text"
        ]

        sms_df["label"] = sms_df[
            "label"
        ].map(
            {
                "ham": 0,
                "spam": 1
            }
        )

        return sms_df

    # -------------------------
    # Clean Email Dataset
    # -------------------------

    def prepare_email_dataset(self, email_df):

        email_df = email_df.rename(
            columns={
                "spam": "label"
            }
        )

        email_df = email_df[
            [
                "label",
                "text"
            ]
        ].copy()

        return email_df

    # -------------------------
    # Text Cleaning
    # -------------------------

    def clean_text(self, text):

        text = str(text).lower()

        text = re.sub(
            r"http\S+",
            "",
            text
        )

        text = re.sub(
            r"\S+@\S+",
            "",
            text
        )

        text = re.sub(
            r"\d+",
            "",
            text
        )

        text = text.translate(
            str.maketrans(
                "",
                "",
                string.punctuation
            )
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        words = []

        for word in text.split():

            if word not in self.stop_words:
                words.append(word)

        return " ".join(words)

    # -------------------------
    # Merge
    # -------------------------

    def merge_datasets(
        self,
        sms_df,
        email_df
    ):

        merged = pd.concat(
            [
                sms_df,
                email_df
            ],
            ignore_index=True
        )

        return merged

    # -------------------------
    # Final Cleaning
    # -------------------------

    def process_dataset(
        self,
        dataset
    ):

        dataset["clean_text"] = dataset[
            "text"
        ].apply(
            self.clean_text
        )

        dataset.drop_duplicates(
            subset="clean_text",
            inplace=True
        )

        dataset = dataset[
            dataset["clean_text"] != ""
        ]

        dataset.reset_index(
            drop=True,
            inplace=True
        )

        return dataset