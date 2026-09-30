import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

# ----------------------------
# Load Datasets
# ----------------------------

sms = pd.read_csv(
    "data/raw/spam_sms.csv",
    encoding="latin-1"
)

email = pd.read_csv(
    "data/raw/spam_email.csv"
)

# ----------------------------
# Clean SMS Dataset
# ----------------------------

sms = sms[["v1", "v2"]]

sms.columns = [
    "label",
    "text"
]

sms["label"] = sms["label"].map({
    "ham": 0,
    "spam": 1
})

# ----------------------------
# Clean Email Dataset
# ----------------------------

email = email.rename(
    columns={
        "spam": "label"
    }
)

email = email[
    ["label", "text"]
]

# ----------------------------
# Merge Datasets
# ----------------------------

df = pd.concat(
    [sms, email],
    ignore_index=True
)

print("Merged Dataset Shape:", df.shape)

# ----------------------------
# Text Cleaning
# ----------------------------

stop_words = set(
    stopwords.words("english")
)

def clean_text(text):

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

    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_text"] = df["text"].apply(
    clean_text
)

# ----------------------------
# Remove duplicates
# ----------------------------

df.drop_duplicates(
    subset="clean_text",
    inplace=True
)

# ----------------------------
# Remove empty rows
# ----------------------------

df = df[
    df["clean_text"] != ""
]

# ----------------------------
# Save Dataset
# ----------------------------

df.to_csv(
    "data/processed/clean_dataset.csv",
    index=False
)

print()

print(df.head())

print()

print("Final Shape:", df.shape)

print()

print("Saved Successfully!")