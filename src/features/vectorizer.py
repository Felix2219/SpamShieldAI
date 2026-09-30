import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from config.config import VECTORIZER_FILE


class TfidfFeatureExtractor:
    """
    Converts cleaned email text into TF-IDF numerical features.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=10000
        )

    def fit_transform(self, texts):
        """
        Learn the vocabulary from training data and transform it.
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """
        Transform new/test text using the already-fitted vectorizer.
        """
        return self.vectorizer.transform(texts)

    def save(self):
        """
        Save the trained TF-IDF vectorizer.
        """
        VECTORIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, VECTORIZER_FILE)

    def load(self):
        """
        Load a previously saved TF-IDF vectorizer.
        """
        self.vectorizer = joblib.load(VECTORIZER_FILE)