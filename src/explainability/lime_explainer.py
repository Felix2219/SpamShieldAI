import joblib
import numpy as np

from lime.lime_text import LimeTextExplainer

from config.config import MODEL_FILE, VECTORIZER_FILE
from src.data.preprocessor import DataPreprocessor


class SpamLimeExplainer:
    """
    Generates LIME explanations for spam/ham predictions.
    """

    def __init__(self):

        # Load trained model
        self.model = joblib.load(MODEL_FILE)

        # Load trained TF-IDF vectorizer
        self.vectorizer = joblib.load(VECTORIZER_FILE)

        # Use the same preprocessing as training
        self.preprocessor = DataPreprocessor()

        # LIME text explainer
        self.explainer = LimeTextExplainer(
            class_names=["HAM", "SPAM"],
            random_state=42
        )

    # --------------------------------------------------
    # Prediction Function for LIME
    # --------------------------------------------------

    def predict_proba_for_lime(self, texts):
        """
        Convert SVM decision scores into probability-like
        values that LIME can use.
        """

        cleaned_texts = [
            self.preprocessor.clean_text(text)
            for text in texts
        ]

        features = self.vectorizer.transform(cleaned_texts)

        # Linear SVM decision scores
        scores = self.model.decision_function(features)

        # Sigmoid conversion
        spam_probability = 1 / (1 + np.exp(-scores))

        ham_probability = 1 - spam_probability

        # LIME expects:
        # [HAM probability, SPAM probability]

        probabilities = np.column_stack(
            (ham_probability, spam_probability)
        )

        return probabilities

    # --------------------------------------------------
    # Generate Explanation
    # --------------------------------------------------

    def explain(self, text, num_features=10):
        """
        Generate a LIME explanation for a message.
        """

        explanation = self.explainer.explain_instance(
            text,
            self.predict_proba_for_lime,
            num_features=num_features
        )

        return explanation

    # --------------------------------------------------
    # Get Explanation as List
    # --------------------------------------------------

    def get_feature_weights(self, text, num_features=10):
        """
        Return important words and their LIME weights.
        """

        explanation = self.explain(
            text,
            num_features
        )

        return explanation.as_list()