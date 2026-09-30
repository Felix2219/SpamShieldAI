import joblib
import numpy as np

from config.config import MODEL_FILE, VECTORIZER_FILE
from src.data.preprocessor import DataPreprocessor


class SpamPredictor:
    """
    Loads the trained spam classifier and TF-IDF vectorizer
    to make predictions on new email messages.
    """

    def __init__(self):
        # Load trained model
        self.model = joblib.load(MODEL_FILE)

        # Load TF-IDF vectorizer
        self.vectorizer = joblib.load(VECTORIZER_FILE)

        # Use the same preprocessing used during training
        self.preprocessor = DataPreprocessor()

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    def predict(self, text):
        """
        Predict whether a message is spam or ham.
        """

        # Clean the incoming message
        cleaned_text = self.preprocessor.clean_text(text)

        # Convert text into TF-IDF features
        features = self.vectorizer.transform([cleaned_text])

        # Make prediction
        prediction = self.model.predict(features)[0]

        if prediction == 1:
            label = "SPAM"
        else:
            label = "HAM"

        return label

    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    def get_confidence(self, text):
        """
        Calculate a confidence score for the prediction.

        LinearSVC does not provide predict_proba(), so we
        convert its decision score into a probability-like
        confidence value using a sigmoid function.
        """

        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)

        # TF-IDF transformation
        features = self.vectorizer.transform([cleaned_text])

        # Get decision score
        decision_score = self.model.decision_function(features)[0]

        # Convert decision score to probability-like value
        probability = 1 / (1 + np.exp(-decision_score))

        prediction = self.model.predict(features)[0]

        if prediction == 1:
            confidence = probability
        else:
            confidence = 1 - probability

        return confidence

    # --------------------------------------------------
    # Complete Prediction
    # --------------------------------------------------

    def predict_with_confidence(self, text):
        """
        Return prediction and confidence together.
        """

        prediction = self.predict(text)

        confidence = self.get_confidence(text)

        return {
            "prediction": prediction,
            "confidence": confidence
        }