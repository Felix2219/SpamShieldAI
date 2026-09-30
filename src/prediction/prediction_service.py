from src.prediction.predictor import SpamPredictor
from src.explainability.lime_explainer import SpamLimeExplainer


class PredictionService:
    """
    Combines prediction, confidence scoring,
    and LIME explainability.
    """

    def __init__(self):

        self.predictor = SpamPredictor()
        self.explainer = SpamLimeExplainer()

    # --------------------------------------------------
    # Complete Prediction
    # --------------------------------------------------

    def analyze(self, text, num_features=10):
        """
        Analyze a message and return prediction,
        confidence, and explanation.
        """

        # Get prediction and confidence
        prediction_result = (
            self.predictor.predict_with_confidence(text)
        )

        # Get LIME explanation
        explanation = (
            self.explainer.get_feature_weights(
                text,
                num_features=num_features
            )
        )

        return {
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "explanation": explanation
        }