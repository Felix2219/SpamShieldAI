from src.prediction.predictor import SpamPredictor


def main():

    predictor = SpamPredictor()

    messages = [
        "Congratulations! You have won a free prize. Click here to claim it.",
        "Hey, are we still meeting tomorrow at 10 AM?",
        "URGENT! You have won $1000. Call now to claim your reward."
    ]

    for message in messages:

        result = predictor.predict_with_confidence(message)

        print()
        print("=" * 60)
        print("Message:")
        print(message)

        print()
        print("Prediction:", result["prediction"])
        print(
            "Confidence:",
            f"{result['confidence'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()