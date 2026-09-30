from src.prediction.prediction_service import PredictionService


def main():

    service = PredictionService()

    message = (
        "Congratulations! You have won a free prize. "
        "Click here to claim it."
    )

    result = service.analyze(
        message,
        num_features=10
    )

    print("=" * 60)
    print("SPAMSHIELD AI ANALYSIS")
    print("=" * 60)

    print()
    print("Message:")
    print(message)

    print()
    print("Prediction:", result["prediction"])

    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print()
    print("LIME Explanation:")
    print("-" * 60)

    for word, weight in result["explanation"]:

        direction = (
            "SPAM"
            if weight > 0
            else "HAM"
        )

        print(
            f"{word:<20} "
            f"{weight:+.4f} "
            f"-> {direction}"
        )


if __name__ == "__main__":
    main()