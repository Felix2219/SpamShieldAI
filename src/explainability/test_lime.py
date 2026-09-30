from src.explainability.lime_explainer import SpamLimeExplainer


def main():

    explainer = SpamLimeExplainer()

    message = (
        "Congratulations! You have won a free prize. "
        "Click here to claim it."
    )

    print("=" * 60)
    print("LIME EXPLANATION")
    print("=" * 60)

    print()
    print("Message:")
    print(message)

    print()
    print("Important words:")

    features = explainer.get_feature_weights(
        message,
        num_features=10
    )

    for word, weight in features:

        direction = "SPAM" if weight > 0 else "HAM"

        print(
            f"{word:<20} "
            f"{weight:+.4f} "
            f"-> {direction}"
        )


if __name__ == "__main__":
    main()