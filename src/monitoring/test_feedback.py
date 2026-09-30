from src.monitoring.feedback_manager import FeedbackManager


def main():

    manager = FeedbackManager()

    # Example feedback
    manager.save_feedback(
        text="Congratulations! You won a free prize.",
        predicted_label=1,
        correct_label=1
    )

    manager.save_feedback(
        text="Hey, are we meeting tomorrow?",
        predicted_label=0,
        correct_label=1
    )

    print("=" * 60)
    print("FEEDBACK SYSTEM TEST")
    print("=" * 60)

    print()
    print("Feedback file:")
    print(manager.feedback_file)

    print()
    print("Total feedback:", manager.get_feedback_count())

    print(
        "Correct predictions:",
        manager.get_correct_count()
    )

    print(
        "Incorrect predictions:",
        manager.get_incorrect_count()
    )

    print()
    print("Feedback data:")
    print(manager.load_feedback())


if __name__ == "__main__":
    main()