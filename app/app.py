import sys
from pathlib import Path

import streamlit as st


# ==================================================
# Project Path
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# Project Imports
# ==================================================

from src.prediction.prediction_service import PredictionService
from src.monitoring.feedback_manager import FeedbackManager
from src.monitoring.retrainer import ModelRetrainer
from src.monitoring.model_version import ModelVersionManager


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# Load Services
# ==================================================

@st.cache_resource
def load_prediction_service():
    return PredictionService()


@st.cache_resource
def load_feedback_manager():
    return FeedbackManager()


service = load_prediction_service()
feedback_manager = load_feedback_manager()

retrainer = ModelRetrainer()

version_manager = ModelVersionManager()
model_metadata = version_manager.initialize()


# ==================================================
# Header
# ==================================================

st.title("SpamShield AI")

st.subheader(
    "Intelligent Spam Detection and Explainable AI"
)

st.write(
    "Enter an email or message below. SpamShield AI "
    "will classify it, estimate confidence, and explain "
    "the important words influencing the prediction."
)


# ==================================================
# Message Input
# ==================================================

message = st.text_area(
    "Enter email / message",
    height=200,
    placeholder=(
        "Example: Congratulations! You have won a free "
        "prize. Click here to claim it."
    )
)


# ==================================================
# Analyze Button
# ==================================================

if st.button("Analyze Message", type="primary"):

    if not message.strip():

        st.warning(
            "Please enter a message first."
        )

    else:

        with st.spinner("Analyzing message..."):

            result = service.analyze(
                message,
                num_features=10
            )

        # Store result in session
        st.session_state["prediction_result"] = result
        st.session_state["analyzed_message"] = message

        # Allow feedback for the new message
        st.session_state.pop(
            "feedback_submitted_for",
            None
        )


# ==================================================
# Display Prediction
# ==================================================

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    analyzed_message = (
        st.session_state["analyzed_message"]
    )

    prediction = result["prediction"]

    confidence = result["confidence"]


    # ==================================================
    # Prediction Result
    # ==================================================

    st.divider()

    st.header("Prediction")

    col1, col2 = st.columns(2)

    with col1:

        if prediction == "SPAM":

            st.error("SPAM")

        else:

            st.success("HAM")

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ==================================================
    # Feedback
    # ==================================================

    st.divider()

    st.header("Was this prediction correct?")

    feedback_col1, feedback_col2 = st.columns(2)


    # ==================================================
    # Correct Feedback
    # ==================================================

    with feedback_col1:

        if st.button(
            "Yes, correct",
            key="feedback_yes"
        ):

            if (
                st.session_state.get(
                    "feedback_submitted_for"
                )
                != analyzed_message
            ):

                correct_label = (
                    1 if prediction == "SPAM" else 0
                )

                predicted_label = (
                    1 if prediction == "SPAM" else 0
                )

                feedback_manager.save_feedback(
                    text=analyzed_message,
                    predicted_label=predicted_label,
                    correct_label=correct_label
                )

                st.session_state[
                    "feedback_submitted_for"
                ] = analyzed_message

                st.success(
                    "Thanks. Your feedback has been recorded."
                )


                # ------------------------------------------
                # Automatic Retraining Check
                # ------------------------------------------

                retrain_result = (
                    retrainer.check_and_retrain()
                )


                # ------------------------------------------
                # Model Promoted
                # ------------------------------------------

                if retrain_result["status"] == "promoted":

                    st.success(
                        "A new model was trained and "
                        "promoted because it performed "
                        "better than the current model."
                    )

                    # Clear cached services
                    # so the new model is loaded.
                    st.cache_resource.clear()

                    st.rerun()


                # ------------------------------------------
                # Candidate Rejected
                # ------------------------------------------

                elif retrain_result["status"] == "rejected":

                    st.info(
                        "Retraining was completed, but "
                        "the new model did not outperform "
                        "the current model. The existing "
                        "model has been retained."
                    )


            else:

                st.info(
                    "Feedback for this message has "
                    "already been recorded."
                )


    # ==================================================
    # Incorrect Feedback
    # ==================================================

    with feedback_col2:

        if st.button(
            "No, incorrect",
            key="feedback_no"
        ):

            if (
                st.session_state.get(
                    "feedback_submitted_for"
                )
                != analyzed_message
            ):

                predicted_label = (
                    1 if prediction == "SPAM" else 0
                )

                correct_label = (
                    0 if prediction == "SPAM" else 1
                )

                feedback_manager.save_feedback(
                    text=analyzed_message,
                    predicted_label=predicted_label,
                    correct_label=correct_label
                )

                st.session_state[
                    "feedback_submitted_for"
                ] = analyzed_message

                st.success(
                    "Thanks. Your correction has been recorded."
                )


                # ------------------------------------------
                # Automatic Retraining Check
                # ------------------------------------------

                retrain_result = (
                    retrainer.check_and_retrain()
                )


                # ------------------------------------------
                # Model Promoted
                # ------------------------------------------

                if retrain_result["status"] == "promoted":

                    st.success(
                        "A new model was trained and "
                        "promoted because it performed "
                        "better than the current model."
                    )

                    # Clear cached services
                    # so the new model is loaded.
                    st.cache_resource.clear()

                    st.rerun()


                # ------------------------------------------
                # Candidate Rejected
                # ------------------------------------------

                elif retrain_result["status"] == "rejected":

                    st.info(
                        "Retraining was completed, but "
                        "the new model did not outperform "
                        "the current model. The existing "
                        "model has been retained."
                    )


            else:

                st.info(
                    "Feedback for this message has "
                    "already been recorded."
                )


    # ==================================================
    # Model Information
    # ==================================================

    st.divider()

    st.header("Model Information")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:

        st.write(
            f"**Model:** {model_metadata['model']}"
        )

    with model_col2:

        st.write(
            f"**Feature Engineering:** "
            f"{model_metadata['feature_engineering']}"
        )

    with model_col3:

        st.write(
            f"**Model Version:** "
            f"{model_metadata['version']}"
        )


    # ==================================================
    # Model Metrics
    # ==================================================

    version_col1, version_col2, version_col3 = st.columns(3)

    with version_col1:

        st.metric(
            "Model F1 Score",
            f"{model_metadata['f1_score']:.4f}"
        )

    with version_col2:

        st.metric(
            "Training Samples",
            model_metadata["training_samples"]
        )

    with version_col3:

        st.metric(
            "Feedback Used",
            model_metadata["feedback_used"]
        )


    # ==================================================
    # LIME Explanation
    # ==================================================

    st.divider()

    st.header(
        "Why did the model make this prediction?"
    )

    st.write(
        "LIME identifies words that locally influence "
        "the model's decision."
    )

    explanation = result["explanation"]

    for word, weight in explanation:

        if weight > 0:

            st.write(
                f"**{word}** → influences **SPAM** "
                f"({weight:+.4f})"
            )

        else:

            st.write(
                f"**{word}** → influences **HAM** "
                f"({weight:+.4f})"
            )


    # ==================================================
    # Feedback Statistics
    # ==================================================

    st.divider()

    st.header("Feedback Statistics")

    total_feedback = (
        feedback_manager.get_feedback_count()
    )

    correct_feedback = (
        feedback_manager.get_correct_count()
    )

    incorrect_feedback = (
        feedback_manager.get_incorrect_count()
    )

    stat1, stat2, stat3 = st.columns(3)

    with stat1:

        st.metric(
            "Total Feedback",
            total_feedback
        )

    with stat2:

        st.metric(
            "Correct",
            correct_feedback
        )

    with stat3:

        st.metric(
            "Incorrect",
            incorrect_feedback
        )


    # ==================================================
    # Continuous Learning
    # ==================================================

    st.divider()

    st.header("Continuous Learning")

    feedback_count = (
        retrainer.get_feedback_count()
    )

    last_retrained_count = (
        retrainer.get_last_retrained_count()
    )

    feedback_since_retraining = (
        feedback_count - last_retrained_count
    )

    remaining_feedback = max(
        retrainer.RETRAIN_THRESHOLD
        - feedback_since_retraining,
        0
    )


    # --------------------------------------------------
    # Retraining Metrics
    # --------------------------------------------------

    retrain_col1, retrain_col2, retrain_col3 = (
        st.columns(3)
    )

    with retrain_col1:

        st.metric(
            "Retraining Threshold",
            retrainer.RETRAIN_THRESHOLD
        )

    with retrain_col2:

        st.metric(
            "Since Last Retraining",
            feedback_since_retraining
        )

    with retrain_col3:

        st.metric(
            "Feedback Until Retraining",
            remaining_feedback
        )


    # --------------------------------------------------
    # Retraining Message
    # --------------------------------------------------

    if remaining_feedback == 0:

        st.warning(
            "Retraining threshold reached. "
            "The next feedback submission will "
            "trigger automatic model retraining."
        )

    else:

        st.info(
            f"{remaining_feedback} more feedback "
            "record(s) are required before automatic "
            "retraining is triggered."
        )


    # ==================================================
    # Technical Details
    # ==================================================

    with st.expander("Technical Details"):

        st.write(
            "The message is cleaned using the same "
            "preprocessing pipeline used during training."
        )

        st.write(
            "The cleaned text is converted into TF-IDF "
            "features and passed to the trained Linear SVM."
        )

        st.write(
            "LIME generates a local explanation showing "
            "which words contributed to the prediction."
        )

        st.write(
            "User feedback is stored for future model "
            "retraining and continuous improvement."
        )

        st.write(
            "When the feedback threshold is reached, "
            "a candidate model is trained and evaluated "
            "against the current production model."
        )

        st.write(
            "The candidate model is promoted only if "
            "its F1-score is better than the current model."
        )

        st.write(
            "Each successfully promoted model receives "
            "a new version number such as v2, v3, and v4."
        )