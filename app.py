import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# Constants
CLASS_NAMES = ["Normal", "Pneumonia"]
IMG_SIZE = 224

# Model paths
YOLO_MODEL_PATH = "yolov8-cls-pneumonia-classification-model.pt"
RESNET_MODEL_PATH = "resnet-50-pneumonia-classification-model.keras"


@st.cache_resource
def load_resnet_model():
    """Load ResNet50 Keras model for classification."""
    import keras

    model_path = Path(RESNET_MODEL_PATH)
    if model_path.exists():
        model = keras.models.load_model(str(model_path))
        return model
    else:
        st.error(f"ResNet model not found at: {model_path}")
        return None


@st.cache_resource
def load_yolo_model():
    """Load YOLO classification model."""
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model_path = Path(YOLO_MODEL_PATH)
    if model_path.exists():
        model = YOLO(str(model_path))
        return model
    else:
        st.error(f"YOLO model not found at: {model_path}")
        return None


THRESHOLD = 0.3


def predict_resnet(model, image: Image.Image):
    """
    Predict pneumonia from a chest X-ray image using ResNet model.

    Args:
        model: Trained Keras model
        image: PIL Image

    Returns:
        tuple: (prediction label, confidence percentage)
    """
    if model is None:
        return None, 0

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to 224x224
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array and normalize by dividing by 255.0
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    # Apply custom threshold
    pred_class = 1 if prediction > THRESHOLD else 0

    # Determine label and confidence
    label = 'PNEUMONIA' if pred_class == 1 else 'NORMAL'
    confidence = prediction if pred_class == 1 else 1 - prediction

    return label, float(confidence) * 100


def predict_yolo(model, image: Image.Image):
    """
    Predict pneumonia from a chest X-ray image using YOLO model.

    Args:
        model: Trained YOLO model
        image: PIL Image

    Returns:
        tuple: (prediction label, confidence percentage)
    """
    if model is None:
        return None, 0

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Make prediction with imgsz=224
    result = model.predict(image, verbose=False, imgsz=IMG_SIZE)

    # Get probabilities
    probs = result[0].probs.data.cpu().numpy()
    pneumonia_prob = probs[1]

    # Apply custom threshold
    pred_class = 1 if pneumonia_prob > THRESHOLD else 0

    # Determine label and confidence
    label = 'PNEUMONIA' if pred_class == 1 else 'NORMAL'
    confidence = pneumonia_prob if pred_class == 1 else 1 - pneumonia_prob

    return label, float(confidence) * 100


def main():
    st.title("🩺 Chest X-Ray Pneumonia Detection")
    st.markdown("Classify chest X-rays as **Normal** or **Pneumonia** using deep learning models.")
    st.markdown("---")

    # Step 1: Model Selection
    st.subheader("Step 1: Select Model")
    model_choice = st.selectbox(
        "Choose Classification Model",
        options=["-- Select a model --", "YOLOv8-cls", "ResNet-50"],
        index=0,
        help="Select which trained model to use for prediction"
    )

    if model_choice == "-- Select a model --":
        st.info("Please select a model to begin.")

        # Show model comparison
        st.markdown("### Model Performance Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**ResNet-50**")
            st.markdown("- Accuracy: 86.22%")
            st.markdown("- Precision: 84.70%")
            st.markdown("- Recall: 95.13%")
            st.markdown("- F1 Score: 89.61%")

        with col2:
            st.markdown("**YOLOv8-cls**")
            st.markdown("- Accuracy: 94.87%")
            st.markdown("- Precision: 93.23%")
            st.markdown("- Recall: 98.97%")
            st.markdown("- F1 Score: 96.02%")
        return

    # Show selected model info
    if model_choice == "YOLOv8-cls":
        st.success(f"Selected: **YOLOv8-cls** (Accuracy: 94.87%, F1: 96.02%)")
    else:
        st.success(f"Selected: **ResNet-50** (Accuracy: 86.22%, F1: 89.61%)")

    # Step 2: Image Upload
    st.markdown("---")
    st.subheader("Step 2: Upload Chest X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a chest X-ray image for classification"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)

        # Step 3: Prediction
        with col2:
            st.subheader("Step 3: Prediction Results")

            with st.spinner("Loading model and making prediction..."):
                try:
                    if model_choice == "YOLOv8-cls":
                        model = load_yolo_model()
                        prediction, confidence = predict_yolo(model, image)
                    else:  # ResNet-50
                        model = load_resnet_model()
                        prediction, confidence = predict_resnet(model, image)

                    # Display results
                    if prediction:
                        # Result card
                        if "pneumonia" in prediction.lower():
                            st.error(f"🔴 **Diagnosis: {prediction.upper()}**")
                        else:
                            st.success(f"🟢 **Diagnosis: {prediction.upper()}**")

                        # Confidence meter
                        st.metric(label="Confidence", value=f"{confidence:.2f}%")
                        st.progress(confidence / 100)

                        # Additional info
                        st.markdown("---")
                        st.caption(f"Model: **{model_choice}** | Threshold: 0.3")
                    else:
                        st.error("Failed to make prediction. Check if the model file exists.")

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
    else:
        st.info("📤 Please upload a chest X-ray image to get a prediction.")

    # Footer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer**: This tool is for educational purposes only and should not be used for medical diagnosis.")


if __name__ == "__main__":
    main()
