import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# Application configuration
# ============================================================
st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌱",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "plant_disease_cnn.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"
LEAF_VALIDATOR_PATH = BASE_DIR / "models" / "leaf_validator.keras"

IMG_SIZE = (128, 128)

# Must match the binary validator's training configuration.
# The Colab example provided earlier uses 160x160 and a
# Rescaling(1/127.5, offset=-1) layer INSIDE the model.
VALIDATION_SIZE = (160, 160)

# The Colab binary dataset was ordered as:
# ['leaf', 'non_leaf'].
# A sigmoid output is therefore interpreted as P(non_leaf)
# only if the training code actually used this label convention.
NON_LEAF_THRESHOLD = 0.5
LEAF_THRESHOLD = 0.5


# ============================================================
# Load models and class labels
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_leaf_validator():
    if not LEAF_VALIDATOR_PATH.is_file():
        return None

    return tf.keras.models.load_model(LEAF_VALIDATOR_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Image preprocessing
# ============================================================
def prepare_rgb_image(image):
    """Correct image orientation and convert to RGB."""
    return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_disease_image(image):
    """Preprocessing for the existing five-class CNN."""
    image = prepare_rgb_image(image)
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # The existing CNN contains Rescaling(1/255).
    # Do not normalize a second time.
    return image_array


def preprocess_validator_image(image):
    """Prepare the image for the separately trained binary model."""
    image = prepare_rgb_image(image)
    image = image.resize(
        VALIDATION_SIZE,
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


# ============================================================
# Image validation
# ============================================================
def validate_image(image, leaf_validator):
    """
    Returns:
        status: 'leaf', 'non_leaf', or 'uncertain'
        leaf_probability: probability for the leaf class, if available

    Expected binary model:
        Input: RGB image in the range 0..255
        Output: one sigmoid probability
        Training label order: ['leaf', 'non_leaf']

    The validator must have been trained using compatible
    preprocessing and this exact label convention.
    """
    if leaf_validator is None:
        return "uncertain", None

    input_array = preprocess_validator_image(image)

    output = np.asarray(
        leaf_validator.predict(input_array, verbose=0)
    )

    if output.size != 1:
        raise ValueError(
            "The binary validator must output exactly one sigmoid value."
        )

    value = float(output.reshape(-1)[0])

    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            "The validator output must be a finite probability between 0 and 1."
        )

    # IMPORTANT:
    # This interpretation is valid only when the binary model was trained
    # with the sigmoid target 0=leaf, 1=non_leaf.
    non_leaf_probability = value
    leaf_probability = 1.0 - non_leaf_probability

    if non_leaf_probability >= NON_LEAF_THRESHOLD:
        return "non_leaf", leaf_probability

    if leaf_probability > LEAF_THRESHOLD:
        return "leaf", leaf_probability

    return "uncertain", leaf_probability


# ============================================================
# Optional ImageNet diagnostic predictions
# ============================================================
@st.cache_resource
def load_imagenet_diagnostic_model():
    return tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True,
    )


def get_imagenet_diagnostics(image, diagnostic_model):
    """
    These predictions are for display only.
    They must never determine whether the image is a leaf.
    """
    image = prepare_rgb_image(image)
    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(image, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = diagnostic_model.predict(
        image_array,
        verbose=0,
    )

    return tf.keras.applications.mobilenet_v2.decode_predictions(
        probabilities,
        top=5,
    )[0]


# ============================================================
# User interface
# ============================================================
st.title("🌱 Plant Disease Prediction")

st.write(
    "Upload a tomato leaf image to predict its category "
    "using a CNN trained on the PlantVillage dataset."
)

st.info(
    "This prototype supports five tomato-leaf categories. "
    "Its predictions are for educational purposes and are not "
    "a definitive agricultural diagnosis."
)

if not MODEL_PATH.is_file() or not CLASS_PATH.is_file():
    st.error(
        "Required disease-model files are missing. Place "
        "'plant_disease_cnn.keras' and 'class_names.json' "
        "inside the project's models folder."
    )
    st.stop()

try:
    model = load_model()
    leaf_validator = load_leaf_validator()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Unable to load the models or labels: {error}")
    st.stop()

if len(class_names) != model.output_shape[-1]:
    st.error("The number of class labels does not match the disease model.")
    st.stop()

if leaf_validator is None:
    st.warning(
        "The binary leaf validator is not installed. Disease prediction "
        "is disabled until a trained validator is added."
    )
    st.caption(
        f"Expected validator file: {LEAF_VALIDATOR_PATH}"
    )

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = prepare_rgb_image(image)

        st.image(
            image,
            caption="Uploaded image",
            width="stretch",
        )

        if st.button("Predict Disease", type="primary"):
            with st.spinner("Validating the uploaded image..."):
                validation_status, leaf_probability = validate_image(
                    image,
                    leaf_validator,
                )

            # Show validator information for transparency.
            with st.expander("Validation details", expanded=False):
                if leaf_probability is None:
                    st.warning(
                        "No binary validator is available, so this image "
                        "cannot be validated."
                    )
                else:
                    st.metric(
                        "Validator leaf probability",
                        f"{leaf_probability * 100:.2f}%",
                    )

                    st.caption(
                        "This score comes from the separately trained "
                        "binary model. It is not a guarantee that the "
                        "image contains a tomato leaf."
                    )

                    try:
                        diagnostic_model = load_imagenet_diagnostic_model()
                        diagnostic_predictions = get_imagenet_diagnostics(
                            image,
                            diagnostic_model,
                        )

                        st.caption(
                            "Additional ImageNet guesses (diagnostic only; "
                            "not used to accept or reject the image):"
                        )

                        for _, label, score in diagnostic_predictions:
                            st.write(
                                f"**{label}** — {float(score) * 100:.2f}%"
                            )

                    except Exception as diagnostic_error:
                        st.caption(
                            "Optional ImageNet diagnostics are unavailable."
                        )
                        # Details remain in the terminal, not the user interface.
                        print("ImageNet diagnostic error:", diagnostic_error)

            # Do not run the disease CNN unless validation accepts the image.
            if validation_status == "non_leaf":
                st.warning(
                    "Please upload a clear tomato leaf image "
                    "to detect plant diseases."
                )
                st.stop()

            if validation_status == "uncertain":
                st.warning(
                    "We could not validate this image confidently. "
                    "Please upload a clear tomato leaf image."
                )
                st.stop()

            # Only accepted images reach the five-class CNN.
            with st.spinner("Analyzing the tomato leaf..."):
                input_array = preprocess_disease_image(image)
                probabilities = model.predict(
                    input_array,
                    verbose=0,
                )[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index])

            st.subheader("Prediction Result")

            if predicted_class == "Tomato___healthy":
                st.success("Predicted category: Healthy tomato leaf")
            else:
                disease_name = (
                    predicted_class
                    .replace("Tomato___", "")
                    .replace("_", " ")
                )
                st.warning(f"Predicted category: {disease_name}")

            st.metric(
                "Disease model confidence",
                f"{confidence * 100:.2f}%",
            )

            st.caption(
                "The disease model's confidence is its output probability, "
                "not a guarantee that its prediction is correct."
            )

            st.subheader("Class Probabilities")

            for name, probability in zip(class_names, probabilities):
                display_name = (
                    name.replace("Tomato___", "")
                    .replace("_", " ")
                )

                st.write(
                    f"**{display_name}** — {probability * 100:.2f}%"
                )
                st.progress(float(probability))

    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error("Unable to process this image. Try another image.")
        print("Image processing error:", error)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        print("Prediction error:", error)

st.divider()

st.caption(
    "Deep Learning course project | CNN | TensorFlow | Streamlit"
)
