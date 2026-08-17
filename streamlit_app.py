"""
Streamlit app: draw a digit, predict it with your MNIST CNN.

WHAT THIS APP DOES
1. Loads your trained model back from mnist_cnn_model.pkl
2. Shows a black canvas the user can draw a white digit on
   (matches MNIST's format: white digit on black background)
3. Resizes/preprocesses the drawing to look like an MNIST image (28x28)
4. Runs it through the model and shows the predicted digit + confidence

SETUP (run these once, locally or in your requirements.txt):
    pip install streamlit streamlit-drawable-canvas tensorflow pillow numpy

RUN LOCALLY:
    streamlit run streamlit_app.py

DEPLOY:
    Push this file + mnist_cnn_model.pkl + requirements.txt to a GitHub repo,
    then deploy on streamlit.io/cloud (Community Cloud) pointing at this file.
"""

import pickle

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

# ----------------------------------------------------------------------
# 1. Load the model from the .pkl file (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model(pkl_path="mnist_cnn_model.pkl"):
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    if bundle["class_name"] == "Sequential":
        model = keras.Sequential.from_config(bundle["config"])
    else:
        model = keras.Model.from_config(bundle["config"])

    model.set_weights(bundle["weights"])
    return model


model = load_model()

# ----------------------------------------------------------------------
# 2. Page layout
# ----------------------------------------------------------------------
st.set_page_config(page_title="Digit Recognizer", page_icon="✏️")
st.title("✏️ Handwritten Digit Recognizer")
st.write("Draw a single digit (0-9) below, then click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Draw here")
    # Canvas settings: white stroke on a black background, thick brush
    # (thick brush matters — MNIST digits are fairly bold/thick strokes)
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# ----------------------------------------------------------------------
# 3. Preprocess the drawing to match MNIST's format, then predict
# ----------------------------------------------------------------------
def preprocess_canvas_image(image_data):
    """
    Convert the raw canvas drawing (RGBA numpy array) into the same
    format the model was trained on: a 28x28 grayscale image, pixel
    values scaled to [0, 1], shaped (1, 28, 28, 1).
    """
    # Canvas gives us an RGBA image; convert to grayscale ("L" mode)
    img = Image.fromarray(image_data.astype("uint8"), mode="RGBA").convert("L")

    # Resize down to 28x28, same size as MNIST images.
    # LANCZOS gives smoother resizing than the default.
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to a NumPy array and normalize to [0, 1]
    img_array = np.array(img).astype("float32") / 255.0

    # Reshape to match the model's expected input: (batch, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


with col2:
    st.subheader("Prediction")

    if st.button("Predict", type="primary"):
        # Make sure the user actually drew something
        if canvas_result.image_data is None or canvas_result.image_data[..., :3].sum() == 0:
            st.warning("Please draw a digit first.")
        else:
            processed = preprocess_canvas_image(canvas_result.image_data)

            # Show exactly what the model sees, for sanity-checking
            st.image(
                processed.reshape(28, 28),
                caption="What the model sees (28x28)",
                width=140,
            )

            # Run the prediction
            predictions = model.predict(processed, verbose=0)[0]
            predicted_digit = int(np.argmax(predictions))
            confidence = float(np.max(predictions)) * 100

            st.markdown(f"### Predicted digit: **{predicted_digit}**")
            st.write(f"Confidence: {confidence:.1f}%")

            # Show the full probability breakdown for all 10 digits
            st.bar_chart(
                {str(i): float(predictions[i]) for i in range(10)}
            )

st.divider()
st.caption(
    "Tip: draw a bold, centered digit that fills most of the canvas — "
    "that's what the model was trained on."
)
