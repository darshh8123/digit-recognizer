import streamlit as st
import sqlite3
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from datetime import datetime

# Load the Keras model
model = tf.keras.models.load_model("digit_model.h5")

# Create/connect to SQLite database
conn = sqlite3.connect("digit_db.sqlite")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    predicted_digit INTEGER,
    image BLOB
)
""")

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    return img_array

# Function to insert into the DB
def insert_prediction(predicted_digit, image):
    # Convert image to binary (BLOB)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    cursor.execute("""
    INSERT INTO predictions (timestamp, predicted_digit, image)
    VALUES (?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), predicted_digit, img_bytes))
    conn.commit()

# Streamlit UI
st.title("🧠 Digit Recognizer with Database")

# File uploader to upload digit image
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img_input = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(img_input)
    predicted_digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {predicted_digit}")

    # Save to DB button
    if st.button("Save to Database"):
        insert_prediction(predicted_digit, image)
        st.success("Saved to database!")

# Optional: View DB records
if st.checkbox("Show Past Predictions"):
    cursor.execute("SELECT timestamp, predicted_digit FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    for row in rows:
        st.write(f"🕒 {row[0]} → Predicted: {row[1]}")

