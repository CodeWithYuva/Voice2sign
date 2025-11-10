# app_with_import.py
import streamlit as st
import os
import cv2
from PIL import Image
from live_prediction import live_predict

# import your live prediction function
try:
    from live_prediction import live_predict
    HAS_LIVE_PRED = True
except Exception as e:
    HAS_LIVE_PRED = False
    live_import_error = str(e)

st.set_page_config(page_title="ISL Communication Tool", page_icon="🤟")
st.title("ISL Indian Sign Language Communication Tool")

mode = st.tabs(["Text to ISL", "ISL to Text"])

# Text to ISL (same as before)...
with mode[0]:
    st.subheader("🔤 Text to ISL Conversion")
    input_text = st.text_input("Enter text (A-Z only):").upper()
    if st.button("Convert to ISL") and input_text:
        for char in input_text:
            if char.isalpha():
                img_path = f"C:/Users/HP/Desktop/ISL prototype/lil/signs/signs/{char}.jpg"
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, caption=f"Sign: {char}", width=200)
                else:
                    st.write(f"⚠️ No sign image found for '{char}'")
            else:
                st.write(f"🚫 Unsupported character: {char}")

# ISL to Text
# =============== ISL to Text Mode =============== #
with mode[1]:
    st.subheader("🖐️ ISL to Text (Live Prediction)")

    st.write("This mode will open a separate OpenCV window for live ISL sign recognition.")
    st.write("Press **q** in that window to stop prediction.")

    if st.button("Start Live Prediction"):
        st.warning("Camera starting... check the new window")
        live_predict()  # ✅ Direct call to your function
        st.success("Prediction ended. You can start again.")
