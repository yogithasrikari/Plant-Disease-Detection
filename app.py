import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

MODEL_PATH = "models/plant_disease_model.keras"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# RECOMMENDATIONS
# ============================================================

RECOMMENDATIONS = {

    "Apple___Apple_scab":
        "Remove infected leaves and fruit, improve air circulation, and follow appropriate local agricultural disease-management guidance.",

    "Apple___Black_rot":
        "Remove infected plant material and fruit, prune affected branches, and maintain good orchard sanitation.",

    "Apple___Cedar_apple_rust":
        "Remove affected leaves and fruit, improve air circulation, and follow local disease-management guidance.",

    "Apple___healthy":
        "The leaf appears healthy. Continue regular monitoring and maintain proper plant nutrition and watering.",

    "Blueberry___healthy":
        "The leaf appears healthy. Continue regular monitoring and good plant-care practices.",

    "Cherry_(including_sour)___Powdery_mildew":
        "Improve air circulation, avoid excessive humidity, and remove severely affected plant material.",

    "Cherry_(including_sour)___healthy":
        "The leaf appears healthy. Continue regular monitoring and maintain good growing conditions.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Remove heavily infected material where practical and improve field sanitation and airflow.",

    "Corn_(maize)___Common_rust_":
        "Monitor rust development and follow local agricultural recommendations for disease management.",

    "Corn_(maize)___Northern_Leaf_Blight":
        "Remove infected residue where possible and follow recommended crop-rotation and disease-management practices.",

    "Corn_(maize)___healthy":
        "The leaf appears healthy. Continue regular field monitoring and proper crop management.",

    "Grape___Black_rot":
        "Remove infected berries and leaves, improve canopy airflow, and follow recommended disease-management practices.",

    "Grape___Esca_(Black_Measles)":
        "Remove severely affected plant material and maintain vineyard sanitation. Seek local agricultural guidance.",

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Improve airflow, remove affected leaves where practical, and follow recommended disease-management practices.",

    "Grape___healthy":
        "The leaf appears healthy. Continue regular monitoring and maintain good vineyard management.",

    "Orange___Haunglongbing_(Citrus_greening)":
        "Inspect the tree carefully and seek local agricultural guidance promptly. Management should follow appropriate citrus disease-management practices.",

    "Peach___Bacterial_spot":
        "Remove severely affected material where practical and maintain good orchard sanitation and airflow.",

    "Peach___healthy":
        "The leaf appears healthy. Continue regular monitoring and maintain proper orchard care.",

    "Pepper,_bell___Bacterial_spot":
        "Remove heavily infected material, avoid unnecessary leaf wetness, and maintain field sanitation.",

    "Pepper,_bell___healthy":
        "The leaf appears healthy. Continue regular monitoring and proper watering and nutrition.",

    "Potato___Early_blight":
        "Remove infected foliage where practical, avoid prolonged leaf wetness, and follow local disease-management guidance.",

    "Potato___Late_blight":
        "Remove affected plant material promptly and seek local agricultural guidance for appropriate disease management.",

    "Potato___healthy":
        "The leaf appears healthy. Continue regular crop monitoring and good cultivation practices.",

    "Raspberry___healthy":
        "The leaf appears healthy. Continue monitoring and maintain good plant hygiene and growing conditions.",

    "Soybean___healthy":
        "The leaf appears healthy. Continue regular crop monitoring and maintain proper field management.",

    "Squash___Powdery_mildew":
        "Improve airflow, reduce excessive humidity around foliage, and remove severely affected leaves where practical.",

    "Strawberry___Leaf_scorch":
        "Remove severely affected leaves, improve plant spacing and airflow, and maintain good field sanitation.",

    "Strawberry___healthy":
        "The leaf appears healthy. Continue regular monitoring and proper plant care.",

    "Tomato___Bacterial_spot":
        "Remove severely affected leaves, avoid overhead watering, and maintain good sanitation around plants.",

    "Tomato___Early_blight":
        "Remove affected leaves, avoid prolonged leaf wetness, improve airflow, and follow local disease-management guidance.",

    "Tomato___Late_blight":
        "Remove infected plant material promptly and seek local agricultural guidance for appropriate disease-management measures.",

    "Tomato___Leaf_Mold":
        "Improve ventilation and reduce prolonged humidity around the foliage. Remove severely affected leaves where practical.",

    "Tomato___Septoria_leaf_spot":
        "Remove affected leaves, improve airflow, avoid overhead watering, and maintain garden sanitation.",

    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Inspect the undersides of leaves and follow local integrated pest-management guidance.",

    "Tomato___Target_Spot":
        "Remove affected foliage, improve airflow, avoid prolonged leaf wetness, and follow local disease-management guidance.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Inspect for whiteflies and follow local integrated pest-management guidance. Remove severely affected plants where recommended.",

    "Tomato___Tomato_mosaic_virus":
        "Remove severely infected plants and maintain strict sanitation of hands, tools, and plant-contact surfaces.",

    "Tomato___healthy":
        "The leaf appears healthy. Continue regular monitoring and proper watering, nutrition, and sanitation."
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image to detect the most likely "
    "disease using a deep-learning model."
)

st.info(
    "Model: MobileNetV2-based image classifier | "
    "38 PlantVillage classes | Test Accuracy: 92.52%"
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

        # ----------------------------------------------------
        # PREPROCESS IMAGE
        # ----------------------------------------------------

        resized_image = image.resize(
            IMAGE_SIZE
        )

        image_array = np.array(
            resized_image,
            dtype=np.float32
        )

        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index]
        ) * 100

        # ----------------------------------------------------
        # FORMAT DISPLAY NAME
        # ----------------------------------------------------

        display_name = predicted_class.replace(
            "___",
            " - "
        ).replace(
            "_",
            " "
        )

        # ----------------------------------------------------
        # PREDICTION RESULT
        # ----------------------------------------------------

        st.subheader("Prediction")

        st.success(
            f"🌱 {display_name}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # ----------------------------------------------------
        # CONFIDENCE MESSAGE
        # ----------------------------------------------------

        if confidence >= 90:

            st.success(
                "High-confidence prediction"
            )

        elif confidence >= 70:

            st.warning(
                "Moderate-confidence prediction"
            )

        else:

            st.warning(
                "Low-confidence prediction. "
                "Consider verifying the image with an expert."
            )

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader(
            "Recommended Action"
        )

        recommendation = RECOMMENDATIONS.get(
            predicted_class,
            "Follow appropriate local agricultural guidance."
        )

        st.write(
            recommendation
        )

        # ----------------------------------------------------
        # TOP 3 PREDICTIONS
        # ----------------------------------------------------

        st.subheader(
            "Top 3 Predictions"
        )

        top_indices = np.argsort(
            predictions
        )[-3:][::-1]

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            name = CLASS_NAMES[
                index
            ].replace(
                "___",
                " - "
            ).replace(
                "_",
                " "
            )

            score = float(
                predictions[index]
            ) * 100

            st.write(
                f"**{rank}. {name} — {score:.2f}%**"
            )

            st.progress(
                float(
                    predictions[index]
                )
            )

    except Exception as e:

        st.error(
            "Unable to process the uploaded image."
        )

        st.exception(
            e
        )


# ============================================================
# INFORMATION
# ============================================================

st.markdown("---")

st.caption(
    "Plant Disease Detection using Deep Learning | "
    "MobileNetV2 + PlantVillage"
)

st.caption(
    "For educational and research purposes. "
    "Predictions should be verified before making agricultural decisions."
)