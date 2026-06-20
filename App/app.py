import streamlit as st
import torch
from torchvision.models import resnet18
from torch import nn
from torchvision import transforms
from PIL import Image

# Page Title
st.set_page_config(
    page_title="Visual Defect Detection",
    layout="centered"
)

st.title(" Steel Surface Defect Detection System")

st.success(
    "Industrial AI Solution for Automated Surface Defect Inspection"
)
st.markdown(
    """
    ### AI-Based Industrial Quality Inspection

    Upload a steel surface image and the trained ResNet18 model
    will automatically identify manufacturing defects.
    """
)

with st.sidebar:

    st.title("Project Details")

    st.write(
        """
        **Model:** ResNet18

        **Validation Accuracy:** 99.17%

        **Dataset:** NEU Surface Defect Database

        **Defect Classes:**

        • Crazing

        • Inclusion

        • Patches

        • Pitted Surface

        • Rolled-in Scale

        • Scratches
        """
    )

# Classes
classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

descriptions = {
    "crazing":
        "Network of fine surface cracks.",

    "inclusion":
        "Foreign material embedded in the steel surface.",

    "patches":
        "Irregular patches affecting surface quality.",

    "pitted_surface":
        "Small cavities or pits formed on the surface.",

    "rolled-in_scale":
        "Oxide scale pressed into the steel during rolling.",

    "scratches":
        "Linear marks caused by mechanical damage."
}

# Load Model
@st.cache_resource
def load_model():

    model = resnet18(weights=None)

    model.fc = nn.Linear(512, 6)

    model.load_state_dict(
        torch.load(
            "../models/resnet18_final.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Upload
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    predicted_class = classes[prediction.item()]

    with col2:

        st.subheader("Prediction Result")

        st.success(predicted_class)

        st.metric(
            "Confidence Score",
            f"{confidence.item()*100:.2f}%"
        )

    st.info(
        descriptions[predicted_class]
    )

st.markdown("""
### Supported Defects

✔ Crazing

✔ Inclusion

✔ Patches

✔ Pitted Surface

✔ Rolled-in Scale

✔ Scratches
""")
st.caption(
    "AI-Powered Visual Defect Detection System"
)