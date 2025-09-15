import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
import torch.nn.functional as F
import torchvision.transforms as T

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_uploaded_file(file):
    """Convert uploaded file into OpenCV BGR image"""
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR format
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    # Convert to RGB
    return img_rgb


def align_face_mtcnn(image):
    """Detect and align face using MTCNN"""
    mtcnn = MTCNN(image_size=224, margin=20, device=device, post_process=True)

    print("Original image type: ", type(image))

    # Convert PIL → NumPy
    img = np.array(image, dtype=np.uint8)
    print("Converted image type: ", type(img))
    print("Converted image shape: ", img.shape)


    # Ensure uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)


    aligned_face = mtcnn(img)
    return aligned_face




# --- Age Prediction Model ---
def build_age_model(model_name='resnet50', pretrained=True, fine_tune=True):
    """Builds a pre-trained model for age regression"""
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final layer for regression (1 output)
    if 'resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif 'densenet' in model_name:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)

    return model





class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features, self.out_features, self.s, self.m = in_features, out_features, s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = np.cos(m), np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, embedding, label):
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        phi = cosine * self.cos_m - torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1)) * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_dim=512, pretrained='vggface2'):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained)
        self.arc_face = ArcFaceLoss(in_features=embedding_dim, out_features=160)

    def forward(self, image, label):
        embedding = self.backbone(image)
        return self.arc_face(embedding, label)

    def get_embedding(self, image):
        return self.backbone(image)

# --- Load Models ---
@st.cache_resource
def load_models():
    """Load pre-trained models"""
    # Age prediction model
    age_model = build_age_model('resnet50', pretrained=True, fine_tune=False)
    # Load your trained weights here
    age_model.load_state_dict(torch.load('best_age_model.pth', map_location=device))
    age_model.eval()
    print("Age model loaded.")
    
    # Face recognition model
    face_model = FaceRecognitionModel()
    # Load your trained weights here
    face_model.load_state_dict(torch.load('best_aifr_model.pth', map_location=device))
    face_model.eval()
    print("Face recognition model loaded.")
    
    return age_model.to(device), face_model.to(device)

# --- Image Transformations ---
age_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Prediction Functions ---
def predict_age(image, model, transform):
    """Predict age from image"""
    # If image is already a tensor, skip the transform or apply only necessary parts
    if isinstance(image, torch.Tensor):
        print("Image is already a tensor.")
        # Ensure tensor is in right format and on correct device
        if image.dim() == 3:
            print("Image dimension is 3")
            image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(device)
    else:
        print("Image not tensor")
        # Apply full transform for PIL/numpy inputs
        image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image)
        print("Raw age prediction: ", prediction)
    return int(prediction.item())

def verify_faces(image1, image2, model, transform, threshold=0.5):
    """Verify if two faces belong to the same person"""
    # Transform images
    if isinstance(image1, torch.Tensor):
        # Ensure tensor is in right format and on correct device
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)  # Add batch dimension
        image1 = image1.to(device)
    else:
        # Apply full transform for PIL/numpy inputs
        image1 = transform(image1).unsqueeze(0).to(device)

    if isinstance(image2, torch.Tensor):
        # Ensure tensor is in right format and on correct device
        if image2.dim() == 3:
            image2 = image2.unsqueeze(0)  # Add batch dimension
        image2 = image2.to(device)
    else:
        # Apply full transform for PIL/numpy inputs
        image2 = transform(image2).unsqueeze(0).to(device)

    
    
    # Get embeddings
    with torch.no_grad():
        emb1 = model.get_embedding(image1)
        emb2 = model.get_embedding(image2)
    
    # Calculate cosine similarity
    similarity = F.cosine_similarity(emb1, emb2).item()
    
    # Determine if same person
    is_same = similarity > threshold
    
    return is_same, similarity

# --- Streamlit App ---
def main():
    st.title("Face Analysis App")
    st.write("Upload two facial images to predict ages and verify if they're the same person")
    
    # Load models
    age_model, face_model = load_models()
    
    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        image1_file = st.file_uploader("Upload first image", type=['jpg', 'jpeg', 'png'])
    with col2:
        image2_file = st.file_uploader("Upload second image", type=['jpg', 'jpeg', 'png'])
    
    if image1_file and image2_file:
        # Load images
        image1 = read_uploaded_file(image1_file)
        image2 = read_uploaded_file(image2_file)
        
        # Display original images
        st.subheader("Original Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, caption="Image 1", use_column_width=True)
        with col2:
            st.image(image2, caption="Image 2", use_column_width=True)

        print("Image 1 data type: ", image1.dtype)
        print("Image 2 data type: ", image2.dtype)
        
        # Align faces
        st.subheader("Face Alignment")
        with st.spinner("Aligning faces..."):
            aligned_face1 = align_face_mtcnn(image1)
            aligned_face2 = align_face_mtcnn(image2)
        


        to_pil = T.ToPILImage()

        if aligned_face1 is not None and aligned_face2 is not None:
            # Display aligned faces
            # col1, col2 = st.columns(2)
            # with col1:
            #     st.image(to_pil(aligned_face1), caption="Aligned Face 1", use_column_width=True)
            # with col2:
            #     st.image(to_pil(aligned_face2), caption="Aligned Face 2", use_column_width=True)
            
            # Predict ages
            st.subheader("Age Prediction")
            with st.spinner("Predicting ages..."):
                age1 = predict_age(aligned_face1, age_model, age_transform)
                age2 = predict_age(aligned_face2, age_model, age_transform)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Age (Image 1)", f"{age1} years")
            with col2:
                st.metric("Predicted Age (Image 2)", f"{age2} years")
            
            # Face verification
            st.subheader("Face Verification")
            with st.spinner("Verifying if same person..."):
                is_same, similarity = verify_faces(aligned_face1, aligned_face2, face_model, face_transform)
                
            similarity = max(0.0, min(1.0, similarity))
            if is_same:
                st.success(f"✅ Same person! Similarity score: {similarity:.3f}")
            else:
                st.error(f"❌ Different persons! Similarity score: {similarity:.3f}")
                
            # Display similarity gauge
            st.progress(similarity, text=f"Similarity: {similarity:.3f}")
            
        else:
            st.error("Could not detect faces in one or both images. Please try with different images.")

if __name__ == "__main__":
    main()