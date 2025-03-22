import os
import requests
import torch
import clip
import streamlit as st
from PIL import Image

# Function to download the file from Google Drive
def download_file_from_gdrive(file_url, local_filename):
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(local_filename, "wb") as f:
            f.write(response.content)
        st.success(f"File {local_filename} downloaded successfully!")
    else:
        st.error(f"Failed to download {local_filename}.")

# Set Page Layout
st.set_page_config(layout="wide")

# Google Drive links for the files
file_url_embeddings = "https://drive.google.com/uc?id=1H2ppMXkDakSaAGz3jp1PMm4XqOBpfr3i"
file_url_image_paths = "https://drive.google.com/uc?id=1AC3A4BU1-7bW3IxthEQcvi49h1ylm_Lk"

# File paths for the embeddings and image paths
embeddings_file_path = "wikiart_embeddings.pt"
image_paths_file_path = "wikiart_image_paths.txt"

# Check if the files already exist, if not download them
if not os.path.exists(embeddings_file_path):
    with st.spinner("Downloading the embeddings file..."):
        download_file_from_gdrive(file_url_embeddings, embeddings_file_path)

if not os.path.exists(image_paths_file_path):
    with st.spinner("Downloading the image paths file..."):
        download_file_from_gdrive(file_url_image_paths, image_paths_file_path)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the embeddings file
image_embeddings = torch.load(embeddings_file_path, weights_only=True, map_location=torch.device(device))

# Load the image paths file
with open(image_paths_file_path, "r", encoding="utf-8") as f:
    image_names = f.read().splitlines()

# Title and Team Credits
st.title("🎨 WikiArt Search Engine")
st.subheader("Made by Kishan, Rashad, Cita, and Lara")

# Search Box
query = st.text_input("Enter a description (e.g., 'a surreal dreamlike painting')")

# Handle Search
if st.button("Search") and query:
    with torch.no_grad():
        tokens = clip.tokenize([query]).to(device)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

        similarity = (text_features.float() @ image_embeddings.T.float()).squeeze(0)
        top_indices = similarity.argsort(descending=True)[:5]

    st.subheader("Top 5 Results:")
    cols = st.columns(5)
    for i, idx in enumerate(top_indices):
        img = Image.open(image_names[idx])
        cols[i].image(img, caption=f"Score: {similarity[idx]:.4f}", use_column_width=True)
