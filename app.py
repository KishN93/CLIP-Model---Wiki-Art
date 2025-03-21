import torch
import clip
import streamlit as st
from PIL import Image

# Set Page Layout
st.set_page_config(layout="wide")

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Title and Team Credits
st.title("🎨 WikiArt Search Engine")
st.subheader("Made by Kishan, Rashad, Cita, and Lara")

# Search Box
query = st.text_input("Enter a description (e.g., 'a surreal dreamlike painting')")

# Load embeddings and image paths
image_embeddings = torch.load("wikiart_embeddings.pt", weights_only=True, map_location=torch.device("cpu"))
with open("wikiart_image_paths.txt", "r", encoding="utf-8") as f:
    image_names = f.read().splitlines()

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
