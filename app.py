import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

custom_objects = {
    # Add custom objects if your model uses any custom layers or functions
    # 'MyCustomLayer': MyCustomLayer
}

# Function to ignore unsupported arguments when deserializing layers
def ignore_groups(**kwargs):
    kwargs.pop('groups', None)  # Remove the 'groups' argument
    return tf.keras.layers.Conv2DTranspose(**kwargs)

# Load the model using custom_objects and ignoring unsupported arguments
generator = tf.keras.models.load_model('generator.h5', custom_objects={'Conv2DTranspose': ignore_groups})

# Function to generate and display images
def generate_images(generator, num_images, latent_dim, labels_dim):
    noise = np.random.uniform(-1, 1, size=(num_images, latent_dim))
    labels = tf.keras.utils.to_categorical(np.random.choice(range(labels_dim), size=(num_images)), num_classes=labels_dim)
    generated_images = generator.predict([noise, labels])
    return generated_images, labels

def get_label_name(label):
    if label == 0:
        return 'Normal'
    else:
        return 'Pneumonia'

st.title("GAN-generated Chest X-ray Images")

# Sidebar for user inputs
st.sidebar.header("Generation Settings")
num_images = st.sidebar.slider("Number of images to generate", min_value=1, max_value=100, value=16)
latent_dim = st.sidebar.number_input("Latent space dimension", min_value=10, max_value=500, value=100)
labels_dim = st.sidebar.number_input("Number of classes", min_value=2, max_value=10, value=2)

if st.sidebar.button("Generate Images"):
    generated_images, labels = generate_images(generator, num_images, latent_dim, labels_dim)

    st.subheader("Generated Images")
    fig, axes = plt.subplots(2, 8, figsize=(15, 4))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(generated_images[i])
            ax.axis('off')
            ax.set_title(get_label_name(np.argmax(labels[i])))  # Display label name
    st.pyplot(fig)
