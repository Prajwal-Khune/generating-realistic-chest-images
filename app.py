import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

# User input for the number of images
num_images = st.slider("Number of images to generate", min_value=1, max_value=100, value=16)

latent_dim = 100  # Latent space dimension
labels_dim = 2  # Number of classes

if st.button("Generate Images"):
    generated_images, labels = generate_images(generator, num_images, latent_dim, labels_dim)

    st.subheader("Generated Images")
    # Determine the number of rows and columns for the plot grid
    rows = (num_images + 7) // 8  # 8 images per row
    fig, axes = plt.subplots(rows, 8, figsize=(15, rows * 2))
    axes = axes.flatten() if rows > 1 else [axes]  # Ensure axes is iterable

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(generated_images[i])
            ax.axis('off')
            ax.set_title(get_label_name(np.argmax(labels[i])))  # Display label name
        else:
            ax.axis('off')  # Turn off unused axes

    st.pyplot(fig)
