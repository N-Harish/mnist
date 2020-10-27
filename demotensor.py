import streamlit as st

import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

if st.button("predict"):
    model = tf.keras.models.load_model("model.h5")
    test_img = test_images[0] / 255.0
    pred = model.predict(test_img)

    st.success(pred)