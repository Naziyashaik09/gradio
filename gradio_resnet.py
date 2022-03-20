import tensorflow as tf

resnet=tf.keras.applications.resnet50.ResNet50()

import requests
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def classify_image(inp):
  inp = inp.reshape((-1, 224, 224, 3))
  inp = tf.keras.applications.resnet50.preprocess_input(inp)
  prediction = resnet.predict(inp).flatten()
  confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
  return confidences

import gradio as gr

gr.Interface(fn=classify_image, 
             inputs=gr.inputs.Image(shape=(224, 224)),
             outputs=gr.outputs.Label(num_top_classes=3),
             examples=["gold_fish.jpg", "abacus.jpg"]).launch(share=True)

