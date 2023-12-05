from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Loading the model
model_path = 'D:\miniproject\my_trained_model.h5'
loaded_model = keras.models.load_model(model_path)

# Function to preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    img = keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the artist
def predict_artist(image_array):
    prediction = loaded_model.predict(image_array)
    prediction_probability = np.amax(prediction)
    prediction_idx = np.argmax(prediction)
    return prediction_idx, prediction_probability

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction_text="No file found")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction_text="No selected file")

        # Save the uploaded file
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Process the uploaded image and make a prediction
        image_array = preprocess_image(file_path)
        prediction_idx, prediction_probability = predict_artist(image_array)

        probability_threshold = 0.9


        # Get the predicted label
        labels = {0: 'Vincent van Gogh', 1: 'Edgar Degas', 2: 'Pablo Picasso', 3: 'Pierre-Auguste Renoir'}  # Replace with your actual labels
        

        predicted_artist = labels.get(prediction_idx)

        if prediction_probability < probability_threshold or predicted_artist not in labels.values():
            return redirect(url_for('unknown_artist'))
        # Redirect to corresponding artist page
        elif predicted_artist == 'Vincent van Gogh':
            return redirect(url_for('vangogh'))
        elif predicted_artist == 'Edgar Degas':
            return redirect(url_for('degas'))
        elif predicted_artist == 'Pablo Picasso':
            return redirect(url_for('picasso'))
        elif predicted_artist == 'Pierre-Auguste Renoir':
            return redirect(url_for('renoir'))
        else:
            return redirect(url_for('unknown_artist'))

    return render_template('index.html')

# Define routes for each artist
@app.route('/vangogh')
def vangogh():
    return render_template('vangogh.html')

@app.route('/degas')
def degas():
    return render_template('degas.html')

@app.route('/picasso')
def picasso():
    return render_template('picasso.html')

@app.route('/renoir')
def renoir():
    return render_template('renoir.html')

@app.route('/unknown_artist')
def unknown_artist():
    return render_template('unknown_artist.html')

if __name__ == '__main__':
    app.run(debug=True)

