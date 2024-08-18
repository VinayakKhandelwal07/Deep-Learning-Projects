from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import io
 
app = Flask(__name__)

model = tf.keras.models.load_model('models/CNN_model.h5')

@app.route('/')
def upload():
    return render_template('Home.html')

@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['file']
    
    # Load and preprocess the image
    image = Image.open(file.stream)
    image = image.resize((224, 224))  # Make sure this is the correct size for your model
    image_array = np.array(image) / 255.0
    
    # If the image is grayscale, convert it to RGB by adding an extra dimension
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    
    # Expand dimensions to match the model input
    image_array = np.expand_dims(image_array, axis=0)

    # Predict using the model
    prediction = model.predict(image_array)

    # For binary classification, use a threshold of 0.5
    if model.output_shape[1] == 1:
        out = (prediction > 0.5).astype(int).ravel()
    else:
        # For multi-class classification, use argmax to get the class with the highest probability
        out = np.argmax(prediction, axis=1)
    
    class_names = ['DOG', 'CAT']
    result = class_names[out[0]]

    return render_template('Home.html', result=result)
if __name__ == '__main__':
    app.run(debug=True)
