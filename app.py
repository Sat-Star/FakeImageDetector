from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images/'

# Load the model
meso = load_model('mesonet_model1.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        likelihood = predict_image(file_path)
        
        # Determine the result text
        if likelihood >= 0.5:
            result_text = f'The model predicts that the image is real with a likelihood of {(likelihood)*100:.2f}%'
        else:
            result_text = f'The model predicts that the image is fake with a likelihood of {(1 - likelihood) * 100:.2f}%'

        return render_template('result.html', image_path=file_path.split('static/')[1], result_text=result_text)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = meso.predict(img_array)
    likelihood = prediction[0][0]
    return likelihood

if __name__ == '__main__':
    app.run(debug=True)
