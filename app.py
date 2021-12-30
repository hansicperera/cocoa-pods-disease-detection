import os

from flask import Flask, render_template, request, url_for, redirect
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

UPLOAD_FOLDER = 'static'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = keras.models.load_model('keras_model.h5')
class_names = ['healthy', 'moniliasis', 'phytopthora disease']

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def prediction():
    img = request.files['img']
    # img.save('img.jpg')
    filename = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
    img.save(filename)

    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img) / 255
    resized_img_np = np.expand_dims(x, axis=0)
    prediction = model.predict(resized_img_np)

    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * (np.max(prediction[0])), 2)
    # p1 = 1 - prediction
    # if prediction > p1:
    #     pred = "Dog"
    # else:
    #     pred = "Cat"

    return render_template("index.html", data=predicted_class,c=confidence, filename=filename)


if __name__ == "__main__":
    app.run(debug=True)