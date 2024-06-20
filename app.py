import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import cv2
from keras.models import load_model # type: ignore
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
app=Flask(__name__)

model =load_model('Minor_Project_CNN.h5')

labels=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
'Blueberry___healthy','Cherry___Powdery_mildew','Cherry___healthy',
'Corn___Cercospora_leaf_spot_or_Gray_leaf_spot','Corn___Common_rust','Corn___Northern_Leaf_Blight','Corn___healthy',
'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
'Pepper___bell___Bacterial_spot','Pepper___bell___healthy',
'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot','Tomato_Yellow_Leaf_Curl_Virus','Tomato_mosaic_virus','Tomato___healthy']


def getResult(image_path):
    img = image.load_img(image_path, target_size=(256,256))
    x = image.img_to_array(img)
    x=x/255.0

    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads',f.filename)
        f.save(file_path)

        predictions=getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        output=str(predicted_label)
        return render_template('index.html', prediction_text='The Result is : {}'.format(output))

if __name__=="__main__":
    app.run(debug=True,port=8000)