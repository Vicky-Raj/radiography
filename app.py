import json
from flask import Flask,flash,jsonify,request,render_template,redirect,url_for
import PIL
import numpy as np
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import uuid


UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = load_model("./Covid.h5")

@app.route('/',methods=["GET", "POST"])
def runhome():
    
	return render_template('upload.html')

@app.route('/showresult', methods=["GET","POST"])
def show():
    pic=request.files["pic"]
    name = str(uuid.uuid4())
    path = os.path.join(os.getcwd(),app.config['UPLOAD_FOLDER'],f"{name}.png")
    pic.save(path)
    image = load_img(path, target_size=(224, 224))
    img = img_to_array(image)
    img = img.reshape((1, 224, 224, 3))
    result = model.predict(img)
    result = np.argmax(result, axis=-1)
    output = None
    if result == 0:
        output = "Normal"
    elif result == 1:
        output = "Viral Pneumonia"
    else:
        output = "COVID"
    return jsonify({"prediction":output})



if __name__ == "__main__":
    app.run(host= '0.0.0.0', port=5000, debug=True)
