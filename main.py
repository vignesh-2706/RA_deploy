#importng the libraries
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('SavedModel_V2.h5')  #loading the model
class_names = ['Negative', 'Positive']  #assigning the class_names

app = Flask(__name__)

#funtion to predict the image
def predict(image):
    img = Image.open(image)     #reading the image
    img = img.resize((180,180))     #resizing the image
    img = np.array(img)     #converting the image to numpy array
    img = np.expand_dims(img, axis=0)   #expanding the dimension
    img = img[:,:,:,0:3]     #converting 4 channel to 3 channel
    prediction = class_names [np.argmax(model.predict(img))]    #predicting the image
    a = model.predict(img)
    confidence = max(a)[np.argmax(a)]*100   #calculating the confidence
    return prediction, confidence.round(2)  #returning the prediction and confidence

# index route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')    #returning the index.html file

# @app.route('/index1.html', methods=['GET'])
# def index1():
#     return render_template('index1.html') 

@app.route('/home.html', methods=['GET'])
def home():
    return render_template('home.html') 


# predict route
@app.route('/predict', methods=['POST'])
def predictt():
    img = request.files['img']    #getting the image from the user
    img.save('static\img.jpg')     #saving the image
    prediction, confidence = predict("static\img.jpg")     #predicting the image from previously defined predict function and sending the image as an argument
    return render_template('predict.html', data=[prediction, confidence],usr_img = img)   #returning the prediction and confidence to the user



# # predict route
# @app.route('/prediction', methods=['POST'])
# def prediction():
#     img = request.files['img']    #getting the image from the user
#     img.save('static\img.jpg')     #saving the image
#     prediction, confidence = predict("static\img.jpg")     #predicting the image from previously defined predict function and sending the image as an argument
#     return render_template('prediction.html', data=[prediction, confidence],usr_img = img)   #returning the prediction and confidence to the user



if __name__ == '__main__':
    app.run(debug=True)