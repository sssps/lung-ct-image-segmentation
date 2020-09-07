# from __future__ import division, print_function
# coding=utf-8
from __future__ import division, print_function
from flask import Flask, render_template, request,session,logging,flash,url_for,redirect,jsonify,Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from flask_mail import Mail
import os
import secrets
import json
import pickle


import sys
import os
import glob
import re
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')

CATEGORIES = ['NORMAL','CANCER']
# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify,Response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


with open('config.json', 'r') as c:
    params = json.load(c)["params"]
# Define a flask app
local_server = True
app = Flask(__name__,template_folder='template')
app.secret_key = 'super-secret-key'

app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = params['gmail_user']
app.config['MAIL_PASSWORD'] = params['gmail_password']
mail = Mail(app)

if(local_server):
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']

db = SQLAlchemy(app)

# Model saved with Keras model.save()
MODEL_PATH = '64x3-CNN.model'

# model = load_model('model_vgg19.h5')
# img = image.load_img('val/PNEUMONIA/person1946_bacteria_4874.jpeg', target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# img_data = preprocess_input(x)
# classes = model.predict(img_data)

#Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# # You can also use pretrained model from Keras
# # Check https://keras.io/applications/
# #from keras.applications.resnet50 import ResNet50
# #model = ResNet50(weights='imagenet')
# #model.save('')
# print('Model loaded. Check http://127.0.0.1:5000/')


class Contact(db.Model):
    '''
    sno, name phone_num, msg, date, email
    '''
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(20), nullable=False)
    phone_num = db.Column(db.String(12), nullable=False)
    message = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(12), nullable=True)


class Register(db.Model):
    '''
    sno, name phone_num, msg, date, email
    '''
    rno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(12), nullable=False)
    password2 = db.Column(db.String(120), nullable=False)


@app.route("/")
def home():
    return render_template('index.html',params=params)

  
@app.route("/about")
def about():
    return render_template('about.html',params=params)  


@app.route("/contact", methods = ['GET', 'POST'])
def contact():
    sendmessage=""
    errormessage=""
    if(request.method=='POST'):
        '''Add entry to the database'''
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('contact')
        message = request.form.get('message')
        try:
            entry = Contact(name=name, phone_num= phone, message = message, email = email,date= datetime.now() )
            db.session.add(entry)
            sendmessage="Thank you for contacting us !.Your message has been sent."
        except Exception as e:
            errormessage="Error : "+ str(e)
        finally:
             db.session.commit()


    return render_template('contact.html',params=params ,sendmessage=sendmessage,errormessage=errormessage)


@app.route("/register", methods=['GET','POST'])
def register():
    if(request.method=='POST'):
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        password2 = request.form.get('password2')

        if (password==password2):
            entry = Register(name=name,email=email,password=password, password2=password2)
            db.session.add(entry)
            db.session.commit()
            return redirect(url_for('login'))
        else:
            flash("plz enter right password")
    return render_template('register.html',params=params)

@app.route("/login",methods=['GET','POST'])
def login():
    if('email' in session and session['email']):
        return render_template('lungindex.html',params=params)

    if (request.method== "POST"):
        email = request.form["email"]
        password = request.form["password"]
        
        login = Register.query.filter_by(email=email, password=password).first()
        if login is not None:
            session['email']=email
            return render_template('lungindex.html',params=params)
        else:
            flash("plz enter right password")
    return render_template('login.html',params=params)


def model_predict(img_path, model):
    # import pdb;pdb.set_trace();
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    x=new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # 
    preds=model.predict(x)
    return preds



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('lungindex.html', params=params)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # import pdb;pdb.set_trace()
        # Get the file from post request

        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        #here segment  code
        img = cv2.imread(file_path) # in BGR mode
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        plt.close()

        #second step
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print("Threshold limit: " + str(ret))

        plt.axis('off')
        plt.imshow(thresh, cmap = 'gray')
        plt.show()
        plt.close()
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations = 3)

        # sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        plt.imshow(dist_transform, cmap = 'gray')
        plt.show()
        plt.close()

        # to change figsize
        fig = plt.figure(figsize = (20, 10)) # to change figsize
        plt.subplot(131)
        plt.imshow(sure_bg, cmap = 'gray')
        plt.title('Sure background, dilated')

        plt.subplot(132)
        plt.imshow(sure_fg, cmap = 'gray')
        plt.title('Sure foreground, eroded')

        plt.subplot(133)
        plt.imshow(unknown, cmap = 'gray')
        plt.title('Subtracted image, black - sure bg & fg')
        plt.tight_layout()
        plt.show()
        plt.close()

        #marker
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==255] = 0
        fig = plt.figure(figsize = (20, 10)) # to change figsize
        plt.subplot(121)
        plt.imshow(markers, cmap = 'gray')
        plt.subplot(122)
        plt.imshow(markers)
        plt.show()
        plt.close()
        #marker
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [0, 255,0]
        plt.imshow(img)
        plt.show()
        plt.close()

        preds = model_predict(file_path, model)
        print(preds[0][0])
        result=CATEGORIES[int(preds[0][0])]
        print(result)
        return jsonify(result)
    return None


@app.route("/logout", methods = ['GET','POST'])
def logout():
    session.pop('email')
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
