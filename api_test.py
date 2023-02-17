from flask import Flask, flash, request, redirect, url_for, render_template, Response
import urllib.request

from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from time import strftime
from datetime import datetime
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2
from numpy import savez_compressed
from numpy import asarray
from os import listdir
from numpy import load
from numpy import reshape
from keras.models import load_model
import numpy as np
import csv
import face_recognition
from numpy import array
from numpy import expand_dims, max
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import shutil

from flask import Flask, flash, request, redirect, url_for, render_template, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from time import strftime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from werkzeug.utils import secure_filename
import time
from flask_cors import CORS
import json
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2
from os import listdir

from dataclasses import dataclass
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'hardsecretkey'


# SqlAlchemy Database Configuration With Mysql
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


db = SQLAlchemy(app)


# This is our model
@dataclass
class StudentInfo(db.Model):
    username: str
    password:str
    fname: str
    lname: str
    gender :str
    email :str
    pno:int
    sem:int
    dob:str

    username = db.Column(db.String(100), unique=True, primary_key=True)
    password = db.Column(db.String(100), nullable=False)
    fname = db.Column(db.String(100), nullable=False)
    lname = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    pno = db.Column(db.Integer, nullable=False)
    sem = db.Column(db.Integer, nullable=False)

    def __init__(self, username, password, fname, lname, dob, gender, email, pno, sem):
        self.username = username
        self.password = password
        self.fname = fname
        self.lname = lname
        self.dob = dob
        self.gender = gender
        self.email = email
        self.pno = pno
        self.sem = sem

    # def __repr__(self) -> str:
    #     return f"{self.username} - {self.pno}"


# This is our model
@dataclass
class TeacherInfo(db.Model):
    username : str
    password : str
    fname : str
    lname : str
    dob : str
    gender : str
    email : str
    pno : int
    subject : str

    username = db.Column(db.String(100), unique=True, primary_key=True)
    password = db.Column(db.String(100), nullable=False)
    fname = db.Column(db.String(100), nullable=False)
    lname = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    pno = db.Column(db.Integer, nullable=False)
    subject = db.Column(db.String(100), nullable=False)

    def __init__(self, username, password, fname, lname, dob, gender, email, pno, subject):
        self.username = username
        self.password = password
        self.fname = fname
        self.lname = lname
        self.dob = dob
        self.gender = gender
        self.email = email
        self.pno = pno
        self.subject = subject

    def __repr__(self) -> str:
        return f"{self.username} - {self.subject}"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    # ---------clearing the cropped folder to store new images ---------------------#

    for x in os.listdir('static/cropped/'):
        if os.path.isfile('static/cropped/' + x):
            print("files existed and removed")
            os.remove('static/cropped/' + x)
        else:
            print(x)
            print("files not found")

    # ------------------------------------------------------------------------------#

    # -----------clearing Uploads folder first -------------------------------------#

    for x in os.listdir('static/uploads/'):
        if os.path.isfile('static/uploads/' + x):
            print("files existed and removed")
            os.remove('static/uploads/' + x)
        else:
            print(x)
            print("files not found")

    # Uploading Extracting from request
    if 'file' not in request.files:
        flash('No file part')
        return { 'status_of_file': 'failed' }
    
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return { 'status_of_file': 'failed' }
    
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print('upload_image filename: ' + filename)

    # --------Cropping of Images---------------------------------------------------#

    def save_faces(filecname, result_list):
        # load the image
        data = pyplot.imread(filecname)
        # plot each face as a subplot
        for i in range(len(result_list)):
            # get coordinates
            x1, y1, width, height = result_list[i]['box']
            x2, y2 = x1 + width, y1 + height
            cv2.imwrite("static/cropped/img_{}.png".format(i), data[y1:y2, x1:x2])

    fname = UPLOAD_FOLDER + filename
    filecname = fname
    # load image from file
    pixels = pyplot.imread(filecname)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    # display faces on the original image
    save_faces(filecname, faces)
    print("cropped saved")

    # -------------Check if image is face else delete----------------------------------#

    for x in os.listdir('static/cropped/'):
        test_image = 'static/cropped/' + x
        image = face_recognition.load_image_file(test_image)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) <= 0:
            print('There is no face in the provided image, Please select another image.' + x)
            os.remove('static/cropped/' + x)

        elif len(face_locations) > 1:
            print('There are multiple faces in the provided image, Please select another image.' + x)

        else:
            print('This is a perfect image.' + x)

    # ---------------------------------------------------------------------------------#

    # ------------ Take Attendance ----------------------------------------------------#

    std_list = []

    # extracting embeddings
    def extract_embeddings(model, face_pixels):
        face_pixels = face_pixels.astype('float32')
        mean = face_pixels.mean()
        std = face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = model.predict(samples)
        return yhat[0]

    directory = 'static/cropped/'
    print(os.listdir('static/cropped/'))
    dirs = os.listdir(directory)
    model = load_model('facenet_keras.h5', compile=False)
    data1 = load('DataSet.npz')
    train_x, train_y = data1['arr_0'], data1['arr_1']

    data = load('Embeddings.npz')
    trainx, trainy = data['arr_0'], data['arr_1']

    name = []
    # data rows of csv file
    
    unfacecount = 0
    for filename in dirs:
        Img = directory + filename

        # load data and reshape the image
        img1 = Image.open(Img)
        img1 = img1.convert('RGB')
        img1 = img1.resize((160, 160))
        pixels = asarray(img1)
        testx = pixels.reshape(-1, 160, 160, 3)
        # print("Input test data shape: ",testx.shape)

        # find embeddings
        new_testx = list()
        for test_pixels in testx:
            embeddings = extract_embeddings(model, test_pixels)
            new_testx.append(embeddings)

        new_testx = asarray(new_testx)
        # print("Input test embedding shape: ",new_testx.shape)
        # print(trainy.shape[0])
        # print("Loaded data: Train=%d , Test=%d"%(trainx.shape[0],new_testx.shape[0]))

        # normalize the input data
        in_encode = Normalizer(norm='l2')
        trainx = in_encode.transform(trainx)
        new_testx = in_encode.transform(new_testx)

        # create a label vector
        new_testy = trainy
        out_encode = LabelEncoder()
        out_encode.fit(trainy)
        trainy1 = out_encode.transform(trainy)
        new_testy = out_encode.transform(new_testy)

        # define svm classifier model
        model1 = SVC(kernel='linear', probability=True)
        model1.fit(trainx, trainy1)

        # predict
        predict_train = model1.predict(trainx)
        predict_test = model1.predict(new_testx)

        # get the confidence score
        probability = model1.predict_proba(new_testx)
        confidence = max(probability)
        # print(confidence)

        if confidence > 0.45:

            acc_train = accuracy_score(trainy1, predict_train)

            # display
            trainy_list = list(trainy1)
            p = int(predict_test)
            if p in trainy_list:
                val = trainy_list.index(p)
                _trainy = out_encode.inverse_transform(trainy1)
                string = strftime('%H:%M:%S: %p')
                name.append([_trainy[val], string])
                std_list.append(_trainy[val])
            # print(std_list)
            # print(trainy[val])
        else:
            unfacecount += 1
            print("Unknown Face")
    if len(name) > 0:
        header = ['Name', 'Time']
        with open('Attendance.csv', 'w') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            for w in name:
                writer.writerow(w)

    return { 'status_of_file': 'success', "attendance" : name }
    

@app.route('/uploads', methods=['POST'])
def upload_images():
    print('-1', request.files)
    print('0', dir(request.files.getlist))
    print('1', request.files.getlist('file'))
    
    filelist = request.files.getlist('file')
    image_count = 90
    minimum_image_required = 85
    # saving all files
    for file in filelist:
        print('2', file.filename)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)
    
    return { 'status_of_file': 'success', 'count' : image_count } if image_count >= minimum_image_required else { 'status_of_file': 'failed', 'count' : image_count }
    

@app.route('/student/list', methods=['POST', 'GET'])
def studentlist():
    student = StudentInfo.query.all()
    return jsonify(student)

@app.route('/student/modify/<string:username>', methods=['GET', 'POST'])
def updateStudent(username):
    username = request.form['username']
    fname = request.form['firstname']
    lname = request.form['lastname']
    dob = request.form['date']
    gender = request.form['gender']
    email = request.form['email']
    pno = request.form['phoneno']
    sem = request.form['sem']
    student = StudentInfo.query.filter_by(username=username).first()
    student.fname = fname
    student.lname = lname
    student.dob = dob
    student.gender = gender
    student.email = email
    student.pno = pno
    student.sem = sem
    db.session.add(student)
    db.session.commit()

@app.route('/student/delete/<string:username>', methods=['POST', 'GET'])
def deleteStudent(username):
    print(username)
    student = StudentInfo.query.filter_by(username=username).first()
    # db.session.delete(student)
    # db.session.commit()

    return { 'status': 'success'}


@app.route('/teacher/list', methods=['POST', 'GET'])
def viewTeacher():
    teacher = TeacherInfo.query.all()
    return jsonify(teacher)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
