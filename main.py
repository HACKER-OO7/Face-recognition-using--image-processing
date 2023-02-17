from email import message
from tokenize import Name
from flask import Flask, flash, request, redirect, url_for, render_template, Response
import urllib.request
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from time import strftime
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time 
from time import strftime
from datetime import datetime
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2
from numpy import savez_compressed
from numpy import asarray
from os import listdir
from numpy import load
from numpy import expand_dims
from numpy import reshape
from keras.models import load_model 
import numpy as np
import csv
from numpy import array
from numpy import expand_dims
from numpy import max
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import shutil
import pyttsx3  
from werkzeug.security import check_password_hash, generate_password_hash
import serial
import face_recognition


app = Flask(__name__)
#-------------------------------------------------------------------------------------------------------------------------
#database

app.config['SECRET_KEY'] = 'hardsecretkey'


#SqlAlchemy Database Configuration With Mysql
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# engine = pyttsx3.init('sapi5')  
# voices = engine.getProperty('voices')
# print(voices[1].id)
# engine.setProperty('voice', voices[1].id)

db = SQLAlchemy(app)


#This is our model
class StudentInfo(db.Model):
    username = db.Column(db.String(100), unique = True,primary_key = True)
    password = db.Column(db.String(100), nullable=False)
    fname = db.Column(db.String(100), nullable=False)
    lname = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    pno = db.Column(db.Integer, nullable=False)
    sem = db.Column(db.Integer, nullable=False)


    

    def __init__(self, username, password , fname , lname , dob , gender , email , pno , sem):
        self.username = username
        self.password = password
        self.fname = fname
        self.lname = lname
        self.dob = dob
        self.gender = gender
        self.email = email
        self.pno = pno
        self.sem = sem
        


    def __repr__(self) -> str:  
        return f"{self.username} - {self.pno}"



#This is our model
class TeacherInfo(db.Model):
    username = db.Column(db.String(100), unique = True,primary_key = True)
    password = db.Column(db.String(100), nullable=False)
    fname = db.Column(db.String(100), nullable=False)
    lname = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    pno = db.Column(db.Integer, nullable=False)
    subject = db.Column(db.String(100), nullable=False)


    

    def __init__(self, username, password , fname , lname , dob , gender , email , pno , subject):
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

#-------------------------------------------------------------------------------------------------------------------------
#Starts Home Page and include UPLOAD-IMAGE



UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
detector = MTCNN() 
 
@app.route('/')
def home():
    return render_template('index.html')


def send_msg( pno , n):
    phone = serial.Serial("COM3",  9600, timeout=5)
    try:
        time.sleep(1)
        phone.write(b'ATZ\r')
        time.sleep(1)
        phone.write(b'AT+CMGF=1\r')
        time.sleep(1)
        recipient = "+91"+str(pno)
        phone.write(b'AT+CMGS="' + recipient.encode() + b'"\r')
        time.sleep(1)
        message = str(n)+" is present in the class."
        phone.write(message.encode() + b"\r")
        time.sleep(1)
        phone.write(bytes([26]))
        time.sleep(1)
    finally:
        phone.close()


@app.route('/', methods=['POST'])
def upload_image():

    #---------clearing the cropped folder to store new images ---------------------#
    student= StudentInfo.query.all() 
                
    for x in os.listdir('static/cropped/'):
        if os.path.isfile('static/cropped/'+x):
            print("files existed and removed")
            os.remove('static/cropped/'+x)
        else:
            print(x)
            print("files not found")
    
    #------------------------------------------------------------------------------#

    #-----------clearing Uploads folder first -------------------------------------#        
    
    for x in os.listdir('static/uploads/'):
        if os.path.isfile('static/uploads/'+x):
            print("files existed and removed")
            os.remove('static/uploads/'+x)
        else:
            print(x)
            print("files not found")
    
    #--------------------------------------------------------------------------------#

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        print('upload_image filename: ' + filename)
       

        #--------Cropping of Images---------------------------------------------------#

        def save_faces(filecname, result_list):
            # load the image
            data = pyplot.imread(filecname)
            # plot each face as a subplot
            for i in range(len(result_list)):
                # get coordinates
                x1, y1, width, height = result_list[i]['box']
                x2, y2 = x1 + width, y1 + height
                cv2.imwrite("static/cropped/img_{}.png".format(i),data[y1:y2, x1:x2])
       
        fname = UPLOAD_FOLDER+ filename
        filecname = fname
        # load image from file
        pixels = pyplot.imread(filecname)
        # create the detector, using default weights
        
        # detect faces in the image
        faces = detector.detect_faces(pixels)
        # display faces on the original image
        save_faces(filecname, faces)
        print("cropped saved")
        
        #-------------Check if image is face else delete----------------------------------#

        # for x in os.listdir('static/cropped/'):
        #     test_image = 'static/cropped/' + x
        #     image = face_recognition.load_image_file(test_image)
        #     face_locations = face_recognition.face_locations(image)
        #     if len(face_locations) <= 0:
        #         print('There is no face in the provided image, Please select another image.'+x)
        #         os.remove('static/cropped/'+x)

        #     elif len(face_locations) > 1:
        #         print('There are multiple faces in the provided image, Please select another image.'+x)

        #     else:
        #         print('This is a perfect image.' +x)

        for x in os.listdir('static/cropped/'):
            test = 'static/cropped/' + x
            test_image = cv2.imread(test)
            faces = detector.detect_faces(test_image)

            if not faces:
                print('There is no face in the provided image, removing... '+x)
                os.remove('static/cropped/'+x)

            else:
                print('This is a perfect image.' +x)
        
        
        #---------------------------------------------------------------------------------#
        
        #------------ Take Attendance ----------------------------------------------------#

        std_list = []
        #extracting embeddings
        def extract_embeddings(model,face_pixels):
            face_pixels = face_pixels.astype('float32')
            mean = face_pixels.mean()
            std  = face_pixels.std()
            face_pixels = (face_pixels - mean)/std
            samples = expand_dims(face_pixels,axis=0)
            yhat = model.predict(samples)
            return yhat[0]
        directory='static/cropped/'
        print(os.listdir('static/cropped/'))
        dirs = os.listdir( directory )
        model = load_model('facenet_keras.h5',compile=False)
        data1 = load('DataSet.npz')
        train_x,train_y = data1['arr_0'],data1['arr_1']

        data = load('Embeddings.npz')
        trainx,trainy= data['arr_0'],data['arr_1']

        # i=0
        name=[[]]
        # data rows of csv file

        global unfacecount
        unfacecount = 0 
        for filename in dirs:
            Img = directory + filename

            #load data and reshape the image
            img1 = Image.open(Img)            
            img1 = img1.convert('RGB')  
            img1 = img1.resize((160,160))        
            pixels = asarray(img1)    
            testx = pixels.reshape(-1,160,160,3)
            # print("Input test data shape: ",testx.shape)

            #find embeddings
            new_testx = list()
            for test_pixels in testx:
                embeddings = extract_embeddings(model,test_pixels)
                new_testx.append(embeddings)

            new_testx = asarray(new_testx)  
            # print("Input test embedding shape: ",new_testx.shape)
            # print(trainy.shape[0])
            # print("Loaded data: Train=%d , Test=%d"%(trainx.shape[0],new_testx.shape[0]))

            #normalize the input data
            in_encode = Normalizer(norm='l2')
            trainx = in_encode.transform(trainx)
            new_testx = in_encode.transform(new_testx)

            #create a label vector
            new_testy = trainy 
            out_encode = LabelEncoder()
            out_encode.fit(trainy)
            trainy1 = out_encode.transform(trainy)
            new_testy = out_encode.transform(new_testy)
            
            #define svm classifier model 
            model1 =SVC(kernel='linear', probability=True)
            model1.fit(trainx,trainy1)

            #predict
            predict_train = model1.predict(trainx)
            predict_test = model1.predict(new_testx)


            #get the confidence score
            probability = model1.predict_proba(new_testx)
            confidence = max(probability)
            # print(confidence)

            if confidence > 0.45:

                acc_train = accuracy_score(trainy1,predict_train)

                #display
                trainy_list = list(trainy1)
                p=int(predict_test)
                if p in trainy_list:
                    val = trainy_list.index(p)
                    _trainy = out_encode.inverse_transform(trainy1)
                    string = strftime('%H:%M:%S: %p')
                    n=_trainy[val]
                    #student= StudentInfo.query.filter_by(username=n).first()
                    #pn = student.pno
                    #print(pn)
                    
                    name.append([n,string])
                    std_list.append(_trainy[val])
                    #print("send_msg(pn,n)")
                    #send_msg(pn,n)
                    #del(pn)
                    #del(n)
                #print(std_list)
                #print(trainy[val])
            else:
                #----------------------------unknown face counter-------------------------------#
                unfacecount+=1
                print("Unknown Face")

        if len(name) > 0:
            header = ['Name','Time']
            with open('Attendance.csv', 'w') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)
                for w in name:
                    writer. writerow(w)

                
        flash('Image successfully uploaded and displayed below')
        
     
       
        return render_template('index.html' , filename=filename , std_list=std_list, st1="Unknown Student : "+str(unfacecount))
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)



#-------------------------------------------------------------------------------------------------------------------------------
#Live Attendance

@app.route('/liveatt', methods=['POST','GET'])
def select_image():
        
        student= StudentInfo.query.all() 

        #cam = cv2.VideoCapture('rtsp://admin:L2E66E94@192.168.0.108/cam/realmonitor?channel=1&subtype=00')
        cam = cv2.VideoCapture(0)
        hasFrame, frame = cam.read()
        if not hasFrame :
            print("camera not accessible.")
            cam.release()
        else:
            for x in os.listdir('static/cropped/'):
                if os.path.isfile('static/cropped/'+x):
                    print("files existed and removed")
                    os.remove('static/cropped/'+x)

            for x in os.listdir('static/uploads/'):
                if os.path.isfile('static/uploads/'+x):
                    print("files existed and removed")
                    os.remove('static/uploads/'+x)

                else:
                    print(x)
                    print("files not found")
                    print("camera starts.")
                cv2.imwrite("static/uploads/img.jpg",frame)

        #--------Cropping of Images---------------------------------------------------#
        #---------------------------------------------------------------------------------#
        


        def save_faces(filecname, result_list):
            # load the image
            data = pyplot.imread(filecname)
            # plot each face as a subplot
            for i in range(len(result_list)):
                # get coordinates
                x1, y1, width, height = result_list[i]['box']
                x2, y2 = x1 + width, y1 + height
                cv2.imwrite("static/cropped/img_{}.png".format(i),data[y1:y2, x1:x2])
       
        fname = UPLOAD_FOLDER+ "img.jpg"
        filecname = fname
        # load image from file
        pixels = pyplot.imread(filecname)
        # create the detector, using default weights
        
        # detect faces in the image
        faces = detector.detect_faces(pixels)
        # display faces on the original image
        save_faces(filecname, faces)
        print("cropped saved")
        
        #-------------Check if image is face else delete----------------------------------#

        for x in os.listdir('static/cropped/'):
            test = 'static/cropped/' + x
            test_image = cv2.imread(test)
            faces = detector.detect_faces(test_image)

            if not faces:
                print('There is no face in the provided image, removing... '+x)
                os.remove('static/cropped/'+x)

            else:
                print('This is a perfect image.' +x)
        
       
        #------------ Take Attendance ----------------------------------------------------#

        std_list = []
        #extracting embeddings
        def extract_embeddings(model,face_pixels):
            face_pixels = face_pixels.astype('float32')
            mean = face_pixels.mean()
            std  = face_pixels.std()
            face_pixels = (face_pixels - mean)/std
            samples = expand_dims(face_pixels,axis=0)
            yhat = model.predict(samples)
            return yhat[0]
        directory='static/cropped/'
        print(os.listdir('static/cropped/'))
        dirs = os.listdir( directory )
        model = load_model('facenet_keras.h5',compile=False)
        data1 = load('DataSet.npz')
        train_x,train_y = data1['arr_0'],data1['arr_1']

        data = load('Embeddings.npz')
        trainx,trainy= data['arr_0'],data['arr_1']

        # i=0
        name=[[]]
        # data rows of csv file

        global unfacecount
        unfacecount = 0 
        for filename in dirs:
            Img = directory + filename

            #load data and reshape the image
            img1 = Image.open(Img)            
            img1 = img1.convert('RGB')  
            img1 = img1.resize((160,160))        
            pixels = asarray(img1)    
            testx = pixels.reshape(-1,160,160,3)
            # print("Input test data shape: ",testx.shape)

            #find embeddings
            new_testx = list()
            for test_pixels in testx:
                embeddings = extract_embeddings(model,test_pixels)
                new_testx.append(embeddings)

            new_testx = asarray(new_testx)  
            # print("Input test embedding shape: ",new_testx.shape)
            # print(trainy.shape[0])
            # print("Loaded data: Train=%d , Test=%d"%(trainx.shape[0],new_testx.shape[0]))

            #normalize the input data
            in_encode = Normalizer(norm='l2')
            trainx = in_encode.transform(trainx)
            new_testx = in_encode.transform(new_testx)

            #create a label vector
            new_testy = trainy 
            out_encode = LabelEncoder()
            out_encode.fit(trainy)
            trainy1 = out_encode.transform(trainy)
            new_testy = out_encode.transform(new_testy)
            
            #define svm classifier model 
            model1 =SVC(kernel='linear', probability=True)
            model1.fit(trainx,trainy1)

            #predict
            predict_train = model1.predict(trainx)
            predict_test = model1.predict(new_testx)


            #get the confidence score
            probability = model1.predict_proba(new_testx)
            confidence = max(probability)
            # print(confidence)

            if confidence > 0.45:

                acc_train = accuracy_score(trainy1,predict_train)

                #display
                trainy_list = list(trainy1)
                p=int(predict_test)
                if p in trainy_list:
                    val = trainy_list.index(p)
                    _trainy = out_encode.inverse_transform(trainy1)
                    string = strftime('%H:%M:%S: %p')
                    n=_trainy[val]
                    # student= StudentInfo.query.filter_by(username=n).first()
                    # pn = student.pno
                    name.append([n,string])
                    std_list.append(_trainy[val])
                    #send_msg(pn,n)
                    
                #print(std_list)
                #print(_trainy[val])
            else:
                #----------------------------unknown face counter-------------------------------#
                unfacecount+=1
                print("Unknown Face")

        if len(name) > 0:
            header = ['Name','time']
            with open('Attendance.csv', 'w') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)
                for w in name:
                    writer. writerow(w)

    
        flash('Image successfully uploaded and displayed below')
               
        return render_template('index.html' , filename="img.jpg" , std_list=std_list , st1=unfacecount)


#-------------------------------------------------------------------------------------------------------------------------------
#Generate Dataset

global switch,user_folder,username,camera
switch=0

def face_cropped(img):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        face_cropped = img[y:y+h, x:x+w]
        return face_cropped  

def gen_frames():  # generate frame by frame from camera
    img_id=0
    #camera = cv2.VideoCapture("rtsp://admin:L2E66E94@192.168.0.108/cam/realmonitor?channel=1&subtype=00")  # use 0 for web camera
    camera = cv2.VideoCapture(0)  # use 0 for web camera

    while True:
        if(switch):
            # Capture frame-by-frame
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                # frame = buffer.tobytes()
                facec = face_cropped(frame)
                if facec is not None:
                    img_id += 1
                    face = cv2.resize(facec, (450, 450))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    file_name_path=user_folder+"/" +username+"."+str(img_id)+".jpg"   
                    cv2.imwrite(file_name_path, face)
                    cv2.putText(face, str(img_id), (50, 50),cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    print(img_id)
                    if cv2.waitKey(1) == 13 or int(img_id) == 100:  # wait key ==delay
                        print("Dataset Created Successfully!")
                        camera.release()
                        cv2.destroyAllWindows()
                        exit()
                        
                        

                    frame = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    for x in os.listdir('new/'+ username+'/'):
        test = 'new/'+ username+'/' + x
        test_image = cv2.imread(test)
        faces = detector.detect_faces(test_image)

        if not faces:
            print('There is no face in the provided image, Please select another image.'+x)
            os.remove('new/'+ username+'/'+x) 
        coun=0
        for x in os.listdir('new/'+ username+'/'):
            coun+=1
        print("There are " + str(coun) + " number of images there out of 100")
        if coun < 85:
            print("Warning!! Images are less than 85% please regenerte the dataset immediately!")

@app.route('/gen',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        gen_frames()
        
    else:
        return render_template('student.html')
    return render_template('student.html')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera,username,user_folder,NEW_FOLDER

    if request.method == 'POST':
        NEW_FOLDER = 'new/'
        username = request.form.get("username")
        password = request.form.get("password")
        hashed_password = generate_password_hash(password, method = 'sha256')
        password = hashed_password

        fname=request.form.get("firstname")
        lname=request.form.get("lastname")
        dob = request.form.get("date")
        gender = request.form.get("gender")
        email= request.form.get("email")
        pno= request.form.get("phoneno")
        sem= request.form.get("sem")


        # print(username + password + fname + lname + str(dob) + gender + email + str(pno) + str(sem))

        new_register =StudentInfo(username=username, password=password, fname=fname, lname=lname, dob=dob, gender=gender, email=email, pno=pno, sem=sem)

        db.session.add(new_register)
        # print("before commit")
        db.session.commit()
        # print("after commit")
        student = StudentInfo.query.all() 

    
    def skip():
        print("folder exists and overwritting")

    def skip1():
        print("folder created")
    if request.method == "POST":
        username = username.strip().capitalize()
        user_folder = os.path.join(NEW_FOLDER, username)
        
        if os.path.exists(user_folder):
            skip()
        else:
            os.mkdir(user_folder)
            skip1()

        switch=1
             
                 
    elif request.method=='GET':
        return render_template('student.html' , student = student)
    return render_template('student.html' , student = student)



#-----------------------------------------------------------------------------------------------------------------------
#View search delete update Student

@app.route('/viewstudent',methods=['POST','GET'])
def viewstd():
    student = StudentInfo.query.all() 
    # if request.form.get("search")==None:
    #     student = StudentInfo.query.all()
    # else:
    #     student = StudentInfo.query.get_or_404(request.form.get("search"))
    return render_template('viewstudent.html' , student = student)


@app.route('/updatestd/<string:username>', methods=['GET', 'POST'])
def updatestd(username):
    if request.method=='POST':
        username = request.form['username']
        fname=request.form['firstname']
        lname=request.form['lastname']
        dob = request.form['date']
        gender = request.form['gender']
        email= request.form['email']
        pno= request.form['phoneno']
        sem= request.form['sem']
        student = StudentInfo.query.filter_by(username = username).first()
        student.fname = fname
        student.lname=lname
        student.dob = dob
        student.gender = gender
        student.email= email
        student.pno= pno
        student.sem= sem
        db.session.add(student)
        db.session.commit()
        return redirect("/viewstudent")
        
    student = StudentInfo.query.filter_by(username = username).first()
    return render_template('updatestudent.html', student = student)


@app.route('/deletestd/<string:username>')
def deletestd(username):
    student = StudentInfo.query.filter_by(username=username).first()
    db.session.delete(student)
    db.session.commit()
    return redirect("/viewstudent")


#-----------------------------------------------------------------------------------------------------------------------
#Add new Student


@app.route('/addnew')
def addnew():
    print("AddNewStudent")

    def extract_image(image):
        store_face=[]
        img1 = Image.open(image)            
        img1 = img1.convert('RGB')          
        pixels = asarray(img1)              
        detector = MTCNN()                  
        f = detector.detect_faces(pixels)
        if f != []:
            x1,y1,w,h = f[0]['box']             
            x1, y1 = abs(x1), abs(y1)
            x2 = abs(x1+w)
            y2 = abs(y1+h)
            #locate the co-ordinates of face in the image
            store_face = pixels[y1:y2,x1:x2]
        # plt.imshow(store_face)
        image1 = Image.fromarray(store_face,'RGB')    #convert the numpy array to object
        image1 = image1.resize((160,160))             #resize the image
        face_array = asarray(image1)                  #image to array
        return face_array

    def load_faces(directory):
        face = []
        i=1
        for filename in listdir(directory):
            path = directory + filename
            faces = extract_image(path)
            face.append(faces)
        return face

    def load_dataset(directory):
        x, y = [],[]
        i=1
        for subdir in listdir(directory):
            path = directory + subdir + '/'
            #load all faces in subdirectory
            faces = load_faces(path)
            #create labels
            labels = [subdir for _ in range(len(faces))]
            # print(labels)
            #summarize
            print("%d There are %d images in the class %s:"%(i,len(faces),subdir))
            x.extend(faces)
            y.extend(labels)
            i=i+1
        return asarray(x),asarray(y)  

    print(listdir('new/'))
    #load the datasets
    trainX,trainY = load_dataset('new/')
    print(trainX.shape,trainY.shape)
    # print(trainY)
    #compress the data
    savez_compressed('newED/ANSDataSet.npz',trainX,trainY)

    #Generalize the data and extract the embeddings
    def extract_embeddings(model,face_pixels):
        face_pixels = face_pixels.astype('float32')  #convert the entire data to float32(base)
        mean = face_pixels.mean()                    #evaluate the mean of the data
        std  = face_pixels.std()                     #evaluate the standard deviation of the data
        face_pixels = (face_pixels - mean)/std       
        samples = expand_dims(face_pixels,axis=0)    #expand the dimension of data 
        yhat = model.predict(samples)
        return yhat[0]

    #load the compressed dataset and facenet keras model
    data = load('newED/ANSDataSet.npz')
    trainx, trainy = data['arr_0'],data['arr_1']
    # print(trainy)
    print(trainx.shape, trainy.shape)
    model = load_model('facenet_keras.h5',compile=False)

    #get the face embeddings
    new_trainx = list()
    for train_pixels in trainx:
        embeddings = extract_embeddings(model,train_pixels)
        new_trainx.append(embeddings)
    new_trainx = asarray(new_trainx)             #convert the embeddings into numpy array
    print(new_trainx.shape)

    #compress the 128 embeddings of each face 
    savez_compressed('newED/ANSEmbeddings.npz',new_trainx,trainy)




    data_1 = np.load('newED/DataSet.npz')
    data_2 = np.load('newED/ANSDataSet.npz')
    arr_0 = np.concatenate([data_1['arr_0'], data_2['arr_0']])
    arr_1 = np.concatenate([data_1['arr_1'], data_2['arr_1']])
    np.savez('newED/FinalDataSet.npz', arr_0, arr_1)

    #For Combinning Embeddings
    data_1 = np.load('newED/Embeddings.npz')
    data_2 = np.load('newED/ANSembeddings.npz')
    arr_0 = np.concatenate([data_1['arr_0'], data_2['arr_0']])
    arr_1 = np.concatenate([data_1['arr_1'], data_2['arr_1']])
    np.savez('newED/Finalembeddings.npz', arr_0, arr_1)
    msg="New student added"
    print(msg)
    
    fldpath="new/"
    srcpath="old/"
    for filename in os.listdir(fldpath):
        path = fldpath + filename
        shutil.move(path,srcpath)
    
    
    return render_template('index.html',msg=msg) 


#------------------------------------------------------------------------------------------------------------------------
#Train Model
@app.route('/train')
def Train():
    print("Train")
    def extract_image(image):
        store_face=[]
        img1 = Image.open(image)            
        img1 = img1.convert('RGB')          
        pixels = asarray(img1)              
        detector = MTCNN()                  
        f = detector.detect_faces(pixels)
        if f != []:
            x1,y1,w,h = f[0]['box']             
            x1, y1 = abs(x1), abs(y1)
            x2 = abs(x1+w)
            y2 = abs(y1+h)
            #locate the co-ordinates of face in the image
            store_face = pixels[y1:y2,x1:x2]
        # plt.imshow(store_face)
        image1 = Image.fromarray(store_face,'RGB')    #convert the numpy array to object
        image1 = image1.resize((160,160))             #resize the image
        face_array = asarray(image1)                  #image to array
        return face_array

    def load_faces(directory):
        face = []
        i=1
        for filename in listdir(directory):
            path = directory + filename
            faces = extract_image(path)
            face.append(faces)
        return face

    def load_dataset(directory):
        x, y = [],[]
        i=1
        for subdir in listdir(directory):
            path = directory + subdir + '/'
            #load all faces in subdirectory
            faces = load_faces(path)
            #create labels
            labels = [subdir for _ in range(len(faces))]
            # print(labels)
            #summarize
            print("%d There are %d images in the class %s:"%(i,len(faces),subdir))
            x.extend(faces)
            y.extend(labels)
            i=i+1
        return asarray(x),asarray(y)  

    print(listdir('new/'))
    #load the datasets
    trainX,trainY = load_dataset('new/')
    print(trainX.shape,trainY.shape)
    # print(trainY)
    #compress the data
    savez_compressed('newED/DataSet.npz',trainX,trainY)

    #Generalize the data and extract the embeddings
    def extract_embeddings(model,face_pixels):
        face_pixels = face_pixels.astype('float32')  #convert the entire data to float32(base)
        mean = face_pixels.mean()                    #evaluate the mean of the data
        std  = face_pixels.std()                     #evaluate the standard deviation of the data
        face_pixels = (face_pixels - mean)/std       
        samples = expand_dims(face_pixels,axis=0)    #expand the dimension of data 
        yhat = model.predict(samples)
        return yhat[0]

    #load the compressed dataset and facenet keras model
    data = load('newED/DataSet.npz')
    trainx, trainy = data['arr_0'],data['arr_1']
    # print(trainy)
    print(trainx.shape, trainy.shape)
    model = load_model('facenet_keras.h5',compile=False)

    #get the face embeddings
    new_trainx = list()
    for train_pixels in trainx:
        embeddings = extract_embeddings(model,train_pixels)
        new_trainx.append(embeddings)
    new_trainx = asarray(new_trainx)             #convert the embeddings into numpy array
    print(new_trainx.shape)

    #compress the 128 embeddings of each face 
    savez_compressed('newED/Embeddings.npz',new_trainx,trainy)

    msg2="Model has been trained"
    print(msg2)
    return render_template('index.html', msg2=msg2)


#-------------------------------------------------------------------------------------------------------------------------------
#Add Teacher
@app.route('/teacher',methods=['POST','GET'])
def teacher():
    return render_template('teacher.html')

@app.route('/addteacher',methods=['POST','GET'])
def addteacher():
    
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")
        hashed_password = generate_password_hash(password, method = 'sha256')
        password = hashed_password
        fname=request.form.get("firstname")
        lname=request.form.get("lastname")
        dob = request.form.get("date")
        gender = request.form.get("gender")
        email= request.form.get("email")
        pno= request.form.get("phoneno")
        subject= request.form.get("subject")

        new_register =TeacherInfo(username=username, password=password, fname=fname, lname=lname, dob=dob, gender=gender, email=email, pno=pno, subject = subject)

        db.session.add(new_register)
        db.session.commit()
   
    elif request.method=='GET':
        return render_template('teacher.html')
    return render_template('teacher.html')



#-----------------------------------------------------------------------------------------------------------------------
#View search delete update teacher

@app.route('/viewteacher',methods=['POST','GET'])
def viewtech():
    teacher = TeacherInfo.query.all()
    return render_template('viewteacher.html' , teacher=teacher)


@app.route('/update/<string:username>', methods=['GET', 'POST'])
def update(username):
    if request.method=='POST':
        username = request.form.get("username")
        fname=request.form.get("firstname")
        lname=request.form.get("lastname")
        dob = request.form.get("date")
        gender = request.form.get("gender")
        email= request.form.get("email")
        pno= request.form.get("phoneno")
        subject= request.form.get("subject")
        # username = request.form['username']
        # fname=request.form['firstname']
        # lname=request.form['lastname']
        # dob = request.form['date']
        # gender = request.form['gender']
        # email= request.form['email']
        # pno= request.form['phoneno']
        # subject= request.form['subject']
        teacher = TeacherInfo.query.filter_by(username = username).first()
        teacher.fname = fname
        teacher.lname=lname
        teacher.dob = dob
        teacher.gender = gender
        teacher.email= email
        teacher.pno= pno
        teacher.subject= subject
        db.session.add(teacher)
        db.session.commit()
        return redirect("/viewteacher")
        
    teacher = TeacherInfo.query.filter_by(username = username).first()
    return render_template('updateteacher.html', teacher = teacher)

@app.route('/delete/<string:username>')
def delete(username):
    teacher = TeacherInfo.query.filter_by(username=username).first()
    db.session.delete(teacher)
    db.session.commit()
    return redirect("/viewteacher")

# def searchtech():
#     teacher = TeacherInfo.query.get_or_404(username) 
#     return render_template('viewstudent.html' , teacher = teacher)


#-------------------------------------------------------------------------------------------------------------------------


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run( host='0.0.0.0', port=8080 ,debug=True)



