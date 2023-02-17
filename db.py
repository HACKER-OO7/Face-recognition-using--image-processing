from flask import Flask, render_template, flash, request, redirect, url_for
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash


#create the object of Flask
app  = Flask(__name__)

app.config['SECRET_KEY'] = 'hardsecretkey'


#SqlAlchemy Database Configuration With Mysql
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False



db = SQLAlchemy(app)


class StudentInfo(db.Model):
    username = db.Column(db.String(100), unique = True, primary_key = True)
    password = db.Column(db.String(100), nullable=False)
    fname = db.Column(db.String(100), nullable=False)
    lname = db.Column(db.String(100), nullable=False)
    #dob = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    pno = db.Column(db.Integer(10), nullable=False)
    sem = db.Column(db.Integer, nullable=False)    

    def __init__(self, username, password):
        self.username = username
        self.password = password


#This is our model
class TeacherInfo(db.Model):
    username = db.Column(db.String(100), unique = True,primary_key = True)
    password = db.Column(db.String(100), nullable=False)
    fname = db.Column(db.String(100), nullable=False)
    lname = db.Column(db.String(100), nullable=False)
    #dob = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    pno = db.Column(db.Integer(10), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    totlec = db.Column(db.Integer , nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.password = password


    

    def __init__(self, username, password):
        self.username = username
        self.password = password



#creating our routes
@app.route('/')
def index():

    return render_template('index.html')





#run flask app
if __name__ == "__main__":
    app.run(debug=False)