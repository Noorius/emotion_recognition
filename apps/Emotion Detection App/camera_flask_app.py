from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import pickle
import dlib
from functions import predict, predict2, predict3
from get_cnn_model import get_cnn_model
from flask_socketio import SocketIO
from tensorflow import keras
import mediapipe as mp


global capture,rec_frame, grey, switch, neg, face, rec, out, cnn, mlp, gradboost
capture=0
# grey=0
# neg=0
face=0
switch=1
rec=0
cnn = mlp = gradboost = 0
emotions = []

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

model_gradboost = pickle.load(open("model9_nose_boost185", 'rb'))
model_mlp = keras.models.load_model("mlp_dataset_60%_5emotions.h5")
model_cnn = get_cnn_model()

face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
img_shape = 48
# std = np.load('std.npy')
# mean = np.load('mean.npy')
scaler = pickle.load(open("scaler-train-dataset-184.pkl", 'rb'))

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
app.config['SECRET_KEY'] = 'mysecretkey'
socketio = SocketIO(app)

camera = cv2.VideoCapture(0)
    

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
#     global net
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#         (300, 300), (104.0, 177.0, 123.0))   
#     net.setInput(blob)
#     detections = net.forward()
#     confidence = detections[0, 0, 0, 2]

#     if confidence < 0.5:            
#             return frame           

#     box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
#     (startX, startY, endX, endY) = box.astype("int")
#     try:
#         frame=frame[startY:endY, startX:endX]
#         (h, w) = frame.shape[:2]
#         r = 480 / float(h)
#         dim = ( int(w * r), 480)
#         frame=cv2.resize(frame,dim)
#     except Exception as e:
#         pass
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces):
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        frame = frame[y : y + h, x : x + w]
    
    return frame
 
@socketio.on('connect')
def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame, emotions
    while True:
        success, frame = camera.read() 
        if success:
            a_face = detect_face(frame)
            
            if(cnn):
                emotions = predict(model_cnn, a_face)
            elif(mlp):
                emotions = predict2(model_mlp, a_face, img_shape, scaler, face_mesh, mp_face_mesh)
            elif(gradboost):
                emotions = predict3(model_gradboost, a_face, img_shape, scaler, face_mesh, mp_face_mesh)
            
            socketio.emit('update_chart', emotions)
            
            if(face):                
                frame = a_face
#             if(grey):
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             if(neg):
#                 frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        
        global cnn, mlp, gradboost
        if request.form.get('models') == 'cnn':
            mlp = gradboost = 0
            cnn = 1
        elif request.form.get('models') == 'mlp':
            cnn = gradboost = 0
            mlp = 1
        elif request.form.get('models') == 'gradboost':
            cnn = mlp = 0 
            gradboost = 1
        
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     