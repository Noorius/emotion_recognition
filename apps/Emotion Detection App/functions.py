import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import dlib
from imutils import face_utils
import joblib
import itertools


def euc(a, b):
    return np.sqrt( (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) )

def calc_dist(shape):
    V1 = (euc(shape[1],shape[5]) + euc(shape[2],shape[4])) / 2
    V2 = euc(shape[0], shape[3])            
    V3 = (euc(shape[7],shape[11]) + euc(shape[8],shape[10])) / 2
    V4 = euc(shape[6], shape[9])            
    V5 = euc(shape[12], shape[14])
    V6 = euc(shape[15], shape[17])            
    V7 = euc(shape[19], shape[25])
    V8 = (euc(shape[1],shape[13]) + euc(shape[2],shape[13])) / 2            
    V9 = (euc(shape[7],shape[16]) + euc(shape[8],shape[16])) / 2
    V10 = euc(shape[18], shape[22])            
    V11 = (euc(shape[5],shape[19]) + euc(shape[4],shape[19])) / 2
    V12 = (euc(shape[11],shape[25]) + euc(shape[10],shape[25])) / 2            
    V13 = euc(shape[3], shape[14])
    V14 = euc(shape[6], shape[15])            
    V15 = euc(shape[20], shape[30])
    V16 = euc(shape[21], shape[29])            
    V17 = euc(shape[22], shape[28])
    V18 = euc(shape[23], shape[27])            
    V19 = euc(shape[24], shape[26])
    V20 = euc(shape[19], shape[18])            
    V21 = euc(shape[25], shape[18])
    V22 = (euc(shape[33],shape[34]) + euc(shape[35],shape[38]) + euc(shape[36],shape[37])) / 3            
    V23 = euc(shape[3],shape[19])
    V24 = euc(shape[6],shape[20])
    
    return V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24

def landmarks(frame, img_shape, std, mean):
    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (img_shape, img_shape), interpolation=cv2.INTER_AREA)
    roi_gray = cv2.GaussianBlur(roi_gray, (3,3), cv2.BORDER_DEFAULT)
    
def euc2d(a, b):
    return np.sqrt( (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) )
    
def euc3d(a, b):
    return np.sqrt( (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]) )
    
def predict(model, frame):
    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    
    if np.sum([roi_gray]) != 0:
        roi = tt.functional.to_pil_image(roi_gray)
        roi = tt.functional.to_grayscale(roi)
        roi = tt.ToTensor()(roi).unsqueeze(0)

        # make a prediction on the ROI
        tensor = model(roi)
        tensor = torch.nn.functional.softmax(tensor, dim=1).detach().numpy()[0]
        
        #pred = torch.max(tensor, dim=1)[1].tolist()
    
    return tensor.tolist()

def predict2(model, frame, img_shape, scaler, face_mesh, mp_face_mesh):
#     roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     roi_gray = cv2.resize(roi_gray, (img_shape, img_shape), interpolation=cv2.INTER_AREA)
#     roi_gray = cv2.GaussianBlur(roi_gray, (3,3), cv2.BORDER_DEFAULT)

#     landmarks = predictor(roi_gray, dlib.rectangle(0, 0, img_shape, img_shape))
    
#     shape = face_utils.shape_to_np(landmarks)
#     shape = shape[[36, 37, 38, 39, 40, 41] + [42, 43, 44, 45, 46, 47] + [17, 19, 21] + [22, 24, 26] + [30, 31, 35] + [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 67, 62, 63, 65, 66]]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    LEFT_EYE = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
    RIGHT_EYE = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
    LEFT_EYEBROW = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
    RIGHT_EYEBROW = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
    LIPS = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
    CONTOURS = list(set(itertools.chain(*mp_face_mesh.FACEMESH_CONTOURS)))
    OTHER = [1] #0, 1, 6, 8, 10
    
    frame = cv2.resize(frame, (img_shape, img_shape), interpolation=cv2.INTER_AREA)
    
    results = face_mesh.process(frame)
    
    values = np.zeros(92 * 2).reshape(1, -1)
    
    if results.multi_face_landmarks:
        
        shape = [(lmk.x, lmk.y, lmk.z) for lmk in results.multi_face_landmarks[0].landmark]
        shape = np.array(shape)
        nose = shape[1]
        shape = shape[LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + LIPS]

        #distances2d = distance.cdist(shape[1], shape, 'euclidean')

        distances2d = [round(euc2d(nose, x), 6) for x in shape]
        distances3d = [round(euc3d(nose, x), 6) for x in shape]

        #df.append(pd.DataFrame([np.append(distances.flatten(), i)], columns=columns))

        values = np.array(distances2d + distances3d).reshape(1, -1)
        
        values = scaler.transform(values)
       
    
    pred = model.predict(values, verbose=0)
    
    #print(pred)
    
    #print(pred.tolist()[0])
    
    return pred.tolist()[0]

def predict3(model, frame, img_shape, scaler, face_mesh, mp_face_mesh):
#     roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     roi_gray = cv2.resize(roi_gray, (img_shape, img_shape), interpolation=cv2.INTER_AREA)
#     roi_gray = cv2.resize(roi_gray, (img_shape, img_shape), interpolation=cv2.INTER_AREA)
#     roi_gray = cv2.GaussianBlur(roi_gray, (3,3), cv2.BORDER_DEFAULT)
    
#     landmarks = predictor(roi_gray, dlib.rectangle(0, 0, img_shape, img_shape))
    
#     shape = face_utils.shape_to_np(landmarks)
#     shape = shape[[36, 37, 38, 39, 40, 41] + [42, 43, 44, 45, 46, 47] + [17, 19, 21] + [22, 24, 26] + [30, 31, 35] + [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 67, 62, 63, 65, 66]]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    LEFT_EYE = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
    RIGHT_EYE = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
    LEFT_EYEBROW = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
    RIGHT_EYEBROW = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
    LIPS = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
    CONTOURS = list(set(itertools.chain(*mp_face_mesh.FACEMESH_CONTOURS)))
    OTHER = [1] #0, 1, 6, 8, 10

    frame = cv2.resize(frame, (img_shape, img_shape), interpolation=cv2.INTER_AREA)
    
    results = face_mesh.process(frame)
    
    values = np.zeros(92 * 2).reshape(1, -1)
    
    if results.multi_face_landmarks:
        
        shape = [(lmk.x, lmk.y, lmk.z) for lmk in results.multi_face_landmarks[0].landmark]
        shape = np.array(shape)
        nose = shape[1]
        shape = shape[LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + LIPS]

        #distances2d = distance.cdist(shape[1], shape, 'euclidean')

        distances2d = [round(euc2d(nose, x), 6) for x in shape]
        distances3d = [round(euc3d(nose, x), 6) for x in shape]

        #df.append(pd.DataFrame([np.append(distances.flatten(), i)], columns=columns))

        values = np.array(distances2d + distances3d).reshape(1, -1)
        
        values = scaler.transform(values)
    
    pred = model.predict_proba(values)
    
    #print(pred.tolist())
    
    return pred.tolist()[0]
