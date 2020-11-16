import cv2
from PIL import  Image
import os
import numpy as np
import random
import time


def extract_face(filename):
    '''
    get faces from given image using haar cascades
    :param filename:
    :param required_size:
    :return:
    '''    
    #read file as grescale image
    image = cv2.imread(filename,0)
    # Loading classifiers
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    #extract feature coodinates
    features = faceCascade.detectMultiScale(image, scaleFactor = 1.1, minNeighbors = 10)
    coords = []
    
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    # drawing rectangle around the feature and labeling it
    
    for (x, y, w, h) in features:
        #cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)
        coords = [x, y, w, h]

    if len(coords) == 4:
        # Updating region of interest by cropping image
        roi_img = image[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
    elif len(coords) > 4:
        return -2
    else:
        return -1
    cv2.imwrite("selena.jpg", roi_img)
    face_array = np.asarray(roi_img)
    return face_array


def get_embeddings(filename):
    '''
    gets image embeddings using given model
    :param filename:
    :param model:
    :return:
    '''

    return 
    
def validate(ground_truth_file_path, candidate_file_path):
    '''
    Validate a candidate against an image using a given model
    :param ground_truth_file_path:
    :param candidate_file_path:
    :param model: defaults to Resnet50
    :return: score between 0 and 1 (lower is better)
    '''
    faces=[]
    ids=[]
    # get classifier file filenames
    clf = cv2.face.LBPHFaceRecognizer_create()    
    # extract faces
    face_ground = extract_face(ground_truth_file_path)
    if type(face_ground) is int:
        return face_ground
    else:
        imageNp = np.array(face_ground, 'uint8')
        faces.append(imageNp)
        # perform prediction
        ids.append(1)
        clf.train(faces, np.array(ids))
        try:
            _, con = clf.predict(np.array(extract_face(candidate_file_path), 'uint8'))
        except:
            con = -3
        return con

def recognise(ground_truth_file_directory, candidate_file_path):
    '''
    Validate a candidate against an image using a given model
    :param ground_truth_file_path:
    :param candidate_file_path:
    :param model: defaults to Resnet50
    :return: score between 0 and 1 (lower is better)
    '''
    faces=[]
    ids=[]
    
    files = os.listdir(ground_truth_file_directory)
    files = [os.path.join(ground_truth_file_directory, f) for f in files]
    #teeestss = random.sample(files, 1)
    #files.remove(teeestss[0])
    
    # get classifier file filenames
    clf = cv2.face.LBPHFaceRecognizer_create()  
    
    # extract faces
    for path in files:
        face_ground = extract_face(path)
        imageNp = np.array(face_ground , 'uint8')
        faces.append(imageNp)
        # perform prediction
        ids.append(1)
    
    clf.train(faces, np.array(ids))
    
    id, con = clf.predict(extract_face(candidate_file_path)) 
    return con




if __name__ == "__main__":
    
    #test = extract_face('00005/00005_941121_fa.ppm')
    #test = recognise('00003','Selena_Gomez1.jpg')
    test = validate('00003/00003_940307_fb_a.ppm','mueez3.JPEG')
    
    print(test)