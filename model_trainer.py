import face_recognition
from imutils import paths
import os
import pickle
import cv2

print("Start processing faces. . . ")
image_paths = list(paths.list_images('dataset'))
knownEncodings = []
knownNames = []

for (i, image_path) in enumerate(image_paths):
    print(f"Proceessing image {i + 1}/{len(image_paths)}")
    name = image_path.split(os.path.sep)[-2]
    
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model="hog") # hog is default, faster for GPUs can also use cnn
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
        
print("Serialze encodings. . .")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))
    
print("FINISHED: saved encodings in encodings.pickle")