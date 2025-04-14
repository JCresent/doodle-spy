import face_recognition
import cv2
import numpy as np
import time
import pickle

# Load pre-trained face encodings
print("Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Grab webcam
video_capture = cv2.VideoCapture(0) 

cv_scaler = 2  # integer
seen_face_locations = []
seen_face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

def process_frame(frame):
    global seen_face_locations, seen_face_encodings, face_names
    
    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    seen_face_locations = face_recognition.face_locations(rgb_resized_frame, model="hog")
    seen_face_encodings = face_recognition.face_encodings(rgb_resized_frame, seen_face_locations, model='large')
    
    face_names = []
    for face_encoding in seen_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Intruder"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    
    return frame

def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name in zip(seen_face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        if name == "Intruder":
               
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left -3, top - 35), (right+3, top), (0, 0, 255), cv2.FILLED)
            
        else:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left -3, top - 35), (right+3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break
    
    processed_frame = process_frame(frame)
    
    display_frame = draw_results(processed_frame)
    
    # Track FPS
    current_fps = calculate_fps()
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show in frame
    cv2.imshow('Video', display_frame)
    
    # Quit key
    if cv2.waitKey(1) == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()