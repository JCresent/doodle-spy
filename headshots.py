import cv2
import os
import time

def create_path(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder
    
def capture_photos(name):
    folder = create_path(name)
    
    cam = cv2.VideoCapture(0)
    #if not cam.isOpend():
        #print("ERROR: Can't open webcam")
        #return
    
    time.sleep(2) #CAM WARM UP
    
    photo_count = 0
    
    print(f"Taking photos for {name}. Press space to take picture or q to quit")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        cv2.imshow('Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            photo_count += 1
            filename = name+"_"+str(photo_count)+".jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Photo {photo_count} saved:{filepath}")
            
        elif key == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    PERSON_NAME = input("What is your name: ")
    print(PERSON_NAME)
    capture_photos(PERSON_NAME)