# doodle-spy
Real Time Video Surveillance 

## Dependencies
```
dlib    
cmake    
face_recognition    
opencv
```

## Usage:  
Ran on raspberry pi 3B (running raspberrian) with a freetalk webcam.     

Running scripts:
1. Use the headshots.py to take many headshots of desired recognized face    
```
python headshots.py     

What is your name: <INPUT NAME HERE>

```
2. Train model with model_trainer.py     
```
python model_trainer.py     

Start processing faces. . .     
Proceessing image 1/n       
. . .      
Serialze encodings. . .     
FINISHED: saved encodings in encodings.pickle

```      
3. Run the face_recognizer.py    
```
python face_recognizer.py     

Loading encodings...
```
