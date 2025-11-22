import cv2
import numpy as np
from keras.models import load_model
import os

try:
    model = load_model('models/model_file_30epochs.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if faceDetect.empty():
    print("Error: Haar cascade file not found")
    exit()

labels_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open camera. Trying alternative camera...")
    video = cv2.VideoCapture(1)
    if not video.isOpened():
        print("Error: No camera available")
        exit()

print("Camera opened successfully. Press 'q' to quit.")

frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        print("Warning: Could not read frame, retrying...")
        continue
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for x, y, w, h in faces:
        try:
            sub_face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            
            if frame_count % 5 == 0:  # Predict every 5th frame for performance
                result = model.predict(reshaped, verbose=0)
                label = np.argmax(result, axis=1)[0]
                emotion = labels_dict[label]
                confidence = np.max(result)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 255, 0), -1)
            cv2.putText(frame, f"{emotion} {confidence:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e:
            continue
    
    cv2.imshow("Real-time Emotion Detection", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Application closed successfully")