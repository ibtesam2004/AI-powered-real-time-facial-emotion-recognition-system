import cv2
import numpy as np
from keras.models import load_model

model = load_model('data/models/model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
labels_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

frame = cv2.imread('data/image2.png')
if frame is None:
    print("Error: Could not load image")
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No faces detected")
else:
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped, verbose=0)
        label = np.argmax(result, axis=1)[0]
        emotion = labels_dict[label]
        confidence = np.max(result)
        print(f"Detected: {emotion} ({confidence:.2f})")
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 255, 0), -1)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imshow("Emotion Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()