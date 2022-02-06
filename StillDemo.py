import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
img=cv2.imread('sad.jpg')
#img=cv2.imread('happy.jpg')
predictions=DeepFace.analyze(img)
#print(predictions['dominant_emotion'])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
result = DeepFace.analyze(img, actions=['emotion'])
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,result['dominant_emotion'],(50,100),font,3,
(0,0,255),2,cv2.LINE_4)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()