import cv2
face_Classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # getting the face detection Model 
img = cv2.imread('./Images/sad.jpg',1) 
grayScaleImage = cv2.cvtColor(img,0) 

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
faces = face_Classifier.detectMultiScale(grayScaleImage,1.3,2)

for (x,y,w,h) in faces :
    #drawing rectange over the faces 
    cv2.rectangle(img,(x,y),(x+w,y+h),(100,25,70),5,cv2.LINE_AA)

cv2.imshow("Detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()