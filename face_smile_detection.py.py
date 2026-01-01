import cv2

face_casecade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_casecade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

smile_casecade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )


webcam = cv2.VideoCapture(0)

while True:
    ret, img = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_casecade.detectMultiScale(gray, 1.5, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

#****EYES DETECTION****#

    eye = eye_casecade.detectMultiScale(roi_gray, 1.5, 4)  
    if len(eye) > 0:
        cv2.putText(img, "Eyes Detected", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)  


#****SMILE DETECTION****#

    smiles = smile_casecade.detectMultiScale(roi_gray, 1.5, 4)

    if len(smiles) > 0:
        cv2.putText(img, "Smiling", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2) 
 


    cv2.imshow("Smart Face Detection", img)

    if cv2.waitKey(10) == 27:
        break


webcam.release()
cv2.destroyAllWindows()
