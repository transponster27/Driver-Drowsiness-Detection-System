import imutils
import cv2
from imutils import face_utils
import dlib
from scipy.spatial import distance

def eye_aspect_ratio(eye):                           #calculate the open eye distance and watch out for the closed eye distance which drops A+B when closed
    A = distance.euclidean(eye(1), eye(5))
    B = distance.euclidean(eye(2), eye(4))
    C = distance.euclidean(eye(0), eye(3))
    ear = (A+B)/(2.0*C)
    return ear                                       #differentiate blink
(lStart, lEnd) = face_utils.FACIAL_LANDMARK_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARK_68_IDXS['right_eye']


detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"C:\Users\vrehm\OneDrive\Desktop\shape_predictor_68_face_landmarks.dat")    #face landmarks
cap = cv2.VideoCapture(0)       

while True:
    ret, frame = cap.read()                  #read returns boolean value if the frame avails or not-ret, image- frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subjects)              #predict landmarks on face
        shape = face_utils.shape_to_np(shape)        #converting face to a list of xy coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)
        ear = (leftEar + rightEar) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)         #minimum distance that covers the eye
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, {0, 255, 0}, 1)                                              #contours that join the convex hull(outline)
        cv2.drawContours(frame, [rightEyeHull], -1, {0, 255, 0}, 1)                                              #contours that join the convex hull(outline)


    cv2.imshow("Frame", frame)               #displays image stored in frame var
    cv2.waitKey(1)                           #waits for a time duration to stop display
