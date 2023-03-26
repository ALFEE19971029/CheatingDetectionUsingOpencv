import cv2
from gaze_tracking import GazeTracking
import os


models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

path = "./"
file_list = os.listdir(path)
file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
tempList = []

while True:
    # We get a new frame from the webcam
    retval, frame = webcam.read()
    frame = cv2.flip(frame,1)   #좌우반전

    if not(retval):   # 프레임정보를 정상적으로 읽지 못하면
        break  # while문을 빠져나가기

    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text_warning = ""
    text_ok = ""
    check_second = False

    if gaze.is_right():
        text_warning = "Warning! Looking Right"
    elif gaze.is_left():
        text_warning = "Warning! Looking Left"
    elif gaze.is_center():
        check_second = True

    if check_second == True:
        cv2.putText(frame, "Doing Well", (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (147, 58, 31), 2)
    elif check_second == False:
        cv2.putText(frame, text_warning, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
