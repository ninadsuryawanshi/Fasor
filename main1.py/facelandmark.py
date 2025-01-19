import cv2
import dlib

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        # Extract key points for face shape detection
        left_cheek = (face_landmarks.part(4).x, face_landmarks.part(4).y)
        right_cheek = (face_landmarks.part(14).x, face_landmarks.part(14).y)
        chin = (face_landmarks.part(8).x, face_landmarks.part(8).y)
        forehead = (face_landmarks.part(0).x, face_landmarks.part(0).y)
        
        # Calculate distances between key points
        cheek_width = right_cheek[0] - left_cheek[0]
        face_height = chin[1] - forehead[1]

        # Based on distances, classify face shape
        if cheek_width / face_height > 0.8:
            face_shape = "Round"
        elif cheek_width / face_height < 0.7:
            face_shape = "Oval"
        else:
            face_shape = "Square"

        # Draw face shape label on the frame
        cv2.putText(frame, face_shape, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
