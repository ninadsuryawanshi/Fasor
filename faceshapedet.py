import cv2
import numpy as np

# Load the face landmark model
face_model = cv2.face.createFacemarkLBF()

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('img3_round.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Check if a single face is detected
if len(faces) == 1:
    (x, y, w, h) = faces[0]
    
    # Extract the face region
    face_region = gray[y:y+h, x:x+w]
    
    # Detect face landmarks in the face region

    # face_model.fit(img, faces=[np.array([(x, y, w, h)])])
    if not isinstance(img, np.ndarray):
     raise ValueError("img should be a NumPy array")

    if not isinstance(faces, list):
     raise ValueError("faces should be a list")

    for face in faces:
     if not isinstance(face, np.ndarray):
        raise ValueError("Each element in faces should be a NumPy array")
    landmarks = face_model.getFaces()[0]
    
    # Get the (x, y) coordinates of the landmarks
    landmarks_coords = landmarks.reshape(-1, 2)
    
    # Calculate the distances between the landmarks to determine the face shape
    distances = [
        np.linalg.norm(landmarks_coords[i] - landmarks_coords[j])
        for i in [30, 45, 48]
        for j in [30, 45, 48]
    ]
    
    # Determine the face shape based on the distances
    if np.mean(distances) < 80:
        face_shape = 'Round'
    elif np.mean(distances) < 100:
        face_shape = 'Oval'
    elif np.mean(distances) < 120:
        face_shape = 'Square'
    else:
        face_shape = 'Rectangular'
    
    # Draw the landmarks on the image
    for p in landmarks_coords:
        cv2.circle(img, (int(p[0] + x), int(p[1] + y)), 1, (0, 255, 0), -1)
    
    # Draw the face shape text on the image
    cv2.putText(img, face_shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the output image with the detected face shape
    cv2.imshow('Face Shape Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected or multiple faces detected. Please try again with a single face image.")