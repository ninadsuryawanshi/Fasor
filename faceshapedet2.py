# import cv2
# import numpy as np

# # Load the face landmark model
# face_model = cv2.face.createFacemarkLBF()
# face_model.loadModel('lbfmodel.yaml')  # Make sure to load the pre-trained model

# # Load the face detection cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Read the input image
# img = cv2.imread('testimgrect.jpeg')

# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Detect faces in the image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # Check if a single face is detected
# if len(faces) == 1:
#     (x, y, w, h) = faces[0]

#     # Convert the face bounding box to a numpy array
#     face_list = np.array([[x, y, w, h]], dtype=np.int32)

#     # Detect face landmarks in the face region
#     success, landmarks = face_model.fit(gray, face_list)

#     if success:
#         # Get the (x, y) coordinates of the landmarks
#         landmarks_coords = landmarks[0][0]

#         # Calculate the distances between the landmarks to determine the face shape
#         distances = [
#             np.linalg.norm(landmarks_coords[i] - landmarks_coords[j])
#             for i in [30, 45, 48]
#             for j in [30, 45, 48]
#             if i != j
#         ]

#         # Determine the face shape based on the distances
#         if np.mean(distances) < 80:
#             face_shape = 'Round'
#         elif np.mean(distances) < 100:
#             face_shape = 'Oval'
#         elif np.mean(distances) < 120:
#             face_shape = 'Square'
#         else:
#             face_shape = 'Rectangular'

#         # # Draw the landmarks on the image
#         # for p in landmarks_coords:
#         #     cv2.circle(img, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)

#         # # Draw the face shape text on the image
#         # cv2.putText(img, face_shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # # Display the output image with the detected face shape
#         # cv2.imshow('Face Shape Detection', img)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()

# else:
#     print("No face or multiple faces detected.")
# print(face_shape)

import cv2
import numpy as np

# Load the face landmark model
face_model = cv2.face.createFacemarkLBF()
face_model.loadModel('lbfmodel.yaml')  # Make sure to load the pre-trained model

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('testimgoval.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Check if a single face is detected
if len(faces) == 1:
    (x, y, w, h) = faces[0]

    # Convert the face bounding box to a numpy array
    face_list = np.array([[x, y, w, h]], dtype=np.int32)

    # Detect face landmarks in the face region
    success, landmarks = face_model.fit(gray, face_list)

    if success:
        # Get the (x, y) coordinates of the landmarks
        landmarks_coords = landmarks[0][0]

        # Define the key points based on landmarks
        chin = landmarks_coords[8]  # Chin
        left_cheek = landmarks_coords[3]  # Left cheek
        right_cheek = landmarks_coords[13]  # Right cheek
        forehead_center = landmarks_coords[27]  # Forehead center
        left_jaw = landmarks_coords[5]  # Left jaw
        right_jaw = landmarks_coords[11]  # Right jaw
        nose_tip = landmarks_coords[30]  # Nose tip
        left_temporal = landmarks_coords[0]  # Left temporal
        right_temporal = landmarks_coords[16]  # Right temporal

        # Calculate key distances and ratios
        jaw_width = np.linalg.norm(left_jaw - right_jaw)
        cheek_width = np.linalg.norm(left_cheek - right_cheek)
        face_height = np.linalg.norm(forehead_center - chin)
        nose_to_chin = np.linalg.norm(nose_tip - chin)
        temporal_width = np.linalg.norm(left_temporal - right_temporal)

        # Calculate ratios for classification
        jaw_to_cheek_ratio = jaw_width / cheek_width
        height_to_width_ratio = face_height / cheek_width
        nose_to_face_height_ratio = nose_to_chin / face_height
        temporal_to_jaw_ratio = temporal_width / jaw_width

        # Print ratios for debugging
        print(f"jaw_to_cheek_ratio: {jaw_to_cheek_ratio}")
        print(f"height_to_width_ratio: {height_to_width_ratio}")
        print(f"nose_to_face_height_ratio: {nose_to_face_height_ratio}")
        print(f"temporal_to_jaw_ratio: {temporal_to_jaw_ratio}")

        # Classification based on ratios and distances
        if height_to_width_ratio < 1.15 and jaw_to_cheek_ratio > 0.7 and temporal_to_jaw_ratio > 0.45 and nose_to_face_height_ratio < 0.45:
         face_shape = 'Round'
        elif height_to_width_ratio > 1 and jaw_to_cheek_ratio < 0.7 and temporal_to_jaw_ratio < 0.45 and nose_to_face_height_ratio < 0.5:
         face_shape = 'Oval'
        elif height_to_width_ratio > 1 and jaw_to_cheek_ratio > 0.8 and temporal_to_jaw_ratio < 0.45 and nose_to_face_height_ratio > 0.5:
         face_shape = 'Square'
        elif height_to_width_ratio > 1 and jaw_to_cheek_ratio < 0.8 and temporal_to_jaw_ratio < 0.45 and nose_to_face_height_ratio < 0.55:
         face_shape = 'Rectangular'
        elif height_to_width_ratio > 1.3 and jaw_to_cheek_ratio < 0.8 and temporal_to_jaw_ratio < 0.45 and nose_to_face_height_ratio < 0.6:
         face_shape = 'Oblong'
        elif height_to_width_ratio > 1 and jaw_to_cheek_ratio > 0.8 and temporal_to_jaw_ratio > 0.45 and nose_to_face_height_ratio > 0.6:
         face_shape = 'Heart'
        else:
         face_shape = 'Unknown'

        print(face_shape)
else:
    print("No face or multiple faces detected.")
    
