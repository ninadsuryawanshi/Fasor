import cv2
import numpy as np
import json

# Load the face landmark model
face_model = cv2.face.createFacemarkLBF()
face_model.loadModel('lbfmodel.yaml')  # Make sure to load the pre-trained model

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
input_image_path = 'ranbirface3.jpeg'
img = cv2.imread(input_image_path)

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

        # Draw landmarks on the image
        for (lx, ly) in landmarks_coords:
            cv2.circle(img, (int(lx), int(ly)), 1, (0, 255, 0), -1)

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

        # Print ratios for debugging
        print(f"jaw_to_cheek_ratio: {jaw_width / cheek_width}")
        print(f"height_to_width_ratio: {face_height / cheek_width}")
        print(f"nose_to_face_height_ratio: {nose_to_chin / face_height}")
        print(f"temporal_to_jaw_ratio: {temporal_width / jaw_width}")

        
        if face_height / cheek_width < 1.15 and jaw_width / cheek_width > 0.7 and temporal_width / jaw_width > 0.45 and nose_to_chin / face_height < 0.45:
            face_shape = 'round'
        elif face_height / cheek_width > 1 and jaw_width / cheek_width < 0.7 and temporal_width / jaw_width < 0.45 and nose_to_chin / face_height < 0.5:
            face_shape = 'oval'
        elif face_height / cheek_width > 1 and jaw_width / cheek_width > 0.8 and temporal_width / jaw_width < 0.45 and nose_to_chin / face_height > 0.5:
            face_shape = 'Square'
        elif face_height / cheek_width > 1 and jaw_width / cheek_width < 0.8 and temporal_width / jaw_width < 0.45 and nose_to_chin / face_height < 0.55:
            face_shape = 'Rectangular'
        elif face_height / cheek_width > 1.3 and jaw_width / cheek_width < 0.8 and temporal_width / jaw_width < 0.45 and nose_to_chin / face_height < 0.6:
            face_shape = 'Oblong'
        elif face_height / cheek_width > 1 and jaw_width / cheek_width > 0.8 and temporal_width / jaw_width > 0.45 and nose_to_chin / face_height > 0.6:
            face_shape = 'Heart'
        else:
            face_shape = 'oval'

        # Display face shape on the image
        cv2.putText(img, face_shape, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Show the input image with landmarks and face shape
        cv2.imshow('Detected Face with Landmarks', img)

        # Load image collection based on face shape
        with open('image_collection.json') as f:
            face_shape_to_images = json.load(f)

        images = face_shape_to_images.get(face_shape.lower())

        if images:
    # Set a fixed size for the images
          img_width, img_height = 600, 600

    # Display each image in a separate window
          for idx, image_path in enumerate(images):
              image = cv2.imread(image_path)
              if image is not None:
              # Resize the image to the fixed size
               image = cv2.resize(image, (img_width, img_height))
               cv2.imshow(f"Image {idx}", image)
              else:
               print("No images found for face shape:", face_shape)

cv2.waitKey(0)  # wait for key press to close windows
cv2.destroyAllWindows()
