import cv2
import json

with open('image_collection.json') as f:
    face_shape_to_images = json.load(f)

face_shape = "round"  # output from shapedet

images = face_shape_to_images.get(face_shape)

if images:
    # Create a window to display the images
    cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Images", 800, 600)  # set the window size

    for image in images:
        img = cv2.imread(image)
        # Resize the image to fit the window
        img = cv2.resize(img, (400, 300))  # adjust the size as needed
        cv2.imshow("Images", img)
        cv2.waitKey(1000)  # wait for 1 second before displaying the next image
    cv2.destroyAllWindows()
else:
    print("No images found for face shape:", face_shape)
    
    
    
    # import cv2
# import json

# with open('image_collection.json') as f:
#     face_shape_to_images = json.load(f)

# face_shape = "round"  # output from shapedet

# images = face_shape_to_images.get(face_shape)

# if images:
#     # display the images
#     for image in images:
#         img = cv2.imread(image)
#         cv2.imshow("Image", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
# else:
#     print("No images found for face shape:", face_shape)