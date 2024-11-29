import cv2

def capture_image():
    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)
    
    print("Capturing image. Look at the camera.")
    ret, frame = video_capture.read()
    
    # Release the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
    return frame

def detect_faces(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    face_locations = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return face_locations

def display_detected_faces(image, face_locations):
    # Loop over each face found in the image
    for (x, y, w, h) in face_locations:
        # Draw a box around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the resulting image
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Capture an image from the webcam
    image = capture_image()
    
    # Detect faces in the image
    face_locations = detect_faces(image)
    
    if len(face_locations) > 0:
        print(f"Found {len(face_locations)} face(s) in the image.")
        # Display the image with detected faces
        display_detected_faces(image, face_locations)
    else:
        print("No faces found in the image.")

if __name__ == "__main__":
    main()
