import cv2
import matplotlib.pyplot as plt
import os

def capture_image():
    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open video device.")
        return None
    
    print("Capturing image. Look at the camera.")
    ret, frame = video_capture.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        video_capture.release()
        return None
    
    # Release the webcam
    video_capture.release()
    
    return frame

def detect_faces(image):
    if image is None:
        print("Error: No image to detect faces.")
        return []
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    face_locations = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return face_locations

def display_detected_faces(image, face_locations, save_path):
    if image is None:
        print("Error: No image to display faces.")
        return
    
    # Loop over each face found in the image
    for (x, y, w, h) in face_locations:
        # Draw a box around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Convert BGR image to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the resulting image
    plt.imshow(rgb_image)
    plt.axis('off')  # Hide axes
    plt.title('Face Detection')
    plt.show()
    
    # Save the resulting image
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")

def main():
    # Capture an image from the webcam
    image = capture_image()
    
    if image is None:
        print("No image captured.")
        return
    
    # Detect faces in the image
    face_locations = detect_faces(image)
    
    if len(face_locations) > 0:
        print(f"Found {len(face_locations)} face(s) in the image.")
        # Define the path to save the output image
        save_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'cgmini.py')
        # Display the image with detected faces and save it to the desktop
        display_detected_faces(image, face_locations, save_path)
    else:
        print("No faces found in the image.")

if __name__ == "__main__":
    main()
