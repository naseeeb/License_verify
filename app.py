from flask import Flask, render_template, request, jsonify
import cv2
import pytesseract
import face_recognition
import os

app = Flask(__name__, template_folder='.')

UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to extract text from an image using Tesseract OCR
def extract_text(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return None
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform text extraction using OCR (Tesseract)
    text = pytesseract.image_to_string(gray)
    
    return text

# Function to detect faces and extract facial embeddings
def extract_face_embeddings(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return None
    
    # Convert image to RGB (face_recognition uses RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find all face locations in the image
    face_locations = face_recognition.face_locations(rgb_image)
    
    # If no face is found, return None
    if not face_locations:
        return None
    
    # Extract facial embeddings for each face found
    face_embeddings = face_recognition.face_encodings(rgb_image, face_locations)
    
    return face_embeddings

# Function to match faces using facial embeddings
def match_faces(person_image, license_image):
    # Extract facial embeddings from person's photo and license photo
    person_embeddings = extract_face_embeddings(person_image)
    license_embeddings = extract_face_embeddings(license_image)
    
    # If no face is detected in either image, return False
    if person_embeddings is None or license_embeddings is None:
        return False
    
    # Calculate face distance (lower distance means better match)
    face_distance = face_recognition.face_distance(person_embeddings, license_embeddings[0])
    
    # Set a threshold for matching
    threshold = 0.6
    
    # If face distance is below threshold, consider it a match
    if face_distance < threshold:
        return 'Face Matched with the person'
    else:
        return 'Face not matched'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    # Get file paths from form data
    person_photo = request.files['person_photo']
    license_photo = request.files['license_photo']
    
    # Save uploaded images temporarily
    person_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], 'person_photo.jpg')
    license_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], 'license_photo.jpg')
    person_photo.save(person_photo_path)
    license_photo.save(license_photo_path)
    
    # Extract text from license photo
    license_text = extract_text(license_photo_path)
    
    # Match faces in person and license photos
    face_match_result = match_faces(person_photo_path, license_photo_path)
    
    return jsonify({'license_text': license_text, 'face_match_result': face_match_result})

if __name__ == '__main__':
    app.run(debug=True)
