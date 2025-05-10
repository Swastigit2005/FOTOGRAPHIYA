import cv2
import numpy as np
import dlib
import os


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def compute_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2]) / 255.0

def compute_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray) / 255.0

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return min(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0, 1.0)

def estimate_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.var(gray - cv2.medianBlur(gray, 3)) / 255.0

def check_white_balance(image):
    b, g, r = cv2.mean(image)[:3]
    balance = 1 - abs(r - g) / 255.0
    return np.clip(balance, 0, 1)

def check_eyes_open_and_smile(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    eyes_score, smile_score, gaze_score = 0, 0, 0

    if len(faces) == 0:
        return eyes_score, smile_score, gaze_score

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        if landmarks.shape[0] < 68:
            continue

        # Eyes landmarks (left: 36-41, right: 42-47)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])
        right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[4])
        eye_open_ratio = (left_eye_height + right_eye_height) / 2.0

        eyes_score = min(eye_open_ratio / 10.0, 1.0)  # normalize

        # Smile landmarks (mouth: 48-67)
        mouth = landmarks[48:68]
        smile_width = np.linalg.norm(mouth[0] - mouth[6])
        smile_height = np.linalg.norm(mouth[3] - mouth[9])
        smile_score = min((smile_height / smile_width) * 2.0, 1.0)

        eye_center_left = np.mean(left_eye, axis=0)
        eye_center_right = np.mean(right_eye, axis=0)
        face_center = np.mean(landmarks[0:27], axis=0)
        gaze_score = 1.0 - (abs(eye_center_left[0] - face_center[0]) + abs(eye_center_right[0] - face_center[0])) / (2 * face.width())

    return eyes_score, smile_score, gaze_score

def estimate_depth_placeholder(image):
    # TODO: Replace with real monocular depth estimator (MiDaS)
    return 0.5  # dummy normalized value

def compute_aesthetic_placeholder(image):
    # TODO: Replace with pretrained NIMA or aesthetic model
    return 0.5  # dummy normalized value

input_folder = 'wedding_photos'
output_folder = 'best_photos'
os.makedirs(output_folder, exist_ok=True)

# to clear output folder
for file in os.listdir(output_folder):
    file_path = os.path.join(output_folder, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

image_scores = {}

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        continue

    # calculating parameters
    brightness = compute_brightness(image)
    contrast = compute_contrast(image)
    sharpness = compute_sharpness(image)
    noise = 1 - estimate_noise(image)
    white_balance = check_white_balance(image)
    eyes_open, smile, gaze = check_eyes_open_and_smile(image)
    depth = estimate_depth_placeholder(image)
    aesthetic = compute_aesthetic_placeholder(image)

    # Skip images where eyes are not sufficiently open
    if eyes_open < 0.7:  
        continue

    #final score calculation
    score = (0.2 * depth + 0.1 * sharpness + 0.05 * brightness + 0.05 * contrast +
             0.05 * noise + 0.05 * white_balance + 0.1 * eyes_open + 0.1 * smile +
             0.05 * gaze + 0.15 * aesthetic)

    image_scores[image_path] = score

# Sort images by score
top_images = sorted(image_scores, key=image_scores.get, reverse=True)[:20]

# Copy to output folder
for path in top_images:
    filename = os.path.basename(path)
    cv2.imwrite(os.path.join(output_folder, filename), cv2.imread(path))

print("✅ Top 20 wedding photos saved to:", output_folder)
