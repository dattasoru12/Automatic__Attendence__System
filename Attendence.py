import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Absolute path to the directory containing known faces
KNOWN_FACES_DIR = 'known_people'
# Path to the CSV file to log attendance
ATTENDANCE_FILE = 'Attendence.csv'

# Initialize lists for known face encodings and names
known_face_encodings = []
known_face_names = []


# Load known faces and their encodings
def load_known_faces():
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Error: The directory '{KNOWN_FACES_DIR}' does not exist.")
        return

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            encodings = face_recognition.face_encodings(image_rgb)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
    if not known_face_encodings:
        print("No faces found in the known faces directory.")


# Function to log attendance with error handling
def mark_attendance(name):
    try:
        with open(ATTENDANCE_FILE, 'a+') as f:
            f.seek(0)
            data = f.readlines()
            name_list = [line.split(',')[0] for line in data]
            if name not in name_list:
                now = datetime.now()
                dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
                f.writelines(f'{name},{dt_string}\n')
                print(f'Attendance marked for {name}')
            elif not data:  # If file is empty, write header
                f.write('Name,Time\n')
    except FileNotFoundError:
        print(f"Error: The file '{ATTENDANCE_FILE}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied to access '{ATTENDANCE_FILE}'. Please check file permissions.")
    except Exception as e:
        print(f"An error occurred while marking attendance: {e}")


# Function to process each frame
def process_frame(frame):
    frame_resized = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            color = (0, 255, 0)  # Green for known face
            mark_attendance(name)
        else:
            name = 'Unknown'
            color = (0, 0, 255)  # Red for unknown face

        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def main():
    load_known_faces()
    if not known_face_encodings:
        print("Exiting due to no known faces loaded.")
        return

    print('Known faces loaded and encoded')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image from webcam.")
            break

        processed_frame = process_frame(frame)
        cv2.imshow('Webcam', processed_frame)

        if cv2.waitKey(10) == 13:  # Enter key to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
