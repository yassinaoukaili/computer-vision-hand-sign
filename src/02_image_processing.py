import os
import cv2
import pickle
from pathlib import Path, PosixPath
import mediapipe as mp


def process_hand_landmarks(samples_dir: PosixPath,
                           output_file: str = 'data.pkl',
                           static_image_mode: bool = True,
                           min_detection_confidence: float = 0.3):
    # Initialize MediaPipe hands solution
    mp_hands = mp.solutions.hands

    # Create a hands detector object with the provided configuration
    hands = mp_hands.Hands(static_image_mode=static_image_mode, min_detection_confidence=min_detection_confidence)

    samples_dir = Path(samples_dir)
    if not samples_dir.exists() or not samples_dir.is_dir():
        raise ValueError(f"The provided directory path {samples_dir} is invalid or does not exist.")

    data = []
    labels = []

    for sub_dir in os.listdir(samples_dir):
        sub_dir_path = samples_dir / sub_dir
        if not os.path.isdir(sub_dir_path):
            continue

        for img_path in os.listdir(sub_dir_path):
            img_path_full = sub_dir_path / img_path
            if not img_path_full.is_file():
                continue

            try:
                # Read the image file
                img = cv2.imread(str(img_path_full))
                # Ensure the image is read correctly
                if img is None:
                    raise ValueError(f"Unable to read image {img_path_full}. Skipping this file.")

                # Convert the image from BGR (OpenCV default) to RGB (MediaPipe requirement)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Process the image using MediaPipe's hand detection
                results = hands.process(img_rgb)

                data_aux = []
                x_ = []  # To store x-coordinates of landmarks
                y_ = []  # To store y-coordinates of landmarks

                # If hand landmarks are detected
                if multi_hl := results.multi_hand_landmarks:
                    # Iterate over each detected hand
                    for hand_landmarks in multi_hl:
                        # Store coordinates
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            y_.append(y)

                        # Normalize the coordinates
                        # Ensures all landmarks are in a consistent range relative to each hand
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    data.append(data_aux)
                    labels.append(sub_dir)

            except Exception as e:
                print(f"Error processing image {img_path_full}: {e}")
                continue

    output_path = Path.cwd().parent / output_file
    try:
        with open(output_path, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")


if __name__ == '__main__':
    samples_dir = Path.cwd().parent / 'data'
    process_hand_landmarks(samples_dir=samples_dir)
