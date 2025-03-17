import os
import cv2
import time
import logging
from pathlib import Path, PosixPath


def create_samples_directory():
    samples_dir = Path(Path.cwd()).parent / 'data'
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    return samples_dir


def create_samples(number_of_classes: int, dataset_size: int, sample_dir: PosixPath, logger: logging.Logger):
    if not isinstance(number_of_classes, int) or not isinstance(dataset_size, int):
        logger.error("number_of_classes and dataset_size should be integers.")
        return

    if number_of_classes <= 0 or dataset_size <= 0:
        logger.error("number_of_classes and dataset_size must be greater than 0.")
        return

    if not sample_dir.exists():
        logger.error(f"Sample directory {sample_dir} does not exist.")
        return

    if not sample_dir.is_dir():
        logger.error(f"{sample_dir} is not a directory.")
        return

    logger.info(f"Starting data collection with {number_of_classes} classes and {dataset_size} samples per class.")

    # Initialize video capture (webcam 1)
    cap = cv2.VideoCapture(1)

    # Check if the webcam is accessible
    if not cap.isOpened():
        logger.error("Error: Could not access the webcam.")
        return

    try:
        # Loop over each class to collect samples
        for j in range(number_of_classes):
            class_dir = sample_dir / str(j)
            if not class_dir.exists():
                logger.info(f"Creating directory for class {j}.")
                class_dir.mkdir(parents=True)

            logger.info(f"Collecting data for class {j}")

            # Display a message to the user and wait for 'S' to begin capturing
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Failed to capture image for class {j}.")
                    break
                cv2.putText(frame, 'Press "S" to start', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0),
                            3, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            logger.info(f"Starting sample collection for class {j}.")
            counter = 0
            start_time = time.time()

            # Capture the required number of samples for this class
            while counter < dataset_size:
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Failed to capture image for class {j} at sample {counter}.")
                    break

                # Save the captured frame as an image in the class directory
                sample_path = class_dir / f"{counter}.jpg"
                cv2.imwrite(str(sample_path), frame)

                # Display the captured frame to the user
                cv2.imshow('frame', frame)
                cv2.waitKey(25)  # Wait before capturing the next image

                counter += 1

                # Log progress every 10 images
                if counter % 10 == 0:
                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"Class {j}: Collected {counter}/{dataset_size} samples. Time elapsed: {elapsed_time:.2f}s")

            logger.info(f"Finished collecting {counter} samples for class {j}.")

    finally:
        # Ensure that the webcam is released and all OpenCV windows are destroyed at the end of the process
        cap.release()
        cv2.destroyAllWindows()

    logger.info(f"Data collection complete for all classes.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    samples_dir = create_samples_directory()
    create_samples(number_of_classes=3, dataset_size=100, sample_dir=samples_dir, logger=logger)
