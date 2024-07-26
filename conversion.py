import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time




def create_directory_structure(base_path):
    os.makedirs(os.path.join(base_path, 'intrinsic'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'extrinsic'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'pose'), exist_ok=True)

def write_matrix_to_file(matrix, file_path):
    with open(file_path, 'w') as f:
        for row in matrix:
            f.write(' '.join(f"{val:.6f}" for val in row) + '\n')

def compute_affine_transformation(position, rotation):
    # Convert quaternion to rotation matrix
    rot = R.from_quat(rotation).as_matrix()
    
    # Create affine transformation matrix
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rot
    affine_matrix[:3, 3] = position
    
    return affine_matrix

def conversion_function(position, rotation, principalX, principalY, focalX, focalY, index, base_path):
    create_directory_structure(base_path)
    
    # Intrinsic matrices
    intrinsic_matrix = np.array([
        [focalX, 0, principalX, 0],
        [0, focalY, principalY, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    write_matrix_to_file(intrinsic_matrix, os.path.join(base_path, 'intrinsic', 'intrinsic_color.txt'))
    write_matrix_to_file(intrinsic_matrix, os.path.join(base_path, 'intrinsic', 'intrinsic_depth.txt'))
    
    # Extrinsic matrices (identity matrix in this case)
    extrinsic_matrix = np.eye(4)
    
    write_matrix_to_file(extrinsic_matrix, os.path.join(base_path, 'extrinsic', 'extrinsic_color.txt'))
    write_matrix_to_file(extrinsic_matrix, os.path.join(base_path, 'extrinsic', 'extrinsic_depth.txt'))
    
    # Affine transformation matrix for the given index
    affine_matrix = compute_affine_transformation(position, rotation)
    pose_file_path = os.path.join(base_path, 'pose', f"{index:04d}.txt")
    write_matrix_to_file(affine_matrix, pose_file_path)

    # copy image
    image_path = os.path.join(image_folder, f"{index:05d}.jpg")
    shutil.copy(image_path, os.path.join(base_path, 'color', f"{index:04d}.jpg"))




class CSVHandler(FileSystemEventHandler):
    def __init__(self, file_path, base_path):
        self.file_path = file_path
        self.base_path = base_path
        self.processed_lines = 0

    def on_modified(self, event):
        if event.src_path == self.file_path:
            self.process_new_lines()

    def process_new_lines(self):
        print("Processing poses")
        with open(self.file_path, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            new_lines = lines[self.processed_lines:]
            self.processed_lines = len(lines)

            for line in new_lines:
                print("Line: ",line)
                frame_id = int(line[0])
                timestamp = float(line[1])
                position = [float(line[2]), float(line[3]), float(line[4])]
                rotation = [float(line[5]), float(line[6]), float(line[7]), float(line[8])]
                principalX = float(line[9])
                principalY = float(line[10])
                focalX = float(line[11])
                focalY = float(line[12])
                skew = float(line[13])  # Assuming skew is not used in conversion_function

                conversion_function(position, rotation, principalX, principalY, focalX, focalY, frame_id, self.base_path)

if __name__ == "__main__":
    base_path = '/capture/rtg_slam'
    csv_file_path = '/capture/FullPoses.csv'
    image_folder = '/capture/images'

    event_handler = CSVHandler(csv_file_path, base_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(csv_file_path), recursive=False)
    observer.start()
    event_handler.process_new_lines()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()