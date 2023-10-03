# Import necessary libraries
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Import custom modules
from plot import Annotator
from solve_pnp import SolvePNP
from min_clusters import generate_minimum_clusters
from centroidtracker import CentroidTracker

# Initialize the YOLO model for object detection
model = YOLO("yolov8n-pose.onnx")

# Open a video capture device (0 for default camera, or specify a video file)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Initialize instances of custom classes
annot = Annotator(cap.read())  # Used for annotating images
solver = SolvePNP(cap.read())    # Used for solving PnP pose estimation
person_ct = CentroidTracker()   # Centroid tracking for people
cluster_ct = CentroidTracker()  # Centroid tracking for clusters of people

# Initialize the radius for talking clusters
talk_radius = 200  # in centimeters

# Create a window to display the video feed
cv2.namedWindow('output')

interest_zone = np.array([[200, 0,0], [200,0,200], [0, 0,200], [0, 0,0]], dtype='float32')
zone_translation = np.array([-100,120,400],dtype='float32')
interest_zone = interest_zone + zone_translation

# Main loop for processing video frames
while cap.isOpened():
    ret, img_orig = cap.read()
    if not ret:
        break

    # Create a blank image for additional annotations
    blank_image = np.ones((img_orig.shape[0], img_orig.shape[1], 3), np.uint8) * 160

    start_time = time.time()

    # Perform object detection using YOLO model
    results = model(img_orig, save_txt=False, stream=True, verbose=False, imgsz=(224, 224))

    # Process keypoints detected by the YOLO model
    for r in results:
        kpoints = r.keypoints.data

    detection_dict = {}
    people_on_image = []

    # Process keypoints and perform pose estimation for each person
    for i, keyPoints in enumerate(kpoints):
        left_wrist = keyPoints[10]
        right_wrist = keyPoints[9]
        rvec, tvec = solver.solve(keyPoints.numpy())

        # Annotate the image with keypoints and pose estimation
        img_orig = annot.kpts(img_orig, keyPoints)

        detection_dict[i] = {'rvec': rvec, 'tvec': tvec}
        image_points = solver.project_on_roof_camera(tvec)[0]
        people_on_image.append(image_points)

    # Update centroid tracking for individual people
    person_points = person_ct.update(people_on_image)

    # Annotate the image with tracked people
    for i, (objectID, centroid) in enumerate(person_points[0].items()):
        cv2.circle(blank_image, (int(centroid[0]), int(centroid[1])), annot.lw, (0, 0, 139), -1)
        cv2.putText(blank_image, str(objectID), (int(centroid[0]), int(centroid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 139), 2)

    # Extract and process translation vectors for clustering
    tvec_list = [detection_dict[i]['tvec'] for i in range(len(detection_dict))]
    tvec_array = np.array(tvec_list).reshape(-1, 3)

    # Generate minimum clusters of people based on their positions
    clusters = generate_minimum_clusters(tvec_array, max_radius=talk_radius, min_points_per_cluster=1, plot_2d=False)

    # Annotate the image with cluster information
    for i, cluster in enumerate(clusters):
        cluster_points = np.array([tvec_list[idx] for idx in cluster])
        cluster_center = np.mean(cluster_points, axis=0)

        radius_distance = np.array([0, 0, talk_radius], dtype='float32').reshape(3, 1)
        cluster_radius = cluster_center + radius_distance
        cluster_radius_on_image = solver.project_on_roof_camera(cluster_radius)[0]

        cluster_center_on_image = solver.project_on_roof_camera(cluster_center)[0]
        distance = np.linalg.norm(cluster_radius_on_image - cluster_center_on_image)

        cv2.circle(blank_image, (int(cluster_center_on_image[0]), int(cluster_center_on_image[1])),
                   int(distance), (14, 57, 43), annot.lw)

        if len(cluster_points) >= 2:
            cv2.putText(blank_image, 'Roda de Conversa',
                        (int(cluster_center_on_image[0] + distance / 2), int(cluster_center_on_image[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 0, 0), 1)
        else:
            cv2.putText(blank_image, 'Pessoa', (int(cluster_center_on_image[0] + distance / 2),
                                                int(cluster_center_on_image[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (139, 0, 0), 1)
    interest_zone_on_image = solver.project_on_roof_camera(interest_zone)
    print(interest_zone_on_image)

    # interest_zone_on_image = interest_zone_on_image.reshape(-1, 2)
    cv2.polylines(blank_image, np.int32([interest_zone_on_image]), True, (0, 0, 0), 2)
    #project on original camera
    interest_zone_on_image = solver.project_on_camera(interest_zone)

    cv2.polylines(img_orig, np.int32([interest_zone_on_image]), True, (0, 0, 0), 2)
    # Display the annotated images
    cv2.imshow('output', img_orig)
    cv2.imshow('blank_image', blank_image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
