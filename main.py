from ultralytics import YOLO
import cv2
import numpy as np
import time
from plot import Annotator
from solve_pnp import SolvePNP
from min_clusters import generate_minimum_clusters
from centroidtracker import CentroidTracker
model = YOLO("yolov8n-pose.onnx")

# cap = cv2.VideoCapture('output_video.mp4')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
annot = Annotator(cap.read())
solver = SolvePNP(cap.read())
person_ct = CentroidTracker()
cluster_ct = CentroidTracker()

# Initialize the slider position

# Create a callback function for the trackbar
talk_radius = 200 #cm
# Create a window to display the video feed
cv2.namedWindow('output')

# Create a trackbar to control the slider's position

while cap.isOpened():
    ret, img_orig = cap.read()
    if not ret:
        break
    blank_image = np.ones((img_orig.shape[0],img_orig.shape[1],3), np.uint8)*160

    start_time = time.time()
    results = model(img_orig, save_txt=False, stream=True, verbose=False, imgsz=(224, 224))
    
    for r in results:
        kpoints = r.keypoints.data
    detection_dict={}
    person_on_image = []
    for i,keyPoints in enumerate(kpoints):
        # print(keyPoints)
        #circle the left and right wrist
        left_wrist = keyPoints[10]
        right_wrist = keyPoints[9]
        rvec,tvec=solver.solve(keyPoints.numpy())
        img_orig = annot.kpts(img_orig, keyPoints)
        # cv2.circle(img_orig, (int(keyPoints[10][0]), int(keyPoints[10][1])), annot.lw, (0, 0, 255), -1)
        # cv2.circle(img_orig, (int(keyPoints[9][0]), int(keyPoints[9][1])), annot.lw, (0, 255, 0), -1)
        #plot on the image the rotation vector and translation vector
        detection_dict[i]={'rvec':rvec,'tvec':tvec}
        #project the points into a virtual camera above the real camera to see the 3d points in a 2d image
        image_points = solver.project_on_roof_camera(tvec)
        person_on_image.append(image_points)
        # print(image_points)
        # image_points = person_ct.update(image_points)
        # print(image_points)
        # cv2.circle(blank_image, (int(image_points[0]), int(image_points[1])), annot.lw, (0, 0, 139), -1)
        
        # for i in range(3):
        #     cv2.putText(img_orig, str(round(tvec[i][0],2)), (10, 30+30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 139), 2)
    person_points=person_ct.update(person_on_image)
    #loop through the person points ordereddict and plot the points on the image
    for i, (objectID, centroid) in enumerate(person_points[0].items()):
        # print(objectID)
        # print(centroid)
        cv2.circle(blank_image, (int(centroid[0]), int(centroid[1])), annot.lw, (0, 0, 139), -1)
        cv2.putText(blank_image, str(objectID), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 139), 2)
    # print(detection_dict)
    tvec_list = [detection_dict[i]['tvec'] for i in range(len(detection_dict))]

    # Convert the list of arrays into a numpy array
    tvec_array = np.array(tvec_list).reshape(-1, 3)

    # Generate clusters
    clusters = generate_minimum_clusters(tvec_array, max_radius=talk_radius, min_points_per_cluster=1,plot_2d=False) #max_radius=50 (cm), min_points_per_cluster=1
    
    for i, cluster in enumerate(clusters):
        cluster_points = np.array([tvec_list[idx] for idx in cluster])
        cluster_center = np.mean(cluster_points, axis=0)

        radius_distance = np.array([0,0,talk_radius],dtype='float32').reshape(3, 1)
        cluster_radius = cluster_center + radius_distance
        cluster_radius_on_image = solver.project_on_roof_camera(cluster_radius)

        cluster_center_on_image = solver.project_on_roof_camera(cluster_center)
        distance = np.linalg.norm(cluster_radius_on_image-cluster_center_on_image)#compute the distance between the center and the radius in pixels
        cv2.circle(blank_image, (int(cluster_center_on_image[0]), int(cluster_center_on_image[1])), int(distance), (14, 57, 43),annot.lw)
        if len(cluster_points)>=2:
            cv2.putText(blank_image, 'Roda de Conversa', (int(cluster_center_on_image[0]+distance/2), int(cluster_center_on_image[1])), cv2.FONT_HERSHEY_SIMPLEX,0.6, (139, 0, 0), 1)
        else:
            cv2.putText(blank_image, 'Pessoa', (int(cluster_center_on_image[0]+distance/2), int(cluster_center_on_image[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 0, 0), 1)
    # Display the image
    cv2.imshow('output', img_orig)
    #flip the image to be easier to understand (mirror image)
    # blank_image = cv2.flip(blank_image, 1)
    cv2.imshow('blank_image', blank_image)

    # print("FPS: ", 1.0 / (time.time() - start_time))
    # print("ms ", (time.time() - start_time))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

