import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
from io import BytesIO



def generate_minimum_clusters(points, max_radius, min_points_per_cluster,plot_3d=False,plot_2d=False,x_lim=500,y_lim=1000,plot_size=(640,640)):
    # Create a KD-Tree from the points
    kdtree = cKDTree(points)

    clusters = []

    # Iterate through unassigned points
    unassigned_points = set(range(len(points)))

    while unassigned_points:
        cluster = []
        point_idx = unassigned_points.pop()

        # Start a new cluster with the current point
        cluster.append(point_idx)

        # Find nearby points within the maximum radius
        nearby_points = kdtree.query_ball_point(points[point_idx], max_radius)

        for nearby_idx in nearby_points:
            if nearby_idx in unassigned_points:
                cluster.append(nearby_idx)
                unassigned_points.remove(nearby_idx)

        # If the cluster meets the minimum size requirement, add it to the list of clusters
        if len(cluster) >= min_points_per_cluster:
            clusters.append(cluster)
    
    if plot_2d:
        # Create a white OpenCV image
        img = np.ones((plot_size[1], plot_size[0], 3), np.uint8) * 255

        # Generate a list of unique colors for clusters
        cluster_colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in clusters]

        for i, point in enumerate(points):
            # Convert 3D point to image coordinates
            image_point = np.array([int(plot_size[0] / 2 + (point[0] / x_lim) * plot_size[0]),
                                    int((point[2] / y_lim) * plot_size[1])], dtype="int32")
            cv2.circle(img, (image_point[0], image_point[1]), 6, (0, 0, 255), -1)

        for i, cluster in enumerate(clusters):
            cluster_points = np.array([points[idx] for idx in cluster])
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_image_center = np.array([int(plot_size[0] / 2 + (cluster_center[0] / x_lim) * plot_size[0]),
                                            int((cluster_center[2] / y_lim) * plot_size[1])], dtype="int32")
            cluster_image_radius = int(max_radius / x_lim * plot_size[0])

            # Draw a circle representing the cluster with a unique color
            cv2.circle(img, (cluster_image_center[0], cluster_image_center[1]), cluster_image_radius,cluster_colors[i], 3)

        # Flip the image 180 degrees
        img = cv2.rotate(img, cv2.ROTATE_180)

        # Display the image
        cv2.imshow('Clusters', img)
        cv2.waitKey(1)

    return clusters

if __name__ == '__main__':
    # Example usage:
    points = np.random.rand(10, 3)  # Replace with your 3D point data
    print(points.shape)
    max_radius = 0.2  # Adjust the maximum radius as needed
    min_points_per_cluster = 1  # Adjust the minimum points per cluster as needed
    clusters = generate_minimum_clusters(points, max_radius, min_points_per_cluster)
    print(clusters)

    # Each cluster in 'clusters' contains a list of point indices that belong to that cluster.
    # Plotting the points and clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Points')

    # Plot clusters as translucent spheres
    for i, cluster in enumerate(clusters):
        cluster_points = np.array([points[idx] for idx in cluster])
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_radius = max_radius

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = cluster_radius * np.outer(np.cos(u), np.sin(v)) + cluster_center[0]
        y = cluster_radius * np.outer(np.sin(u), np.sin(v)) + cluster_center[1]
        z = cluster_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + cluster_center[2]

        ax.plot_surface(x, y, z, color=f'C{i}', alpha=0.3, label=f'Cluster {i+1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.legend()
    plt.show()