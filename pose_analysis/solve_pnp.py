#solve pnp problem to estimate the pose of the camera relative to the object
import cv2
import numpy as np

# 3D model points.
import cv2
import numpy as np

# 3D model points.
import cv2
import numpy as np
import cv2
import numpy as np


class SolvePNP:
    def __init__(self,img_orig):
        self.body_pointes = np.array([  (0.0, 0.0, 0.0),  #left shoulder
                                        (60.0,0.0,0.0),  #right shoulder
                                        (55,102,0.0),
                                        (5,102,0.0)],dtype='float32')  #right side of waist 
        size = img_orig[1].shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        self.camera_matrix = np.array(  [[focal_length, 0, center[0]],
                                        [0, focal_length, center[1]],
                                        [0, 0, 1]], dtype = "float32")
        self.dist_coeffs = np.zeros((4,1), dtype="float32")
        self.projection_camera_height = 2000 #height of the camera from the ground (cm)
        self.projection_camera_distance = 100  #forward distance of the projection from the camera (cm) 
        self.roof_camera_translation = np.array([0,self.projection_camera_distance,self.projection_camera_height],dtype='float32') #create the translation vector to a projection image above the camera (xz opencv camera plane)
        self.roof_camera_rotation = np.array([np.pi/2,0,0],dtype='float32') #create the rotation vector to a projection image above the camera (xz opencv camera plane)

    def solve(self,keyPoints):
        image_points = np.array([keyPoints[6][:2], keyPoints[5][:2], keyPoints[11][:2],keyPoints[12][:2]], dtype="float32")

        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.body_pointes, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        return rotation_vector, translation_vector
    
    def pose_matrix(self,translation, rotation_matrix):
        if rotation_matrix.shape != (3, 3):
            rotation_matrix = cv2.Rodrigues(rotation_matrix)[0]
    # Create a 4x4 identity matrix
        pose = np.eye(4)
        # Fill in the translation part (top 3x1 corner)
        pose[:3, 3] = translation
        # Fill in the rotation part (top-left 3x3 corner)
        pose[:3, :3] = rotation_matrix 
        return pose
    
    def project_on_roof_camera(self,point_tvec):
        image_points = cv2.projectPoints(point_tvec, self.roof_camera_rotation, self.roof_camera_translation, self.camera_matrix, self.dist_coeffs)[0].reshape(-1, 2)
        return image_points
    
    def project_on_camera(self,point_tvec):                                 #camera angle is 30 degrees (0.523599 radians)
        image_points = cv2.projectPoints(point_tvec, np.eye(3), np.array([0,-0.523599,0],dtype='float32'), self.camera_matrix, self.dist_coeffs)[0].reshape(-1, 2)
        return image_points
