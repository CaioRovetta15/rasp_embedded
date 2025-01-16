from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=10, minDistance=50):
        # Initialize the CentroidTracker class
        self.nextObjectID = 0
        self.objects = OrderedDict()  # Stores object IDs and their centroids
        self.originRects = OrderedDict()  # Stores object IDs and their original rectangles
        self.disappeared = OrderedDict()  # Stores object IDs and the number of consecutive frames they've disappeared
        
        # Parameters for tracking
        self.maxDisappeared = maxDisappeared  # Maximum consecutive frames an object can disappear before deregistration
        self.minDistance = minDistance  # Minimum distance for associating a new centroid with an existing object
    
    def register(self, centroid, rect):
        # Register a new object with its centroid and original rectangle
        self.originRects[self.nextObjectID] = rect
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Deregister an object (remove it from tracking)
        del self.originRects[objectID]
        del self.objects[objectID]
        del self.disappeared[objectID]

    def get_id(self, rect):
        # Get the ID of an object based on the closest centroid to a given rectangle
        (x, y) = rect
        cX = x
        cY = y

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        
        D = dist.cdist(np.array(objectCentroids), [(cX, cY)])  # Calculate distance between centroids and the given point

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        objectID = None

        for (row, col) in zip(rows, cols):
            objectID = objectIDs[row]
            break
        return objectID

    def update(self, centroids):
        # Update the object tracking with new centroids
        
        if len(centroids) == 0:
            # Handle the case when no centroids are detected in the current frame
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.originRects
        
        inputCentroids = np.zeros((len(centroids), 2), dtype="int")
        for (i, center) in enumerate(centroids):
            cX = int(center[0])
            cY = int(center[1])
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            # If there are no existing tracked objects, register the new centroids as objects
            for i in range(0, len(inputCentroids)):
                centroid = inputCentroids[i]
                cent = centroids[i]
                self.register(centroid, cent)
        
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)  # Calculate distance matrix

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.originRects[objectID] = centroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                # Handle the case when there are more existing objects than new centroids
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                # Handle the case when there are more new centroids than existing objects
                for col in unusedCols:
                    centroid = inputCentroids[col]
                    rect = centroids[col]
                    self.register(centroid, rect)
                    
        return self.objects, self.originRects
