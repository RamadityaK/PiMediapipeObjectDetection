import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time

#Variables to benchmark inference performance
startTime = 0
inferenceTime = 0

# You can import any jpg image and change this filename to test it out!
IMAGE_FILE = 'image.jpg'

#Configure the Mediapipe Object to perform inferencing
base_options = python.BaseOptions(model_asset_path = 'efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

#Process the image into a format that Mediapipe likes
image = mp.Image.create_from_file(IMAGE_FILE)

#Run the inferencing
startTime = time.time()
detection_result = detector.detect(image)
inferenceTime = time.time() - startTime

#You can ignore the below code, it's mainly used for debugging
#print(detection_result)
#print('-----------------------')

#Parse through each object detected (ironically, detection is also an object, see the Mediapipe Documentation for more information)
for detection in detection_result.detections:
    category = detection.categories[0] #This line grabs the category of the detection with the highest probablity (This is an object)
    bbox = detection.bounding_box # This line grabs the bounding box of the detection (This is an object)
    #Display the results of the detection and misc. info.
    print(category.category_name, ", with prob: ", str(round(category.score,2))
          , "and centroid: ", str(bbox.origin_x+(bbox.width/2)),",",str(bbox.origin_y+(bbox.height/2)))

#Print the time it takes to perform inference on the image for benchmarking
print('Inference Time: ',inferenceTime)