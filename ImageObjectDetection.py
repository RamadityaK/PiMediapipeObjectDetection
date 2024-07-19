import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
#from utils import visualize

IMAGE_FILE = 'pigImage.jpg'

base_options = python.BaseOptions(model_asset_path = 'test.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

image = mp.Image.create_from_file(IMAGE_FILE)

detection_result = detector.detect(image)
print(detection_result)
print('-----------------------')
for detection in detection_result.detections:
    category = detection.categories[0]
    bbox = detection.bounding_box
    print(category.category_name, ", with prob: ", str(round(category.score,2))
          , "and centroid: ", str(bbox.origin_x+(bbox.width/2)),",",str(bbox.origin_y+(bbox.height/2)))

image_copy = np.copy(image.numpy_view())
# annotated_image = visualize(image_copy,detection_result)
# rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
# cv2.imshow(rgb_annotated_image)

cv2.destroyAllWindows()
