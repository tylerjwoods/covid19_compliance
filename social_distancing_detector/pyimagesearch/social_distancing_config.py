# base model to YOLO directory
model_path = "yolo-coco"

# initialize minimum prob to filter weak detections along with
# the threshold when applying non-maxima suppression
min_conf = 0.3
nms_thresh = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
use_gpu = False

# define the minimum safe distance (in pixels) that two people can be
# from each other
min_distance = 50