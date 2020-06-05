# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.model_path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.model_path, "yolov3.weights"])
configPath = os.path.sep.join([config.model_path, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.use_gpu:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# testing to see what the average height is across all frames
all_ave_heights = []
ave_person_height = 5.5 # feet

# loop over the frames from the video stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# instead of using 50 pixels as the minimum distance,
	# let's instead use the average height of the people in the frame
	heights = []
	for i, result in enumerate(results):
		heights.append(results[i][3])
    # change it to maximum height
	ave_height_pixels = np.mean(heights)
	all_ave_heights.append(ave_height_pixels)

	# initialize the set of indexes that violate the minimum social
	# distance
	violate = set()

	# initalize the set of tuples for neighbors that violate minimum social 
	# distance
	neighbors = set()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the height of that person
				if (((D[i, j] < results[i][3]) or (D[i, j] < results[j][3])) and 
                (abs(results[i][1][2] - results[j][1][2]) < 30)):
					# update our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)

					# update the neighbors
					if (i,j) not in neighbors or (j, i) not in neighbors:
						neighbors.add((i, j))

	# loop over the results
	for (i, (prob, bbox, centroid, h)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then
		# update the color
		if i in violate:
			color = (0, 0, 255)

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)
		#text = str(i)
		#cv2.putText(frame, text, (cX, cY),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	# loop over the neighbors set
	# and draw a line from the bad neighbors
	for each_neighbor in neighbors:
		cX_0, cY_0 = results[each_neighbor[0]][2]
		cX_1, cY_1 = results[each_neighbor[1]][2]
		cv2.line(frame, (cX_0, cY_0), (cX_1, cY_1), (0, 0, 255), 2)

		# calculate the distance from the two points in pixels
		neighbor_dist = dist.cdist(np.array([cX_0, cY_0]).reshape(1, -1), 
			np.array([cX_1, cY_1]).reshape(1, -1), metric="euclidean")[0][0]
		# calculate the distance from the two points in feet
		neighbor_dist_feet = neighbor_dist * (ave_person_height/ave_height_pixels)
		location_for_distance_X = int((cX_0 + cX_1) / 2)
		location_for_distance_Y = int((cY_0 + cY_1) / 2)
		feet = int(neighbor_dist_feet)
		inches = int((neighbor_dist_feet - int(neighbor_dist_feet)) * 12)
		# put text of average distance
		text = "{}'{}\"".format(feet, inches)
		cv2.putText(frame, text, (location_for_distance_X, location_for_distance_Y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
		

	# draw the total number of social distancing violations on the
	# output frame
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	# check to see if the output frame should be displayed to our
	# screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)

#print('...Average height is {} pixels'.format(np.mean(all_ave_heights)))