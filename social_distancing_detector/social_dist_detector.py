'''
This script uses an object detector from YOLOv3
to find people in frames in a video and creates
a bounding box around those people.

Then using some references, determines whether or not
people are adhering to social distant protocols.

This script is best run when the view from the camera
is from about 45 degrees.

To use this function, run
python social_dist_detector.py --input pedestrians.mp4 --output output.avi
'''

# import packages
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

def draw_line_print_distances(frame, results, neighbors, 
                                frame_number, neighbor_distances,
                                ave_height_pixels):
    '''
    input
    ------
    frame: frame of video
    results: detections from the frame
    neighbors: indexes of neighbors who are too close
    frame_number: frame number (0, 1, 2, ...etc)
    neighbor_distances: list of lists containing the text and location for the text

    returns
    -----
    neighbor_distances
    '''
    # define average person height
    ave_person_height = 5.6 # feet

    # initialize a number to reset the text
    frame_reset = 10

    # if the frame number is divisible by frame_reset, reset the neighbor distances
    if frame_number % frame_reset == 0:
        neighbor_distances = list()

    # loop over the neighbors set
    # and draw a line from the bad neighbors
    for i, each_neighbor in enumerate(neighbors):
        cX_0, cY_0 = results[each_neighbor[0]][2]
        cX_1, cY_1 = results[each_neighbor[1]][2]
        cv2.line(frame, (cX_0, cY_0), (cX_1, cY_1), (0, 0, 255), 2)

        # calculate the distance from the two points in pixels
        # if the frame number is divisible by the frame reset
        if frame_number % frame_reset == 0:
            neighbor_dist = dist.cdist(np.array([cX_0, cY_0]).reshape(1, -1), 
                np.array([cX_1, cY_1]).reshape(1, -1), metric="euclidean")[0][0]
            neighbor_dist_feet = neighbor_dist * (ave_person_height/ave_height_pixels)
            location_for_distance_X = int((cX_0 + cX_1) / 2)
            location_for_distance_Y = int((cY_0 + cY_1) / 2)
            feet = int(neighbor_dist_feet)
            inches = int((neighbor_dist_feet - int(neighbor_dist_feet)) * 12)
            # put text of average distance
            text = "{}'{}\"".format(feet, inches)
            # store the text and location in neighbor_distances
            neighbor_distances.append([text, location_for_distance_X, location_for_distance_Y])
        
    # print the neighbor distances on the frame
    for distances in neighbor_distances:
        cv2.putText(frame, distances[0], (distances[1], distances[2]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    # return neighbor_distances
    return neighbor_distances

def draw_boxes(frame, results, violate):
    '''
    input
    ------
    frame: frame of video
    results: detections from the frame

    returns
    ------
    None

    Draws boxes on the frame around people detections
    '''
    for (i, (prob, bbox, centroid, y)) in enumerate(results):
        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        (startX, startY, endX, endY) = bbox 
        (cX, cY) = centroid 
        color = (0, 255, 0)

        # if the index pair exists within the violation set, then
        # update the color
        if i in violate:
            color = (0, 0, 255)
        
        # draw (1) a bounding box around the person and (2) a circle 
        # at the centroid coords of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)


def calc_heights(results):
    '''
    input
    ------
    results: detections from the frame

    returns
    ------
    np.array of average height all people in the frame
    '''
    heights = []
    for i, result in enumerate(results):
        heights.append(results[i][3])
    return np.mean(heights)


def video_detection(net, video_input, video_output):
    # determine only the *output layer names that we need
    # from YOLOv3
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initalize the video stream and pointer to the output video file
    print("...Accessing video stream...")
    vs = cv2.VideoCapture(video_input)
    writer = None

    # initalize a frame number for putting text
    frame_number = 0

    # loop over the frames from the video stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if grabbed is False, then we have reach the end
        # of the stream
        if not grabbed:
            break

        # resize the frame and then detect people (only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

        # use average height of people as the minimum distance required between 
        # people. Call function calc_heights to calc the average height
        # of people in the frame
        ave_height_pixels = calc_heights(results)

        # initialize the set of indexes that violate
        # the minimum social distance (i.e., ave_height_pixels)
        violate = set()

        # initialize the set of tuples for neighbors that violate minimum
        # social distance (i.e., ave_height_pixels)
        neighbors = set()

        # ensure there are *at least* two people detections (required in 
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute
            # the Euclidean distance between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric='euclidean')

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the average height
                    if D[i, j] < ave_height_pixels:
                        # update violation set with the indexes of 
                        # the centroid pairs
                        violate.add(i)
                        violate.add(j)

                        # update the neighbors set
                        if (i, j) not in neighbors or (j, i) not in neighbors:
                            neighbors.add((i, j))

        # draw bounding box using draw_boxes function
        draw_boxes(frame, results, violate)

        # initialize a list to store the neighbor distances
        # and clear it every 10th frame
        if frame_number % 10 == 0:
            neighbor_distances = list()

        # use function draw_lines to draw a line from the bad neighbors
        neighbor_distances = draw_line_print_distances(frame, results, neighbors, 
                                frame_number, neighbor_distances, ave_height_pixels)
        
        # draw the total number of social distancing violations on the
        # output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        # if an output video file path has been supplied and the video
        # writer has not been itialized, do so now
        if video_output != "" and writer is None:
            # itialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(video_output, fourcc, 25,
                (frame.shape[1], frame.shape[0]), True)
        
        # if the video writer is not None, write the frame to the output
        # video file
        if writer is not None:
            writer.write(frame)

        # counter for frame number
        frame_number += 1

                


if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="", 
        help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="",
        help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1, 
        help="whether or not output frame should be displayed")
    args = vars(ap.parse_args())

    # define path where model is saved
    model_path = "yolo-coco"

    # load the COCO class labels and YOLO model
    labelsPath = os.path.sep.join([config.model_path, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model config
    weightsPath = os.path.sep.join([config.model_path, "yolov3.weights"])
    configPath = os.path.sep.join([config.model_path, "yolov3.cfg"])

    # Load YOLO object detector trained on COCO dataset (80 classes)
    print("...Loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # define input and output video path from args
    video_input = args["input"] if args["input"] else 0
    video_output = args["output"] if args["output"] else 0

    # call the video_detection function and pass
    # net, video_input, and video_output
    video_detection(net, video_input, video_output)
