"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# File formats
IMAGE_EXTENSIONS = [".bmp", ".jpeg", ".jpg", ".png", ".webp", ".tiff", ".tif"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm"]


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocess(input, net_input_shape):
    p_input = cv2.resize(input, (net_input_shape[3], net_input_shape[2]))
    p_input = p_input.transpose((2, 0, 1))
    p_input = p_input.reshape(1, *p_input.shape)
    return p_input

def draw_boxes(frame, output, conf_threshold, width, height):
    color = (255, 0, 0)
    count = 0

    for box in output[0][0]:
        conf = box[2]
        if conf >= conf_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            count += 1
    
    return frame, count

def get_file_extension(path_string):
    filename = os.path.basename(path_string)
    _, ext = os.path.splitext(filename)
    return ext

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    model_xml = args.model
    device = args.device
    input = args.input

    ### Load the model through `infer_network`
    infer_network.load_model(model_xml, device)
    network_input_shape = infer_network.get_input_shape()

    # Initialize variable for VideoCapture
    cap = None

    ### Handle the input stream
    file_ext = get_file_extension(input).lower()

    # If input is an image file
    if (file_ext in IMAGE_EXTENSIONS):
        print("Image file fed.")
        
        # Read and preprocess image
        image = cv2.imread(input)
        height, width, _ = image.shape
        p_image = preprocess(image, network_input_shape)

        # Infer and draw boxes, then save as .jpg
        infer_network.exec_net(p_image)
        if infer_network.wait() == 0:
            output = infer_network.get_output()
            out_image, _ = draw_boxes(image, output, prob_threshold, width, height)
            cv2.imwrite("out.jpg", out_image)

        # Do not proceed
        return

    elif (file_ext in VIDEO_EXTENSIONS):
        cap = cv2.VideoCapture(input)
    
    elif (input.lower() == "cam"):
        input = 0
        cap = cv2.VideoCapture(input)
    
    else:
        print("ERROR: Unsupported input provided!")
        print("Please provide a valid image or video file, or 'cam' for camera feed")
        sys.exit(1)

    cap.open(input)

    input_width = int(cap.get(3))
    input_height = int(cap.get(4))

    # Total no of people couted
    total_count = 0

    # To store past value of "current_count"
    last_count_value = 0

    # No of consecutive frames when inference yiels same result for "current_count"
    same_count_streak = 0

    # Last value of "current_count" when there was consecutive same value for
    # "current_count" variable for at least 5 times
    last_streak_count_value = 0

    # To store start time of event when at least one person appears in frame
    start_time = None

    # Duration during which at least one person is in frame
    duration = 0

    # Determine if a person completed a duration by exiting out of the frame
    is_duration_complete = False

    ### Loop until stream is over
    while(cap.isOpened()):

        ### Read from the video capture 
        flag, frame = cap.read()
        if not flag:
            break

        # Break stream if Esc key is pressed
        if cv2.waitKey(1) == 27:
            break

        # Display input if read from camera
        if input == 0:
            cv2.imshow("input", frame)

        ### Pre-process the image as needed 
        p_frame = preprocess(frame, network_input_shape)

        ### Start asynchronous inference for specified request
        infer_network.exec_net(p_frame)

        ### Wait for the result
        if infer_network.wait() == 0:

            ### Get the results of the inference request
            output = infer_network.get_output()

            ### Extract any desired stats from the results
            out_frame, current_count = draw_boxes(frame, output, prob_threshold, input_width, input_height)

            if (last_count_value == current_count):
                # Increment streak if same value observed
                same_count_streak += 1
            else:
                # Reset if different value observed
                same_count_streak = 1

            last_count_value = current_count

            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            # Unless same 5 values are observed, do not add to total
            # This is to avoid miscounting due to momentarily
            # random false low predictions
            if (same_count_streak == 5):

                # Add only if additional people are detected in frame.
                # Not when partial no of people get out of frame
                if (last_streak_count_value <= current_count):
                    total_count += current_count - last_streak_count_value

                # If at least one person appears in frame while there were none,
                # initiate a new start_time with timestamp of current frame
                if (current_count != 0 and last_streak_count_value == 0):
                    start_time = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Else reset start_time if frame contains no person at the time and 
                # and set is_duration_complete to True, if current count changed from non-zero to 0 value,
                # as duration is completed then for person in frame
                elif (current_count == 0):
                    start_time = None
                    if last_streak_count_value != 0:
                        is_duration_complete = True

                # Set last streak count's value to current value    
                last_streak_count_value = current_count
            
            # If there are people in frame calculate duration in seconds
            if (not start_time is None):
                duration = round((cap.get(cv2.CAP_PROP_POS_MSEC) - start_time)/1000.0, 2)

            client.publish("person", json.dumps({"count": current_count, "total": total_count}))
            # If a duration completed publish it and set duration to 0 and is_duration_complete to False
            if is_duration_complete:
                client.publish("person/duration", json.dumps({"duration": duration}))
                is_duration_complete = False
                duration = 0


        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()

    ### Release resources
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()    


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
