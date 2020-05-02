'''
This is written to get stats from plain model
'''

import numpy as np
import tensorflow as tf
import cv2
import time
import sys

# Frame durations (in ms) during which model must detect a person
pos_frames = ((6300, 19600), (22900, 44500), (50600, 69600), (74800, 86500), (92500, 119600), (123900, 135600))

conf_threshold = 0.5

def preprocess(frame):
    inp = cv2.resize(frame, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    return inp

# Determine if inference is wrongly predicted
def is_wrong_infer(output):
    num_detections = int(output[0][0])
    for i in range(num_detections):
        score = float(output[1][0][i])
        if (score > conf_threshold):
            return False
    return True

# Determine if frame is positive example (ie contains person)
def is_frame_pos(frame_dur):
    for dur in pos_frames:
        if frame_dur in range(*dur):
            return True
    return False

input ="../resources/sample_vid.mp4"
model = "PATH_TO_MODEL_FROZEN_INFERENCE_GRAPH"

if not model.endswith(".pb"):
    print("ERROR: Please set 'model' variable to appropriate .pb file of saved model in the script.")
    sys.exit(1)

cap = cv2.VideoCapture(input)

n_frames = 0
n_pos_frames = 0
n_wrong_infer = 0

cap.open(input)

start = 0
end = 0
total = 0

# Read the graph.
with tf.gfile.FastGFile(model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    while(cap.isOpened()):
        flag, frame = cap.read()

        if not flag:
            break

        n_frames += 1

        frame_dur = cap.get(cv2.CAP_PROP_POS_MSEC)
        is_pos = is_frame_pos(frame_dur)

        if (is_pos):
            n_pos_frames += 1

        p_frame = preprocess(frame)

        start = time.time()

        output = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': p_frame.reshape(1, p_frame.shape[0], p_frame.shape[1], 3)})

        end = time.time()

        if (is_pos and is_wrong_infer(output)):
            n_wrong_infer += 1


        total += (end - start)
        print("Frames: " + str(n_frames) + " Pos Frames: " + str(n_pos_frames) + " Wrong Infer: " + str(n_wrong_infer))

print("PLAIN MODEL BENCHMARKS")
print("\n")
print("TOTAL NO. OF FRAMES: " + str(n_frames))
print("TOTAL TIME TAKEN (s): " + str(total))
print("AVERAGE PER FRAME (ms): " + str( (float(total) / n_frames) * 1000.0 ))
print("\n")
print("TOTAL POSITIVE FRAMES: " + str(n_pos_frames))
print("TOTAL WRONG INFERENCES: " + str(n_wrong_infer))
print("ACCURACY: " + str( (float(n_pos_frames - n_wrong_infer)) / (float(n_pos_frames)) ))




        # rows = img.shape[0]
        # cols = img.shape[1]
        # inp = cv.resize(img, (300, 300))
        # inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model


    # Visualize detected bounding boxes.
    # num_detections = int(out[0][0])
    # for i in range(num_detections):
    #     classId = int(out[3][0][i])
    #     score = float(out[1][0][i])
    #     bbox = [float(v) for v in out[2][0][i]]
    #     if score > 0.3:
    #         x = bbox[1] * cols
    #         y = bbox[0] * rows
    #         right = bbox[3] * cols
    #         bottom = bbox[2] * rows
    #         cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

# cv.imshow('TensorFlow MobileNet-SSD', img)
# cv.waitKey()
