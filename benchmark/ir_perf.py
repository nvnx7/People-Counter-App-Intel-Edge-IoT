'''
This is written to get stats from IR representation
of the model
'''

from inference import Network
import cv2
import time
import sys

# Frame durations (in ms) during which model must detect a person
pos_frames = ((6300, 19600), (22900, 44500), (50600, 69600), (74800, 86500), (92500, 119600), (123900, 135600))

conf_threshold = 0.5

def preprocess(input, net_input_shape):
    p_input = cv2.resize(input, (net_input_shape[3], net_input_shape[2]))
    p_input = p_input.transpose((2, 0, 1))
    p_input = p_input.reshape(1, *p_input.shape)
    return p_input

# Determine if inference is wrongly predicted
def is_wrong_infer(output):
    for box in output[0][0]:
        if box[2] > conf_threshold:
            return False
    return True

# Determine if frame is positive example (ie contains person)
def is_frame_pos(frame_dur):
    for dur in pos_frames:
        if frame_dur in range(*dur):
            return True
    return False

model_xml = "PATH_TO_IR_OF_MODEL"

if not model_xml.endswith(".xml"):
    print("ERROR: Please set 'model_xml' variable to appropriate .xml file of model IR in the script.")
    sys.exit(1)

device = "CPU"
input ="../resources/sample_vid.mp4"

infer_network = Network()
infer_network.load_model(model_xml, device)
network_input_shape = infer_network.get_input_shape()

cap = cv2.VideoCapture(input)

n_frames = 0 # int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n_pos_frames = 0
n_wrong_infer = 0

cap.open(input)

start = 0
end = 0
total = 0

while(cap.isOpened()):
    flag, frame = cap.read()

    if not flag:
        break

    n_frames += 1

    frame_dur = cap.get(cv2.CAP_PROP_POS_MSEC)
    is_pos = is_frame_pos(frame_dur)

    if (is_pos):
        n_pos_frames += 1

    p_frame = preprocess(frame, network_input_shape)

    start = time.time()

    infer_network.exec_net(p_frame)
    if infer_network.wait() == 0:
        output = infer_network.get_output()

    end = time.time()

    if (is_pos and is_wrong_infer(output)):
        n_wrong_infer += 1


    total += (end - start)
    print("Frames: " + str(n_frames) + " Pos Frames: " + str(n_pos_frames) + " Wrong Infer: " + str(n_wrong_infer))

print("MODEL IR BENCHMARKS")
print("\n")
print("TOTAL NO. OF FRAMES: " + str(n_frames))
print("TOTAL TIME TAKEN (s): " + str(total))
print("AVERAGE PER FRAME (ms): " + str( (float(total) / n_frames) * 1000.0 ))
print("\n")
print("TOTAL POSITIVE FRAMES: " + str(n_pos_frames))
print("TOTAL WRONG INFERENCES: " + str(n_wrong_infer))
print("ACCURACY: " + str( (float(n_pos_frames - n_wrong_infer)) / (float(n_pos_frames)) ))


