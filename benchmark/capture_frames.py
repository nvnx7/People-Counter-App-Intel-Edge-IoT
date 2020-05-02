'''
This script is used to generate images from video sample to
benchmark models on
'''
import cv2

# Frame timestamps (ms) around which to capture frame (these are either start or end times approx.
# when a person enters/exits)
pos_frames = (6300, 19200, 23000, 44400, 50800, 69500, 74800, 86400, 92600, 118900, 124000, 135600)

dT = 3000

# Durations (start to end) during which to capture frames
durations = []
for x in pos_frames:
    durations.append((x - dT, x + dT))

input ="../resources/sample_vid.mp4"

cap = cv2.VideoCapture(input)
cap.open(input)

while(cap.isOpened()):
    flag, frame = cap.read()

    if not flag:
        break

    frame_dur = cap.get(cv2.CAP_PROP_POS_MSEC)
    print(frame_dur)

    for dur in durations:
        if int(frame_dur) in range(*dur):
            cv2.imwrite(str(frame_dur) + ".jpg", frame)
            break
