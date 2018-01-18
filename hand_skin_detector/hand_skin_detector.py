import cv2
import judger_hand
import numpy as np
import skimage.io


def nice_contour(contour, height=140):
    min_y = min([c[0][1] for c in contour])
    new_contour = [c for c in contour if c[0][1] < min_y + height]
    return np.array(new_contour)


def find_hand_by_color(img, thre_area=2000, color_range=[[0, 87, 134], [255, 137, 150]], height=140):
    img = np.array(img)
    # Constants for finding range of skin color in YCrCb
    min_YCrCb = np.array(color_range[0], np.uint8)
    max_YCrCb = np.array(color_range[1], np.uint8)
    # Find region with skin tone in YCrCb image
    imageYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    # Do contour detection on skin region
    _, contours, _ = cv2.findContours(
        skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_box = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > thre_area:
            contours[i] = nice_contour(contours[i], height)
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(img, contours, i, (0, 0, 255), 2)
            candidate_box.append((w * h, [x, y, w, h]))

    candidate_box = sorted(candidate_box, key=lambda x: -x[0])[:2]
    bbox = {'bbox': {}}
    for _, box in candidate_box:
        coor = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        if box[0] + box[2] > 0.6 * img.shape[1]:
            bbox['bbox']['R'] = coor
        else:
            bbox['bbox']['L'] = coor
    print(img.shape)
    print(candidate_box)
    print(bbox)
    return bbox, img


test_files = judger_hand.get_file_names()
output_f = judger_hand.get_output_file_object()
bbox_ = {'bbox': {}}
for fname in test_files:
    img = skimage.io.imread(fname)
    bbox, img = find_hand_by_color(img)
    if not bbox['bbox']:
        bbox = bbox_
    for hand, box in bbox['bbox'].items():
        hand = 0 if hand == 'L' else 1
        out = '%s %d %d %d %d %d 1.0 \n' % (
            fname, box[0], box[1], box[2], box[3], hand)
        print(out)
        output_f.write(out.encode())
    bbox_ = bbox
judger_hand.judge()

""" For demo use
# Camera
camera = cv2.VideoCapture(0)

while(1):
    # Capture frame from camera
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame,5,50,100)
    bbox, frame = find_hand_by_color(frame, color_range=[[0, 140, 80], [255,180,128]], height=300)
    
    cv2.imshow('Hand Detection',frame)
    interrupt=cv2.waitKey(10)
"""
