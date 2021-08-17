import cv2

img = cv2.imread('monyet.png')

classNames = 'Monkey.names'

with open(classNames, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configpath = 'custom-yolov4-tiny-detector.cfg'
weightpath = 'custom-yolov4-tiny-detector_best.weights'

net = cv2.dnn_DetectionModel(weightpath, configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=0.5)
print(classIds, bbox)

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    cv2.rectangle(img, box, color=(0,255,0), thickness=2)
    cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)

cv2.imshow("Output", img)
cv2.waitKey(0) 
