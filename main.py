# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import PySimpleGUI as sg

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

layout = 	[
		[sg.Text('YOLOv4')],
		[sg.Text('Path to image'), sg.In(r'C:/Python/PycharmProjects/YoloObjectDetection/images/baggage_claim.jpg',size=(40,1), key='image'), sg.FileBrowse()],
		[sg.Text('Yolo base path'), sg.In(r'yolo-orange',size=(40,1), key='yolo'), sg.FolderBrowse()],
		[sg.Text('Confidence'), sg.Slider(range=(0,10),orientation='h', resolution=1, default_value=5, size=(15,15), key='confidence')],
		[sg.Text('Threshold'), sg.Slider(range=(0,10), orientation='h', resolution=1, default_value=3, size=(15,15), key='threshold')],
		[sg.OK(), sg.Cancel(), sg.Stretch()]
			]

win = sg.Window('YOLOv4 Orange Detector',
				default_element_size=(14,1),
				text_justification='right',
				auto_size_text=False).Layout(layout)

while True:
	event, values = win.Read()
	if event == sg.WIN_CLOSED or event == 'Exit':
		break

	args = values

	args['threshold'] = float(args['threshold']/10)
	args['confidence'] = float(args['confidence']/10)

	labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	# np.random.seed(42)
	# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	#	dtype="uint8")
	COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([args["yolo"], "yolo-obj_last.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolo-obj.cfg"])

	print("weightsPath", weightsPath)
	print("configPath", configPath)

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNet(configPath, weightsPath)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)


	model = cv2.dnn_DetectionModel(net)
	model.setInputParams(size=(416, 416), scale=1/255)

	# load our input image and grab its spatial dimensions
	image = cv2.imread(args["image"])

	(H, W) = image.shape[:2]


	start = time.time()
	classes, scores, boxes = model.detect(image, args['confidence'] , args['threshold'])
	end = time.time()

	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	start_drawing = time.time()
	for (classid, score, box) in zip(classes, scores, boxes):
		color = COLORS[int(classid) % len(COLORS)]
		print(color)
		label = "%s : %f" % (LABELS[classid[0]], score)
		cv2.rectangle(image, box, color, 2)
		cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	end_drawing = time.time()

	print("[INFO] " + "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000))

	if H > 900:
		image = ResizeWithAspectRatio(image, height = 900)
	if W > 1600:
		image = ResizeWithAspectRatio(image, width = 1600)

	# show the output image
	imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto

	layout = 	[
			[sg.Text('Yolo Output')],
			[sg.Image(data=imgbytes)],
			[sg.OK(), sg.Cancel(), sg.Stretch()]
				]

	win2 = sg.Window('Orange Detection YOLOv4',
					text_justification='right',
					resizable = True,
					auto_size_text=False).Layout(layout)
	event, values = win2.Read()
	win2.Close()


	# cv2.imshow("Image", image)
	cv2.waitKey(0)

win.close()
