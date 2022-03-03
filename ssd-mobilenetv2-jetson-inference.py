import cv2
import jetson.inference
import jetson.utils

import argparse
import sys
import os
import time
import numpy as np

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
								 epilog=jetson.inference.detectNet.Usage() + jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="lines,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.4, help="minimum detection threshold to use")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

def distance(pixel_height, focal_length, average_height, cls):
	min_height = average_height - 0.2
	max_height = average_height + 0.2
	if cls == 0:  # human
		average_height = average_height
		min_height = min_height
		max_height = max_height
	elif cls == 2:  # car
		average_height = 1.5
		min_height = 1.4
		max_height = 1.9
	elif cls == 5:  # bus
		average_height = 3.0
		min_height = 2.5
		max_height = 4.0
	elif cls == 8:  # truck
		average_height = 3.0
		min_height = 2.0
		max_height = 4.0
	else:
		average_height = 0
	if average_height != 0:
		d = focal_length * average_height / pixel_height
		d_min = focal_length * min_height / pixel_height
		d_max = focal_length * max_height / pixel_height
	else:
		d = 0
		d_min = 0
		d_max = 0
	return d, d_min, d_max

SAVE_TXT = True
TXT_PATH = "output.txt"
SHOW_VID = True

# create video output object
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
# net = jetson.inference.detectNet(argv=['--model=/home/ff/Desktop/jetson-inference/build/aarch64/bin/networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff', '--labels=/home/ff/Desktop/personal_project/classes.txt'])

# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

focal_length = 1000
average_height = 1.8
video_height = input.GetHeight()
video_weight = input.GetWidth()
if video_height != 1080:
	focal_l = focal_length * (video_height / 1080)
else:
	focal_l = focal_length

frame_count = 0
# process frames until the user exits
while True:
	frame_count += 1
	# capture the next image
	img = input.Capture()

	# Convert image to numpy.
	img_np_tmp = jetson.utils.cudaToNumpy(img)
	img_np = img_np_tmp.copy()

	# Convert frame from BGR to RGB color format again.
	frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

	# get the start time
	start_time = time.time()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		print(detection)
		label = detection.ClassID
		class_name = net.GetClassDesc(label)
		score = detection.Confidence
		bbox_left = detection.Left
		bbox_top = detection.Top
		bbox_right = detection.Right
		bbox_bottom = detection.Bottom
		bbox_height = detection.Height
		bbox_width = detection.Width
		bbox_area = detection.Area
		bbox_center = detection.Center

		d = distance(bbox_height, focal_l, average_height, label)

		if SAVE_TXT:
			# Check if file is empty. If it is, add column names.
			if not os.path.isfile(TXT_PATH) or os.stat(TXT_PATH).st_size == 0:
				with open(TXT_PATH, 'a') as f:
					out_txt = f"timestamp,frame_id,id_type,id,confidence,bbox_left,bbox_top,bbox_w,bbox_h," \
								f"avg_distance,min_distance,max_distance"
					f.write(out_txt)

			# to MOT format
			d_avg = float(d[0])
			d_min = float(d[1])
			d_max = float(d[2])
			# Write MOT compliant results to file
			with open(TXT_PATH, 'a') as f:
				out_txt = f"{start_time},{frame_count},{class_name},{label},{score},{bbox_left}," \
							f"{bbox_top},{bbox_width},{bbox_height},{d_avg},{d_min},{d_max}\n"
				f.write(out_txt)

	# render the image
	output.Render(img)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break