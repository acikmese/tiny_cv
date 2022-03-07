import torchvision
import cv2
import torch
import argparse
import time
import detect_utils

# # construct the argument parser
# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--input', help='path to input video')
# args = vars(parser.parse_args())
# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the model
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, pretrained_backbone=False)
# model = torch.jit.script(model)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
# model.conf = 0.5  # confidence threshold (0-1)
# model.iou = 0.45  # NMS IoU threshold (0-1)
# model.classes = [0, 2, 5, 8, 15]
# load the model onto the computation device
model = model.eval().to(device)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print('Error while trying to read video. Please check path again')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = "output_video"
# define codec and create VideoWriter object
out = cv2.VideoWriter(f"{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height)
                      )
frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second

img_counter = 0
frame_set = []
start_time = time.time()

# read until end of video
while cap.isOpened():
	# if time.time() - start_time >= 1:  # <---- Check if 5 sec passed
	if 1 > 0:
		# capture each frame of the video
		ret, frame = cap.read()
		if ret:
			# get the start time
			start_time = time.time()
			with torch.no_grad():
				# get predictions for the current frame
				boxes, classes, labels = detect_utils.predict(frame, model, device, 0.5)

			# draw boxes and show current frame on screen
			image = detect_utils.draw_boxes(boxes, classes, labels, frame)
			# get the end time
			end_time = time.time()
			# get the fps
			fps = 1 / (end_time - start_time)
			# add fps to total fps
			total_fps += fps
			# increment frame count
			frame_count += 1
			# write the FPS on the current frame
			cv2.putText(image, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
			            0.5, (0, 255, 0), 2
			            )
			# press `q` to exit
			wait_time = max(1, int(fps / 4))
			# convert from BGR to RGB color format
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			cv2.imshow('image', image)
			out.write(image)
			if cv2.waitKey(wait_time) & 0xFF == ord('q'):
				break

			# img_name = "opencv_frame_{}.png".format(img_counter)
			# cv2.imwrite(img_name, image)
			# print("{} written!".format(img_counter))
			start_time = time.time()
		else:
			break
		img_counter += 1

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
