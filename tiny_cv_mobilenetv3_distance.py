import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from coco_names import coco_category_names as coco_names
import time
import os


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

SAVE_TXT = True
TXT_PATH = "output.txt"
SHOW_VID = True
# Time interval for taking shots.
time_interval = 0  # in seconds
# Model detection threshold.
threshold = 0.4

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def distance(pixel_height, focal_length, average_height, cls):
	min_height = average_height - 0.2
	max_height = average_height + 0.2
	if cls == 1:  # human
		average_height = average_height
		min_height = min_height
		max_height = max_height
	elif cls == 3:  # car
		average_height = 1.5
		min_height = 1.4
		max_height = 1.9
	elif cls == 6:  # bus
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


# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Half precision only supported on CUDA
half = True if device != 'cpu' else False
# Create colors for each class in coco dataset.
colors = np.random.uniform(0, 255, size=(len(coco_names), 3))
# Create model.
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, pretrained_backbone=False)
if half:
	model.half()
model = model.eval().to(device)

# Transform image.
transform = transforms.Compose([
	transforms.ToTensor(),
]
)
# Set camera.
# cap = cv2.VideoCapture(0)
flip = 2
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=flip), cv2.CAP_GSTREAMER)
# Set camera.
#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# To count total frame processed.
frame_count = 0
# To get final frames per second.
total_fps = 0
# Start time to use camera according to time interval.
start_time = time.time()

focal_length = 1000
average_height = 1.8

while cap.isOpened():
	# Check if specified seconds passed.
	if time.time() - start_time >= time_interval:
		# Capture each frame of the video
		ret, frame = cap.read()
		if not ret:
			raise RuntimeError("Failed to read frame!")

		# get the start time
		start_time = time.time()

		video_height = frame.shape[0]
		video_width = frame.shape[1]

		if video_height != 1080:
			focal_l = focal_length * (video_height / 1080)
		else:
			focal_l = focal_length

		# Transform image to tensor.
		image = transform(frame).to(device)
		# Add a batch dimension.
		image = image.unsqueeze(0)
		if half:
			image = image.half()
		with torch.no_grad():
			# Get predictions.
			outputs = model(image)
		# Get labels as numbers.
		labels = outputs[0]['labels']
		# Get class names of labels.
		classes = np.array([coco_names[i] for i in outputs[0]['labels'].cpu().numpy()])
		# Get score for all the predicted objects
		scores = outputs[0]['scores'].detach().cpu().numpy()
		# Get all the predicted bounding boxes
		bboxes = outputs[0]['boxes'].detach().cpu().numpy()
		# Get boxes above the threshold score
		boxes = bboxes[scores >= threshold].astype(np.int32)
		# Get founded classes.
		found = classes[scores >= threshold]
		found_unique, found_counts = np.unique(found, return_counts=True)
		print(np.asarray((found_unique, found_counts)).T)

		# Convert frame from BGR to RGB color format again.
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Get end time of the process.
		end_time = time.time()
		# Calculate fps.
		fps = 1 / (end_time - start_time)
		print(f"FPS: {fps}")
		frame_count += 1
		total_fps += frame_count

		# Convert frame from BGR to RGB color format.
		frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
		# Draw boxes.
		for i, box in enumerate(boxes):
			# Height of the box.
			height = box[3] - box[1]
			# Distance
			d = distance(height, focal_l, average_height, labels[i])

			if SHOW_VID:
				color = colors[labels[i]]  # Get color of box.
				# Draw rectangle on frame.
				cv2.rectangle(
					frame,
					(int(box[0]), int(box[1])),
					(int(box[2]), int(box[3])),
					color, 1
				)
				# Put class name on box.
				cv2.putText(frame, f"{classes[i]}: {scores[i]:.2f}: {d[0]:.2f}m", (int(box[0]), int(box[1] - 5)),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
				            lineType=cv2.LINE_AA
				            )

				# Write the FPS on the current frame.
				cv2.putText(frame, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
				            0.5, (0, 255, 0), 2
				            )
				cv2.imshow('frame', frame)

			if SAVE_TXT:
				# Check if file is empty. If it is, add column names.
				if not os.path.isfile(TXT_PATH) or os.stat(TXT_PATH).st_size == 0:
					with open(TXT_PATH, 'a') as f:
						out_txt = f"timestamp,frame_id,id_type,id,confidence,bbox_left,bbox_top,bbox_w,bbox_h," \
						          f"avg_distance,min_distance,max_distance"
						f.write(out_txt)

				# to MOT format
				bbox_left = int(box[0])
				bbox_top = int(box[1])
				bbox_w = int(box[2] - box[0])
				bbox_h = int(box[3] - box[1])
				d_avg = float(d[0])
				d_min = float(d[1])
				d_max = float(d[2])
				conf_value = float(scores[i])
				id_type = str(classes[i])
				idx = int(labels[i].cpu().detach().numpy())
				# Write MOT compliant results to file
				with open(TXT_PATH, 'a') as f:
					out_txt = f"{start_time},{frame_count},{id_type},{idx},{conf_value},{bbox_left}," \
					          f"{bbox_top},{bbox_w},{bbox_h},{d_avg},{d_min},{d_max}\n"
					f.write(out_txt)

		# Generate wait time for imshow.
		wait_time = max(1, int(fps / 4))
		if cv2.waitKey(wait_time) & 0xFF == ord('q'):
			break

# Release VideoCapture()
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()
# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
