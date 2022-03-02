import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import draw_bounding_boxes
import numpy as np
import time

np.random.seed(1)

# Time interval for taking shots.
time_interval = 0  # in seconds
# Model detection threshold.
threshold = 0.4
# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create model.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.4  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference
model = model.eval().to(device)

# Create colors for each class in coco dataset.
colors = np.random.uniform(0, 255, size=(len(model.names), 3))

# Set camera.
cap = cv2.VideoCapture(0)
# To count total frame processed.
frame_count = 0
# To get final frames per second.
total_fps = 0
# Start time to use camera according to time interval.
start_time = time.time()

while cap.isOpened():
	# Check if specified seconds passed.
	if time.time() - start_time >= time_interval:
		# Capture each frame of the video
		ret, frame = cap.read()
		if not ret:
			raise RuntimeError("Failed to read frame!")

		# get the start time
		start_time = time.time()

		with torch.no_grad():
			# Get predictions.
			outputs = model(frame)

		results = outputs.pandas().xyxy[0]
		labels = results['class'].values.tolist()
		classes = results['name'].values.tolist()
		scores = results['confidence'].values.tolist()
		boxes = results.iloc[:, :4].values

		# Convert frame from BGR to RGB color format.
		frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)

		# Draw boxes.
		for i, box in enumerate(boxes):
			color = colors[labels[i]]  # Get color of box.
			# Draw rectangle on frame.
			cv2.rectangle(
				frame,
				(int(box[0]), int(box[1])),
				(int(box[2]), int(box[3])),
				color, 1
			)
			# Put class name on box.
			cv2.putText(frame, f"{classes[i]}: {scores[i]:.2f}", (int(box[0]), int(box[1] - 5)),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
			            lineType=cv2.LINE_AA
			            )
		# Convert frame from BGR to RGB color format again.
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Get end time of the process.
		end_time = time.time()
		# Calculate fps.
		fps = 1 / (end_time - start_time)
		frame_count += 1
		total_fps += frame_count
		# Write the FPS on the current frame.
		cv2.putText(frame, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
		            0.5, (0, 255, 0), 2
		            )

		# Generate wait time for imshow.
		wait_time = max(1, int(fps / 4))
		cv2.imshow('frame', frame)
		if cv2.waitKey(wait_time) & 0xFF == ord('q'):
			break

# Release VideoCapture()
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()
# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
