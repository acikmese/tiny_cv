import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from coco_names import coco_category_names as coco_names
import time

# Time interval for taking shots.
time_interval = 0  # in seconds
# Model detection threshold.
threshold = 0.5
# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create colors for each class in coco dataset.
colors = np.random.uniform(0, 255, size=(len(coco_names), 3))
# Create model.
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, pretrained_backbone=False)
model = model.eval().to(device)

# Transform image.
transform = transforms.Compose([
	transforms.ToTensor(),
]
)
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

		# Transform image to tensor.
		image = transform(frame).to(device)
		# Add a batch dimension.
		image = image.unsqueeze(0)
		with torch.no_grad():
			# Get predictions.
			outputs = model(image)
		# Get labels as numbers.
		labels = outputs[0]['labels']
		# Get class names of labels.
		classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
		# Get score for all the predicted objects
		scores = outputs[0]['scores'].detach().cpu().numpy()
		# Get all the predicted bounding boxes
		bboxes = outputs[0]['boxes'].detach().cpu().numpy()
		# Get boxes above the threshold score
		boxes = bboxes[scores >= threshold].astype(np.int32)

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
