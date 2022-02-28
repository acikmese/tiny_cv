import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import time

# Preprocess module for MobileNet
preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
)

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model = torch.jit.script(model)
model.eval()

image_size = 224  # 640 or 320 for YOLOv5, 300 for MobileNetV2, 224 for MobileNetV3

# Set CUDA or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('Using device:', device)

cap = cv2.VideoCapture(0)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
# model.conf = 0.5  # confidence threshold (0-1)
# model.iou = 0.45  # NMS IoU threshold (0-1)
# # model.classes = [0, 2, 5, 8, 15]
model.to(device)

if device == 'cuda':
	# FP16 supported on limited backends with CUDA
	model.model.half()

started = time.time()
last_logged = time.time()
frame_count = 0

while cap.isOpened():
	# # Get FPS.
	# fps = cap.get(cv2.CAP_PROP_FPS)

	# Read frame
	ret, image = cap.read()
	if not ret:
		raise RuntimeError("failed to read frame")

	# convert opencv output from BGR to RGB
	image = image[:, :, [2, 1, 0]]
	permuted = image

	# preprocess
	input_tensor = preprocess(image)

	# create a mini-batch as expected by the model
	input_batch = input_tensor.unsqueeze(0)

	# # Convert BGR image to RGB image
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# # Define a transform to convert image to a Torch tensor
	# transform = transforms.Compose([transforms.ToTensor()])
	# frame = transform(frame)

	# # Make detections (with inference size)
	# results = model(frame, size=image_size)
	# results.print()

	# run model
	output = model(input_batch)

	# log model performance
	frame_count += 1
	now = time.time()
	fps = 0
	if now - last_logged > 1:
		fps = frame_count / (now - last_logged)
		print(f"{frame_count / (now - last_logged)} fps")
		last_logged = now
		frame_count = 0

	# Display the resulting frame
	cv2.imshow(f"FPS: {fps}", np.squeeze(output.render()))

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
