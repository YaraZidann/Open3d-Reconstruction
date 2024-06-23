import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import time

start = time.time()
# Q matrix camera calibration
Q = np.array(([1.0, 0.0, 0.0, 0.0],
              [0.0, -1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0 / 90.0, 0.0]), dtype=np.float32)

midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Input folder containing original images
input_folder = 'D:/downloads/livingroom1-color/'
output_folder = 'D:/downloads/ddd4epth/'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for image_file in os.listdir(input_folder):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image_path = os.path.join(input_folder, image_file)
        start = time.time()

        image = cv2.imread(image_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize the depth map to be in the range [0, 1]
        depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

        # Invert the depth map so that closer objects are brighter (0 is far, 1 is close)
        depth_map_inverted = 1.45 - depth_map_normalized

        # Convert to 8-bit grayscale (0 is black, 255 is white)
        depth_map_gray = (depth_map_inverted * 255).astype(np.uint16)

        # Invert again so that closer objects are darker
        depth_map_gray = 255 - depth_map_gray
        depth_map_gray = cv2.normalize(depth_map_gray, None, 0, 20, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)


        # Display or save the image
        # cv2.imshow('Depth Image', depth_map_gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        depth_map_path = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_depth.png')
        cv2.imwrite(depth_map_path, depth_map_gray)

end = time.time()

finaltime = end - start
print("Total Run Time for depth creation: ", finaltime)
