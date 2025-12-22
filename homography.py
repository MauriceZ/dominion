import numpy as np
import cv2

FILE_PATH = "road_iowa.png.points.csv"

data = np.loadtxt(FILE_PATH, delimiter=",", skiprows=1)

src_pts = data[:, 2:4].astype(np.float32)
dst_pts = data[:, 0:2].astype(np.float32)

H, mask = cv2.findHomography(src_pts, dst_pts, method=0)

print(H)
