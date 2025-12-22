import cv2
import contextily as ctx
import numpy as np
import ultralytics.engine.results

# Precomputed homography matrix
H_MATRIX = np.array([
    [1.44148602e+05, -2.98484538e+06, -1.00982323e+07],
    [-7.47233399e+04, 1.54733778e+06, 5.22176396e+06],
    [-1.42749785e-02, 2.95577907e-01, 1.00000000e+00],
])

class Car:
    def __init__(self, car_id, x, y, frame_num):
        self.car_id = car_id
        self.positions = [(x, y, frame_num)] # tuples of (x, y, last_seen_frame_num)

    def add_position(self, x, y, last_seen_frame):
        self.positions.append((x, y, last_seen_frame))

    def get_last_seen_frame(self):
        return self.positions[-1][2]


class Map:
    def __init__(self, fps):
        self.fps = fps

        # The map bounds in web mercator
        self.x_min, self.x_max, self.y_min, self.y_max = -10098545.5, -10097885.3, 5234813.7, 5235224.6

        img_rgb, extent = ctx.bounds2img(self.x_min, self.y_min, self.x_max, self.y_max,
                                         zoom=18, source=ctx.providers.CartoDB.DarkMatter)

        self.img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self.h_px, self.w_px = self.img_bgr.shape[:2]

        self.cars = {}

    def meters_to_image_pixels(self, x, y):
        u = (x - self.x_min) / (self.x_max - self.x_min) * (self.w_px - 1)
        v = (self.y_max - y) / (self.y_max - self.y_min) * (self.h_px - 1)
        return int(round(u)), int(round(v))

    def _upsert_car(self, car_id, x, y, frame):
        if car_id in self.cars:
            car = self.cars[car_id]
            car.add_position(x, y, frame)
        else:
            self.cars[car_id] = Car(car_id, x, y, frame)

    def add_car(self, bounding_box, frame_num):
        x1, y2, x2, y1 = bounding_box.xyxy[0].cpu().numpy()

        # Invert the y-coords to be positive -- this matches the web mercator CRS
        y2 = -y2
        y1 = -y1

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        point_on_camera = np.array([[[center_x, center_y]]], dtype=np.float32)
        point_on_map = cv2.perspectiveTransform(point_on_camera, H_MATRIX)
        map_x, map_y = point_on_map[0][0]

        map_x -= 30

        # The homography matrix isn't perfect so manually shift the positions down when beyond an x threshold.
        x_shift_threshold = -10098330.5
        if map_x >= x_shift_threshold:
            map_y -= 0.4 * (map_x - x_shift_threshold)

        self._upsert_car(bounding_box.id, map_x, map_y, frame_num)

        u, v = self.meters_to_image_pixels(map_x, map_y)
        cv2.circle(self.img_bgr, (u, v), 1, (0, 0, 255), thickness=-1)

        cv2.imshow("Traffic Map", self.img_bgr)

    # Called after each frame is done processing. For now, only removes stale cars
    def refresh(self, frame_num):
        pass
