import logging
import math

import contextily as ctx
import cv2
import numpy as np

# Precomputed homography matrix
H_MATRIX = np.array([
    [1.44148602e+05, -2.98484538e+06, -1.00982323e+07],
    [-7.47233399e+04, 1.54733778e+06, 5.22176396e+06],
    [-1.42749785e-02, 2.95577907e-01, 1.00000000e+00],
])

CLASS_LABELS = {
    0: 'Pickup',
    1: 'Sedan',
    2: 'SUV',
    3: 'Truck',
    4: 'Van',
}

CLASS_COLORS_BGR = {
    0: (245, 41, 16),
    1: (232, 216, 100),
    2: (243, 243, 243),
    3: (100, 31, 20),
    4: (178, 212, 97),
}


class Car:
    def __init__(self, car_id, x, y, cls, frame_num, fps):
        self.fps = fps
        self.car_id = car_id
        self.positions = [(x, y, cls, frame_num)]  # tuples of (x, y, cls, last_seen_frame_num)

    def add_position(self, x, y, cls, frame_num):
        self.positions.append((x, y, cls, frame_num))

    def get_last_seen_frame(self):
        return self.positions[-1][3]

    def get_speed(self):
        if len(self.positions) < 5:
            logging.debug("no car speed")
            return None

        x1, y1, _, f1 = self.positions[-5]
        x2, y2, _, f2 = self.positions[-1]

        dx = x2 - x1
        dy = y2 - y1

        # scale mercator distance to true distance using the latitude
        dist = np.hypot(dx, dy)
        scale = np.cos(np.deg2rad(42.4932261))
        dist_scaled = math.floor(dist * scale)

        t = (1 / self.fps) * (f2 - f1)

        speed_mps = dist_scaled / t
        speed_kmph = speed_mps / 1000 * 60 * 60

        return speed_kmph

    @staticmethod
    def get_line_color(cls):
        if not isinstance(cls, int):
            cls = cls.int()[0].item()
        return CLASS_COLORS_BGR[cls]


class TrafficMap:
    def __init__(self, fps):
        self.fps = fps

        # The map bounds in web mercator
        self.x_min, self.x_max, self.y_min, self.y_max = -10098545.5, -10097885.3, 5234813.7, 5235224.6

        img_rgb, extent = ctx.bounds2img(self.x_min, self.y_min, self.x_max, self.y_max,
                                         zoom=18, source=ctx.providers.CartoDB.DarkMatter)

        self.img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self.h_px, self.w_px = self.img_bgr.shape[:2]

        hist_x, hist_y = self.meters_to_image_pixels(self.x_max, self.y_min)
        w = 400

        self.histogram = Histogram(pos=(hist_x - w, hist_y), width=w, unit_height=3)

        self.cars = {}

    def meters_to_image_pixels(self, x, y):
        u = (x - self.x_min) / (self.x_max - self.x_min) * (self.w_px - 1)
        v = (self.y_max - y) / (self.y_max - self.y_min) * (self.h_px - 1)
        return int(round(u)), int(round(v))

    def _upsert_car(self, car_id, x, y, cls, frame):
        if not isinstance(car_id, int):
            car_id = car_id.int()[0].item()
        if not isinstance(cls, int):
            cls = cls.int()[0].item()

        if car_id in self.cars:
            car = self.cars[car_id]
            car.add_position(x, y, cls, frame)
        else:
            logging.debug(f"adding car id={car_id}")
            self.cars[car_id] = Car(car_id, x, y, cls, frame, self.fps)

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

        # The homography matrix isn't perfect so manually shift the positions a bit
        map_x -= 30
        x_shift_threshold = -10098350.5
        if map_x >= x_shift_threshold:
            map_y -= 0.3 * (map_x - x_shift_threshold)

        self._upsert_car(bounding_box.id, map_x, map_y, bounding_box.cls, frame_num)

    def get_car_speed(self, car_id):
        if not isinstance(car_id, int):
            car_id = car_id.int()[0].item()

        if car_id not in self.cars:
            return None

        car = self.cars[car_id]
        return car.get_speed()

    # Called after each frame is done processing. For now, only removes stale cars
    def refresh(self, cur_frame):
        to_del = set()
        img = self.img_bgr.copy()

        for car_id, car in self.cars.items():
            last_x, last_y, _, last_f = car.positions[-1]
            secs_ago = (1 / self.fps) * (cur_frame - last_f)

            if secs_ago >= 5:
                to_del.add(car_id)
                continue

            for i in range(len(car.positions)):
                x, y, cls, _ = car.positions[i]
                u, v = self.meters_to_image_pixels(x, y)

                if i == 0:
                    cv2.circle(img, (u, v), 1, car.get_line_color(cls), thickness=-1)
                else:
                    last_x, last_y, last_cls, _ = car.positions[i - 1]
                    last_u, last_v = self.meters_to_image_pixels(last_x, last_y)
                    cv2.line(img, (last_u, last_v), (u, v), car.get_line_color(last_cls), thickness=2, lineType=16)

        u, v = self.meters_to_image_pixels(self.x_min, self.y_max)
        self.histogram.render(img, self.cars)

        cv2.imshow("Traffic 2D Map", img)

        for car_id in to_del:
            del self.cars[car_id]


class Histogram:
    def __init__(self, pos, width, unit_height):
        self.pos = pos # bottom left of histogram
        self.width = width
        self.unit_height = unit_height

    def render(self, img, cars):
        x_start, y_start = self.pos

        counts = {}

        for car in cars.values():
            car_cls = car.positions[-1][2]
            cls_count = counts.get(car_cls, 0) + 1
            counts[car_cls] = cls_count

        bar_w = self.width // len(CLASS_LABELS.keys())

        label_height = 20

        for i, cls in enumerate(sorted(CLASS_LABELS.keys())):
            count = counts.get(cls, 0)

            bar_x = x_start + bar_w * i
            bar_y = y_start - label_height - (count * self.unit_height)

            cv2.rectangle(img, (bar_x, bar_y), (x_start + bar_w * (i + 1), y_start - label_height), CLASS_COLORS_BGR[cls], -1)

            # Calculate margin to center the label text under the bar
            label = f"{CLASS_LABELS[cls]}s"
            (label_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_margin = (bar_w - label_w) // 2
            cv2.putText(img, f"{CLASS_LABELS[cls]}s", (bar_x + label_margin, y_start - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Calculate margin to center the count text over the bar
            (count_text_w, _), _ = cv2.getTextSize(str(count), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_margin = (bar_w - count_text_w) // 2
            cv2.putText(img, str(count), (bar_x + text_margin, bar_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
