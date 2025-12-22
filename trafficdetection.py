import logging

import cv2

from trafficmap import TrafficMap


class TrafficDetector:
    def __init__(self, model, video_url):
        self.model = model
        self.video_url = video_url
        self.traffic_map = TrafficMap(fps=7.95)

    def start(self):
        cap = cv2.VideoCapture(self.video_url)

        if not cap.isOpened():
            logging.error("could not open video stream.")
            return

        frame_num = 0
        while True:
            success, frame = cap.read()
            frame_num += 1

            if not success:
                logging.debug("Failed to read frame or stream ended.")
                break

            results = self.model.track(frame, persist=True)
            annotated_frame = results[0].plot()

            for box in results[0].boxes:
                self.traffic_map.add_car(box, frame_num)
                self.traffic_map.add_car(box, frame_num)
                annotated_frame = self._draw_car_speeds(annotated_frame, box)

            self.traffic_map.refresh(frame_num)

            cv2.imshow("Live Traffic Detection", annotated_frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

    def _draw_car_speeds(self, frame, box):
        car_speed = self.traffic_map.get_car_speed(box.id)
        if car_speed is None:
            return frame

        label = f"{round(car_speed)} km/h"

        x1, y2, x2, y1 = box.xyxy[0].cpu().int().numpy()

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)

        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (36,135,33), -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        return frame
