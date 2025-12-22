import cv2
from trafficmap import Map


class TrafficDetector:
    def __init__(self, model, video_url):
        self.model = model
        self.video_url = video_url

    def start(self):
        cap = cv2.VideoCapture(self.video_url)

        if not cap.isOpened():
            print("Error: could not open video stream.")
            return

        traffic_map = Map(fps=7.95)

        frame_num = 0
        while True:
            success, frame = cap.read()
            frame_num += 1

            if not success:
                print("Failed to read frame or stream ended.")
                break

            results = self.model.track(frame, persist=True)
            annotated_frame = results[0].plot()

            for box in results[0].boxes:
                traffic_map.add_car(box, frame_num)

            traffic_map.refresh(frame_num)

            cv2.imshow("Live Traffic Detection", annotated_frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()
