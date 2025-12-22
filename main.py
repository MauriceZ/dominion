import cv2
from ultralytics import YOLO
from trafficmap import Map


def main():
    model = YOLO("/Users/maurice/Developer/dominion/models/traffic_2.v1i.yolov11/weights/best.pt")
    stream_url = "/Users/maurice/cedar_day_2.mp4"

    cap = cv2.VideoCapture(stream_url)

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

        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        for box in results[0].boxes:
            traffic_map.add_car(box, frame_num)

        # traffic_map.refresh(frame_num)

        # print("hello?")
        cv2.imshow("Live Traffic Detection", annotated_frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
