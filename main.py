from ultralytics import YOLO
from trafficdetection import TrafficDetector


def main():
    model = YOLO("/Users/maurice/Developer/dominion/models/traffic_2.v1i.yolov11/weights/best.pt")
    video_url = "/Users/maurice/cedar_day_2.mp4"
    td = TrafficDetector(model, video_url)
    td.start()


if __name__ == "__main__":
    main()
