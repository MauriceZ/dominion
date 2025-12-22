from ultralytics import YOLO

from trafficdetection import TrafficDetector

MODEL_PATH = "/Users/maurice/Developer/dominion/models/traffic_2.v1i.yolov11/weights/best.pt"
VIDEO_URL = "/Users/maurice/cedar_day_2.mp4"

def main():
    model = YOLO(MODEL_PATH)
    video_url = VIDEO_URL
    td = TrafficDetector(model, video_url)
    td.start()


if __name__ == "__main__":
    main()
