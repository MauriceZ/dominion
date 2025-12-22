from ultralytics import YOLO

from trafficdetection import TrafficDetector

MODEL_PATH = "models/traffic_2.v1i.yolov11/weights/best.pt"
# VIDEO_URL = "https://drive.google.com/uc?export=download&id=1npPgva8HNwcM_B56UqeZ7s3xKZL0vcE_"
VIDEO_URL = "videos/iowa_trimmed.mp4"
# VIDEO_URL = "https://iowadotsfs2.us-east-1.skyvdn.com/rtplive/dqtv16lb/playlist.m3u8" # warning: need better night time data!

def main():
    model = YOLO(MODEL_PATH)
    video_url = VIDEO_URL
    td = TrafficDetector(model, video_url)
    td.start()


if __name__ == "__main__":
    main()
