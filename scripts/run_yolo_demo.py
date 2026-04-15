import argparse
from pathlib import Path

import cv2

from altair_hss.config.settings import AppSettings
from altair_hss.vision.preprocess import VisionPreprocessor
from altair_hss.vision.video_source import VideoFileSource
from altair_hss.vision.yolo_detector import YoloDetector


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=str,default=None,help="Video dosyası yolu")
    parser.add_argument("--model",type=str,default=None,help="Yolo .pt model yolu")
    parser.add_argument("--device",type=str,default=None,help="cpu/cuda")
    return parser.parse_args()

def fit_to_scr(frame,max_width=1280,max_height=720):
    h,w=frame.shape[:2]
    scale=min(max_width/w,max_height/h)
    new_w=int(w*scale)
    new_h=int(h*scale)
    return cv2.resize(frame,(new_w,new_h))


def main():
    args=parse_args()
    settings=AppSettings()
    input_source=args.input or settings.input_source
    model_path=args.model or str(settings.yolo.model_path)
    device=args.device or settings.yolo.device


    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")
    if not Path(input_source).exists():
        raise FileNotFoundError(f"Input video bulunamadı: {input_source}")
    
    source=VideoFileSource(input_source)
    preprocessor=VisionPreprocessor()
    detector=YoloDetector(
        model_path=model_path,
        conf_threshold=settings.yolo.conf_threshold,
        image_size=settings.yolo.image_size,
        device=device

    )
    window_name="ALTAIR YOLO demo"
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name,1280,720)

    source.start()
    try:
        while True:
            frame=source.read()
            if frame is None:
                break
            processed=preprocessor.run(frame)
            detection=detector.detect(processed)

            rendered=processed.copy()
            for det in detection:
                x,y,w,h=det.bbox
                cv2.rectangle(rendered,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(
                    rendered,
                    f"{det.label.value}:{det.confidence:.2f}",
                    (x,max(20,y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )
            display_frame=fit_to_scr(rendered,1280,720)
            cv2.imshow(window_name,display_frame)
            key=cv2.waitKey(1) & 0xFF
            if key==ord("q"):
                break
    finally:
        source.stop()
        cv2.destroyAllWindows()



if __name__=="__main__":
    main()