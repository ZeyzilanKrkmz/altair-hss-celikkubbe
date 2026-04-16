''' cv2.putText(
                display_frame,
                f"FPS: {fps:.1f} | Dets: {len(detections)} | Tracks: {len(tracks)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                display_frame,
                primary_text,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2,
        )'''

import argparse
import time
from pathlib import Path

import cv2

from altair_hss.config.settings import AppSettings
from altair_hss.vision.preprocess import VisionPreprocessor
from altair_hss.vision.video_source import VideoFileSource
from altair_hss.vision.yolo_detector import YoloDetector
from altair_hss.decision.target_selector import TargetSelector
from altair_hss.tracking.tracker import NearestCentroidTracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Video dosyası yolu")
    parser.add_argument("--model", type=str, default=None, help="Yolo .pt model yolu")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda")
    return parser.parse_args()


def fit_to_scr(frame, max_width=1280, max_height=720):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))

def draw_trsnprnt_pnl(image,top_left,bottom_right,color=(30,30,30),alpha=0.55):
    overlay=image.copy()
    cv2.rectangle(overlay,top_left,bottom_right,color,-1)
    return cv2.addWeighted(overlay,alpha,image,1-alpha,0)

def draw_txt_with_shdw(image,
                       text,
                       org,
                       font_scale=0.62,
                       color=(235,235,235),
                       thickness=1,
                       shadow_color=(20,20,20),
                       shadow_offset=(1,1)):
    x,y=org
    sx,sy=shadow_offset
    cv2.putText(
        image,
        text,
        (x+sx,y+sy),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        shadow_color,
        thickness+1,
        cv2.LINE_AA,

    )
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    settings = AppSettings()

    input_source = args.input or settings.input_source
    model_path = args.model or str(settings.yolo.model_path)
    device = args.device or settings.yolo.device

    selector = TargetSelector()

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")
    if not Path(input_source).exists():
        raise FileNotFoundError(f"Input video bulunamadı: {input_source}")

    source = VideoFileSource(input_source)
    preprocessor = VisionPreprocessor()
    detector = YoloDetector(
        model_path=model_path,
        conf_threshold=settings.yolo.conf_threshold,
        image_size=settings.yolo.image_size,
        device=device,
    )
    tracker = NearestCentroidTracker(max_distance_px=120.0, max_missed_frames=12)

    window_name = "ALTAIR YOLO demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    source.start()
    prev_time = time.time()

    try:
        while True:
            frame = source.read()
            if frame is None:
                break

            processed = preprocessor.run(frame)
            detections = detector.detect(processed)
            tracks = tracker.update(detections)

            frame_height, frame_width = processed.shape[:2]
            primary_target, ranked_tracks = selector.select_primary_target(
                tracks,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            score_map = {track.track_id: score for track, score in ranked_tracks}

            rendered = processed.copy()

            for track in tracks:
                x, y, w, h = track.detection.bbox
                color = (80,220,120)
                thickness = 2

                if primary_target is not None and track.track_id == primary_target[0].track_id:
                    color = (70,110,255)
                    thickness = 3

                cv2.rectangle(rendered, (x, y), (x + w, y + h), color, thickness)

                status = "APP" if track.approaching else "STB"
                score = score_map.get(track.track_id, 0.0)

                label = (
                    f"ID:{track.track_id} "
                    f"{track.detection.label.value} "
                    #f"{track.detection.confidence:.2f} "
                    f"S:{score:.2f} "
                    f"{status}"
                )

                cv2.putText(
                    rendered,
                    label,
                    (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA
                )

                cx, cy = int(track.center_xy[0]), int(track.center_xy[1])
                cv2.circle(rendered, (cx, cy), 4, color, -1)

            now = time.time()
            dt = max(now - prev_time, 1e-6)
            fps = 1.0 / dt
            prev_time = now

            primary_text = "Primary: None"
            if primary_target is not None:
                pt_track, pt_score = primary_target
                primary_text = (
                    f"Primary: ID {pt_track.track_id} | "
                    f"{pt_track.detection.label.value} | "
                    f"{pt_score:.2f}"
                )

            display_frame = fit_to_scr(rendered, 1280, 720)
            display_frame=draw_trsnprnt_pnl(
                display_frame,
                top_left=(12,12),
                bottom_right=(430,92),
                color=(24,28,32),
                alpha=0.58,
            )

        #buraya yukarıdaki yorum bloğunda bulunan funclar geliyordu ama sonra değiştirildi.
        
            draw_txt_with_shdw(
                display_frame,
                f"FPS: {fps:.1f} | Dets: {len(detections)} | Tracks: {len(tracks)}",
                (24,38),
                font_scale=0.62,
                color=(235,235,235),
                thickness=1
            )
        
            draw_txt_with_shdw(
                display_frame,
                primary_text,
                (24,68),
                font_scale=0.58,
                color=(120,220,255),
                thickness=1
            )
        
         
        
            

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        source.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()