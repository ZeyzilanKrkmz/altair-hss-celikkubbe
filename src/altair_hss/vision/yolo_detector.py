from __future__ import annotations #kullanım amacı: henüz tanımlanmamış sınıflara referans verebilmeyi kolaylaştırmak ve runtime performansını arttırmaktır.


from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

from altair_hss.core.models import Detection,TargetClass



YOLO_TO_TARGET_CLASS={
    "f16":TargetClass.F16,
    "helicopter":TargetClass.HELICOPTER,
    "ballistic_missile":TargetClass.BALLISTIC_MISSILE,
    "uav":TargetClass.UAV,
    "micro_uav":TargetClass.MICRO_UAV,
    "friend":TargetClass.FRIEND
}

class YoloDetector:
    def __init__(
            self,
            model_path:str|Path,
            conf_threshold:float=0.4,
            image_size:int=640,
            device:str="cpu"
    )->None:
        self.model_path=str(model_path)
        self.conf_threshold=conf_threshold
        self.image_size=image_size
        self.device=device
        self.model=YOLO(self.model_path)

    
    def detect(self,image)->list[Detection]:
        results=self.model.predict(
            source=image,
            conf=self.conf_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False
        )
        return self._parse_results(results)
    

    def _parse_results(self,results:Iterable)->list[Detection]:
        detection:list[Detection]=[]
        for result in results:
            names=result.names
            for box in result.boxes:
                cls_id=int(box.cls[0].item())
                conf=float(box.conf[0].item())
                x1,y1,x2,y2=[int(v) for v in box.xyxy[0].tolist()]
                w=max(0,x2-x1)
                h=max(0,y2-y1)
                label_name=str(names.get(cls_id,"unknown")).lower()
                detection.append(
                    Detection(
                        bbox=(x1,y1,w,h),
                        confidence=conf,
                        label=YOLO_TO_TARGET_CLASS.get(label_name,TargetClass.UNKNOWN),
                    )
                )
        return detection
