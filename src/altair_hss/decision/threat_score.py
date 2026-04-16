from altair_hss.tracking.tracker import TrackState

class ThreatScorer:
    def __init__(self):
        self.class_weights={#başlangıç tehdit ağırlıkları, neden string tabanlı?-> yolov8n.pt ile çoğu şey unknown olabilir, çökmeden puan üretimi için
            "f16":0.95,
            "helicopter":0.85,
            "ballistic_missile":1.00,
            "uav":0.70,
            "micro_uav":0.55,
            "friend":-1.00,
            "unknown":0.20
        }

    def score(self,track:TrackState,frame_width:int,frame_height:int)->float:
        label=track.detection.label.value
        class_score=self.class_weights.get(label,0.10)
        cx,cy=track.center_xy
        center_dx=abs(cx-frame_width/2.0)/max(frame_width/2.0,1.0)
        center_dy=abs(cy-frame_height/2.0)/max(frame_height/2.0,1.0)
        center_offset=(center_dx+center_dy)/2.0
        center_score=1.0-center_offset# görüntü merkezine daha yakın olan hedef kritik olarak ele alındı.


        vx,vy=track.velocity_px
        motion_score=min(((vx**2+vy**2)**0.5)/100.0,1.0)#hedefin frameler arasında ne kadar hareket ettiği

        persistence_score=min(track.age/15.0,1.0)#bir hedef uzun süre takipte kalıyorsa güven skoru yükseltilir

        approach_bonus=0.15 if track.approaching else 0.0#yaklaşma eğiliminde olan hedefe ekstra ouan

        total=(
            0.45* class_score+
            0.20*center_score+
            0.15*motion_score+
            0.20*persistence_score+
            approach_bonus
        )
        return round(total,3)