from __future__ import annotations
from dataclasses import dataclass ,field
from math import hypot
from typing import Optional
import time

from altair_hss.core.models import Detection

@dataclass
class TrackState:
    track_id:int
    detection:Detection
    center_xy:tuple[float,float]
    prev_center_xy:Optional[tuple[float,float]]=None
    approaching:bool=False
    velocity_px:tuple[float,float]=(0.0,0.0)
    age:int=1
    missed_frames:int=0
    last_update:float=field(default_factory=time.time)


class NearestCentroidTracker:
    def __init__(self,max_distance_px:float=80.0,max_missed_frames:int=10):
        self.max_distance_px=max_distance_px
        self.max_missed_frames=max_missed_frames
        self.next_track_id=1
        self.tracks:dict[int,TrackState]={}


    @staticmethod
    def _center_of(det:Detection)->tuple[float,float]:
        x,y,w,h=det.bbox
        return (x+w/2.0,y+h/2.0)
    
    def _distance(self,a:tuple[float,float],b:tuple[float,float])->float:
        return hypot(a[0]-b[0],a[1]-b[1])
    
    def _spawn_track(self,det:Detection)->TrackState:
        center=self._center_of(det)
        track=TrackState(
            track_id=self.next_track_id,
            detection=det,
            center_xy=center
        )
        self.tracks[self.next_track_id]=track
        self.next_track_id+=1
        return track
    
    def update(self,detections:list[Detection])->list[TrackState]:
        if not self.tracks:
            for det in detections:
                self._spawn_track(det)
            return list(self.tracks.values())
        
        unmatched_track_ids=set(self.tracks.keys())
        unmatched_detection_indices=set(range(len(detections)))
        matches:list[tuple[int,int]]=[]

        candidate_pairs:list[tuple[float,int,int]]=[]
        for track_id,track in self.tracks.items():
            for det_idx,det in enumerate(detections):
                det_center=self._center_of(det)
                dist=self._distance(track.center_xy,det_center)
                candidate_pairs.append((dist,track_id,det_idx))

        candidate_pairs.sort(key=lambda x:x[0])

        for dist,track_id,det_idx in candidate_pairs:
            if dist>self.max_distance_px:
                continue
            if track_id not in unmatched_track_ids:
                continue
            if det_idx not in unmatched_detection_indices:
                continue
            matches.append((track_id,det_idx))
            unmatched_detection_indices.remove(det_idx)
            unmatched_track_ids.remove(track_id)

        for track_id,det_idx in matches:
            det=detections[det_idx]
            track=self.tracks[track_id]
            new_center=self._center_of(det)
            prev_center=track.center_xy

            vx=new_center[0]-prev_center[0]
            vy=new_center[1]-prev_center[1]


            approaching=False

            if det.distance_m is not None and track.detection.distance_m is not None:
                approaching=det.distance_m <track.detection.distance_m

            track.prev_center_xy=prev_center
            track.center_xy=new_center
            track.velocity_px=(vx,vy)
            track.approaching=approaching
            track.detection=det
            track.age+=1
            track.missed_frames=0
            track.last_update=time.time()

        for track_id in list(unmatched_track_ids):
            track=self.tracks[track_id]
            track.missed_frames+=1
            if track.missed_frames>self.max_missed_frames:
                del self.tracks[track_id]

        for det_idx in unmatched_detection_indices:
            self._spawn_track(detections[det_idx])

        return list(self.tracks.values())

