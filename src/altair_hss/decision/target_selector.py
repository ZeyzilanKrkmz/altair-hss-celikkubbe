from altair_hss.tracking.tracker import TrackState
from altair_hss.decision.threat_score import ThreatScorer

class TargetSelector:
    def __init__(self):
        self.scorer=ThreatScorer()

    def rank_tracks(
            self,
            tracks:list[TrackState],
            frame_width:int,
            frame_height:int,
    )->list[tuple[TrackState,float]]:
        scored_tracks=[]
        for track in tracks:
            score=self.scorer.score(track,frame_width,frame_height)
            scored_tracks.append((track,score))

        scored_tracks.sort(key=lambda item:item[1],reverse=True)
        return scored_tracks
    

    def select_primary_target(
            self,
            tracks:list[TrackState],
            frame_width:int,
            frame_height:int,
    ):
        ranked=self.rank_tracks(tracks,frame_width,frame_height)
        if not ranked:
            return None,[]
        return ranked[0], ranked