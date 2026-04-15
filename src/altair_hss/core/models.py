from dataclasses import dataclass
from enum import Enum
from typing import Optional,Tuple

class TargetClass(str,Enum):
    F16="f16"
    HELICOPTER="helicopter"
    BALLISTIC_MISSILE="ballistic_missile"
    UAV="uav"
    MICRO_UAV="micro_uav"
    FRIEND="friend"
    UNKNOWN="unknown"

@dataclass
class Detection:
    bbox:Tuple[int,int,int,int]
    confidence:float
    label:TargetClass=TargetClass.UNKNOWN
    distance_m:Optional[float]=None