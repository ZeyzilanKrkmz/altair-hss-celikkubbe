import cv2

class VisionPreprocessor:
    def run(self,frame):
        frame=cv2.GaussianBlur(frame,(5,5),0)
        return frame
    
    