import cv2

class VideoFileSource:
    def __init__(self,path:str):
        self.path=path
        self.cap=None

    def start(self)->None:
        self.cap=cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {self.path}")
        

    def read(self):
        ok,frame=self.cap.read()
        if not ok:
            return None
        return frame
    
    def stop(self)->None:
        if self.cap is not None:
            self.cap.release()
            self.cap=None
