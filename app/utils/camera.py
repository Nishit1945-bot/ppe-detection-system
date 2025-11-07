"""
Camera handling utilities
"""
import cv2
from typing import Optional, Tuple

class Camera:
    """Camera capture and management"""
    
    def __init__(
        self,
        source: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        """
        Initialize camera
        
        Args:
            source: Camera index or RTSP URL
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
    
    def start(self) -> bool:
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        return True
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Read frame from camera"""
        if self.cap is None:
            return False, None
        
        return self.cap.read()
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
