"""
Core PPE detection module using YOLOv8
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO

class PPEDetector:
    """PPE Detection class for helmet, glasses, and gloves"""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize PPE Detector
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence_threshold: Minimum confidence for detection
            device: Device to run inference on (cpu/cuda)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        
        self.required_ppe = ["helmet", "safety_glasses", "gloves"]
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect PPE in frame
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        results = self.model(frame, conf=self.confidence_threshold)[0]
        
        detected_items = self._parse_detections(results)
        compliance_status = self._check_compliance(detected_items)
        
        return {
            "detected_items": detected_items,
            "compliance_status": compliance_status,
            "missing_items": self._get_missing_items(detected_items),
            "raw_results": results
        }
    
    def _parse_detections(self, results) -> Dict:
        """Parse YOLO detection results"""
        detected = {
            "helmet": {"detected": False, "confidence": 0.0, "boxes": []},
            "safety_glasses": {"detected": False, "confidence": 0.0, "boxes": []},
            "gloves": {"detected": False, "confidence": 0.0, "boxes": []}
        }
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = results.names[class_id].lower().replace(" ", "_")
            
            if class_name in detected:
                detected[class_name]["detected"] = True
                detected[class_name]["boxes"].append({
                    "confidence": confidence,
                    "bbox": box.xyxy[0].tolist()
                })
                
                if confidence > detected[class_name]["confidence"]:
                    detected[class_name]["confidence"] = confidence
        
        return detected
    
    def _check_compliance(self, detected_items: Dict) -> bool:
        """Check if all required PPE is detected"""
        return all(
            detected_items[item]["detected"]
            for item in self.required_ppe
        )
    
    def _get_missing_items(self, detected_items: Dict) -> List[str]:
        """Get list of missing PPE items"""
        return [
            item for item in self.required_ppe
            if not detected_items[item]["detected"]
        ]
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        detection_results: Dict
    ) -> np.ndarray:
        """
        Draw annotations on frame
        
        Args:
            frame: Input image
            detection_results: Results from detect()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw bounding boxes
        for item, data in detection_results["detected_items"].items():
            color = (0, 255, 0) if data["detected"] else (0, 0, 255)
            
            for box_data in data["boxes"]:
                bbox = box_data["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                label = f"{item}: {box_data['confidence']:.2f}"
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        
        # Draw compliance status
        status = "COMPLIANT" if detection_results["compliance_status"] else "NON-COMPLIANT"
        status_color = (0, 255, 0) if detection_results["compliance_status"] else (0, 0, 255)
        
        cv2.rectangle(annotated, (0, 0), (frame.shape[1], 60), status_color, -1)
        cv2.putText(
            annotated, status, (20, 40),
            cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 3
        )
        
        return annotated
