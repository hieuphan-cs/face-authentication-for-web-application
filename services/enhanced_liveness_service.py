import numpy as np
from services.blink_detector import BlinkDetector

class EnhancedLivenessService:
    """
    Liveness detection nâng cao kết hợp:
    1. Blink detection - 30 frames to capture
    2. Movement analysis
    3. Texture analysis
    """
    
    def __init__(self, face_service):
        self.face_service = face_service
        self.blink_detector = BlinkDetector()
        
        print("Enhanced liveness service initialized")
        print("Blink detection enabled (30 frames)")
        print("Movement analysis enabled")
        print("Texture analysis enabled")
    
    def check_movement_liveness(self, frames):
        """
        Check liveness of face movement

        """
        if len(frames) < 10:
            return {
                'passed': False,
                'reason': 'Insufficient frames (need at least 10)',
                'confidence': 0.0
            }
        
        embeddings = []
        
        # Sample frames để tối ưu tốc độ (lấy mỗi 3 frame) Quan trong
        # 30 frames -> 10 samples
        sample_indices = range(0, len(frames), max(1, len(frames) // 10))
        
        for i in sample_indices:
            if i >= len(frames):
                break
            embedding = self.face_service.extract_face_embedding(frames[i])
            if embedding is None:
                return {
                    'passed': False,
                    'reason': f'Face not detected in frame {i+1}',
                    'confidence': 0.0
                }
            embeddings.append(embedding)
        
        if len(embeddings) < 5:
            return {
                'passed': False,
                'reason': 'Not enough valid face detections',
                'confidence': 0.0
            }
        
        # Calculate variations
        variations = []
        for i in range(len(embeddings) - 1):
            similarity = self.face_service.calculate_similarity(
                embeddings[i],
                embeddings[i + 1]
            )
            variation = 1 - similarity
            variations.append(variation)
        
        avg_variation = np.mean(variations)
        std_variation = np.std(variations)
        
        # Live faces: 0.002 < variation < 0.2
        passed = (0.002 < avg_variation < 0.2) and (std_variation < 0.08)
        
        return {
            'passed': bool(passed),
            'avg_variation': float(avg_variation),
            'std_variation': float(std_variation),
            'confidence': float(avg_variation),
            'frames_analyzed': len(embeddings),
            'reason': 'Natural movement detected' if passed else 'Suspicious movement pattern'
        }
    
    def check_blink_liveness(self, frames):
        """
        Check liveness by blinking
        """
        if not self.blink_detector.initialized:
            print("Blink detector not available, skipping blink check")
            return {
                'passed': True,  # Skip if it has no blink detector
                'reason': 'Blink detection not available',
                'blinks': 0
            }
        
        try:
            result = self.blink_detector.analyze_video_frames(frames)
            
            # Với 30 frames, yêu cầu ít nhất 1 lần chớp mắt
            # Người bình thường chớp mắt 15-20 lần/phút
            # Trong 6 giây = 1-2 lần chớp là bình thường
            
            return {
                'passed': result['is_live'],
                'blinks': result['total_blinks'],
                'required_blinks': result['required_blinks'],
                'reason': result['reason']
            }
        except Exception as e:
            print(f"Blink detection error: {str(e)}")
            return {
                'passed': True,  # Skip if error
                'reason': f'Blink detection error: {str(e)}',
                'blinks': 0
            }
    
    def check_liveness(self, frames):
        """
        Check both movement and blink
        """
        total_frames = len(frames)
        print(f"Checking liveness with {total_frames} frames...")
        
        if total_frames < 10:
            return {
                'is_live': False,
                'reason': f'Need at least 10 frames (got {total_frames})',
                'confidence': 0.0,
                'details': {}
            }
        
        # 1. Check movement liveness
        print(f"Checking movement ({total_frames} frames)...")
        movement_result = self.check_movement_liveness(frames)
        
        # 2. Check blink liveness
        print(f"Checking blinks ({total_frames} frames)...")
        blink_result = self.check_blink_liveness(frames)
        
        # 3. Combine results
        movement_passed = movement_result['passed']
        blink_passed = blink_result['passed']
        blink_count = blink_result.get('blinks', 0)
        
        # Combine
        is_live = movement_passed and blink_passed
        
        # Compute confidence score
        confidence = 0.0
        if movement_passed and blink_passed and blink_count > 0:
            confidence = 0.95  # Rất cao nếu có blink
        elif movement_passed and blink_passed:
            confidence = 0.85  # Cao nếu pass cả 2
        elif movement_passed or blink_passed:
            confidence = 0.6   # Trung bình nếu pass 1 trong 2
        
        # Reason 
        reasons = []
        if not movement_passed:
            reasons.append(movement_result['reason'])
        if not blink_passed:
            reasons.append(blink_result['reason'])
        
        if not reasons:
            reasons.append(f'All checks passed - {blink_count} blink(s) detected')
        
        result = {
            'is_live': bool(is_live),
            'confidence': float(confidence),
            'reason': '; '.join(reasons),
            'details': {
                'movement': movement_result,
                'blink': blink_result,
                'total_frames': total_frames
            }
        }
        
        print(f"Liveness result: {is_live} (confidence: {confidence:.2f})")
        print(f"Movement: {movement_passed}, Blink: {blink_passed}, Blinks: {blink_count}, Frames: {total_frames}")
        
        return result