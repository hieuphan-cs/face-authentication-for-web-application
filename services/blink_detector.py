import cv2
import numpy as np
from scipy.spatial import distance as dist
import dlib

class BlinkDetector:
    """  
    Blink detection using EAR - Eye Aspect Ratio
    """
    def __init__(self, ear_threshold=0.3, consec_frames=1):
        """
        Docstring for __init__
        
        :param self: Description
        :param ear_threshold: Description
        :param consec_frames: Description
        """
        self.EAR_THRESHOLD = ear_threshold
        self.CONSEC_FRAMES = consec_frames

        self.blink_counter = 0
        self.frame_counter = 0

        # load face detector and landmark predictor
        try:
            # use dlib to detect facial landmarks
            import dlib

            # download shape predictor nếu chưa có
            predictor_path = "models/shape_predictor_68_face_landmarks.dat"
            import os
            if not os.path.exists(predictor_path):
                print("Downloading facial landmark detector...")
                self._download_shape_predictor(predictor_path)

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)

            # eye landmarks indices (68-point model)
            self.LEFT_EYE_INDICES = list(range(42, 48)) # left eye: 42-47
            self.RIGHT_EYE_INDICES = list(range(36, 42)) # right eye: 36-42

            self.initialized = True
            print("Blink detector initialized with dlib")

        except ImportError:
            print("dlib not available, using alternative method")
            self.initialized = False

    def _download_shape_predictor(self, save_path):
        """
        Download shape predictor model
        """
        import urllib.request
        import bz2
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = save_path + ".bz2"

        print("Downloading from dlib.net...")
        urllib.request.urlretrieve(url, compressed_path)

        print("Extracting...")
        with bz2.BZ2File(compressed_path, 'rb') as fr, open(save_path, 'wb') as fw:
            fw.write(fr.read())

        os.remove(compressed_path)
        print("Downloaded facial landmark detector")

    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        # vertical distances 
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

        # horizontal distances
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

        ear = (A + B) / (2.0 * C)

        return ear
    
    def detect_blink_in_frame(self, frame):
        """
        Blink detection in each frame

        Returns:
            dict: {
                'blink_detected': bool,
                'total_blinks': int,
                'ear_left': float,
                'ear_right': float,
                'avg_ear': float
            }
        """

        if not self.initialized:
            return {
                'blink_detected': False,
                'total_blinks': 0,
                'error': 'Detector not initialized' 
            }
        
        # convert to grayscale
        if len(frame.shape) == 3:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            grayscale = frame

        # detect faces
        faces = self.detector(grayscale, 0)

        if len(faces) == 0:
            return {
                'blink_detected': False,
                'total_blinks': self.blink_counter,
                'error': 'No face detected'
            }

        # get first face
        face = faces[0]

        # detect landmarks
        landmarks = self.predictor(grayscale, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # extract eye landmarks
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]

        # calculate EAR for each eye
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        print(f"[BLINK DEBUG] EAR={avg_ear:.3f}, closed={avg_ear < self.EAR_THRESHOLD}")

        # check if eyes are closed
        blink_detected = False
        
        if avg_ear < self.EAR_THRESHOLD:
            self.frame_counter += 1
        else:
            # eyes were closed for enough consecutive frames
            if self.frame_counter >= self.CONSEC_FRAMES:
                self.blink_counter += 1
                blink_detected = True

            self.frame_counter = 0
        
        return {
            'blink_detected': blink_detected,
            'total_blinks': self.blink_counter,
            'ear_left': float(left_ear),
            'ear_right': float(right_ear),
            'avg_ear': float(avg_ear),
            'eyes_closed': avg_ear < self.EAR_THRESHOLD
        }
    
    def reset(self):
        self.blink_counter = 0
        self.frame_counter = 0

    def analyze_video_frames(self, frames):
        """  
        Analyze frames to detect blinks
        """
        self.reset()

        import io
        import base64
        from PIL import Image

        results = []

        for i, frame_data in enumerate(frames):
            if isinstance(frame_data, str):
                if 'base64,' in frame_data:
                    frame_data = frame_data.split('base64,')[1]
                
                image_data = base64.b64decode(frame_data)
                image = Image.open(io.BytesIO(image_data))
                frame = np.array(image)

            else:
                frame = frame_data
        
            # detect blinks
            result = self.detect_blink_in_frame(frame=frame)
            result['frame_idx'] = i
            results.append(result)

        total_blinks = self.blink_counter
        has_blinks = total_blinks > 0

        return {
            'has_blinks': has_blinks,
            'total_blinks': total_blinks,
            'required_blinks': 1,
            'is_live': has_blinks,
            'frame_results': results,
            'reason': f'Detected {total_blinks} blink(s)' if has_blinks else 'No blinks detected - possible photo/video'
        }