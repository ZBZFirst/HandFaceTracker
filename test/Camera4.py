import cv2
import numpy as np
import time
import face_animations
import sys
from enum import Enum

print("="*70)
print("MULTI-MODEL FACE DETECTION SYSTEM")
print("="*70)

# ============================================================================
# BASE DETECTOR CLASS
# ============================================================================

class BaseDetector:
    """Base class that all detectors must inherit from"""
    def __init__(self):
        self.name = "Base Detector"
        self.landmark_count = 0
        self.is_available = False
        self.description = "Base detector description"
        self.color = (255, 255, 255)  # Default color (white)

    def initialize(self):
        """Initialize the detector - returns True if successful"""
        self.is_available = False
        return False

    def detect(self, frame):
        """Detect faces and landmarks in frame"""
        return {'faces': [], 'landmarks': []}

    def draw_results(self, frame, results, show_faces=True, show_landmarks=True, show_connections=True):
        """Draw detection results on frame"""
        return frame.copy()

    def get_info(self):
        """Get detector information"""
        return {
            'name': self.name,
            'landmarks': self.landmark_count,
            'available': self.is_available,
            'description': self.description,
            'color': self.color
        }

# ============================================================================
# DETECTOR IMPLEMENTATIONS
# ============================================================================

class HaarCascadeDetector(BaseDetector):
    """Basic Haar Cascade face detector"""
    def __init__(self):
        super().__init__()
        self.name = "Haar Cascade"
        self.landmark_count = 0
        self.description = "Basic rectangle face detection (no landmarks)"
        self.color = (0, 255, 0)  # Green
        self.face_cascade = None

    def initialize(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.is_available = not self.face_cascade.empty()
            return self.is_available
        except Exception as e:
            print(f"  [Haar] Initialization error: {e}")
            self.is_available = False
            return False

    def detect(self, frame):
        if not self.is_available:
            return {'faces': [], 'landmarks': []}

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return {'faces': faces, 'landmarks': []}
        except Exception as e:
            print(f"  [Haar] Detection error: {e}")
            return {'faces': [], 'landmarks': []}

    def draw_results(self, frame, results, show_faces=True, show_landmarks=True, show_connections=True):
        """Draw detection results on frame with visual toggles"""
        output = frame.copy()

        # Only draw faces if enabled
        if show_faces:
            for (x, y, w, h) in results['faces']:
                cv2.rectangle(output, (x, y), (x+w, y+h), self.color, 2)
                cv2.putText(output, "Face", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
        return output

class Dlib68Detector(BaseDetector):
    """Dlib 68-point facial landmark detector"""
    def __init__(self):
        super().__init__()
        self.name = "Dlib 68-Point"
        self.landmark_count = 68
        self.description = "68 precise facial landmarks"
        self.color = (0, 0, 255)  # Red
        self.detector = None
        self.predictor = None

    def initialize(self):
        try:
            import dlib
            # Initialize face detector
            self.detector = dlib.get_frontal_face_detector()

            # Try to load the predictor model
            try:
                self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                self.is_available = True
                print(f"  [Dlib] Loaded from current directory")
            except:
                # Try alternative path
                try:
                    self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
                    self.is_available = True
                    print(f"  [Dlib] Loaded from models/ directory")
                except Exception as e:
                    print(f"  [Dlib] Model file not found. Please download from:")
                    print(f"         http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                    self.is_available = False

            return self.is_available
        except ImportError:
            print(f"  [Dlib] dlib library not installed. Run: pip install dlib")
            self.is_available = False
            return False
        except Exception as e:
            print(f"  [Dlib] Initialization error: {e}")
            self.is_available = False
            return False

    def detect(self, frame):
        if not self.is_available:
            return {'faces': [], 'landmarks': []}

        try:
            import dlib
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dlib_faces = self.detector(gray)

            faces = []
            landmarks_list = []

            for face in dlib_faces:
                # Convert dlib rectangle to (x, y, w, h)
                x, y = face.left(), face.top()
                w, h = face.width(), face.height()
                faces.append((x, y, w, h))

                # Get landmarks
                shape = self.predictor(gray, face)
                landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                landmarks_list.append(landmarks)

            return {'faces': faces, 'landmarks': landmarks_list}
        except Exception as e:
            print(f"  [Dlib] Detection error: {e}")
            return {'faces': [], 'landmarks': []}

    def draw_results(self, frame, results, show_faces=True, show_landmarks=True, show_connections=True):
        """Draw detection results on frame with visual toggles"""
        output = frame.copy()

        # Draw face rectangles only if enabled
        if show_faces:
            for (x, y, w, h) in results['faces']:
                cv2.rectangle(output, (x, y), (x+w, y+h), self.color, 1)

        # Draw landmarks and connections only if enabled
        if show_landmarks:
            for landmarks in results['landmarks']:
                if not landmarks:
                    continue

                # Draw all landmarks as dots
                if show_landmarks:
                    for (x, y) in landmarks:
                        cv2.circle(output, (x, y), 2, (255, 0, 255), -1)  # Magenta dots

                # Draw connections only if enabled
                if show_connections:
                    # Define connections for 68-point model
                    connections = [
                        (0, 16),  # Jawline
                        (17, 21), (22, 26),  # Left/Right eyebrows
                        (36, 41), (42, 47),  # Eyes
                        (27, 30), (30, 35),  # Nose
                        (48, 59), (60, 67)   # Outer/inner lips
                    ]

                    # Draw connections
                    for start, end in connections:
                        for i in range(start, end):
                            cv2.line(output, landmarks[i], landmarks[i+1], (0, 255, 255), 1)

                    # Connect lips
                    cv2.line(output, landmarks[48], landmarks[59], (0, 255, 255), 1)
                    cv2.line(output, landmarks[60], landmarks[67], (0, 255, 255), 1)

        return output

class MediaPipeDetector(BaseDetector):
    """MediaPipe Face Landmarker using the new Tasks API (Python)"""
    def __init__(self):
        super().__init__()
        # Update name and landmark count for the new model (478 landmarks)
        self.name = "MediaPipe FaceLandmarker"
        self.landmark_count = 478  # New model has 478 landmarks
        self.description = "478-point 3D face mesh (Tasks API)"
        self.color = (255, 0, 0)  # Blue
        self.detector = None
        # Store MediaPipe module reference
        self.mp = None
        self.animator = face_animations.create_default_animator()
        
        # YOUR CUSTOM GROUPS FROM THE PICKER TOOL:
        self.feature_groups = {
            'outline': [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454],
            'forehead': [68, 69, 71, 104, 108, 139, 151, 298, 299, 301, 333, 337, 368],
            'eyebrow1': [46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 156],
            'eyebrow2': [282, 283, 285, 293, 295, 296, 300, 334, 336, 383],
            'eyebrowbridge': [8, 9],
            'iris1': [468, 469, 470, 471, 472],
            'iris2': [473, 474, 475, 476, 477],
            'nosebridge': [5, 6, 168, 195, 197],
            'peaknose': [4],
            'zygomaticpeak1': [50],
            'zygomaticpeak2': [280],
            'eyeball1': [7, 33, 130, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 243, 246, 470],
            'eyeball2': [249, 263, 359, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 463, 466, 475],
            'eyesocket1': [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 56, 110, 112, 113, 124, 143, 189, 190, 193, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 243, 244, 247],
            'eyesocket2': [252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 265, 276, 286, 339, 341, 342, 353, 398, 413, 414, 417, 441, 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453, 464, 467],
            'cheek1': [34, 36, 47, 100, 101, 111, 114, 116, 117, 118, 119, 120, 121, 123, 126, 128, 129, 135, 137, 138, 142, 147, 177, 187, 188, 192, 203, 205, 206, 207, 213, 214, 215, 216, 227, 245],
            'cheek2': [264, 266, 277, 329, 330, 340, 343, 345, 346, 347, 348, 349, 350, 352, 355, 357, 364, 366, 367, 371, 372, 376, 394, 401, 411, 412, 416, 423, 425, 426, 427, 430, 432, 433, 434, 435, 436, 447, 465],
            'nose1': [3, 45, 48, 49, 51, 102, 115, 122, 131, 134, 174, 196, 198, 209, 217, 220, 236],
            'nose2': [248, 275, 279, 281, 351, 360, 363, 399, 419, 420, 429, 437, 440, 456],
            'nare1': [20, 44, 59, 60, 64, 75, 79, 98, 125, 141, 166, 218, 219, 235, 237, 238, 239, 240, 241, 242],
            'nare2': [250, 274, 278, 289, 290, 294, 305, 309, 326, 327, 328, 331, 344, 354, 358, 370, 392, 438, 439, 455, 457, 458, 459, 460, 461, 462],
            'naremiddle': [1, 2, 19, 94],
            'abovemouth': [43, 57, 60, 92, 97, 99, 164, 165, 167, 186, 202, 212, 287, 322, 391, 393, 410, 422],
            'undermouth': [18, 32, 83, 106, 140, 169, 170, 171, 175, 182, 194, 199, 200, 201, 204, 208, 210, 211, 262, 273, 313, 335, 369, 395, 396, 406, 418, 421, 424, 428, 431],
            'mouth': [0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 375, 402, 403, 404, 405, 407, 408, 409, 415],
        }

        # Initialize feature colors for all groups (start all as green)
        self.feature_colors = {}
        for group_name in self.feature_groups.keys():
            self.feature_colors[group_name] = (0, 255, 0)  # Green
        
        self.selected_feature = None  # Which feature to modify
        self.group_list = list(self.feature_groups.keys())  # List of all groups for cycling
        self.group_index = 0  # Current group index for cycling

    def initialize(self):
        """Initialize the FaceLandmarker detector"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self.mp = mp  # Store for use in other methods

            # 1. Path to the downloaded model file (CRITICAL STEP)
            model_path = "face_landmarker.task"  # Ensure this file is in your script's folder

            # 2. Create the options for VIDEO mode (for webcam stream)
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )

            # 3. Create the detector
            self.detector = vision.FaceLandmarker.create_from_options(options)
            self.is_available = True
            print(f"  [MediaPipe] FaceLandmarker initialized (Tasks API)")
            return True

        except ImportError:
            print(f"  [MediaPipe] mediapipe library not installed. Run: pip install mediapipe")
            self.is_available = False
            return False
        except Exception as e:
            print(f"  [MediaPipe] Initialization error: {e}")
            self.is_available = False
            return False

    def detect(self, frame):
        """Detect faces and landmarks in frame using the new API"""
        if not self.is_available or self.detector is None:
            return {'faces': [], 'landmarks': []}

        try:
            # Convert BGR frame (from OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe Image object
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)

            # For VIDEO mode, we need to manage a timestamp
            # Using a simple incrementing timestamp (assuming ~30 FPS)
            if not hasattr(self, '_frame_timestamp_ms'):
                self._frame_timestamp_ms = 0
            self._frame_timestamp_ms += 33  # ~30 FPS

            # Run detection
            detection_result = self.detector.detect_for_video(mp_image, self._frame_timestamp_ms)

            faces = []
            landmarks_list = []

            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    h, w, _ = frame.shape
                    landmark_points = []

                    # Convert normalized landmarks to pixel coordinates
                    for landmark in face_landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmark_points.append((x, y))

                    landmarks_list.append(landmark_points)

                    # Create a simple bounding box from the landmarks
                    if landmark_points:
                        xs = [p[0] for p in landmark_points]
                        ys = [p[1] for p in landmark_points]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        width = x_max - x_min
                        height = y_max - y_min
                        faces.append((x_min, y_min, width, height))

            return {'faces': faces, 'landmarks': landmarks_list}

        except Exception as e:
            print(f"  [MediaPipe] Detection error: {e}")
            return {'faces': [], 'landmarks': []}

    def draw_results(self, frame, results, show_faces=True, show_landmarks=True, show_connections=True):
        if not self.is_available:
            return frame.copy()
        
        output = frame.copy()
        
        # Draw face rectangles (unchanged)
        if show_faces:
            for (x, y, w, h) in results['faces']:
                cv2.rectangle(output, (x, y), (x + w, y + h), self.color, 1)
        
        # Draw landmarks WITH GROUP COLORS
        if show_landmarks:
            for landmarks in results['landmarks']:
                # Get animated landmarks from our animator
                animated_landmarks = self.animator.get_animated_landmarks(landmarks)
                
                # Draw each animated landmark with group-specific colors
                for i, ((x, y), base_color) in enumerate(animated_landmarks):
                    # Check if this point belongs to any feature group
                    point_color = base_color  # Start with animator's color
                    
                    for group_name, indices in self.feature_groups.items():
                        if i in indices:
                            # Use group color if it exists
                            if group_name in self.feature_colors:
                                point_color = self.feature_colors[group_name]
                            break
                    
                    cv2.circle(output, (x, y), 2, point_color, -1)
        
        return output
    
    def toggle_animation(self):
        """Toggle animation on/off - call this from main controls"""
        status = self.animator.toggle_enabled()
        return "ON" if status else "OFF"
    
    def set_animation_speed(self, speed):
        """Set animation speed - call this from main controls"""
        self.animator.set_speed(speed)
    
    def set_animation_style(self, style):
        """Set animation style - call this from main controls"""
        self.animator.set_style(style)
    
    def get_animation_status(self):
        """Get animation status summary"""
        return self.animator.get_config_summary()

    def select_feature(self, feature_name):
        """Select which feature to modify"""
        if feature_name in self.feature_groups:
            self.selected_feature = feature_name
            # Find the index of this feature in the group list
            if feature_name in self.group_list:
                self.group_index = self.group_list.index(feature_name)
            return True
        return False
    
    def select_next_group(self):
        """Select the next group in sequence"""
        if not self.group_list:
            return None
        
        self.group_index = (self.group_index + 1) % len(self.group_list)
        self.selected_feature = self.group_list[self.group_index]
        return self.selected_feature
    
    def select_previous_group(self):
        """Select the previous group in sequence"""
        if not self.group_list:
            return None
        
        self.group_index = (self.group_index - 1) % len(self.group_list)
        self.selected_feature = self.group_list[self.group_index]
        return self.selected_feature
    
    def get_current_feature(self):
        """Get currently selected feature"""
        return self.selected_feature
    
    def get_current_group_info(self):
        """Get info about currently selected group"""
        if self.selected_feature and self.selected_feature in self.feature_groups:
            return {
                'name': self.selected_feature,
                'indices': len(self.feature_groups[self.selected_feature]),
                'color': self.feature_colors.get(self.selected_feature, (0, 255, 0))
            }
        return None
    
    def cycle_feature_color(self, feature_name):
        """Cycle color for a specific facial feature (R→G→B→R...)"""
        if feature_name not in self.feature_colors:
            self.feature_colors[feature_name] = (0, 255, 0)  # Start with green
        
        current = self.feature_colors[feature_name]
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 255, 255) # White
        ]
        
        try:
            current_idx = colors.index(current)
            next_idx = (current_idx + 1) % len(colors)
        except ValueError:
            next_idx = 0
        
        self.feature_colors[feature_name] = colors[next_idx]
        return colors[next_idx]
    
    def reset_feature_selection(self):
        """Reset to no specific feature selected"""
        self.selected_feature = None
        self.group_index = 0

class OpenCVLBFDetector(BaseDetector):
    """OpenCV's LBF facial landmark detector"""
    def __init__(self):
        super().__init__()
        self.name = "OpenCV LBF"
        self.landmark_count = 68
        self.description = "OpenCV's LBF model (68 landmarks)"
        self.color = (0, 255, 255)  # Yellow
        self.facemark = None
        self.face_cascade = None

    def initialize(self):
        try:
            # Check if OpenCV has face module
            if not hasattr(cv2, 'face'):
                print(f"  [OpenCV LBF] OpenCV compiled without face module")
                self.is_available = False
                return False

            # Initialize face detector
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            # Try to create facemark detector
            self.facemark = cv2.face.createFacemarkLBF()

            # Try to load the model
            try:
                self.facemark.loadModel("lbfmodel.yaml")
                self.is_available = True
                print(f"  [OpenCV LBF] Loaded lbfmodel.yaml from current directory")
            except:
                try:
                    self.facemark.loadModel("models/lbfmodel.yaml")
                    self.is_available = True
                    print(f"  [OpenCV LBF] Loaded from models/ directory")
                except Exception as e:
                    print(f"  [OpenCV LBF] Model file not found. Download from:")
                    print(f"         https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml")
                    self.is_available = False

            return self.is_available
        except Exception as e:
            print(f"  [OpenCV LBF] Initialization error: {e}")
            self.is_available = False
            return False

    def detect(self, frame):
        if not self.is_available:
            return {'faces': [], 'landmarks': []}

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            landmarks_list = []

            if len(faces) > 0:
                # Convert faces format for facemark
                faces_for_facemark = []
                for (x, y, w, h) in faces:
                    faces_for_facemark.append([x, y, x+w, y+h])

                # Get landmarks
                ok, landmarks = self.facemark.fit(gray, faces_for_facemark)

                if ok:
                    for landmark_set in landmarks:
                        points = []
                        for point in landmark_set[0]:
                            points.append((int(point[0]), int(point[1])))
                        landmarks_list.append(points)

            return {'faces': faces, 'landmarks': landmarks_list}
        except Exception as e:
            print(f"  [OpenCV LBF] Detection error: {e}")
            return {'faces': [], 'landmarks': []}

    def draw_results(self, frame, results, show_faces=True, show_landmarks=True, show_connections=True):
        """Draw detection results on frame with visual toggles"""
        output = frame.copy()

        # Draw face rectangles only if enabled
        if show_faces:
            for (x, y, w, h) in results['faces']:
                cv2.rectangle(output, (x, y), (x+w, y+h), self.color, 1)

        # Draw landmarks WITH ANIMATION - FIXED VERSION
        if show_landmarks:
            if results['landmarks']:
                # Face detected - animate the landmarks
                for landmarks in results['landmarks']:
                    animated_landmarks = self.animator.get_animated_landmarks(landmarks)
                    
                    # Draw each animated landmark
                    for (x, y), color in animated_landmarks:
                        cv2.circle(output, (x, y), 2, color, -1)
            else:
                # NO face detected - still call animator with empty list for fade-out
                animated_landmarks = self.animator.get_animated_landmarks([])
                
                # Draw any remaining fading landmarks
                for (x, y), color in animated_landmarks:
                    cv2.circle(output, (x, y), 2, color, -1)

        return output

# ============================================================================
# MAIN FACE DETECTION SYSTEM
# ============================================================================

class FaceDetectionSystem:
    def __init__(self):
        self.cap = None
        self.detectors = []
        self.current_detector_index = 0
        self.is_running = False

        # Visual toggles - add these lines
        self.show_red = True
        self.show_green = True
        self.show_blue = True
        self.show_landmarks = True
        self.show_connections = True
        self.show_faces = True
        self.show_text = True

        # Keep these for RAW feed control if needed
        self.show_raw_feed = True
        self.show_processed_feed = True

        # Initialize all detectors
        self.initialize_detectors()

    def initialize_detectors(self):
        """Initialize all available detectors"""
        print("\n" + "="*70)
        print("INITIALIZING DETECTORS")
        print("="*70)

        detector_classes = [
            HaarCascadeDetector,
            Dlib68Detector,
            MediaPipeDetector,
            OpenCVLBFDetector
        ]

        for i, DetectorClass in enumerate(detector_classes):
            print(f"\n[{i+1}] Initializing {DetectorClass.__name__}...")
            detector = DetectorClass()
            if detector.initialize():
                self.detectors.append(detector)
                print(f"    ✓ Available")
            else:
                print(f"    ✗ Not available")

        print(f"\n✅ {len(self.detectors)} of {len(detector_classes)} detectors available")

        if len(self.detectors) == 0:
            print("\n❌ ERROR: No face detectors available!")
            print("Please install at least one of:")
            print("  - OpenCV (always available): pip install opencv-python")
            print("  - Dlib: pip install dlib")
            print("  - MediaPipe: pip install mediapipe")
            sys.exit(1)

    def initialize_camera(self):
        """Initialize the camera"""
        print("\nInitializing camera...")

        try:
            # Try DSHOW backend on Windows first
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                print("  Trying default backend...")
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                print("❌ ERROR: Cannot open camera")
                return False

            # Configure camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"✅ Camera initialized: {width}x{height}")
            return True

        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False

    def get_current_detector(self):
        """Get the currently selected detector"""
        if self.detectors and 0 <= self.current_detector_index < len(self.detectors):
            return self.detectors[self.current_detector_index]
        return None

    def display_feeds(self):
        """Main display loop with dual feeds"""
        print("\n" + "="*70)
        print("FACE DETECTION SYSTEM - DUAL FEED")
        print("="*70)

        # Create windows
        cv2.namedWindow("RAW Camera Feed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("PROCESSED Feed", cv2.WINDOW_NORMAL)

        # Position windows
        cv2.moveWindow("RAW Camera Feed", 100, 100)
        cv2.moveWindow("PROCESSED Feed", 750, 100)

        # Display detector info
        print("\n📋 AVAILABLE DETECTORS:")
        for i, detector in enumerate(self.detectors):
            info = detector.get_info()
            current_marker = " → " if i == self.current_detector_index else "   "
            print(f"{current_marker}[{i+1}] {info['name']} ({info['landmarks']} landmarks)")

        print("\n🎮 VISUALIZATION CONTROLS:")
        print("  [1-4] - Switch detector")
        print("  [e] - Select EYES for color cycling")
        print("  [m] - Select MOUTH for color cycling")
        print("  [n] - Select NOSE for color cycling")
        print("  [a] - Select ALL features (default)")
        print("  [c] - Cycle colors for selected feature")
        print("  [L] - Toggle wireframe visibility (with fade)")
        print("  [S] - Cycle animation styles (FADE/WAVE/NONE)")
        print("  [+/-] - Increase/decrease animation speed")
        print("  [r/g/b] - Toggle camera color channels")
        print("  [f] - Toggle face rectangles")
        print("  [t] - Toggle text overlay")
        print("  [q] - Quit program")
        print("  [i] - Show detector info")

        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        fps = 0
        detection_time = 0

        self.is_running = True

        try:
            while self.is_running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break

                frame_count += 1
                current_time = time.time()

                # Calculate FPS
                if current_time - last_fps_update > 0.5:
                    fps = frame_count / (current_time - last_fps_update)
                    frame_count = 0
                    last_fps_update = current_time

                # Get current detector
                detector = self.get_current_detector()
                detector_info = detector.get_info() if detector else {}

                # ============================================================
                # PROCESS RAW FEED
                # ============================================================
                raw_display = frame.copy()
                height, width = raw_display.shape[:2]

                if self.show_raw_feed:
                    # Add border and title
                    cv2.rectangle(raw_display, (0, 0), (width-1, height-1), (0, 255, 0), 2)
                    cv2.putText(raw_display, "RAW CAMERA FEED", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Add FPS
                    cv2.putText(raw_display, f"FPS: {fps:.1f}", (width-120, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Add detector info
                    if detector:
                        cv2.putText(raw_display, f"Detector: {detector_info.get('name', 'None')}",
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Add instructions
                    cv2.putText(raw_display, "Press [1-4] to switch | [q] to quit",
                              (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # ============================================================
                # PROCESS DETECTOR FEED
                # ============================================================
                if self.show_processed_feed and detector and detector.is_available:
                    try:
                        # Run detection
                        detection_start = time.time()
                        results = detector.detect(frame)
                        detection_time = (time.time() - detection_start) * 1000

                        # Draw results with visual toggles
                        processed_display = frame.copy()  # Start with original frame

                        # Apply color channel filter
                        if not (self.show_red and self.show_green and self.show_blue):
                            # Create a mask for each channel
                            b, g, r = cv2.split(processed_display)
                            if not self.show_blue:
                                b = np.zeros_like(b)
                            if not self.show_green:
                                g = np.zeros_like(g)
                            if not self.show_red:
                                r = np.zeros_like(r)
                            processed_display = cv2.merge([b, g, r])

                        # Draw detector results (pass toggles as parameters)
                        processed_display = detector.draw_results(
                            processed_display,
                            results,
                            show_faces=self.show_faces,
                            show_landmarks=self.show_landmarks,
                            show_connections=self.show_connections
                        )

                        # Add overlay info only if text is enabled
                        if self.show_text:
                            cv2.rectangle(processed_display, (0, 0), (width-1, height-1), detector_info.get('color', (255,255,255)), 2)
                            cv2.putText(processed_display, f"DETECTOR: {detector_info.get('name', 'None')}",
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, detector_info.get('color', (255,255,255)), 2)

                            # Add detection info
                            info_y = 60
                            cv2.putText(processed_display, f"Faces: {len(results['faces'])}",
                                      (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            info_y += 25

                            cv2.putText(processed_display, f"Landmarks: {detector_info.get('landmarks', 0)}",
                                      (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                            info_y += 25

                            cv2.putText(processed_display, f"Time: {detection_time:.1f}ms",
                                      (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

                            # Add color channel status
                            color_status = []
                            if self.show_red: color_status.append("R")
                            if self.show_green: color_status.append("G")
                            if self.show_blue: color_status.append("B")
                            cv2.putText(processed_display, f"Colors: {''.join(color_status) if color_status else 'None'}",
                                      (10, height-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                            # Add visual toggle status
                            visual_status = []
                            if self.show_faces: visual_status.append("F")
                            if self.show_landmarks: visual_status.append("L")
                            if self.show_connections: visual_status.append("C")
                            cv2.putText(processed_display, f"Visuals: {''.join(visual_status) if visual_status else 'None'}",
                                      (10, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                            # Add instructions
                            cv2.putText(processed_display, f"Current: {self.current_detector_index+1}/{len(self.detectors)}",
                                      (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    except Exception as e:
                        # If detector fails, show error
                        processed_display = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(processed_display, "⚠️ DETECTOR ERROR", (width//2-100, height//2-30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(processed_display, f"{str(e)[:50]}...", (width//2-150, height//2+10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        cv2.putText(processed_display, "Check console for details", (width//2-120, height//2+40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    # Show placeholder when no detector or feed disabled
                    processed_display = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(processed_display, "DETECTOR FEED", (width//2-80, height//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                    cv2.putText(processed_display, "Press number keys 1-4 to select detector",
                              (width//2-180, height//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                # ============================================================
                # DISPLAY WINDOWS
                # ============================================================
                if self.show_raw_feed:
                    cv2.imshow("RAW Camera Feed", raw_display)

                if self.show_processed_feed:
                    cv2.imshow("PROCESSED Feed", processed_display)

                # ============================================================
                # HANDLE KEYBOARD INPUT - WITH FEATURE SELECTION
                # ============================================================
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n✅ Quitting...")
                    self.is_running = False
                    break

                # Detector selection
                elif ord('1') <= key <= ord('4'):
                    detector_num = key - ord('1')
                    if detector_num < len(self.detectors):
                        self.current_detector_index = detector_num
                        detector = self.get_current_detector()
                        print(f"\n🔄 Switched to: {detector.name}")

                # Feature selection keys
                elif key == ord('e'):  # Select eyes
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'select_feature'):
                        # For MediaPipe, we need to handle both eyes
                        if detector.name == "MediaPipe FaceLandmarker":
                            detector.selected_feature = 'eyes'
                            # Make sure we have an eyes group
                            if 'eyes' not in detector.feature_groups:
                                detector.feature_groups['eyes'] = (
                                    detector.feature_groups.get('left_eye', []) + 
                                    detector.feature_groups.get('right_eye', [])
                                )
                            if 'eyes' not in detector.feature_colors:
                                detector.feature_colors['eyes'] = (255, 255, 0)  # Cyan
                        else:
                            detector.select_feature('left_eye')
                        print(f"  Selected: EYES - Press 'c' to cycle colors")

                elif key == ord('m'):  # Select mouth/lips
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'select_feature'):
                        detector.select_feature('lips')
                        print(f"  Selected: MOUTH - Press 'c' to cycle colors")

                elif key == ord('n'):  # Select nose
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'select_feature'):
                        detector.select_feature('nose')
                        print(f"  Selected: NOSE - Press 'c' to cycle colors")

                elif key == ord('a'):  # Select all (reset to default)
                    detector = self.get_current_detector()
                    if detector:
                        if hasattr(detector, 'selected_feature'):
                            detector.selected_feature = None
                        print(f"  Selected: ALL FEATURES")

                # Color control - MODIFIED FOR FEATURE SUPPORT
                elif key == ord('c'):
                    detector = self.get_current_detector()
                    if detector:
                        if hasattr(detector, 'selected_feature') and detector.selected_feature:
                            # Cycle color for selected feature
                            colors = [
                                (0, 255, 0),    # Green
                                (255, 0, 0),    # Blue
                                (0, 0, 255),    # Red
                                (255, 255, 0),  # Cyan
                                (255, 0, 255),  # Magenta
                                (0, 255, 255),  # Yellow
                                (255, 255, 255) # White
                            ]
                            
                            # Get current feature color
                            current_color = detector.feature_colors.get(
                                detector.selected_feature, 
                                (0, 255, 0)  # Default to green
                            )
                            
                            # Find next color
                            try:
                                current_index = colors.index(current_color)
                                next_index = (current_index + 1) % len(colors)
                            except ValueError:
                                next_index = 0
                            
                            # Update feature color
                            detector.feature_colors[detector.selected_feature] = colors[next_index]
                            r, g, b = colors[next_index]
                            feature_name = detector.selected_feature.replace('_', ' ').title()
                            print(f"  {feature_name} color: R:{r} G:{g} B:{b}")
                        elif hasattr(detector, 'animator'):
                            # No feature selected, cycle all wireframe (original behavior)
                            colors = [
                                (0, 255, 0),    # Green
                                (255, 0, 0),    # Blue
                                (0, 0, 255),    # Red
                                (255, 255, 0),  # Cyan
                                (255, 0, 255),  # Magenta
                                (0, 255, 255),  # Yellow
                                (255, 255, 255) # White
                            ]
                            current_color = detector.animator.config.wireframe_color
                            
                            # Find next color
                            try:
                                current_index = colors.index(current_color)
                                next_index = (current_index + 1) % len(colors)
                            except ValueError:
                                next_index = 0
                            
                            detector.animator.set_color(*colors[next_index])
                            r, g, b = colors[next_index]
                            print(f"  ALL features color: R:{r} G:{g} B:{b}")

                # Visual toggles - L key now controls ALL wireframe visibility WITH RESET
                elif key == ord('l'):
                    # Toggle landmarks visibility
                    self.show_landmarks = not self.show_landmarks
                    
                    # Get current detector
                    detector = self.get_current_detector()
                    
                    if detector and hasattr(detector, 'animator'):
                        if self.show_landmarks:
                            # When turning ON: reset animation and enable it
                            detector.animator.force_reset()  # NEW: Reset animation state
                            detector.animator.config.enabled = True
                            
                            # Also reset wave start time for wave animation
                            if detector.animator.config.animation_style == "wave":
                                detector.animator.wave_start_time = time.time()
                            
                            print(f"  Wireframe: ON | Animation: RESET")
                        else:
                            # When turning OFF: just update status
                            print(f"  Wireframe: OFF")

                # Cycle animation styles (only FADE, WAVE, NONE)
                elif key == ord('s'):
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'set_animation_style'):
                        styles = ["fade", "wave", "none"]
                        current = detector.animator.config.animation_style
                        next_index = (styles.index(current) + 1) % len(styles) if current in styles else 0
                        detector.set_animation_style(styles[next_index])
                        print(f"  Animation style: {styles[next_index].upper()}")

                # Animation speed control
                elif key == ord('+'):  # Increase speed
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'animator'):
                        new_speed = min(1.0, detector.animator.config.speed + 0.05)
                        detector.animator.set_speed(new_speed)
                        print(f"  Animation speed: {new_speed:.2f}")

                elif key == ord('-'):  # Decrease speed
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'animator'):
                        new_speed = max(0.01, detector.animator.config.speed - 0.05)
                        detector.animator.set_speed(new_speed)
                        print(f"  Animation speed: {new_speed:.2f}")

                # Other toggles
                elif key == ord('r'):
                    self.show_red = not self.show_red
                    status = "ON" if self.show_red else "OFF"
                    print(f"  RED channel: {status}")

                elif key == ord('g'):
                    self.show_green = not self.show_green
                    status = "ON" if self.show_green else "OFF"
                    print(f"  GREEN channel: {status}")

                elif key == ord('b'):
                    self.show_blue = not self.show_blue
                    status = "ON" if self.show_blue else "OFF"
                    print(f"  BLUE channel: {status}")

                elif key == ord('f'):
                    self.show_faces = not self.show_faces
                    status = "ON" if self.show_faces else "OFF"
                    print(f"  Face rectangles: {status}")

                elif key == ord('t'):
                    self.show_text = not self.show_text
                    status = "ON" if self.show_text else "OFF"
                    print(f"  Text overlay: {status}")

                # Show detector info
                elif key == ord('i'):
                    detector = self.get_current_detector()
                    if detector:
                        info = detector.get_info()
                        print(f"\n📊 DETECTOR INFO:")
                        print(f"    Name: {info['name']}")
                        print(f"    Landmarks: {info['landmarks']}")
                        print(f"    Available: {info['available']}")
                        print(f"    Description: {info['description']}")
                        
                        # Show feature info if available
                        if hasattr(detector, 'selected_feature'):
                            if detector.selected_feature:
                                print(f"    Selected feature: {detector.selected_feature}")
                                if hasattr(detector, 'feature_colors'):
                                    color = detector.feature_colors.get(
                                        detector.selected_feature, 
                                        (0, 0, 0)
                                    )
                                    r, g, b = color
                                    print(f"    Feature color: R:{r} G:{g} B:{b}")
                            else:
                                print(f"    Selected feature: ALL")
                        
                        # Show animation status
                        if hasattr(detector, 'animator'):
                            anim_status = detector.get_animation_status()
                            if anim_status:
                                print(f"    Animation: {anim_status}")

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Cleanup complete")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function"""

    # Create the system
    system = FaceDetectionSystem()

    # Initialize camera
    if not system.initialize_camera():
        print("❌ Failed to initialize camera")
        return

    # Display feeds
    system.display_feeds()

if __name__ == "__main__":
    main()
