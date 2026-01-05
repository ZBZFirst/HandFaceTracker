import cv2
import numpy as np
import time
import face_animations
import sys

print("="*70)
print("MEDIAPIPE FACE LANDMARK DETECTION SYSTEM")
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
# MEDIAPIPE DETECTOR IMPLEMENTATION
# ============================================================================

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
        
        # Facial feature groups for MediaPipe 478-point model
        self.feature_groups = {
            'face_outline': [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454],
            'forehead': [68, 69, 71, 104, 108, 139, 151, 298, 299, 301, 333, 337, 368],
            'left_eyebrow': [46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 156],
            'right_eyebrow': [282, 283, 285, 293, 295, 296, 300, 334, 336, 383],
            'eyebrow_bridge': [8, 9],
            'left_iris': [468, 469, 470, 471, 472],
            'right_iris': [473, 474, 475, 476, 477],
            'nose_bridge': [5, 6, 168, 195, 197],
            'nose_tip': [4],
            'left_cheek_peak': [50],
            'right_cheek_peak': [280],
            'left_eye': [7, 33, 130, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 243, 246, 470],
            'right_eye': [249, 263, 359, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 463, 466, 475],
            'left_eyesocket': [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 56, 110, 112, 113, 124, 143, 189, 190, 193, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 243, 244, 247],
            'right_eyesocket': [252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 265, 276, 286, 339, 341, 342, 353, 398, 413, 414, 417, 441, 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453, 464, 467],
            'left_cheek': [34, 36, 47, 100, 101, 111, 114, 116, 117, 118, 119, 120, 121, 123, 126, 128, 129, 135, 137, 138, 142, 147, 177, 187, 188, 192, 203, 205, 206, 207, 213, 214, 215, 216, 227, 245],
            'right_cheek': [264, 266, 277, 329, 330, 340, 343, 345, 346, 347, 348, 349, 350, 352, 355, 357, 364, 366, 367, 371, 372, 376, 394, 401, 411, 412, 416, 423, 425, 426, 427, 430, 432, 433, 434, 435, 436, 447, 465],
            'left_nose': [3, 45, 48, 49, 51, 102, 115, 122, 131, 134, 174, 196, 198, 209, 217, 220, 236],
            'right_nose': [248, 275, 279, 281, 351, 360, 363, 399, 419, 420, 429, 437, 440, 456],
            'left_nostril': [20, 44, 59, 60, 64, 75, 79, 98, 125, 141, 166, 218, 219, 235, 237, 238, 239, 240, 241, 242],
            'right_nostril': [250, 274, 278, 289, 290, 294, 305, 309, 326, 327, 328, 331, 344, 354, 358, 370, 392, 438, 439, 455, 457, 458, 459, 460, 461, 462],
            'nose_septum': [1, 2, 19, 94],
            'upper_lip': [43, 57, 60, 92, 97, 99, 164, 165, 167, 186, 202, 212, 287, 322, 391, 393, 410, 422],
            'lower_lip': [18, 32, 83, 106, 140, 169, 170, 171, 175, 182, 194, 199, 200, 201, 204, 208, 210, 211, 262, 273, 313, 335, 369, 395, 396, 406, 418, 421, 424, 428, 431],
            'mouth': [0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 375, 402, 403, 404, 405, 407, 408, 409, 415],
        }
        
        # Group metadata for display
        self.group_metadata = {
            'face_outline': {"display_name": "Face Outline", "description": "Outer contour of the face"},
            'forehead': {"display_name": "Forehead", "description": "Forehead region"},
            'left_eyebrow': {"display_name": "Left Eyebrow", "description": "Left eyebrow points"},
            'right_eyebrow': {"display_name": "Right Eyebrow", "description": "Right eyebrow points"},
            'eyebrow_bridge': {"display_name": "Eyebrow Bridge", "description": "Between eyebrows"},
            'left_iris': {"display_name": "Left Iris", "description": "Left iris points"},
            'right_iris': {"display_name": "Right Iris", "description": "Right iris points"},
            'nose_bridge': {"display_name": "Nose Bridge", "description": "Bridge of the nose"},
            'nose_tip': {"display_name": "Nose Tip", "description": "Tip of the nose"},
            'left_cheek_peak': {"display_name": "Left Cheek Peak", "description": "Highest point of left cheek"},
            'right_cheek_peak': {"display_name": "Right Cheek Peak", "description": "Highest point of right cheek"},
            'left_eye': {"display_name": "Left Eye", "description": "Left eye including iris"},
            'right_eye': {"display_name": "Right Eye", "description": "Right eye including iris"},
            'left_eyesocket': {"display_name": "Left Eye Socket", "description": "Bone structure around left eye"},
            'right_eyesocket': {"display_name": "Right Eye Socket", "description": "Bone structure around right eye"},
            'left_cheek': {"display_name": "Left Cheek", "description": "Left cheek area"},
            'right_cheek': {"display_name": "Right Cheek", "description": "Right cheek area"},
            'left_nose': {"display_name": "Left Nose", "description": "Left side of nose"},
            'right_nose': {"display_name": "Right Nose", "description": "Right side of nose"},
            'left_nostril': {"display_name": "Left Nostril", "description": "Left nostril area"},
            'right_nostril': {"display_name": "Right Nostril", "description": "Right nostril area"},
            'nose_septum': {"display_name": "Nose Septum", "description": "Center line of nose"},
            'upper_lip': {"display_name": "Upper Lip", "description": "Upper lip area"},
            'lower_lip': {"display_name": "Lower Lip", "description": "Lower lip area"},
            'mouth': {"display_name": "Mouth", "description": "Mouth contours and interior"},
        }

        # Initialize feature colors for all groups (start with distinct colors)
        self.feature_colors = {}
        color_palette = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 255, 255) # White
        ]
        
        # Assign colors to groups
        for i, group_name in enumerate(self.feature_groups.keys()):
            self.feature_colors[group_name] = color_palette[i % len(color_palette)]
        
        self.selected_group = None  # Which group to modify
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
            
            # Display group information
            print(f"  [MediaPipe] Loaded {len(self.feature_groups)} facial feature groups")
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
    
    def get_group_info(self, group_name=None):
        """Get information about a specific group or all groups"""
        if group_name:
            if group_name in self.feature_groups:
                return {
                    'name': group_name,
                    'display_name': self.group_metadata.get(group_name, {}).get('display_name', group_name),
                    'description': self.group_metadata.get(group_name, {}).get('description', 'No description'),
                    'indices': self.feature_groups[group_name],
                    'point_count': len(self.feature_groups[group_name]),
                    'color': self.feature_colors.get(group_name, (0, 255, 0))
                }
            return None
        else:
            # Return info for all groups
            return [self.get_group_info(g) for g in self.feature_groups.keys()]

    def select_group(self, group_name):
        """Select which group to modify"""
        if group_name in self.feature_groups:
            self.selected_group = group_name
            # Find the index of this group in the group list
            if group_name in self.group_list:
                self.group_index = self.group_list.index(group_name)
            return True
        return False
    
    def select_next_group(self):
        """Select the next group in sequence"""
        if not self.group_list:
            return None
        
        self.group_index = (self.group_index + 1) % len(self.group_list)
        self.selected_group = self.group_list[self.group_index]
        return self.selected_group
    
    def select_previous_group(self):
        """Select the previous group in sequence"""
        if not self.group_list:
            return None
        
        self.group_index = (self.group_index - 1) % len(self.group_list)
        self.selected_group = self.group_list[self.group_index]
        return self.selected_group
    
    def get_current_group(self):
        """Get currently selected group"""
        return self.selected_group
    
    def get_current_group_info(self):
        """Get info about currently selected group"""
        if self.selected_group and self.selected_group in self.feature_groups:
            meta = self.group_metadata.get(self.selected_group, {})
            return {
                'name': self.selected_group,
                'display_name': meta.get('display_name', self.selected_group),
                'description': meta.get('description', 'No description'),
                'indices': len(self.feature_groups[self.selected_group]),
                'color': self.feature_colors.get(self.selected_group, (0, 255, 0))
            }
        return None
    
    def cycle_group_color(self, group_name=None):
        """Cycle color for a specific facial feature group (R→G→B→R...)
        If no group specified, uses the currently selected group"""
        target_group = group_name or self.selected_group
        
        if not target_group:
            return None
        
        if target_group not in self.feature_colors:
            self.feature_colors[target_group] = (0, 255, 0)  # Start with green
        
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
            current_idx = colors.index(self.feature_colors[target_group])
            next_idx = (current_idx + 1) % len(colors)
        except ValueError:
            next_idx = 0
        
        self.feature_colors[target_group] = colors[next_idx]
        return colors[next_idx]
    
    def reset_group_selection(self):
        """Reset to no specific group selected"""
        self.selected_group = None
        self.group_index = 0

# ============================================================================
# MAIN FACE DETECTION SYSTEM
# ============================================================================

class FaceDetectionSystem:
    def __init__(self):
        self.cap = None
        self.detectors = []
        self.current_detector_index = 0
        self.is_running = False

        # Visual toggles
        self.show_red = True
        self.show_green = True
        self.show_blue = True
        self.show_landmarks = True
        self.show_connections = True
        self.show_faces = True
        self.show_text = True
        self.show_group_info = False  # New: toggle for group info display

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

        # Only MediaPipe detector
        detector_classes = [MediaPipeDetector]

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
            print("\n❌ ERROR: MediaPipe detector not available!")
            print("Please install MediaPipe and download the model:")
            print("  pip install mediapipe")
            print("  Download face_landmarker.task from:")
            print("  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
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
        print("MEDIAPIPE FACE DETECTION SYSTEM - DUAL FEED")
        print("="*70)

        # Create windows
        cv2.namedWindow("RAW Camera Feed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("PROCESSED Feed", cv2.WINDOW_NORMAL)

        # Position windows
        cv2.moveWindow("RAW Camera Feed", 100, 100)
        cv2.moveWindow("PROCESSED Feed", 750, 100)

        # Display detector info
        print("\n📋 AVAILABLE DETECTOR:")
        detector = self.get_current_detector()
        if detector:
            info = detector.get_info()
            print(f"  → {info['name']} ({info['landmarks']} landmarks)")
            print(f"  Description: {info['description']}")
            
            # Display group information
            print(f"\n🎯 FACIAL FEATURE GROUPS:")
            groups_info = detector.get_group_info()
            if groups_info:
                for group_info in groups_info:
                    if group_info:
                        print(f"  - {group_info['display_name']}: {group_info['point_count']} points")

        print("\n🎮 CONTROL PANEL:")
        print("  GROUP SELECTION:")
        print("    [n] - Select NEXT feature group")
        print("    [p] - Select PREVIOUS feature group")
        print("    [a] - Select ALL groups (default)")
        print("    [g] - Cycle colors for selected group")
        print("    [i] - Show/hide group info overlay")
        
        print("\n  ANIMATION CONTROLS:")
        print("    [L] - Toggle wireframe visibility (with fade)")
        print("    [S] - Cycle animation styles (FADE/WAVE/NONE)")
        print("    [+/-] - Increase/decrease animation speed")
        
        print("\n  VISUALIZATION:")
        print("    [r/g/b] - Toggle camera color channels")
        print("    [f] - Toggle face rectangles")
        print("    [t] - Toggle text overlay")
        
        print("\n  SYSTEM:")
        print("    [q] - Quit program")
        print("    [I] - Show detailed detector info")
        print("    [h] - Show this help screen")

        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        fps = 0
        detection_time = 0
        last_help_time = time.time()

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
                    cv2.putText(raw_display, "Press [h] for help | [q] to quit",
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
                            info_y += 25

                            # Add current group info if a group is selected
                            current_group_info = detector.get_current_group_info()
                            if current_group_info and self.show_group_info:
                                group_y = info_y
                                cv2.putText(processed_display, f"Selected: {current_group_info.get('display_name', 'N/A')}",
                                          (10, group_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_group_info['color'], 2)
                                group_y += 25
                                
                                color = current_group_info['color']
                                cv2.putText(processed_display, f"Color: R:{color[0]} G:{color[1]} B:{color[2]}",
                                          (10, group_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_group_info['color'], 1)
                                group_y += 20
                                
                                cv2.putText(processed_display, f"Points: {current_group_info['indices']}",
                                          (10, group_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_group_info['color'], 1)

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

                            # Add group info toggle status
                            group_status = "ON" if self.show_group_info else "OFF"
                            cv2.putText(processed_display, f"Group Info: {group_status}",
                                      (width-150, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                            # Add instructions
                            cv2.putText(processed_display, f"Press [h] for help | [q] to quit",
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
                    cv2.putText(processed_display, "Press [h] for help",
                              (width//2-80, height//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                # ============================================================
                # DISPLAY WINDOWS
                # ============================================================
                if self.show_raw_feed:
                    cv2.imshow("RAW Camera Feed", raw_display)

                if self.show_processed_feed:
                    cv2.imshow("PROCESSED Feed", processed_display)

                # ============================================================
                # HANDLE KEYBOARD INPUT - WITH GROUP SELECTION
                # ============================================================
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n✅ Quitting...")
                    self.is_running = False
                    break

                # Group selection controls
                elif key == ord('n'):  # Select next group
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'select_next_group'):
                        next_group = detector.select_next_group()
                        group_info = detector.get_current_group_info()
                        if group_info:
                            color = group_info['color']
                            print(f"  Selected: {group_info['display_name']} ({group_info['indices']} points)")
                            print(f"    Color: R:{color[0]} G:{color[1]} B:{color[2]}")

                elif key == ord('p'):  # Select previous group
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'select_previous_group'):
                        prev_group = detector.select_previous_group()
                        group_info = detector.get_current_group_info()
                        if group_info:
                            color = group_info['color']
                            print(f"  Selected: {group_info['display_name']} ({group_info['indices']} points)")
                            print(f"    Color: R:{color[0]} G:{color[1]} B:{color[2]}")

                elif key == ord('a'):  # Select all groups (reset to default)
                    detector = self.get_current_detector()
                    if detector:
                        if hasattr(detector, 'reset_group_selection'):
                            detector.reset_group_selection()
                        print(f"  Selected: ALL FEATURE GROUPS")

                elif key == ord('c'):  # Cycle color for selected group
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'cycle_group_color'):
                        new_color = detector.cycle_group_color()
                        if new_color:
                            current_group = detector.get_current_group()
                            if current_group:
                                group_info = detector.get_group_info(current_group)
                                r, g, b = new_color
                                print(f"  {group_info['display_name']} color: R:{r} G:{g} B:{b}")
                            else:
                                r, g, b = new_color
                                print(f"  ALL groups color: R:{r} G:{g} B:{b}")

                elif key == ord('i'):  # Toggle group info display
                    self.show_group_info = not self.show_group_info
                    status = "ON" if self.show_group_info else "OFF"
                    print(f"  Group info overlay: {status}")

                # Animation controls
                elif key == ord('l'):  # Toggle wireframe visibility
                    # Toggle landmarks visibility
                    self.show_landmarks = not self.show_landmarks
                    
                    # Get current detector
                    detector = self.get_current_detector()
                    
                    if detector and hasattr(detector, 'animator'):
                        if self.show_landmarks:
                            # When turning ON: reset animation and enable it
                            detector.animator.force_reset()
                            detector.animator.config.enabled = True
                            
                            # Also reset wave start time for wave animation
                            if detector.animator.config.animation_style == "wave":
                                detector.animator.wave_start_time = time.time()
                            
                            print(f"  Wireframe: ON | Animation: RESET")
                        else:
                            # When turning OFF: just update status
                            print(f"  Wireframe: OFF")

                elif key == ord('s'):  # Cycle animation styles
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'set_animation_style'):
                        styles = ["fade", "wave", "none"]
                        current = detector.animator.config.animation_style
                        next_index = (styles.index(current) + 1) % len(styles) if current in styles else 0
                        detector.set_animation_style(styles[next_index])
                        print(f"  Animation style: {styles[next_index].upper()}")

                elif key == ord('+'):  # Increase animation speed
                    detector = self.get_current_detector()
                    if detector and hasattr(detector, 'animator'):
                        new_speed = min(1.0, detector.animator.config.speed + 0.05)
                        detector.animator.set_speed(new_speed)
                        print(f"  Animation speed: {new_speed:.2f}")

                elif key == ord('-'):  # Decrease animation speed
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

                # Show detailed detector info
                elif key == ord('I'):
                    detector = self.get_current_detector()
                    if detector:
                        info = detector.get_info()
                        print(f"\n📊 DETECTOR INFO:")
                        print(f"    Name: {info['name']}")
                        print(f"    Landmarks: {info['landmarks']}")
                        print(f"    Available: {info['available']}")
                        print(f"    Description: {info['description']}")
                        
                        # Show current group info
                        current_group = detector.get_current_group()
                        if current_group:
                            group_info = detector.get_group_info(current_group)
                            if group_info:
                                print(f"\n🎯 CURRENT GROUP:")
                                print(f"    Name: {group_info['display_name']}")
                                print(f"    Points: {group_info['point_count']}")
                                color = group_info['color']
                                print(f"    Color: R:{color[0]} G:{color[1]} B:{color[2]}")
                                print(f"    Description: {group_info['description']}")
                        
                        # Show animation status
                        if hasattr(detector, 'animator'):
                            anim_status = detector.get_animation_status()
                            if anim_status:
                                print(f"    Animation: {anim_status}")

                # Show help screen
                elif key == ord('h'):
                    # Show help every 2 seconds to prevent spamming
                    if current_time - last_help_time > 2:
                        print("\n" + "="*70)
                        print("HELP SCREEN")
                        print("="*70)
                        print("  GROUP CONTROLS:")
                        print("    n - Next feature group")
                        print("    p - Previous feature group")
                        print("    a - Select all groups")
                        print("    c - Cycle colors for selected group")
                        print("    i - Toggle group info overlay")
                        print("")
                        print("  ANIMATION CONTROLS:")
                        print("    L - Toggle wireframe visibility")
                        print("    S - Cycle animation styles")
                        print("    +/- - Adjust animation speed")
                        print("")
                        print("  VISUALIZATION:")
                        print("    r/g/b - Toggle color channels")
                        print("    f - Toggle face rectangles")
                        print("    t - Toggle text overlay")
                        print("")
                        print("  SYSTEM:")
                        print("    I - Show detailed info")
                        print("    q - Quit")
                        print("="*70)
                        last_help_time = current_time

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