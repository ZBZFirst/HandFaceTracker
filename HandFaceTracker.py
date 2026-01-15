import cv2
import numpy as np
import mediapipe as mp
import math
import json
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from obswebsocket import obsws, requests

OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "PeppaPig53**"

# Try to import landmark groups from the external file
try:
    from landmark_groups import feature_groups, face_connections
    LANDMARK_GROUPS_AVAILABLE = True
    print("✅ Landmark groups loaded successfully")
except ImportError:
    LANDMARK_GROUPS_AVAILABLE = False
    print("❌ landmark_groups.py not found - face connections will not be drawn")

# ============================================================================
# GESTURE MAPPING SYSTEM
# ============================================================================

class GestureMapping:
    """Maps gestures to actions, separate from radial menu"""

    def __init__(self, tracker):
        self.tracker = tracker
        self.mappings = {}  # gesture_name -> action_function
        self.config_file = "gesture_mappings.json"

        # Initialize with default gesture mappings (but delay action functions)
        self._init_default_mappings()
        self.load_config()

        # Create a reverse lookup for display purposes
        self.action_to_gestures = {}
        self._update_reverse_lookup()

    def _init_default_mappings(self):
        """Initialize default gesture to action mappings (without circular references)"""
        # We'll use placeholder functions that will be replaced later
        self.mappings = {
            'Closed_Fist': self._placeholder_action,
            'Open_Palm': self._placeholder_action,
            'Pointing_Up': self._placeholder_action,
            'Thumb_Down': self._placeholder_action,
            'Thumb_Up': self._placeholder_action,
            'Victory': self._placeholder_action,
            'ILoveYou': self._placeholder_action,
        }

    def _placeholder_action(self):
        """Placeholder that will be replaced with real actions"""
        print("  ⚠️ Gesture action not initialized yet")

    def update_actions(self, gesture_controller):
        """Update the action functions after gesture_controller is created"""
        self.mappings = {
            'Closed_Fist': lambda: self.tracker.toggle('show_red', 'RED channel'),
            'Open_Palm': lambda: self.tracker.toggle('show_green', 'GREEN channel'),
            'Pointing_Up': lambda: self.tracker.toggle('show_blue', 'BLUE channel'),
            'Thumb_Down': gesture_controller.action1,
            'Thumb_Up': gesture_controller.action2,
            'Victory': gesture_controller.take_screenshot,
            'ILoveYou': lambda: self.tracker.toggle('show_text', 'Text overlay'),
        }
        # Reload config to override with saved mappings
        self.load_config()

    def get_available_gestures(self):
        """Return list of all gesture names that can be mapped"""
        return [
            'Closed_Fist',
            'Open_Palm',
            'Pointing_Up',
            'Thumb_Down',
            'Thumb_Up',
            'Victory',
            'ILoveYou',
            'None'  # Empty/placeholder
        ]

    def get_available_actions(self):
        """Return list of all available actions that gestures can be mapped to"""
        return [
            ("toggle_red", "Toggle Red", "🔴", lambda: self.tracker.toggle('show_red', 'RED channel')),
            ("toggle_green", "Toggle Green", "🟢", lambda: self.tracker.toggle('show_green', 'GREEN channel')),
            ("toggle_blue", "Toggle Blue", "🔵", lambda: self.tracker.toggle('show_blue', 'BLUE channel')),
            ("toggle_faces", "Toggle Faces", "😀", lambda: self.tracker.toggle('show_faces', 'Face tracking')),
            ("toggle_hands", "Toggle Hands", "✋", lambda: self.tracker.toggle('show_hands', 'Hand tracking')),
            ("screenshot", "Take Screenshot", "📸", self._get_screenshot_action()),
            ("toggle_text", "Toggle Text", "📝", lambda: self.tracker.toggle('show_text', 'Text overlay')),
            ("gesture_menu", "Open Gesture Menu", "⚙️", self._get_gesture_menu_action()),
            ("empty", "No Action", "○", None),
        ]

    def _get_screenshot_action(self):
        """Get screenshot action (handles circular reference)"""
        if hasattr(self.tracker, 'gesture_controller'):
            return self.tracker.gesture_controller.take_screenshot
        return lambda: print("  📸 Screenshot (gesture controller not ready)")

    def _get_gesture_menu_action(self):
        """Get gesture menu action (handles circular reference)"""
        if hasattr(self.tracker, 'open_gesture_mapping_menu'):
            return self.tracker.open_gesture_mapping_menu
        return lambda: print("  ⚙️ Gesture menu (tracker not ready)")

    def map_gesture(self, gesture_name, action_id):
        """Map a gesture to an action"""
        actions_list = self.get_available_actions()
        action_data = None

        # Find the action
        for action in actions_list:  # Changed from actions
            if action[0] == action_id:
                action_data = action
                break

        if action_data:
            self.mappings[gesture_name] = action_data[3]  # The lambda function
            self.save_config()
            self._update_reverse_lookup()
            return True
        return False

    def get_gesture_action(self, gesture_name):
        """Get the action function for a gesture"""
        return self.mappings.get(gesture_name)

    def execute_gesture(self, gesture_name):
        """Execute the action mapped to a gesture"""
        action = self.get_gesture_action(gesture_name)
        if action:
            action()
            return True
        return False

    def _update_reverse_lookup(self):
        """Update the reverse lookup dictionary for display"""
        self.action_to_gestures = {}
        actions_list = self.get_available_actions()

        for gesture, gesture_action_func in self.mappings.items():
            # Find which action this function corresponds to
            for action_id, name, icon, action_func in actions_list:
                if action_func == gesture_action_func:
                    if name not in self.action_to_gestures:
                        self.action_to_gestures[name] = []
                    self.action_to_gestures[name].append(gesture)
                    break

    def save_config(self):
        """Save gesture mappings to file"""
        # Convert functions to their string IDs
        config = {"mappings": {}, "version": "1.0"}

        for gesture, action_func in self.mappings.items():
            # Find the action ID for this function
            for action_id, name, icon, func in self.get_available_actions():
                if func == action_func:
                    config["mappings"][gesture] = action_id
                    break

        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save gesture mappings: {e}")

    def load_config(self):
        """Load gesture mappings from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

                # Load saved mappings
                saved_mappings = config.get("mappings", {})
                for gesture, action_id in saved_mappings.items():
                    # Don't use map_gesture here to avoid recursion
                    # Just update the mapping directly
                    for action in self.get_available_actions():
                        if action[0] == action_id:
                            self.mappings[gesture] = action[3]
                            break

                print(f"✅ Loaded gesture mappings from {self.config_file}")
                self._update_reverse_lookup()
            except Exception as e:
                print(f"❌ Failed to load gesture mappings: {e}")
                # Fall back to defaults (will be updated later)
        else:
            print(f"⚠️  No gesture mappings found. Using default mappings.")

# ============================================================================
# SCROLLABLE DATA DISPLAY WINDOWS
# ============================================================================

class ScrollableDataWindow:
    """Base class for scrollable data display windows"""

    def __init__(self, window_name, window_width=400, window_height=600):
        self.window_name = window_name
        self.window_width = window_width
        self.window_height = window_height

        # Scroll settings
        self.scroll_offset = 0
        self.max_scroll = 100
        self.row_height = 25
        self.col_width = 120

        # Colors
        self.bg_color = (30, 30, 40)
        self.header_color = (60, 60, 100)
        self.data_color = (200, 200, 220)
        self.grid_color = (60, 60, 70)
        self.scrollbar_color = (100, 100, 150)
        self.scrollbar_active_color = (150, 150, 200)

        # Create the window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        # Display data
        self.display_data = []
        self.column_headers = []

    def update_data(self, data):
        """Update the data to display"""
        self.display_data = data
        self.calculate_scroll_limit()

    def calculate_scroll_limit(self):
        """Calculate maximum scroll based on data rows"""
        visible_rows = (self.window_height - 100) // self.row_height
        total_rows = len(self.display_data)
        self.max_scroll = max(0, total_rows - visible_rows)
        self.scroll_offset = min(self.scroll_offset, self.max_scroll)

    def handle_scroll(self, delta):
        """Handle scroll wheel input"""
        self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset + delta))

    def draw_scrollbar(self, canvas):
        """Draw scrollbar on the right side"""
        if self.max_scroll <= 0:
            return

        scrollbar_width = 15
        scrollbar_x = self.window_width - scrollbar_width - 5

        # Calculate scrollbar thumb position and size
        visible_ratio = min(1.0, (self.window_height - 100) / (len(self.display_data) * self.row_height))
        thumb_height = max(20, int((self.window_height - 100) * visible_ratio))

        if self.max_scroll > 0:
            thumb_position = (self.scroll_offset / self.max_scroll) * (self.window_height - 100 - thumb_height)
        else:
            thumb_position = 0

        # Draw scrollbar track
        cv2.rectangle(canvas, (scrollbar_x, 50),
                     (scrollbar_x + scrollbar_width, self.window_height - 50),
                     self.scrollbar_color, -1)

        # Draw scrollbar thumb
        thumb_y = 50 + int(thumb_position)
        cv2.rectangle(canvas, (scrollbar_x, thumb_y),
                     (scrollbar_x + scrollbar_width, thumb_y + thumb_height),
                     self.scrollbar_active_color, -1)

        # Draw scroll indicator
        if self.max_scroll > 0:
            page_info = f"{self.scroll_offset+1}-{min(self.scroll_offset + (self.window_height-100)//self.row_height, len(self.display_data))}/{len(self.display_data)}"
            cv2.putText(canvas, page_info, (scrollbar_x - 100, self.window_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    def draw_table(self, canvas):
        """Draw the data table with current scroll offset"""
        # Draw title
        cv2.putText(canvas, self.window_name, (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw column headers
        y_pos = 60
        for i, header in enumerate(self.column_headers):
            x_pos = 20 + i * self.col_width
            cv2.rectangle(canvas, (x_pos - 5, y_pos - 20),
                         (x_pos + self.col_width - 5, y_pos + 5), self.header_color, -1)
            cv2.putText(canvas, header, (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        y_pos += 10

        # Draw grid lines for headers
        for i in range(len(self.column_headers) + 1):
            x_line = 15 + i * self.col_width
            cv2.line(canvas, (x_line, 40), (x_line, y_pos), self.grid_color, 1)

        # Draw data rows
        start_row = self.scroll_offset
        visible_rows = (self.window_height - y_pos - 30) // self.row_height

        for row_idx in range(start_row, min(start_row + visible_rows, len(self.display_data))):
            row_data = self.display_data[row_idx]
            row_y = y_pos + (row_idx - start_row) * self.row_height

            # Alternate row colors
            row_color = self.data_color if row_idx % 2 == 0 else (180, 180, 200)

            for col_idx, cell_value in enumerate(row_data):
                x_pos = 20 + col_idx * self.col_width
                cv2.putText(canvas, str(cell_value), (x_pos, row_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, row_color, 1, cv2.LINE_AA)

            # Draw horizontal grid line
            cv2.line(canvas, (15, row_y + 5),
                    (15 + len(self.column_headers) * self.col_width, row_y + 5),
                    self.grid_color, 1)

        # Draw vertical grid lines for data
        for i in range(len(self.column_headers) + 1):
            x_line = 15 + i * self.col_width
            cv2.line(canvas, (x_line, y_pos),
                    (x_line, y_pos + visible_rows * self.row_height), self.grid_color, 1)

        # Draw scrollbar
        self.draw_scrollbar(canvas)

        # Draw instructions
        instructions = "Mouse wheel: Scroll | R: Reset scroll | ESC: Close window"
        cv2.putText(canvas, instructions, (20, self.window_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    def update_window(self):
        """Update and redraw the window"""
        canvas = np.full((self.window_height, self.window_width, 3), self.bg_color, dtype=np.uint8)
        self.draw_table(canvas)
        cv2.imshow(self.window_name, canvas)

    def close(self):
        """Close the window"""
        cv2.destroyWindow(self.window_name)


class FaceDataWindow(ScrollableDataWindow):
    """Window for displaying face landmark data"""

    def __init__(self):
        super().__init__("Face Landmarks", window_width=600, window_height=700)
        self.column_headers = ["ID", "Name", "X", "Y", "Z", "Visibility"]

        # Face landmark names (MediaPipe has 478 face landmarks)
        self.face_names = [
            "Lips Upper Outer", "Lips Upper Inner", "Lips Lower Outer", "Lips Lower Inner",
            "Nose Tip", "Nose Bridge", "Left Eye Outer", "Left Eye Inner",
            "Right Eye Outer", "Right Eye Inner", "Left Ear", "Right Ear",
            "Mouth Left Corner", "Mouth Right Corner", "Forehead", "Chin",
            "Left Cheek", "Right Cheek", "Left Eyebrow Outer", "Left Eyebrow Inner",
            "Right Eyebrow Outer", "Right Eyebrow Inner"
        ]

    def update_from_landmarks(self, face_landmarks_list):
        """Convert face landmarks to display data"""
        if not face_landmarks_list:
            self.update_data([])
            return

        display_rows = []

        # Face has 468 landmarks (or 478 with refine_landmarks=True)
        for i in range(min(len(face_landmarks_list), 468)):
            landmark = face_landmarks_list[i]

            # FIXED: Check both that landmark exists AND visibility is not None
            if landmark is None:
                row = [str(i), f"Point_{i}", "N/A", "N/A", "N/A", "N/A"]
            else:
                name = self.face_names[i] if i < len(self.face_names) else f"Point_{i}"

                # FIXED: Check if visibility exists AND is not None
                if hasattr(landmark, 'visibility') and landmark.visibility is not None:
                    visibility = f"{landmark.visibility:.4f}"
                else:
                    visibility = "N/A"

                row = [
                    str(i),
                    name,
                    f"{landmark.x:.4f}",
                    f"{landmark.y:.4f}",
                    f"{landmark.z:.4f}",
                    visibility  # Now safe
                ]
            display_rows.append(row)

        self.update_data(display_rows)


class HandDataWindow(ScrollableDataWindow):
    """Window for displaying hand landmark data"""

    def __init__(self):
        super().__init__("Hand Landmarks", window_width=700, window_height=600)
        self.column_headers = ["ID", "Name", "X", "Y", "Z", "Dist Wrist", "Angle"]

        # Hand landmark names (21 points)
        self.hand_names = [
            "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
            "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
            "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]

    def update_from_landmarks(self, hand_landmarks, gesture=None):
        """Convert hand landmarks to display data"""
        if not hand_landmarks or len(hand_landmarks) < 21:
            self.update_data([])
            return

        display_rows = []
        wrist = hand_landmarks[0]

        for i in range(len(hand_landmarks)):
            landmark = hand_landmarks[i]
            name = self.hand_names[i] if i < len(self.hand_names) else f"Point_{i}"

            # Calculate distance from wrist
            dist_x = landmark.x - wrist.x
            dist_y = landmark.y - wrist.y
            distance = math.sqrt(dist_x**2 + dist_y**2)

            # Calculate angle from wrist (in degrees)
            angle = math.degrees(math.atan2(dist_y, dist_x)) if i > 0 else 0
            angle = (angle + 360) % 360  # Normalize to 0-360

            # Format the row
            row = [
                str(i),
                name,
                f"{landmark.x:.4f}",
                f"{landmark.y:.4f}",
                f"{landmark.z:.4f}",
                f"{distance:.4f}",
                f"{angle:.1f}°"
            ]
            display_rows.append(row)

        # Add gesture info as first row if available
        if gesture:
            display_rows.insert(0, ["GESTURE", gesture, "", "", "", "", ""])

        self.update_data(display_rows)


class MultiHandDataWindow:
    """Window for displaying multiple hands data with touch detection"""

    def __init__(self):
        self.window_name = "Multi-Hand Data"
        self.window_width = 800
        self.window_height = 500

        # Colors
        self.bg_color = (30, 30, 40)
        self.left_hand_color = (100, 200, 255)  # Cyan for left
        self.right_hand_color = (255, 150, 100)  # Orange for right
        self.text_color = (220, 220, 220)
        self.touch_color = (100, 255, 100)  # Green for touch
        self.no_touch_color = (255, 100, 100)  # Red for no touch

        # Touch detection settings
        self.touch_threshold = 0.02
        self.finger_tip_indices = [4, 8, 12, 16, 20]
        self.finger_names = {
            4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"
        }

        # Touch tracking with visual effects
        self.current_touches = []  # List of current active touches
        self.touch_history = []    # History for visual effects
        self.last_touch_print = 0
        self.cooldown_duration = 6.0  # Seconds before points can touch again

        # Create the window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

    def calculate_distance(self, point1, point2):
        """Calculate 3D Euclidean distance between two landmarks"""
        dx = point1.x - point2.x
        dy = point1.y - point2.y
        dz = point1.z - point2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def detect_touches(self, hand_results):
        """Detect touches between finger tips of different hands"""
        import time
        current_time = time.time()

        # Remove old touches from history
        self.touch_history = [t for t in self.touch_history
                             if current_time - t['touch_time'] < self.cooldown_duration]

        # Get cooldown points (points that recently touched)
        cooldown_points = set()
        for touch in self.touch_history:
            key1 = f"{touch['hand1']}_{touch['finger1']}"
            key2 = f"{touch['hand2']}_{touch['finger2']}"
            cooldown_points.add(key1)
            cooldown_points.add(key2)

        self.current_touches = []

        if not hand_results or not hand_results.hand_landmarks:
            return

        # Need at least 2 hands for inter-hand touches
        if len(hand_results.hand_landmarks) < 2:
            return

        # Get hand labels (left/right)
        hand_labels = []
        for i in range(len(hand_results.hand_landmarks)):
            label = "Unknown"
            if hand_results.handedness and i < len(hand_results.handedness):
                if hand_results.handedness[i]:
                    label = hand_results.handedness[i][0].display_name
            hand_labels.append(label)

        # Check all combinations of hands
        for i in range(len(hand_results.hand_landmarks)):
            for j in range(i + 1, len(hand_results.hand_landmarks)):
                hand1_landmarks = hand_results.hand_landmarks[i]
                hand2_landmarks = hand_results.hand_landmarks[j]
                hand1_label = hand_labels[i]
                hand2_label = hand_labels[j]

                # Check all finger tip combinations
                for idx1 in self.finger_tip_indices:
                    for idx2 in self.finger_tip_indices:
                        # Check if points are in cooldown
                        key1 = f"{hand1_label}_{self.finger_names[idx1]}"
                        key2 = f"{hand2_label}_{self.finger_names[idx2]}"

                        if key1 in cooldown_points or key2 in cooldown_points:
                            continue  # Skip points in cooldown

                        point1 = hand1_landmarks[idx1]
                        point2 = hand2_landmarks[idx2]

                        distance = self.calculate_distance(point1, point2)

                        if distance <= self.touch_threshold:
                            touch_info = {
                                'hand1': hand1_label,
                                'hand2': hand2_label,
                                'finger1': self.finger_names[idx1],
                                'finger2': self.finger_names[idx2],
                                'distance': distance,
                                'hand1_index': i,
                                'hand2_index': j,
                                'point1_idx': idx1,
                                'point2_idx': idx2,
                                'point1': point1,
                                'point2': point2,
                                'touch_time': current_time,
                                'max_distance': distance,  # Track max distance for line stretching
                                'line_color': [0, 255, 0],  # Start white
                                'point1_color': [0, 0, 255],  # Red
                                'point2_color': [0, 0, 255],  # Red
                                'touch_id': f"{hand1_label}_{self.finger_names[idx1]}_{hand2_label}_{self.finger_names[idx2]}"
                            }
                            self.current_touches.append(touch_info)
                            self.touch_history.append(touch_info.copy())

    def update_visual_effects(self, hand_results):
        """Update colors and lines based on touch state"""
        import time
        current_time = time.time()

        for touch in self.current_touches + self.touch_history:
            # Update point colors (fade from red back to normal)
            elapsed = current_time - touch['touch_time']

            if elapsed < self.cooldown_duration:
                # Calculate color fade (red → normal)
                fade_factor = elapsed / self.cooldown_duration

                # Red fades to hand's normal color
                if "Left" in touch['hand1']:
                    normal_color = list(self.left_hand_color)
                else:
                    normal_color = list(self.right_hand_color)

                if "Left" in touch['hand2']:
                    normal_color2 = list(self.left_hand_color)
                else:
                    normal_color2 = list(self.right_hand_color)

                # Interpolate color
                touch['point1_color'] = [
                    int(touch['point1_color'][0] * (1 - fade_factor) + normal_color[0] * fade_factor),
                    int(touch['point1_color'][1] * (1 - fade_factor) + normal_color[1] * fade_factor),
                    int(touch['point1_color'][2] * (1 - fade_factor) + normal_color[2] * fade_factor)
                ]

                touch['point2_color'] = [
                    int(touch['point2_color'][0] * (1 - fade_factor) + normal_color2[0] * fade_factor),
                    int(touch['point2_color'][1] * (1 - fade_factor) + normal_color2[1] * fade_factor),
                    int(touch['point2_color'][2] * (1 - fade_factor) + normal_color2[2] * fade_factor)
                ]

                # Update line properties based on current distance
                if hand_results and len(hand_results.hand_landmarks) > max(touch['hand1_index'], touch['hand2_index']):
                    hand1 = hand_results.hand_landmarks[touch['hand1_index']]
                    hand2 = hand_results.hand_landmarks[touch['hand2_index']]

                    if len(hand1) > touch['point1_idx'] and len(hand2) > touch['point2_idx']:
                        current_point1 = hand1[touch['point1_idx']]
                        current_point2 = hand2[touch['point2_idx']]

                        current_distance = self.calculate_distance(current_point1, current_point2)

                        # Update max distance if current is larger
                        if current_distance > touch['max_distance']:
                            touch['max_distance'] = current_distance

                        # Calculate line properties based on distance
                        # Line fades as distance increases
                        fade_amount = min(1.0, current_distance / (touch['max_distance'] + 0.001))

                        # Line thickness based on distance (thinner when far)
                        touch['line_thickness'] = max(1, int(3 * (1 - fade_amount)))

                        # Line color fades from white to invisible
                        touch['line_color'] = [
                            int(1 * (1 - fade_amount * 0.8)),
                            int(255 * (1 - fade_amount * 0.8)),
                            int(1 * (1 - fade_amount * 0.8))
                        ]

    def update_from_results(self, hand_results):
        """Update display from hand recognition results"""
        canvas = np.full((self.window_height, self.window_width, 3), self.bg_color, dtype=np.uint8)

        # Detect touches
        self.detect_touches(hand_results)

        # Update visual effects
        self.update_visual_effects(hand_results)

        # Print to console
        self.print_touches_to_console()

        # Title
        cv2.putText(canvas, "MULTI-HAND TRACKING", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Display touch status
        touch_status = f"Touch Threshold: {self.touch_threshold}"
        cv2.putText(canvas, touch_status, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Cooldown status
        import time
        active_cooldowns = sum(1 for t in self.touch_history
                              if time.time() - t['touch_time'] < self.cooldown_duration)
        cooldown_text = f"Cooldown: {active_cooldowns} point(s)"
        cv2.putText(canvas, cooldown_text, (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if not hand_results or not hand_results.hand_landmarks:
            cv2.putText(canvas, "No hands detected", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.imshow(self.window_name, canvas)
            return

        y_offset = 120

        # Display touch summary
        if self.current_touches:
            touch_color = self.touch_color
            touch_text = f"ACTIVE TOUCHES: {len(self.current_touches)}"
        else:
            touch_color = self.no_touch_color
            touch_text = "NO ACTIVE TOUCHES"

        cv2.putText(canvas, touch_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, touch_color, 1, cv2.LINE_AA)
        y_offset += 30

        # Display individual touch details
        for touch in self.current_touches[:3]:  # Show first 3 to save space
            touch_detail = f"{touch['hand1']} {touch['finger1']} ↔ {touch['hand2']} {touch['finger2']}"
            cv2.putText(canvas, touch_detail, (40, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.touch_color, 1, cv2.LINE_AA)
            y_offset += 20

        y_offset += 10


        # Draw instructions
        instructions = [
            f"Finger tips: {len(self.finger_tip_indices)} per hand",
            f"Touch threshold: {self.touch_threshold}",
            f"Cooldown: {self.cooldown_duration}s",
            "Line: White→fades | Points: Red→fades"
        ]

        for i, line in enumerate(instructions):
            cv2.putText(canvas, line, (20, self.window_height - 60 + i*15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, canvas)

    def print_touches_to_console(self):
        """Print touch events to console (serial monitor)"""
        import time

        current_time = time.time()

        # Print only if we have new touches and enough time has passed
        if self.current_touches and (current_time - self.last_touch_print > 1.0):
            print("\n" + "="*60)
            print("TOUCH DETECTED - VISUAL FEEDBACK ACTIVATED")
            print("="*60)

            for touch in self.current_touches:
                print(f"  {touch['hand1']} {touch['finger1']} ↔ {touch['hand2']} {touch['finger2']}")
                print(f"  Distance: {touch['distance']:.4f}")
                print(f"  Points now RED (fading over {self.cooldown_duration}s)")
                print(f"  Connecting line drawn")
                print("-" * 40)

            self.last_touch_print = current_time

    def close(self):
        """Close the window"""
        cv2.destroyWindow(self.window_name)

# ============================================================================
# GESTURE MAPPING MENU
# ============================================================================

class GestureMappingMenu:
    """Menu for mapping gestures to actions"""

    def __init__(self, gesture_mapping):
        self.gesture_mapping = gesture_mapping
        self.enabled = False

        # Display settings
        self.width = 400
        self.height = 500

        # Colors
        self.bg_color = (40, 40, 60)
        self.header_color = (60, 60, 100)
        self.cell_color = (50, 50, 70)
        self.cell_hover_color = (70, 70, 90)
        self.text_color = (255, 255, 255)
        self.highlight_color = (100, 200, 255)

        # Selection
        self.selected_gesture = None
        self.selected_action = None

    def toggle(self):
        """Toggle menu visibility"""
        self.enabled = not self.enabled
        self.selected_gesture = None
        self.selected_action = None
        return self.enabled

    def draw(self, frame):
        """Draw the gesture mapping menu"""
        if not self.enabled:
            return frame

        h, w = frame.shape[:2]

        # Calculate position (centered)
        x1 = max(0, (w - self.width) // 2)
        y1 = max(0, (h - self.height) // 2)
        x2 = x1 + self.width
        y2 = y1 + self.height

        # Draw overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw menu background
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Draw title
        title = "GESTURE MAPPING EDITOR"
        cv2.putText(frame, title, (x1 + 20, y1 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw column headers
        cv2.rectangle(frame, (x1 + 20, y1 + 60), (x1 + 190, y1 + 90), self.header_color, -1)
        cv2.putText(frame, "GESTURE", (x1 + 60, y1 + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (x1 + 210, y1 + 60), (x1 + 380, y1 + 90), self.header_color, -1)
        cv2.putText(frame, "ACTION", (x1 + 260, y1 + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw gesture list
        gestures = self.gesture_mapping.get_available_gestures()
        start_y = y1 + 100
        cell_height = 35

        for i, gesture in enumerate(gestures):
            cell_y = start_y + i * cell_height

            # Highlight selected gesture
            cell_color = self.highlight_color if gesture == self.selected_gesture else self.cell_color

            cv2.rectangle(frame, (x1 + 20, cell_y), (x1 + 190, cell_y + cell_height), cell_color, -1)

            # Draw gesture name with emoji
            emoji_map = {
                'Closed_Fist': '✊',
                'Open_Palm': '🖐️',
                'Pointing_Up': '☝️',
                'Thumb_Down': '👎',
                'Thumb_Up': '👍',
                'Victory': '✌️',
                'ILoveYou': '🤟',
                'None': '○'
            }
            emoji = emoji_map.get(gesture, '')
            gesture_text = f"{emoji} {gesture}"

            cv2.putText(frame, gesture_text, (x1 + 30, cell_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)

        # Draw actions list
        actions = self.gesture_mapping.get_available_actions()
        for i, action in enumerate(actions):
            action_id, action_name, action_icon, _ = action
            cell_y = start_y + i * cell_height

            # Highlight selected action
            cell_color = self.highlight_color if action_id == self.selected_action else self.cell_color

            cv2.rectangle(frame, (x1 + 210, cell_y), (x1 + 380, cell_y + cell_height), cell_color, -1)

            # Draw action
            action_text = f"{action_icon} {action_name}"
            cv2.putText(frame, action_text, (x1 + 220, cell_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)

        # Draw current mappings
        mapping_y = y1 + 100 + len(gestures) * cell_height + 20
        cv2.putText(frame, "CURRENT MAPPINGS:", (x1 + 20, mapping_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 1, cv2.LINE_AA)

        mapping_y += 25
        for name, gesture_list in self.gesture_mapping.action_to_gestures.items():
            if gesture_list:
                mapping_text = f"{name}: {', '.join(gesture_list)}"
                cv2.putText(frame, mapping_text, (x1 + 20, mapping_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                mapping_y += 20

        # Draw instructions
        instr_y = y2 - 40
        cv2.putText(frame, "Select gesture → Select action → Press 'S' to save mapping",
                   (x1 + 20, instr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press 'G' to close menu",
                   (x1 + 20, instr_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def handle_click(self, x, y):
        """Handle clicks in the menu"""
        if not self.enabled:
            return False

        h, w = 480, 640  # Default dimensions
        x1 = max(0, (w - self.width) // 2)
        y1 = max(0, (h - self.height) // 2)

        # Check if click is in menu bounds
        if x1 <= x <= x1 + self.width and y1 <= y <= y1 + self.height:
            # Check gesture column
            if x1 + 20 <= x <= x1 + 190:
                start_y = y1 + 100
                cell_height = 35
                gestures = self.gesture_mapping.get_available_gestures()

                for i, gesture in enumerate(gestures):
                    cell_y = start_y + i * cell_height
                    if cell_y <= y <= cell_y + cell_height:
                        self.selected_gesture = gesture
                        return True

            # Check action column
            elif x1 + 210 <= x <= x1 + 380:
                start_y = y1 + 100
                cell_height = 35
                actions = self.gesture_mapping.get_available_actions()

                for i, action in enumerate(actions):
                    cell_y = start_y + i * cell_height
                    if cell_y <= y <= cell_y + cell_height:
                        self.selected_action = action[0]  # action_id
                        return True

            return True  # Click was in menu

        return False  # Click outside menu

    def save_mapping(self):
        """Save the current selection as a mapping"""
        if self.selected_gesture and self.selected_action:
            success = self.gesture_mapping.map_gesture(self.selected_gesture, self.selected_action)
            if success:
                print(f"✓ Mapped {self.selected_gesture} to {self.selected_action}")
                self.selected_gesture = None
                self.selected_action = None
                return True
        return False

# ============================================================================
# SIMPLE RADIAL MENU (for actions only)
# ============================================================================

class SimpleRadialMenu:
    """Simple radial menu for executing actions, not related to gestures"""

    def __init__(self, window_width, window_height):
        self.enabled = False
        self.window_width = window_width
        self.window_height = window_height
        self.center_x = window_width // 2
        self.center_y = window_height // 2

        # Menu settings
        self.radius = min(window_width, window_height) // 4
        self.square_size = 50
        self.square_half = self.square_size // 2

        # Colors
        self.square_color_normal = (100, 100, 200)
        self.square_color_hover = (150, 150, 255)
        self.square_color_active = (200, 200, 255)
        self.square_border_color = (255, 255, 255)
        self.text_color = (255, 255, 255)

        # Menu items
        self.items = []
        self.hovered_item = -1
        self.clicked_item = -1

        # Item labels (simple numbers 1-8)
        self.item_labels = ["1", "2", "3", "4", "5", "6", "7", "8"]

        # Actions for each button (defaults)
        self.button_actions = [
            lambda: print("  🎮 Button 1 clicked"),
            lambda: print("  🎮 Button 2 clicked"),
            lambda: print("  🎮 Button 3 clicked"),
            lambda: print("  🎮 Button 4 clicked"),
            lambda: print("  🎮 Button 5 clicked"),
            lambda: print("  🎮 Button 6 clicked"),
            lambda: print("  🎮 Button 7 clicked"),
            lambda: print("  🎮 Button 8 clicked"),
        ]

        # Create menu layout
        self.create_menu_items()

    def create_menu_items(self):
        """Create 8 menu items arranged in a circle"""
        self.items = []

        for i in range(8):
            # Calculate angle
            angle = (2 * math.pi / 8) * i - math.pi / 8

            # Calculate position
            x = self.center_x + int(self.radius * math.cos(angle))
            y = self.center_y - int(self.radius * math.sin(angle))

            # Create square coordinates
            x1 = x - self.square_half
            y1 = y - self.square_half
            x2 = x + self.square_half
            y2 = y + self.square_half

            self.items.append({
                'center_x': x,
                'center_y': y,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'angle': angle
            })

    def set_button_action(self, button_index, action):
        """Set an action for a specific button"""
        if 0 <= button_index < 8:
            self.button_actions[button_index] = action

    def execute_button_action(self, button_index):
        """Execute the action for a button"""
        if 0 <= button_index < 8 and self.button_actions[button_index]:
            self.button_actions[button_index]()
            return True
        return False

    def toggle(self):
        """Toggle menu visibility"""
        self.enabled = not self.enabled
        self.hovered_item = -1
        self.clicked_item = -1
        return self.enabled

    def update_window_size(self, width, height):
        """Update menu positions when window size changes"""
        self.window_width = width
        self.window_height = height
        self.center_x = width // 2
        self.center_y = height // 2
        self.radius = min(width, height) // 4
        self.create_menu_items()

    def check_hover(self, mouse_x, mouse_y):
        """Check if mouse is hovering over any menu item"""
        if not self.enabled:
            self.hovered_item = -1
            return -1

        self.hovered_item = -1
        for i, item in enumerate(self.items):
            if (item['x1'] <= mouse_x <= item['x2'] and
                item['y1'] <= mouse_y <= item['y2']):
                self.hovered_item = i
                return i
        return -1

    def check_click(self, mouse_x, mouse_y):
        """Check for clicks on menu items"""
        if not self.enabled:
            return -1

        clicked = -1
        for i, item in enumerate(self.items):
            if (item['x1'] <= mouse_x <= item['x2'] and
                item['y1'] <= mouse_y <= item['y2']):
                clicked = i
                break

        if clicked >= 0:
            if self.execute_button_action(clicked):
                self.clicked_item = clicked
                # Flash the button
                import threading
                threading.Timer(0.3, lambda: setattr(self, 'clicked_item', -1)).start()
            return clicked

        return -1

    def draw(self, frame):
        """Draw the radial menu"""
        if not self.enabled:
            return frame

        h, w = frame.shape[:2]

        # Update positions if window size changed
        if w != self.window_width or h != self.window_height:
            self.update_window_size(w, h)

        # Draw center circle
        cv2.circle(frame, (self.center_x, self.center_y), 10, (255, 255, 255), -1)
        cv2.circle(frame, (self.center_x, self.center_y), 10, (50, 50, 200), 2)

        # Draw connecting lines
        for item in self.items:
            cv2.line(frame, (self.center_x, self.center_y),
                    (item['center_x'], item['center_y']),
                    (100, 100, 200), 1)

        # Draw squares and labels
        for i, item in enumerate(self.items):
            # Determine square color
            if i == self.clicked_item:
                color = self.square_color_active
            elif i == self.hovered_item:
                color = self.square_color_hover
            else:
                color = self.square_color_normal

            # Draw square with border
            cv2.rectangle(frame, (item['x1'], item['y1']),
                         (item['x2'], item['y2']), color, -1)
            cv2.rectangle(frame, (item['x1'], item['y1']),
                         (item['x2'], item['y2']), self.square_border_color, 2)

            # Draw label (number)
            label = self.item_labels[i]
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            label_x = item['center_x'] - label_width // 2
            label_y = item['center_y'] + label_height // 2

            cv2.putText(frame, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2, cv2.LINE_AA)

        # Draw menu title
        title = "ACTION MENU"
        (title_width, title_height), baseline = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        title_x = self.center_x - title_width // 2
        title_y = self.center_y + self.radius + 50

        cv2.putText(frame, title, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw instruction text
        instruction = "Click buttons 1-8 to execute actions | Press 'm' to close"
        (inst_width, inst_height), baseline = cv2.getTextSize(
            instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        inst_x = self.center_x - inst_width // 2
        inst_y = title_y + 30

        cv2.putText(frame, instruction, (inst_x, inst_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

# ============================================================================
# TEXT RENDERER (keep as is)
# ============================================================================

class TextRenderer:
    """Helper class for rendering text with consistent styling"""

    def __init__(self, frame):
        self.frame = frame
        self.h, self.w = frame.shape[:2]

        # Font definitions
        self.fonts = {
            'title': cv2.FONT_HERSHEY_COMPLEX,
            'medium': cv2.FONT_HERSHEY_COMPLEX,
            'small': cv2.FONT_HERSHEY_COMPLEX,
            'default': cv2.FONT_HERSHEY_COMPLEX
        }

        # Size definitions
        self.sizes = {
            'title': 0.7,
            'medium': 0.6,
            'small': 0.5,
            'default': 0.5
        }

        # Thickness definitions
        self.thickness = {
            'title': 2,
            'medium': 2,
            'small': 1,
            'default': 1
        }

        # Line type
        self.line_type = cv2.LINE_AA

    def add_text(self, text, x, y, color=(255, 255, 255),
                font_type='default', size_type=None,
                thickness_type=None, shadow=False,
                shadow_color=(0, 0, 0), shadow_offset=1):
        """Add text with specified styling"""

        # Get styling
        font = self.fonts.get(font_type, self.fonts['default'])
        size = self.sizes.get(size_type if size_type else font_type,
                             self.sizes['default'])
        thickness = self.thickness.get(thickness_type if thickness_type else font_type,
                                      self.thickness['default'])

        # Add shadow if requested
        if shadow:
            cv2.putText(self.frame, text, (x+shadow_offset, y+shadow_offset),
                       font, size, shadow_color, thickness, self.line_type)

        # Add main text
        cv2.putText(self.frame, text, (x, y),
                   font, size, color, thickness, self.line_type)

        return self  # For chaining

    def add_status(self, label, value, x, y, color=(255, 255, 255),
                  font_type='default', shadow=True):
        """Add formatted status label: value"""
        text = f"{label}: {value}"
        return self.add_text(text, x, y, color, font_type, shadow=shadow)

    def add_section(self, text, x, y, color=(0, 255, 0),
                   font_type='title', shadow=True):
        """Add section title/header"""
        return self.add_text(text, x, y, color, font_type, shadow=shadow)

    def get_width(self):
        """Get frame width"""
        return self.w

    def get_height(self):
        """Get frame height"""
        return self.h

# ============================================================================
# GESTURE CONTROLLER (updated)
# ============================================================================

class GestureController:
    """Handles gesture-based actions for the tracker"""

    def __init__(self, tracker):
        self.tracker = tracker
        self.enabled = False
        self.verbose = True

        # Gesture state tracking
        self.current_gesture = None
        self.last_action_time = 0
        self.action_cooldown = 1.0  # seconds between actions

        # Use the gesture mapping system instead of hardcoded mappings
        self.gesture_mapping = GestureMapping(tracker)

        # Track which gestures we've seen
        self.detected_gestures = set()

    def update_mapping_actions(self):
        """Update the gesture mapping with actual action functions"""
        self.gesture_mapping.update_actions(self)

    def process_gesture(self, gesture_name, timestamp):
        """Process a detected gesture and trigger appropriate action"""
        if not self.enabled or not gesture_name or gesture_name == 'None':
            return

        continuous = gesture_name in ("Thumb_Up", "Thumb_Down")

        # Cooldown only applies to discrete gestures
        if not continuous:
            if timestamp - self.last_action_time < self.action_cooldown:
                return

        if gesture_name == self.current_gesture and not continuous:
            return

        self.current_gesture = gesture_name

        # Track that we've seen this gesture
        if gesture_name not in self.detected_gestures:
            self.detected_gestures.add(gesture_name)
            if self.verbose:
                print(f"  ✨ New gesture detected: {gesture_name}")

        # Execute the mapped action
        if self.gesture_mapping.execute_gesture(gesture_name):
            self.last_action_time = timestamp

            if self.verbose:
                # Get emoji for gesture
                emoji_map = {
                    'Closed_Fist': '✊',
                    'Open_Palm': '🖐️',
                    'Pointing_Up': '☝️',
                    'Thumb_Down': '👎',
                    'Thumb_Up': '👍',
                    'Victory': '✌️',
                    'ILoveYou': '🤟',
                }
                emoji = emoji_map.get(gesture_name, '')

    # Keep these methods for backward compatibility with radial menu
    def toggle_red(self):
        """Toggle RED channel"""
        self.tracker.show_red = not self.tracker.show_red
        status = "ON" if self.tracker.show_red else "OFF"
        print(f"  RED channel: {status}")

    def toggle_green(self):
        """Toggle GREEN channel"""
        self.tracker.show_green = not self.tracker.show_green
        status = "ON" if self.tracker.show_green else "OFF"
        print(f"  GREEN channel: {status}")

    def toggle_blue(self):
        """Toggle BLUE channel"""
        self.tracker.show_blue = not self.tracker.show_blue
        status = "ON" if self.tracker.show_blue else "OFF"
        print(f"  BLUE channel: {status}")

    def action1(self):
        """Thumb Down → decrease OBS blue"""
        if self.tracker.obs_blue:
            now = cv2.getTickCount() / cv2.getTickFrequency()
            self.tracker.obs_blue.decrease(now)

    def action2(self):
        """Thumb Up → increase OBS blue"""
        if self.tracker.obs_blue:
            now = cv2.getTickCount() / cv2.getTickFrequency()
            self.tracker.obs_blue.increase(now)

    def take_screenshot(self):
        """Take screenshot of current frame"""
        if hasattr(self.tracker, 'last_frame') and self.tracker.last_frame is not None:
            import time
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, self.tracker.last_frame)
            print(f"  📸 Screenshot saved: {filename}")
        else:
            print("  ❌ Could not take screenshot - no frame available")

    def toggle_text(self):
        """Toggle text overlay"""
        self.tracker.show_text = not self.tracker.show_text
        status = "ON" if self.tracker.show_text else "OFF"
        print(f"  Text overlay: {status}")

    def toggle(self):
        """Toggle gesture controller on/off"""
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        print(f"  Gesture Controller: {status}")

    def print_help(self):
        """Print gesture controller help"""
        print("\n" + "="*60)
        print("GESTURE CONTROLLER COMMANDS")
        print("="*60)
        print("Status: " + ("✅ ON" if self.enabled else "❌ OFF"))
        print("\nCurrent Gesture Mappings:")

        # Show current mappings
        for name, gesture_list in self.gesture_mapping.action_to_gestures.items():
            if gesture_list:
                print(f"    {name}: {', '.join(gesture_list)}")

        print("\nController Commands:")
        print("    [g] - Toggle gesture controller")
        print("    [G] - Show gesture mapping menu")
        print("    [C] - Show this help")
        print("="*60)

# ============================================================================
# OBS FILTER PARAMETER (keep as is)
# ============================================================================

class OBSFilterParameter:
    """
    Level-controlled, rate-limited OBS filter parameter.
    Gesture-agnostic. Tracker-agnostic.
    """

    def __init__(self, ws, source_name, filter_name,
                 channel="blue",
                 initial=128,
                 step=3,
                 min_val=0,
                 max_val=255,
                 rate_limit=0.15):

        self.ws = ws
        self.source_name = source_name
        self.filter_name = filter_name

        self.channel = channel
        self.value = initial
        self.step = step
        self.min_val = min_val
        self.max_val = max_val

        self.rate_limit = rate_limit
        self.last_update_time = 0.0

        self.base_settings = {
            "gamma": 0.68,
            "contrast": -0.18,
            "brightness": 0.2837,
            "saturation": 0.18,
            "hue_shift": -63.97,
            "opacity": 0.999,
            "color_multiply": (0 << 16) | (255 << 8) | 0,
        }

    def _clamp(self):
        self.value = max(self.min_val, min(self.value, self.max_val))

    def _rate_ok(self, now):
        return (now - self.last_update_time) >= self.rate_limit

    def increase(self, now):
        if not self._rate_ok(now):
            return
        self.value += self.step
        self._clamp()
        self._push(now)

    def decrease(self, now):
        if not self._rate_ok(now):
            return
        self.value -= self.step
        self._clamp()
        self._push(now)

    def _push(self, now):
        settings = self.base_settings.copy()

        if self.channel == "blue":
            settings["color_add"] = (self.value << 16)

        try:
            self.ws.call(
                requests.SetSourceFilterSettings(
                    sourceName=self.source_name,
                    filterName=self.filter_name,
                    filterSettings=settings
                )
            )
            self.last_update_time = now
            print(f"🎛️ OBS {self.channel.upper()} = {self.value}")
        except Exception as e:
            print(f"❌ OBS update failed: {e}")

# ============================================================================
# POSE TRACKER (NEW)
# ============================================================================

class PoseTracker:
    """Handles full body pose detection and skeleton drawing"""

    def __init__(self, tracker):
        self.tracker = tracker
        self.mp = tracker.mp
        self.pose_detector = None

        # Toggle states
        self.enabled = True
        self.show_skeleton = True
        self.show_landmarks = True
        self.show_boxes = False

        # Colors
        self.skeleton_color = (255, 0, 255)      # Magenta for skeleton
        self.landmark_color = (0, 255, 255)      # Yellow for landmarks
        self.box_color = (255, 0, 255)           # Magenta for boxes

        # Pose data tracking
        self.pose_count = 0
        self.current_poses = []

        # For scrollable data window
        self.data_window = None
        self.show_data_window = False

        # Landmark names (33 total)
        self.landmark_names = [
            "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
            "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR",
            "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
            "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
            "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
            "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP",
            "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
            "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX",
            "RIGHT_FOOT_INDEX"
        ]

    def initialize_detector(self, model_path='pose_landmarker_full.task'):
        """Initialize the MediaPipe pose landmark detector"""
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,  # Can detect multiple people
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False  # Set to True if you want masks
            )

            self.pose_detector = vision.PoseLandmarker.create_from_options(options)
            print("✅ Pose detector initialized (Heavy model)")
            return True

        except Exception as e:
            print(f"❌ Pose detector error: {e}")
            return False

    def detect_poses(self, frame):
        """Detect poses in the frame"""
        if not self.enabled or not self.pose_detector:
            return None

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
            return result

        except Exception as e:
            print(f"Pose detection error: {e}")
            return None

    def draw_skeleton(self, frame, pose_landmarks):
        """Draw the full body skeleton with connections"""
        h, w, _ = frame.shape

        # Define pose connections (MediaPipe Pose 33-point model)
        # This matches the official POSE_CONNECTIONS
        pose_connections = [
            # Face oval
            (0, 1), (1, 2), (2, 3), (3, 7),    # Left eye to left ear
            (0, 4), (4, 5), (5, 6), (6, 8),    # Right eye to right ear
            (9, 10),  # Mouth

            # Upper body (torso)
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Shoulders to hips
            (23, 24),  # Hips

            # Left arm
            (11, 13), (13, 15),  # Shoulder to elbow to wrist
            (15, 17), (15, 19), (15, 21),  # Wrist to fingers
            (17, 19),  # Pinky to index

            # Right arm
            (12, 14), (14, 16),  # Shoulder to elbow to wrist
            (16, 18), (16, 20), (16, 22),  # Wrist to fingers
            (18, 20),  # Pinky to index

            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31),  # Hip to knee to ankle to foot
            (29, 31),  # Heel to foot index

            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32),  # Hip to knee to ankle to foot
            (30, 32),  # Heel to foot index

            # Body center line
            (11, 23), (12, 24),  # Shoulders to hips (already there, but for emphasis)
        ]

        # Draw connections (skeleton)
        if self.show_skeleton:
            for connection in pose_connections:
                start_idx, end_idx = connection

                if (start_idx < len(pose_landmarks) and
                    end_idx < len(pose_landmarks)):

                    start_point = (int(pose_landmarks[start_idx].x * w),
                                 int(pose_landmarks[start_idx].y * h))
                    end_point = (int(pose_landmarks[end_idx].x * w),
                               int(pose_landmarks[end_idx].y * h))

                    # Draw the connection line
                    cv2.line(frame, start_point, end_point,
                            self.skeleton_color, 2, cv2.LINE_AA)

        # Draw landmarks (joints)
        if self.show_landmarks:
            for idx, landmark in enumerate(pose_landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # Draw different sized circles for key points
                if idx in [0, 11, 12, 23, 24]:  # Nose, shoulders, hips
                    radius = 5
                    color = (0, 255, 0)  # Green for major joints
                elif idx in [13, 14, 15, 16, 25, 26, 27, 28]:  # Elbows, wrists, knees, ankles
                    radius = 4
                    color = (255, 255, 0)  # Yellow for medium joints
                else:
                    radius = 3
                    color = self.landmark_color  # Yellow for other points

                cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), radius, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw bounding box
        if self.show_boxes and len(pose_landmarks) > 0:
            landmarks_array = np.array([[lm.x * w, lm.y * h] for lm in pose_landmarks])
            x_min, y_min = landmarks_array.min(axis=0).astype(int)
            x_max, y_max = landmarks_array.max(axis=0).astype(int)

            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                         self.box_color, 2)

        return frame

    def draw_all_poses(self, frame, pose_result):
        """Draw all detected poses in the frame"""
        if not pose_result or not pose_result.pose_landmarks:
            return frame

        self.pose_count = len(pose_result.pose_landmarks)
        self.current_poses = pose_result.pose_landmarks

        for pose_landmarks in pose_result.pose_landmarks:
            frame = self.draw_skeleton(frame, pose_landmarks)

        return frame

    def toggle(self, attribute_name, display_name):
        """Toggle a pose tracking feature"""
        current_value = getattr(self, attribute_name)
        setattr(self, attribute_name, not current_value)
        status = "ON" if not current_value else "OFF"
        print(f"  Pose {display_name}: {status}")
        return True

    def print_help(self):
        """Print pose tracker help"""
        print("\n" + "="*50)
        print("POSE TRACKER CONTROLS")
        print("="*50)
        print("Status: " + ("✅ ON" if self.enabled else "❌ OFF"))
        print("\nCommands:")
        print("    [p] - Toggle pose tracking ON/OFF")
        print("    [5] - Toggle skeleton drawing")
        print("    [6] - Toggle landmark points")
        print("    [7] - Toggle bounding boxes")
        print("    [P] - Toggle pose data window")
        print("="*50)

class PoseDataWindow(ScrollableDataWindow):
    """Window for displaying full body pose landmark data"""

    def __init__(self):
        super().__init__("Pose Landmarks", window_width=800, window_height=700)
        self.column_headers = ["ID", "Name", "X", "Y", "Z", "Visibility", "Angle"]

        # Pose landmark names (33 points)
        self.pose_names = [
            "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
            "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR",
            "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
            "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
            "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
            "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP",
            "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
            "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX",
            "RIGHT_FOOT_INDEX"
        ]

        # Major joints for angle calculations
        self.major_joints = {
            "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
            "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
            "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
            "LEFT_HIP": 23, "RIGHT_HIP": 24,
            "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
            "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28
        }

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points (b is the vertex)"""
        import numpy as np

        # Convert to numpy arrays
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        # Calculate vectors
        ba = a - b
        bc = c - b

        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1, 1)
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    def update_from_landmarks(self, pose_landmarks_list, pose_index=0):
        """Convert pose landmarks to display data"""
        if not pose_landmarks_list or len(pose_landmarks_list) <= pose_index:
            self.update_data([])
            return

        display_rows = []

        # Get the specific pose's landmarks
        pose_landmarks = pose_landmarks_list[pose_index]

        # Calculate angles for major joints
        angles = {}

        # Calculate elbow angles
        if len(pose_landmarks) > 16:
            # Left elbow: shoulder(11) - elbow(13) - wrist(15)
            angles["LEFT_ELBOW"] = self.calculate_angle(
                pose_landmarks[11], pose_landmarks[13], pose_landmarks[15]
            ) if len(pose_landmarks) > 15 else 0

            # Right elbow: shoulder(12) - elbow(14) - wrist(16)
            angles["RIGHT_ELBOW"] = self.calculate_angle(
                pose_landmarks[12], pose_landmarks[14], pose_landmarks[16]
            ) if len(pose_landmarks) > 16 else 0

        # Calculate knee angles
        if len(pose_landmarks) > 28:
            # Left knee: hip(23) - knee(25) - ankle(27)
            angles["LEFT_KNEE"] = self.calculate_angle(
                pose_landmarks[23], pose_landmarks[25], pose_landmarks[27]
            ) if len(pose_landmarks) > 27 else 0

            # Right knee: hip(24) - knee(26) - ankle(28)
            angles["RIGHT_KNEE"] = self.calculate_angle(
                pose_landmarks[24], pose_landmarks[26], pose_landmarks[28]
            ) if len(pose_landmarks) > 28 else 0

        # Create rows for each landmark
        for i in range(min(len(pose_landmarks), 33)):  # Pose has 33 landmarks
            landmark = pose_landmarks[i]
            name = self.pose_names[i] if i < len(self.pose_names) else f"Point_{i}"

            # Check if visibility exists
            if hasattr(landmark, 'visibility') and landmark.visibility is not None:
                visibility = f"{landmark.visibility:.4f}"
            else:
                visibility = "N/A"

            # Get angle for this joint if it's a major one
            angle_text = ""
            for joint_name, joint_id in self.major_joints.items():
                if i == joint_id:
                    angle_text = f"{angles.get(joint_name, 0):.1f}°"
                    break

            row = [
                str(i),
                name,
                f"{landmark.x:.4f}",
                f"{landmark.y:.4f}",
                f"{landmark.z:.4f}",
                visibility,
                angle_text if angle_text else "N/A"
            ]
            display_rows.append(row)

        # Add summary row at the top
        summary_row = ["SUMMARY", f"Pose #{pose_index+1}", f"{len(pose_landmarks)} points", "", "", "", ""]
        display_rows.insert(0, summary_row)

        # Add angle summary row
        if angles:
            angle_summary = ["ANGLES",
                           f"L Elbow: {angles.get('LEFT_ELBOW', 0):.1f}°",
                           f"R Elbow: {angles.get('RIGHT_ELBOW', 0):.1f}°",
                           f"L Knee: {angles.get('LEFT_KNEE', 0):.1f}°",
                           f"R Knee: {angles.get('RIGHT_KNEE', 0):.1f}°",
                           "", ""]
            display_rows.insert(1, angle_summary)

        self.update_data(display_rows)

# ============================================================================
# COMBINED TRACKER (updated)
# ============================================================================

class CombinedTracker:
    def __init__(self):
        self.cap = None
        self.face_detector = None
        self.hand_detector = None
        self.mp = mp

        # RGB channel toggles
        self.show_red = True
        self.show_green = True
        self.show_blue = True

        # Detection toggles
        self.show_faces = True
        self.show_hands = True
        self.show_text = True

        # Bounding box toggles
        self.show_face_boxes = False
        self.show_hand_boxes = False

        # Face landmark toggles
        self.show_face_landmarks = True
        self.show_face_connections = True

        # Hand landmark toggles
        self.show_hand_landmarks = True
        self.show_hand_connections = True

        # Replace the single DataDisplayWindow with separate windows
        self.face_data_window = FaceDataWindow()
        self.hand_data_window = HandDataWindow()
        self.multi_hand_window = MultiHandDataWindow()  # Optional: for multiple hands overview
        # Add pose tracker
        self.pose_tracker = PoseTracker(self)
        self.show_pose_data = False  # NEW: For pose data window

        # Add pose data window
        self.pose_data_window = None  # We'll create this if needed

        # Update colors - add pose color
        self.pose_color = (255, 0, 255)  # Magenta for pose tracking
        # Toggle flags
        self.show_face_data = False
        self.show_hand_data = False
        self.show_multi_hand = False

        # FPS tracking
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0

        # Colors
        self.face_color = (255, 0, 0)    # Blue
        self.hand_color = (0, 255, 0)    # Green

        # Load face connections from imported groups
        if LANDMARK_GROUPS_AVAILABLE:
            # Combine all connection lists from the imported face_connections dictionary
            self.face_connections = []
            for region_name, connections in face_connections.items():
                self.face_connections.extend(connections)
        else:
            # If landmark_groups.py is not found, face connections will be empty
            self.face_connections = []

        # Add gesture controller (which now contains gesture mapping)
        self.gesture_controller = GestureController(self)

        # Update gesture mapping with actual actions
        self.gesture_controller.update_mapping_actions()

        # Add gesture mapping menu
        self.gesture_mapping_menu = GestureMappingMenu(self.gesture_controller.gesture_mapping)
        self.show_gesture_mapping_menu = False

        # Store last frame for screenshots
        self.last_frame = None

        # OBS integration
        self.obs_ws = None
        self.obs_blue = None

        # Create simple radial menu
        self.radial_menu = None
        self.show_radial_menu = False

        # Mouse event handling
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_clicked = False

        # Key actions
        self.key_actions = {
            # ===== SYSTEM CONTROLS =====
            ord('q'): self.quit_action,
            ord('?'): self.display_help,

            # ===== RGB CHANNEL TOGGLES =====
            ord('r'): lambda: self.toggle('show_red', 'RED channel'),
            ord('g'): lambda: self.toggle('show_green', 'GREEN channel'),
            ord('b'): lambda: self.toggle('show_blue', 'BLUE channel'),

            # ===== TRACKING TOGGLES =====
            ord('f'): lambda: self.toggle('show_faces', 'Face tracking'),
            ord('h'): lambda: self.toggle('show_hands', 'Hand tracking'),

            # ===== DATA WINDOW TOGGLES =====
            ord('F'): lambda: self.toggle('show_face_data', 'Face data window'),  # Capital F
            ord('H'): lambda: self.toggle('show_hand_data', 'Hand data window'),  # Capital H
            ord('M'): lambda: self.toggle('show_multi_hand', 'Multi-hand window'),  # Capital M

            # ===== LANDMARK DETAIL TOGGLES =====
            ord('1'): lambda: self.toggle('show_face_landmarks', 'Face landmarks'),
            ord('2'): lambda: self.toggle('show_face_connections', 'Face connections'),
            ord('3'): lambda: self.toggle('show_hand_landmarks', 'Hand landmarks'),
            ord('4'): lambda: self.toggle('show_hand_connections', 'Hand connections'),

            # ===== BOUNDING BOX TOGGLES =====
            ord('B'): lambda: self.toggle('show_face_boxes', 'Face bounding boxes'),
            ord('X'): lambda: self.toggle('show_hand_boxes', 'Hand bounding boxes'),  # Capital X

            # ===== TOUCH DETECTION CONTROLS =====
            ord('+'): lambda: self.adjust_touch_threshold(-1),  # More sensitive
            ord('-'): lambda: self.adjust_touch_threshold(1),   # Less sensitive
            ord('['): lambda: self.adjust_cooldown(-1) if hasattr(self, 'adjust_cooldown') else lambda: None,  # Shorter cooldown
            ord(']'): lambda: self.adjust_cooldown(1) if hasattr(self, 'adjust_cooldown') else lambda: None,   # Longer cooldown
            ord('0'): self.reset_all_touches if hasattr(self, 'reset_all_touches') else lambda: None,  # Reset touches

            # ===== UI TOGGLES =====
            ord('t'): lambda: self.toggle('show_text', 'Text overlay'),
            ord('m'): self.toggle_radial_menu,  # Radial action menu
            ord('G'): self.toggle_gesture_mapping_menu,  # Gesture mapping menu
            ord('p'): lambda: self.pose_tracker.toggle('enabled', 'tracking'),
            ord('5'): lambda: self.pose_tracker.toggle('show_skeleton', 'skeleton'),
            ord('6'): lambda: self.pose_tracker.toggle('show_landmarks', 'landmarks'),
            ord('7'): lambda: self.pose_tracker.toggle('show_boxes', 'boxes'),
            ord('P'): self.toggle_pose_data_window,  # Capital P
            # ===== GESTURE CONTROLLER =====
            ord('c'): self.gesture_controller.toggle,
            ord('C'): self.gesture_controller.print_help,
        }

        # Setup default radial menu actions
        self._setup_default_radial_actions()

    def _setup_default_radial_actions(self):
        """Setup default actions for the radial menu buttons"""
        # These will be set after radial menu is initialized
        self.default_radial_actions = [
            lambda: self.toggle('show_red', 'RED channel'),
            lambda: self.toggle('show_green', 'GREEN channel'),
            lambda: self.toggle('show_blue', 'BLUE channel'),
            lambda: self.toggle('show_faces', 'Face tracking'),
            lambda: self.toggle('show_hands', 'Hand tracking'),
            self.gesture_controller.take_screenshot,
            lambda: self.toggle('show_text', 'Text overlay'),
            self.toggle_radial_menu,  # Close menu
        ]

    def adjust_touch_threshold(self, delta):
        """Adjust the touch threshold sensitivity"""
        if hasattr(self, 'multi_hand_window'):
            new_threshold = self.multi_hand_window.touch_threshold + delta * 0.005
            # Keep threshold within reasonable bounds
            self.multi_hand_window.touch_threshold = max(0.001, min(0.1, new_threshold))
            print(f"  Touch threshold set to: {self.multi_hand_window.touch_threshold:.4f}")

    def open_gesture_mapping_menu(self):
        """Public method to open gesture mapping menu (can be called from gestures)"""
        self.toggle_gesture_mapping_menu()

    def draw_touch_visuals_on_frame(self, frame, hand_result):
        """Draw touch visualization on the main camera frame"""
        if not hasattr(self, 'multi_hand_window') or not self.multi_hand_window:
            return frame

        if not hand_result or not hand_result.hand_landmarks:
            return frame

        frame_height, frame_width = frame.shape[:2]

        # Draw connecting lines and points for current and recent touches
        for touch in (self.multi_hand_window.current_touches +
                      self.multi_hand_window.touch_history):

            # Check if hands still exist in current frame
            if (len(hand_result.hand_landmarks) <= max(touch['hand1_index'], touch['hand2_index'])):
                continue

            hand1 = hand_result.hand_landmarks[touch['hand1_index']]
            hand2 = hand_result.hand_landmarks[touch['hand2_index']]

            if len(hand1) <= touch['point1_idx'] or len(hand2) <= touch['point2_idx']:
                continue

            # Get current points
            point1 = hand1[touch['point1_idx']]
            point2 = hand2[touch['point2_idx']]

            # Convert normalized coordinates to pixel coordinates
            x1 = int(point1.x * frame_width)
            y1 = int(point1.y * frame_height)
            x2 = int(point2.x * frame_width)
            y2 = int(point2.y * frame_height)

            # Draw connecting line with visual properties
            if 'line_thickness' in touch:
                thickness = touch['line_thickness']
            else:
                thickness = 2

            if 'line_color' in touch:
                line_color = tuple(touch['line_color'])
            else:
                line_color = (255, 255, 255)  # White

            # Draw line (with anti-aliasing)
            cv2.line(frame, (x1, y1), (x2, y2), line_color, thickness, cv2.LINE_AA)

            # Draw touch points with current color
            point_radius = 8

            # Point 1
            if 'point1_color' in touch:
                point1_color = tuple(touch['point1_color'])
            else:
                point1_color = (0, 0, 255)  # Default red

            cv2.circle(frame, (x1, y1), point_radius, point1_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x1, y1), point_radius, (255, 255, 255), 1, cv2.LINE_AA)

            # Point 2
            if 'point2_color' in touch:
                point2_color = tuple(touch['point2_color'])
            else:
                point2_color = (0, 0, 255)  # Default red

            cv2.circle(frame, (x2, y2), point_radius, point2_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x2, y2), point_radius, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw distance label near midpoint (optional)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            current_distance = self.multi_hand_window.calculate_distance(point1, point2)
            distance_text = f"{current_distance:.3f}"

            # Add background for text readability
            text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame,
                         (mid_x - text_size[0]//2 - 2, mid_y - text_size[1] - 5),
                         (mid_x + text_size[0]//2 + 2, mid_y + 5),
                         (0, 0, 0), -1)

            cv2.putText(frame, distance_text, (mid_x - text_size[0]//2, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def toggle_radial_menu(self):
        """Toggle the radial action menu"""
        if self.radial_menu:
            self.show_radial_menu = self.radial_menu.toggle()
            status = "ON" if self.show_radial_menu else "OFF"
            print(f"  Radial Menu: {status}")

            if self.show_radial_menu:
                cv2.setMouseCallback("Face & Hand Tracker", self.mouse_callback)
            else:
                cv2.setMouseCallback("Face & Hand Tracker", lambda *args: None)

    def initialize_pose_detector(self):
        """Initialize the pose detector"""
        print("Initializing pose detector...")
        return self.pose_tracker.initialize_detector('pose_landmarker_heavy.task')

    def toggle_gesture_mapping_menu(self):
        """Toggle the gesture mapping menu"""
        self.show_gesture_mapping_menu = self.gesture_mapping_menu.toggle()
        status = "ON" if self.show_gesture_mapping_menu else "OFF"
        print(f"  Gesture Mapping Menu: {status}")

        if self.show_gesture_mapping_menu:
            cv2.setMouseCallback("Face & Hand Tracker", self.mouse_callback)
        elif not self.show_radial_menu:
            cv2.setMouseCallback("Face & Hand Tracker", lambda *args: None)

    def adjust_cooldown(self, delta):
        """Adjust touch cooldown duration"""
        if hasattr(self, 'multi_hand_window'):
            new_duration = self.multi_hand_window.cooldown_duration + delta * 0.5
            # Keep within reasonable bounds (0.5s to 10s)
            self.multi_hand_window.cooldown_duration = max(0.5, min(10.0, new_duration))
            print(f"  Cooldown duration: {self.multi_hand_window.cooldown_duration:.1f}s")

    def reset_all_touches(self):
        """Reset all touch history and cooldowns"""
        if hasattr(self, 'multi_hand_window'):
            self.multi_hand_window.touch_history = []
            self.multi_hand_window.current_touches = []
            print("  ✓ All touch history reset")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_x = x
        self.mouse_y = y

        # Handle gesture mapping menu clicks
        if self.show_gesture_mapping_menu:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.gesture_mapping_menu.handle_click(x, y)

        # Handle radial menu clicks
        elif self.show_radial_menu and self.radial_menu:
            self.radial_menu.check_hover(x, y)

            if event == cv2.EVENT_LBUTTONDOWN:
                result = self.radial_menu.check_click(x, y)
                if result >= 0:
                    print(f"  🎮 Radial button {result + 1} clicked")

    def toggle(self, attribute_name, display_name):
        """Toggle a boolean attribute and print status"""
        current_value = getattr(self, attribute_name)
        setattr(self, attribute_name, not current_value)
        status = "ON" if not current_value else "OFF"
        print(f"  {display_name}: {status}")

    def quit_action(self):
        """Handle quit action"""
        print("\n✅ Quitting...")
        return True  # Signal to quit

    def initialize_camera(self):
        """Initialize the camera"""
        print("Initializing camera...")

        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                print("❌ ERROR: Cannot open camera")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"✅ Camera initialized: {width}x{height}")

            # Initialize radial menu
            self.radial_menu = SimpleRadialMenu(width, height)

            # Set default actions
            for i, action in enumerate(self.default_radial_actions):
                self.radial_menu.set_button_action(i, action)

            return True

        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False

    def initialize_obs(self):
        """Initialize OBS WebSocket connection and parameters"""
        try:
            self.obs_ws = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
            self.obs_ws.connect()
            print("✅ Connected to OBS WebSocket")

            # Create OBS parameter controller
            self.obs_blue = OBSFilterParameter(
                ws=self.obs_ws,
                source_name="Background",   # MUST match OBS
                filter_name="Color",        # MUST match OBS
                channel="blue",
                initial=128,
                step=3,
                rate_limit=0.15
            )

            return True

        except Exception as e:
            print(f"❌ OBS connection failed: {e}")
            self.obs_ws = None
            self.obs_blue = None
            return False

    def initialize_face_detector(self, model_path='face_landmarker.task'):
        """Initialize the MediaPipe face landmark detector"""
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.face_detector = vision.FaceLandmarker.create_from_options(options)
            print("✅ Face detector initialized")
            return True

        except Exception as e:
            print(f"❌ Face detector error: {e}")
            return False

    def initialize_hand_detector(self, model_path='gesture_recognizer.task'):
        """Initialize the MediaPipe gesture recognizer"""
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.hand_detector = vision.GestureRecognizer.create_from_options(options)
            print("✅ Hand detector initialized")
            return True

        except Exception as e:
            print(f"❌ Hand detector error: {e}")
            return False

    def apply_color_filter(self, frame):
        """Apply RGB channel filter based on toggle states"""
        if not (self.show_red and self.show_green and self.show_blue):
            b, g, r = cv2.split(frame)
            if not self.show_blue:
                b = np.zeros_like(b)
            if not self.show_green:
                g = np.zeros_like(g)
            if not self.show_red:
                r = np.zeros_like(r)
            frame = cv2.merge([b, g, r])
        return frame

    def draw_face_landmarks(self, frame, face_landmarks):
        """Draw face landmarks and connections"""
        h, w, _ = frame.shape

        # Draw landmarks as small dots
        if self.show_face_landmarks:
            for landmark in face_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, self.face_color, -1)

        # Draw face connections if available
        if self.show_face_connections and self.face_connections:
            for connection in self.face_connections:
                start_idx, end_idx = connection

                # Check if indices are within range
                if start_idx < len(face_landmarks) and end_idx < len(face_landmarks):
                    start_point = (int(face_landmarks[start_idx].x * w),
                                  int(face_landmarks[start_idx].y * h))
                    end_point = (int(face_landmarks[end_idx].x * w),
                                int(face_landmarks[end_idx].y * h))

                    # Draw the connection line
                    cv2.line(frame, start_point, end_point, self.face_color, 1)

        # Draw bounding box if enabled
        if self.show_face_boxes:
            landmarks_array = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks])
            x_min, y_min = landmarks_array.min(axis=0).astype(int)
            x_max, y_max = landmarks_array.max(axis=0).astype(int)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), self.face_color, 2)

        return frame

    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks and connections"""
        h, w, _ = frame.shape

        # Draw landmarks
        if self.show_hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, self.hand_color, -1)

        # Draw connections (hand skeleton)
        if self.show_hand_connections:
            connections = [
                # Thumb
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Index finger
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Middle finger
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Ring finger
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Pinky finger
                (0, 17), (17, 18), (18, 19), (19, 20),
                # Palm
                (5, 9), (9, 13), (13, 17)
            ]

            for connection in connections:
                start_idx, end_idx = connection
                start_point = (int(hand_landmarks[start_idx].x * w),
                              int(hand_landmarks[start_idx].y * h))
                end_point = (int(hand_landmarks[end_idx].x * w),
                            int(hand_landmarks[end_idx].y * h))
                cv2.line(frame, start_point, end_point, self.hand_color, 2)

        # Draw bounding box if enabled
        if self.show_hand_boxes:
            landmarks_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks])
            x_min, y_min = landmarks_array.min(axis=0).astype(int)
            x_max, y_max = landmarks_array.max(axis=0).astype(int)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), self.hand_color, 2)

        return frame

    def process_frame(self, frame):
        """Process a single frame for face and hand tracking"""
        # Initialize result variables at the start
        face_result = None
        hand_result = None
        pose_result = None  # NEW LINE

        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        self.last_frame = frame.copy()

        # Apply RGB filter
        filtered_frame = self.apply_color_filter(frame.copy())

        # Create display frame
        display_frame = filtered_frame.copy()

        # Add border
        h, w, _ = display_frame.shape
        cv2.rectangle(display_frame, (0, 0), (w-1, h-1), (0, 255, 0), 2)

        # Initialize counters
        face_count = 0
        hand_count = 0
        pose_count = 0  # NEW LINE
        gestures = []

        # Skip detection if menus are showing
        if not self.show_radial_menu and not self.show_gesture_mapping_menu:
            # Face detection
            if self.show_faces and self.face_detector:
                face_result = self.detect_faces(frame)
                if face_result and face_result.face_landmarks:
                    face_count = len(face_result.face_landmarks)
                    for face_landmarks in face_result.face_landmarks:
                        display_frame = self.draw_face_landmarks(display_frame, face_landmarks)

            # Hand detection
            if self.show_hands and self.hand_detector:
                hand_result = self.detect_hands(frame)
                if hand_result and hand_result.hand_landmarks:
                    hand_count = len(hand_result.hand_landmarks)
                    for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                        display_frame = self.draw_hand_landmarks(display_frame, hand_landmarks)

                        # Get gesture for this hand
                        gesture = self.get_gesture(hand_result, idx)
                        if gesture:
                            gestures.append(gesture)

                            # Process gesture for first hand only
                            if idx == 0:
                                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                                self.gesture_controller.process_gesture(gesture, current_time)

            # NEW: Pose detection
            if self.pose_tracker.enabled and self.pose_tracker.pose_detector:
                pose_result = self.pose_tracker.detect_poses(frame)
                if pose_result and pose_result.pose_landmarks:
                    pose_count = len(pose_result.pose_landmarks)
                    display_frame = self.pose_tracker.draw_all_poses(display_frame, pose_result)

        # Update separate data windows
        self.update_data_windows(face_result, hand_result, pose_result, gestures)

        # NEW: DRAW TOUCH VISUALIZATION ON MAIN FRAME
        if hand_result and self.show_multi_hand:
            display_frame = self.draw_touch_visuals_on_frame(display_frame, hand_result)

        # Draw UI menus
        display_frame = self.draw_ui_menus(display_frame)

        # Add text overlay
        if self.show_text and not self.show_radial_menu and not self.show_gesture_mapping_menu:
            display_frame = self.add_text_overlay(display_frame, face_count, hand_count, pose_count, gestures)  # UPDATED

        return display_frame

    # Helper methods to break down the logic
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            return self.face_detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Face detection error: {e}")
            return None

    def detect_hands(self, frame):
        """Detect hands in the frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            return self.hand_detector.recognize_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Hand detection error: {e}")
            return None

    def get_gesture(self, hand_result, hand_index):
        """Extract gesture from hand results"""
        if (hand_result.gestures and
            hand_index < len(hand_result.gestures) and
            hand_result.gestures[hand_index]):
            gesture_data = hand_result.gestures[hand_index][0]
            if gesture_data.score > 0.5:  # Confidence threshold
                return gesture_data.category_name
        return None

    def update_data_windows(self, face_result, hand_result, pose_result, gestures):
        """Update the separate data display windows - UPDATED FOR POSE"""

        # ===== FACE DATA WINDOW =====
        if self.show_face_data:
            try:
                if face_result and face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                    # Get the FIRST FACE's landmarks (list of 468/478 landmarks)
                    face_landmarks_list = face_result.face_landmarks[0]

                    # Check if we actually have landmarks in the list
                    if face_landmarks_list and len(face_landmarks_list) > 0:
                        # Pass the LIST of landmarks to the window
                        self.face_data_window.update_from_landmarks(face_landmarks_list)
                    else:
                        # Empty landmark list
                        self.face_data_window.update_from_landmarks(None)
                else:
                    # No face detected at all
                    self.face_data_window.update_from_landmarks(None)

                self.face_data_window.update_window()

            except Exception as e:
                print(f"Face data window error: {e}")
                import traceback
                traceback.print_exc()  # Show exact error location

        # ===== HAND DATA WINDOW =====
        if self.show_hand_data:
            try:
                if hand_result and hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
                    gesture = gestures[0] if gestures else None
                    # For hands, we pass the FIRST HAND's landmarks (list of 21)
                    hand_landmarks_list = hand_result.hand_landmarks[0]
                    self.hand_data_window.update_from_landmarks(hand_landmarks_list, gesture)
                else:
                    self.hand_data_window.update_from_landmarks(None)
                self.hand_data_window.update_window()
            except Exception as e:
                print(f"Hand data window error: {e}")

        # ===== POSE DATA WINDOW =====
        if self.show_pose_data and hasattr(self, 'pose_data_window') and self.pose_data_window:
            try:
                if pose_result and pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    # Pass the pose landmarks to the window
                    self.pose_data_window.update_from_landmarks(pose_result.pose_landmarks)
                else:
                    # No pose detected
                    self.pose_data_window.update_from_landmarks(None)

                # Update the window display
                self.pose_data_window.update_window()

            except Exception as e:
                print(f"Pose data window error: {e}")
                import traceback
                traceback.print_exc()

        # ===== MULTI-HAND OVERVIEW WINDOW =====
        if self.show_multi_hand:
            try:
                self.multi_hand_window.update_from_results(hand_result)
            except Exception as e:
                print(f"Multi-hand window error: {e}")

    def toggle_pose_data_window(self):
        """Toggle the pose data window"""
        # Make sure attribute exists
        if not hasattr(self, 'show_pose_data'):
            self.show_pose_data = False

        self.show_pose_data = not self.show_pose_data
        status = "ON" if self.show_pose_data else "OFF"
        print(f"  Pose data window: {status}")

        # Create window if it doesn't exist and we're turning it on
        if self.show_pose_data:
            if not hasattr(self, 'pose_data_window') or self.pose_data_window is None:
                self.pose_data_window = PoseDataWindow()
                print("  ✅ Pose data window created")
        else:
            # Optional: Close the window when turning off
            if hasattr(self, 'pose_data_window') and self.pose_data_window:
                try:
                    self.pose_data_window.close()
                except:
                    pass
                self.pose_data_window = None

    def draw_ui_menus(self, frame):
        """Draw UI menus on the frame"""
        if self.show_radial_menu and self.radial_menu:
            frame = self.radial_menu.draw(frame)

        if self.show_gesture_mapping_menu:
            frame = self.gesture_mapping_menu.draw(frame)

        return frame

    def add_text_overlay(self, frame, face_count, hand_count, pose_count, gestures):
        """Add text overlay to the frame - CORRECTED VERSION"""
        renderer = TextRenderer(frame)

        # Title and FPS
        renderer.add_section("FACE & HAND & POSE TRACKER", 10, 30, (0, 255, 0))
        renderer.add_text(f"FPS: {self.fps:.1f}", renderer.get_width()-120, 30, (0, 255, 255), 'medium')

        # Detection counts
        y_offset = 60
        renderer.add_text(f"Faces: {face_count}", 10, y_offset, self.face_color, 'medium')
        y_offset += 25
        renderer.add_text(f"Hands: {hand_count}", 10, y_offset, self.hand_color, 'medium')
        y_offset += 25
        renderer.add_text(f"Poses: {pose_count}", 10, y_offset, self.pose_color, 'medium')
        y_offset += 25

        # Landmark groups status
        if not LANDMARK_GROUPS_AVAILABLE:
            renderer.add_text("Face connections: UNAVAILABLE", 10, y_offset, (255, 50, 50), 'medium')
            y_offset += 25

        # Gestures
        if gestures:
            gesture_text = ", ".join(gestures)
            renderer.add_text(f"Gestures: {gesture_text}", 10, y_offset, (255, 255, 0), 'medium')
            y_offset += 25

        # Status section (bottom left)
        h = renderer.get_height()

        # Start position for status indicators (adjust as needed)
        status_start_y = h - 250

        # Add data window statuses - CHECK THESE ATTRIBUTES EXIST
        face_window_status = "ON" if hasattr(self, 'show_face_data') and self.show_face_data else "OFF"
        face_window_color = (100, 200, 255) if face_window_status == "ON" else (100, 100, 100)
        renderer.add_status("Face Window", face_window_status, 10, status_start_y, face_window_color)
        status_start_y += 25

        hand_window_status = "ON" if hasattr(self, 'show_hand_data') and self.show_hand_data else "OFF"
        hand_window_color = (255, 150, 100) if hand_window_status == "ON" else (100, 100, 100)
        renderer.add_status("Hand Window", hand_window_status, 10, status_start_y, hand_window_color)
        status_start_y += 25

        # IMPORTANT: Add this attribute to your class if it doesn't exist
        if not hasattr(self, 'show_pose_data'):
            self.show_pose_data = False  # Initialize if missing

        pose_window_status = "ON" if self.show_pose_data else "OFF"
        pose_window_color = (255, 0, 255) if pose_window_status == "ON" else (100, 100, 100)
        renderer.add_status("Pose Window", pose_window_status, 10, status_start_y, pose_window_color)
        status_start_y += 25

        multi_window_status = "ON" if hasattr(self, 'show_multi_hand') and self.show_multi_hand else "OFF"
        multi_window_color = (150, 255, 150) if multi_window_status == "ON" else (100, 100, 100)
        renderer.add_status("Multi-Hand", multi_window_status, 10, status_start_y, multi_window_color)
        status_start_y += 25

        # Existing menu statuses
        radial_menu_status = "ON" if hasattr(self, 'show_radial_menu') and self.show_radial_menu else "OFF"
        radial_menu_color = (255, 100, 255) if radial_menu_status == "ON" else (100, 100, 100)
        renderer.add_status("Radial Menu", radial_menu_status, 10, status_start_y, radial_menu_color)
        status_start_y += 25

        gesture_menu_status = "ON" if hasattr(self, 'show_gesture_mapping_menu') and self.show_gesture_mapping_menu else "OFF"
        gesture_menu_color = (100, 255, 255) if gesture_menu_status == "ON" else (100, 100, 100)
        renderer.add_status("Gesture Map", gesture_menu_status, 10, status_start_y, gesture_menu_color)
        status_start_y += 25

        # Gesture controller status
        gc_status = "ON" if hasattr(self, 'gesture_controller') and self.gesture_controller.enabled else "OFF"
        gc_color = (0, 255, 0) if gc_status == "ON" else (100, 100, 100)
        renderer.add_status("Gesture Ctrl", gc_status, 10, status_start_y, gc_color)
        status_start_y += 25

        # RGB status
        color_status = []
        if hasattr(self, 'show_red') and self.show_red: color_status.append("R")
        if hasattr(self, 'show_green') and self.show_green: color_status.append("G")
        if hasattr(self, 'show_blue') and self.show_blue: color_status.append("B")
        renderer.add_status("RGB", ''.join(color_status) if color_status else "OFF",
                          10, status_start_y, (200, 200, 200))
        status_start_y += 25

        # Tracking statuses
        face_status = "ON" if hasattr(self, 'show_faces') and self.show_faces else "OFF"
        renderer.add_status("Face", face_status, 10, status_start_y, self.face_color)
        status_start_y += 15

        hand_status = "ON" if hasattr(self, 'show_hands') and self.show_hands else "OFF"
        renderer.add_status("Hand", hand_status, 10, status_start_y, self.hand_color)
        status_start_y += 15

        # Check if pose_tracker exists
        if hasattr(self, 'pose_tracker'):
            pose_status = "ON" if self.pose_tracker.enabled else "OFF"
            renderer.add_status("Pose", pose_status, 10, status_start_y, self.pose_color)
            status_start_y += 15

        # Updated instructions - added pose info
        instructions = "r/g/b: RGB | f/h/p: Trackers | F/H/P/M: Data Windows | 1-4: Details | 5-7: Pose | t: Text | m: Menu | G: Gesture Map | ?: Help | q: Quit"
        renderer.add_text(instructions, 10, h-5, (200, 200, 200), 'small')

        return renderer.frame

    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        if current_time - self.start_time > 1.0:
            self.fps = self.frame_count / (current_time - self.start_time)
            self.frame_count = 0
            self.start_time = current_time

    def display_help(self):
        """Display help screen"""
        print("\n" + "="*70)
        print("COMBINED FACE & HAND TRACKER")
        print("="*70)

        print("\n🎮 CONTROL PANEL:")

        print("  SYSTEM:")
        print("    [q] - Quit program")
        print("    [?] - Show this help screen")

        print("\n  RGB CONTROLS:")
        print("    [r] - Toggle RED channel")
        print("    [g] - Toggle GREEN channel")
        print("    [b] - Toggle BLUE channel")

        print("\n  TRACKING TOGGLES:")
        print("    [f] - Toggle face tracking")
        print("    [h] - Toggle hand tracking")

        print("\n  POSE CONTROLS:")
        print("    [p] - Toggle pose tracking")
        print("    [5] - Toggle skeleton drawing")
        print("    [6] - Toggle landmark points")
        print("    [7] - Toggle bounding boxes")
        print("    [P] - Toggle pose data window")

        print("\n  DATA WINDOWS:")
        print("    [F] - Toggle Face data window")
        print("    [H] - Toggle Hand data window")
        print("    [M] - Toggle Multi-hand window")

        print("\n  LANDMARK DETAILS:")
        print("    [1] - Toggle face landmarks (dots)")
        print("    [2] - Toggle face connections (mesh)")
        print("    [3] - Toggle hand landmarks")
        print("    [4] - Toggle hand connections")
        print("    [B] - Toggle FACE bounding boxes")
        print("    [X] - Toggle HAND bounding boxes")

        print("\n  TOUCH DETECTION:")
        print("    [+] - Decrease touch threshold (more sensitive)")
        print("    [-] - Increase touch threshold (less sensitive)")
        print("    [[] - Decrease cooldown (faster recovery)")
        print("    []] - Increase cooldown (slower recovery)")
        print("    [0] - Reset all touches")

        print("\n  UI CONTROLS:")
        print("    [t] - Toggle text overlay")
        print("    [m] - Toggle radial menu")
        print("    [G] - Toggle gesture mapping menu")

        print("\n  GESTURE CONTROLLER:")
        print("    [c] - Toggle gesture controller on/off")
        print("    [C] - Show gesture controller help")

        print("\n  RADIAL MENU SHORTCUTS (when menu open):")
        print("    [5][6][7][8] - Activate buttons 5-8")

        print("="*70)

        if not LANDMARK_GROUPS_AVAILABLE:
            print("\n⚠️  IMPORTANT: landmark_groups.py not found!")
            print("   Face connections will not be drawn.")
            print("   Make sure landmark_groups.py is in the same folder.")

        # Show gesture controller status
        gc_status = self.gesture_controller
        print(f"\nGesture Controller: {'✅ ON' if gc_status.enabled else '❌ OFF'}")
        if gc_status.current_gesture:
            print(f"Last gesture: {gc_status.current_gesture}")

        # Show menu statuses
        print(f"Radial Menu: {'✅ ON' if self.show_radial_menu else '❌ OFF'}")
        print(f"Gesture Mapping Menu: {'✅ ON' if self.show_gesture_mapping_menu else '❌ OFF'}")

        print("\nBlue = Face tracking | Green = Hand tracking")
        print("Bounding boxes are OFF by default")

    def run(self):
        """Main loop"""
        # Initialize
        if not self.initialize_camera():
            return

        print("\nInitializing trackers...")
        face_loaded = self.initialize_face_detector()
        hand_loaded = self.initialize_hand_detector()
        pose_loaded = self.initialize_pose_detector()  # NEW LINE

        self.initialize_obs()

        if not face_loaded:
            print("⚠️ Running without face detection (face_landmarker.task not found)")
        if not hand_loaded:
            print("⚠️ Running without hand detection (gesture_recognizer.task not found)")
        if not pose_loaded:  # NEW LINE
            print("⚠️ Running without pose detection")  # NEW LINE

        # Create window
        cv2.namedWindow("Face & Hand Tracker", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Face & Hand Tracker", 100, 100)

        cv2.resizeWindow("Face & Hand Tracker", 800, 640)  # Width, Height


        # Display help
        self.display_help()

        # Initialize FPS tracking
        self.start_time = cv2.getTickCount() / cv2.getTickFrequency()

        print("\n🚀 Starting combined tracker...")
        print("Press '?' for help, 'q' to quit")
        print("Press 'm' for radial menu, 'G' for gesture mapping")
        if not LANDMARK_GROUPS_AVAILABLE:
            print("⚠️  Face connections unavailable - add landmark_groups.py to enable")

        # Main loop
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            # Calculate FPS
            self.calculate_fps()

            # Process frame
            display_frame = self.process_frame(frame)

            # Display frame
            cv2.imshow("Face & Hand Tracker", display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Handle gesture mapping menu keys
            if self.show_gesture_mapping_menu:
                if key == ord('s'):  # Save mapping
                    if self.gesture_mapping_menu.save_mapping():
                        print("✓ Gesture mapping saved")
                elif key == ord('g') or key == ord('G'):  # Close menu
                    self.toggle_gesture_mapping_menu()
                    continue

            # Handle regular key actions
            if key in self.key_actions:
                action = self.key_actions[key]
                if action == self.quit_action:
                    if action():
                        break
                else:
                    action()

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.face_detector:
            self.face_detector.close()
        # Close all data windows
        if hasattr(self, 'face_data_window'):
            self.face_data_window.close()
        if hasattr(self, 'hand_data_window'):
            self.hand_data_window.close()
        if hasattr(self, 'multi_hand_window'):
            self.multi_hand_window.close()
        if self.hand_detector:
            self.hand_detector.close()
        if self.obs_ws:
            self.obs_ws.disconnect()
            print("🔌 Disconnected from OBS")
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function"""
    print("="*70)
    print("COMBINED FACE & HAND TRACKER")
    print("="*70)
    print("Integrated with anatomical landmark groups")
    print("Requires: face_landmarker.task and gesture_recognizer.task")
    print("Optional: landmark_groups.py for face connections")
    print("\nSEPARATED SYSTEMS:")
    print("  1. Radial Menu: 8 clickable buttons for quick actions")
    print("  2. Gesture Mapping: Map gestures to any action")
    print("  3. Gestures are automatically detected and executed")
    print("\nCONTROLS:")
    print("  [m] - Toggle radial menu (click buttons 1-8)")
    print("  [G] - Toggle gesture mapping editor")
    print("  Gesture mappings saved to gesture_mappings.json")
    print("="*70)

    # Create and run the system
    tracker = CombinedTracker()
    tracker.run()

if __name__ == "__main__":
    main()
