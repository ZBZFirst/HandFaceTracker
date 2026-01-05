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
OBS_PASSWORD = "OBSPASSWORDHERE"

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
        self.enabled = True
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
            ord('q'): self.quit_action,
            ord('r'): lambda: self.toggle('show_red', 'RED channel'),
            ord('g'): lambda: self.toggle('show_green', 'GREEN channel'),
            ord('b'): lambda: self.toggle('show_blue', 'BLUE channel'),
            ord('f'): lambda: self.toggle('show_faces', 'Face tracking'),
            ord('h'): lambda: self.toggle('show_hands', 'Hand tracking'),
            ord('1'): lambda: self.toggle('show_face_landmarks', 'Face landmarks'),
            ord('2'): lambda: self.toggle('show_face_connections', 'Face connections'),
            ord('3'): lambda: self.toggle('show_hand_landmarks', 'Hand landmarks'),
            ord('4'): lambda: self.toggle('show_hand_connections', 'Hand connections'),
            ord('B'): lambda: self.toggle('show_face_boxes', 'Face bounding boxes'),
            ord('x'): lambda: self.toggle('show_hand_boxes', 'Hand bounding boxes'),
            ord('t'): lambda: self.toggle('show_text', 'Text overlay'),
            ord('m'): self.toggle_radial_menu,  # Toggle radial action menu
            ord('G'): self.toggle_gesture_mapping_menu,  # Toggle gesture mapping menu
            ord('?'): self.display_help,
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

    def open_gesture_mapping_menu(self):
        """Public method to open gesture mapping menu (can be called from gestures)"""
        self.toggle_gesture_mapping_menu()

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

    def toggle_gesture_mapping_menu(self):
        """Toggle the gesture mapping menu"""
        self.show_gesture_mapping_menu = self.gesture_mapping_menu.toggle()
        status = "ON" if self.show_gesture_mapping_menu else "OFF"
        print(f"  Gesture Mapping Menu: {status}")

        if self.show_gesture_mapping_menu:
            cv2.setMouseCallback("Face & Hand Tracker", self.mouse_callback)
        elif not self.show_radial_menu:
            cv2.setMouseCallback("Face & Hand Tracker", lambda *args: None)

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

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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
        gestures = []

        # Process face detection (only if menus are not showing)
        if not self.show_radial_menu and not self.show_gesture_mapping_menu:
            if self.show_faces and self.face_detector:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                    face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)

                    if face_result.face_landmarks:
                        face_count = len(face_result.face_landmarks)
                        for face_landmarks in face_result.face_landmarks:
                            display_frame = self.draw_face_landmarks(display_frame, face_landmarks)

                except Exception as e:
                    print(f"Face detection error: {e}")

        # Process hand detection (only if menus are not showing)
        if not self.show_radial_menu and not self.show_gesture_mapping_menu:
            if self.show_hands and self.hand_detector:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                    hand_result = self.hand_detector.recognize_for_video(mp_image, timestamp_ms)

                    if hand_result.hand_landmarks:
                        hand_count = len(hand_result.hand_landmarks)
                        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                            display_frame = self.draw_hand_landmarks(display_frame, hand_landmarks)

                            # Get gesture
                            if idx < len(hand_result.gestures):
                                gesture_list = hand_result.gestures[idx]
                                if gesture_list:
                                    gesture = gesture_list[0]
                                    if gesture.score > 0.5:  # Confidence threshold
                                        gestures.append(gesture.category_name)

                                        # Pass to gesture controller (use first hand only)
                                        if idx == 0:
                                            current_time = cv2.getTickCount() / cv2.getTickFrequency()
                                            self.gesture_controller.process_gesture(
                                                gesture.category_name,
                                                current_time
                                            )

                except Exception as e:
                    print(f"Hand detection error: {e}")

        # Draw radial menu if enabled
        if self.show_radial_menu and self.radial_menu:
            display_frame = self.radial_menu.draw(display_frame)

        # Draw gesture mapping menu if enabled
        if self.show_gesture_mapping_menu:
            display_frame = self.gesture_mapping_menu.draw(display_frame)

        # Add text overlay (only if menus are not showing and show_text is True)
        if self.show_text and not self.show_radial_menu and not self.show_gesture_mapping_menu:
            # Create text renderer
            renderer = TextRenderer(display_frame)

            # Title
            renderer.add_section("FACE & HAND TRACKER", 10, 30, (0, 255, 0))

            # FPS
            renderer.add_text(f"FPS: {self.fps:.1f}", renderer.get_width()-120, 30, (0, 255, 255), 'medium')

            # Detection counts
            y_offset = 60
            renderer.add_text(f"Faces: {face_count}", 10, y_offset, self.face_color, 'medium')
            y_offset += 25

            renderer.add_text(f"Hands: {hand_count}", 10, y_offset, self.hand_color, 'medium')
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

            # Menu statuses
            radial_menu_status = "ON" if self.show_radial_menu else "OFF"
            radial_menu_color = (255, 100, 255) if self.show_radial_menu else (100, 100, 100)
            renderer.add_status("Radial Menu", radial_menu_status, 10, h-200, radial_menu_color)

            gesture_menu_status = "ON" if self.show_gesture_mapping_menu else "OFF"
            gesture_menu_color = (100, 255, 255) if self.show_gesture_mapping_menu else (100, 100, 100)
            renderer.add_status("Gesture Map", gesture_menu_status, 10, h-175, gesture_menu_color)

            # Gesture controller status
            gc_status = "ON" if self.gesture_controller.enabled else "OFF"
            gc_color = (0, 255, 0) if self.gesture_controller.enabled else (100, 100, 100)
            renderer.add_status("Gesture Ctrl", gc_status, 10, h-150, gc_color)

            # RGB status
            color_status = []
            if self.show_red: color_status.append("R")
            if self.show_green: color_status.append("G")
            if self.show_blue: color_status.append("B")
            renderer.add_status("RGB", ''.join(color_status) if color_status else "OFF",
                              10, h-125, (200, 200, 200))

            # Face tracking status
            face_status = "ON" if self.show_faces else "OFF"
            renderer.add_status("Face", face_status, 10, h-100, self.face_color)

            # Hand tracking status
            hand_status = "ON" if self.show_hands else "OFF"
            renderer.add_status("Hand", hand_status, 10, h-75, self.hand_color)

            # Bounding box status
            face_box_status = "ON" if self.show_face_boxes else "OFF"
            hand_box_status = "ON" if self.show_hand_boxes else "OFF"
            renderer.add_status("Face Box", face_box_status, 10, h-50, self.face_color)
            renderer.add_status("Hand Box", hand_box_status, 10, h-25, self.hand_color)

            # Instructions
            instructions = "r/g/b: RGB | f/h: Trackers | 1-4: Details | B/x: Boxes | t: Text | m: Radial Menu | G: Gesture Map | ?: Help | q: Quit"
            renderer.add_text(instructions, 10, h-5, (200, 200, 200), 'small')

        return display_frame

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
        print("  RGB CONTROLS:")
        print("    [r] - Toggle RED channel")
        print("    [g] - Toggle GREEN channel")
        print("    [b] - Toggle BLUE channel")
        print("")
        print("  TRACKER TOGGLES:")
        print("    [f] - Toggle face tracking")
        print("    [h] - Toggle hand tracking")
        print("    [1] - Toggle face landmarks (dots)")
        print("    [2] - Toggle face connections (mesh)")
        print("    [3] - Toggle hand landmarks")
        print("    [4] - Toggle hand connections")
        print("    [B] - Toggle FACE bounding boxes")
        print("    [x] - Toggle HAND bounding boxes")
        print("    [t] - Toggle text overlay")
        print("")
        print("  MENU SYSTEMS:")
        print("    [m] - Toggle radial menu (8 clickable buttons)")
        print("    [G] - Toggle gesture mapping menu")
        print("")
        print("  GESTURE CONTROLLER:")
        print("    [c] - Toggle gesture controller on/off")
        print("    [C] - Show gesture controller help")
        print("")
        print("  SYSTEM:")
        print("    [?] - Show this help screen")
        print("    [q] - Quit program")
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

        self.initialize_obs()

        if not face_loaded:
            print("⚠️ Running without face detection (face_landmarker.task not found)")
        if not hand_loaded:
            print("⚠️ Running without hand detection (gesture_recognizer.task not found)")

        # Create window
        cv2.namedWindow("Face & Hand Tracker", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Face & Hand Tracker", 100, 100)

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
