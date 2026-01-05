import cv2
import numpy as np
import time
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("="*70)
print("MEDIAPIPE LANDMARK PICKER TOOL")
print("="*70)
print("Interactive tool to create custom landmark groups")
print("="*70)

class LandmarkPicker:
    def __init__(self):
        self.cap = None
        self.detector = None
        self.mp = mp
        
        # Landmark storage
        self.landmarks = []  # Current detected landmarks
        self.selected_indices = []  # Landmark indices selected
        
        # Groups management
        self.groups = {}  # Dictionary of groups
        self.current_group_name = "outline"  # Start with default name
        
        # UI state
        self.is_running = False
        self.show_numbers = True
        self.show_all_points = True
        self.black_background = False
        self.frame_frozen = False
        self.frozen_frame = None
        self.naming_mode = False
        self.name_input = ""
        self.hide_grouped_points = False  # NEW: Toggle to hide grouped points
        
        # Colors - SIMPLIFIED as requested
        self.colors = {
            'ungrouped': (255, 255, 255),  # WHITE - NOT GROUPED
            'grouped': (255, 0, 0),        # BLUE - GROUPED
            'selected': (0, 255, 255),     # YELLOW - SELECTED
            'text': (255, 255, 255)        # White text
        }
        
        # Initialize MediaPipe
        self.initialize_mediapipe()
        
    def initialize_mediapipe(self):
        """Initialize MediaPipe FaceLandmarker"""
        try:
            model_path = "face_landmarker.task"
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("✅ MediaPipe FaceLandmarker initialized")
            return True
            
        except Exception as e:
            print(f"❌ MediaPipe initialization error: {e}")
            print("Make sure 'face_landmarker.task' is in the same directory")
            return False
    
    def initialize_camera(self):
        """Initialize the camera"""
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("❌ Cannot open camera")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ Camera initialized: {width}x{height}")
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select landmarks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find the closest landmark to the click
            min_distance = 20  # Pixel radius for selection
            closest_index = -1
            
            for i, (lx, ly) in enumerate(self.landmarks):
                distance = np.sqrt((x - lx)**2 + (y - ly)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i
            
            if closest_index != -1:
                # Toggle selection
                if closest_index in self.selected_indices:
                    self.selected_indices.remove(closest_index)
                    print(f"  Deselected landmark {closest_index}")
                else:
                    self.selected_indices.append(closest_index)
                    print(f"  Selected landmark {closest_index}")
    
    def detect_landmarks(self, frame):
        """Detect landmarks in frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
            
            if not hasattr(self, '_frame_timestamp_ms'):
                self._frame_timestamp_ms = 0
            self._frame_timestamp_ms += 33
            
            detection_result = self.detector.detect_for_video(mp_image, self._frame_timestamp_ms)
            
            self.landmarks = []
            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]
                h, w, _ = frame.shape
                
                for landmark in face_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    self.landmarks.append((x, y))
            
            return True
            
        except Exception as e:
            print(f"Detection error: {e}")
            self.landmarks = []
            return False
    
    def get_landmark_color(self, index):
        """Get color for a specific landmark based on its group membership"""
        # Check if currently selected - YELLOW OVERRIDES EVERYTHING
        if index in self.selected_indices:
            return self.colors['selected']
        
        # If hiding grouped points and point is grouped, return None
        if self.hide_grouped_points and self.get_landmark_group(index) is not None:
            return None  # Don't draw grouped points
        
        # Check which group this landmark belongs to
        for group_name, indices in self.groups.items():
            if index in indices:
                return self.colors['grouped']  # BLUE for grouped
        
        # Default color (ungrouped) - WHITE
        return self.colors['ungrouped']
    
    def get_landmark_group(self, index):
        """Get which group a landmark belongs to"""
        for group_name, indices in self.groups.items():
            if index in indices:
                return group_name
        return None
    
    def draw_landmarks(self, frame):
        """Draw landmarks on frame"""
        if self.black_background:
            # Create black background
            output = np.zeros_like(frame)
        else:
            output = frame.copy()
            
        h, w = output.shape[:2]
        
        # Draw all landmarks if enabled
        if self.show_all_points and self.landmarks:
            for i, (x, y) in enumerate(self.landmarks):
                # Get appropriate color
                color = self.get_landmark_color(i)
                
                # Skip if color is None (hiding grouped points)
                if color is None:
                    continue
                
                # Determine size based on status
                if i in self.selected_indices:
                    radius = 5  # Larger for selected points (YELLOW)
                elif self.get_landmark_group(i) is not None:
                    radius = 4  # Medium for grouped points (BLUE)
                else:
                    radius = 2  # Small for ungrouped points (WHITE)
                
                cv2.circle(output, (x, y), radius, color, -1)
                
                # Draw landmark numbers if enabled
                if self.show_numbers and i % 10 == 0:  # Show every 10th number
                    # Skip if this is a grouped point and we're hiding them
                    if not (self.hide_grouped_points and self.get_landmark_group(i) is not None):
                        cv2.putText(output, str(i), (x+5, y-5), 
                                   cv2.FONT_HERSHEY_COMPLEX, 0.3, self.colors['text'], 1)
        
        # Draw UI overlay
        self.draw_ui_overlay(output, w, h)
        
        return output
    
    def draw_ui_overlay(self, frame, width, height):
        """Draw UI information overlay"""
        # Header with frozen indicator
        header = "LANDMARK PICKER TOOL"
        if self.frame_frozen:
            header += " [FROZEN]"
        cv2.putText(frame, header, (10, 30),
                   cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        
        # Current group info
        group_info = f"Current Group: {self.current_group_name}"
        cv2.putText(frame, group_info, (10, 60),
                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)
        
        # Selection info
        selection_info = f"Selected: {len(self.selected_indices)} points"
        cv2.putText(frame, selection_info, (10, 90),
                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 200, 0), 2)
        
        # Groups info
        groups_info = f"Groups Created: {len(self.groups)}/30"
        cv2.putText(frame, groups_info, (10, 120),
                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (200, 200, 255), 2)
        
        # Mode indicators
        bg_mode = "BLACK" if self.black_background else "CAMERA"
        cv2.putText(frame, f"Background: {bg_mode}", (10, 150),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        
        freeze_status = "FROZEN" if self.frame_frozen else "LIVE"
        freeze_color = (0, 0, 255) if self.frame_frozen else (0, 255, 0)
        cv2.putText(frame, f"Frame: {freeze_status}", (10, 170),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, freeze_color, 1)
        
        # Group visibility
        group_vis = "HIDDEN" if self.hide_grouped_points else "VISIBLE"
        group_vis_color = (0, 0, 255) if self.hide_grouped_points else (0, 255, 0)
        cv2.putText(frame, f"Grouped points: {group_vis}", (10, 190),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, group_vis_color, 1)
        
        # Color Legend
        y_pos = 220
        cv2.putText(frame, "COLOR GUIDE:", (10, y_pos),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        # White - Ungrouped
        cv2.circle(frame, (10, y_pos + 5), 5, self.colors['ungrouped'], -1)
        cv2.putText(frame, "White: Ungrouped", (25, y_pos + 10),
                   cv2.FONT_HERSHEY_COMPLEX, 0.4, (200, 200, 200), 1)
        y_pos += 20
        
        # Blue - Grouped
        cv2.circle(frame, (10, y_pos + 5), 5, self.colors['grouped'], -1)
        cv2.putText(frame, "Blue: Grouped", (25, y_pos + 10),
                   cv2.FONT_HERSHEY_COMPLEX, 0.4, (200, 200, 200), 1)
        y_pos += 20
        
        # Yellow - Selected
        cv2.circle(frame, (10, y_pos + 5), 5, self.colors['selected'], -1)
        cv2.putText(frame, "Yellow: Selected", (25, y_pos + 10),
                   cv2.FONT_HERSHEY_COMPLEX, 0.4, (200, 200, 200), 1)
        y_pos += 30
        
        # Show existing groups
        if self.groups:
            cv2.putText(frame, f"Existing Groups ({len(self.groups)}):", (10, y_pos),
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            
            # Show groups in columns
            col1_x = 10
            col2_x = 150
            col3_x = 290
            
            groups_list = list(self.groups.items())
            for i, (group_name, indices) in enumerate(groups_list):
                col = i % 3
                row = i // 3
                
                if row < 6:  # Show max 18 groups (6 rows × 3 columns)
                    x_pos = [col1_x, col2_x, col3_x][col]
                    y_row_pos = y_pos + (row * 20)
                    
                    cv2.circle(frame, (x_pos, y_row_pos + 5), 4, self.colors['grouped'], -1)
                    cv2.putText(frame, f"{group_name}: {len(indices)}", 
                              (x_pos + 15, y_row_pos + 10),
                              cv2.FONT_HERSHEY_COMPLEX, 0.4, (200, 200, 200), 1)
        
        # Naming mode overlay
        if self.naming_mode:
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Naming prompt
            cv2.putText(frame, "ENTER GROUP NAME", (width//2 - 100, height//2 - 50),
                       cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            
            # Input box
            cv2.rectangle(frame, (width//2 - 150, height//2 - 20), 
                         (width//2 + 150, height//2 + 20), (255, 255, 255), 2)
            
            # Current input
            display_text = self.name_input if self.name_input else "Type name..."
            cv2.putText(frame, display_text, (width//2 - 140, height//2 + 10),
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Press ENTER to confirm, ESC to cancel", 
                       (width//2 - 180, height//2 + 70),
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instructions (only show when not in naming mode)
        if not self.naming_mode:
            instructions = [
                "INSTRUCTIONS:",
                "1. Press 'f' to freeze/unfreeze frame",
                "2. Click on white dots to select (yellow)",
                "3. Press 'g' to save to current group (blue)",
                "4. Press 'n' to name new group",
                "5. Press 'h' to hide/show grouped points",
                "6. Press 'c' to clear selection",
                "7. Press 'b' to toggle black background",
                "8. Press 'e' to export groups",
                "9. Press 's' to show/hide all points",
                "10. Press 'd' to show/hide numbers",
                "11. Press 'q' to quit"
            ]
            
            y_pos = height - 260
            for i, line in enumerate(instructions):
                cv2.putText(frame, line, (10, y_pos + i*20),
                           cv2.FONT_HERSHEY_COMPLEX, 0.4, (200, 200, 200), 1)
    
    def start_naming_mode(self):
        """Start GUI naming mode for new group"""
        self.naming_mode = True
        self.name_input = ""
        print("  Entering naming mode... Type group name")
    
    def finish_naming_mode(self, confirm=True):
        """Finish naming mode and create group"""
        self.naming_mode = False
        
        if confirm and self.name_input.strip():
            self.current_group_name = self.name_input.strip()
            
            # Create group if we have selected points
            if self.selected_indices:
                # Check limit when actually creating group
                if len(self.groups) >= 30:
                    print("❌ Maximum 30 groups reached!")
                    return
                    
                self.groups[self.current_group_name] = set(self.selected_indices.copy())
                print(f"✅ Created group: {self.current_group_name}")
                print(f"   Added {len(self.selected_indices)} points")
                
                # Clear selection for next group
                self.selected_indices.clear()
            else:
                # Just set the name, group will be created when points are saved
                print(f"✅ Set group name to: {self.current_group_name}")
                print("   Select points and press 'g' to add them")
        
        self.name_input = ""
    
    def save_to_current_group(self):
        """Save current selection to current group"""
        if not self.current_group_name:
            print("❌ Please name a group first (press 'n')")
            return
        
        if not self.selected_indices:
            print("❌ No points selected")
            return
        
        if len(self.groups) >= 30 and self.current_group_name not in self.groups:
            print("❌ Maximum 30 groups reached!")
            return
        
        if self.current_group_name not in self.groups:
            self.groups[self.current_group_name] = set()
        
        current_set = self.groups[self.current_group_name]
        for idx in self.selected_indices:
            current_set.add(idx)
        
        self.groups[self.current_group_name] = current_set
        print(f"✅ Saved {len(self.selected_indices)} points to group: {self.current_group_name}")
        print(f"   Total in group: {len(current_set)} points")
        
        # Clear selection
        self.selected_indices.clear()
    
    def clear_selection(self):
        """Clear current selection"""
        self.selected_indices.clear()
        print("✅ Cleared current selection")
    
    def toggle_freeze_frame(self):
        """Toggle frame freezing"""
        self.frame_frozen = not self.frame_frozen
        if self.frame_frozen and self.cap is not None:
            # Capture current frame
            ret, self.frozen_frame = self.cap.read()
            if ret:
                # Detect landmarks on frozen frame
                self.detect_landmarks(self.frozen_frame)
        status = "FROZEN" if self.frame_frozen else "LIVE"
        print(f"  Frame: {status}")
    
    def toggle_hide_grouped_points(self):
        """Toggle visibility of grouped points"""
        self.hide_grouped_points = not self.hide_grouped_points
        status = "HIDDEN" if self.hide_grouped_points else "VISIBLE"
        print(f"  Grouped points: {status}")
    
    def export_to_python(self):
        """Export groups as Python code"""
        if not self.groups:
            print("❌ No groups to export")
            return
        
        print("\n" + "="*70)
        print("EXPORTED PYTHON CODE")
        print("="*70)
        
        # Create Python dictionary code
        python_code = "        self.feature_groups = {\n"
        
        for group_name, indices_set in self.groups.items():
            indices = sorted(list(indices_set))
            python_code += f"            '{group_name}': {indices},\n"
        
        python_code += "        }"
        
        print(python_code)
        print("\n" + "="*70)
        
        # Also export to file
        with open("landmark_groups.py", "w") as f:
            f.write("feature_groups = ")
            
            # Convert sets to lists for JSON serialization
            export_dict = {}
            for group_name, indices_set in self.groups.items():
                export_dict[group_name] = sorted(list(indices_set))
            
            f.write(str(export_dict))
        
        print("✅ Also saved to 'landmark_groups.py'")
        
        # Export as JSON for sharing
        with open("landmark_groups.json", "w") as f:
            json.dump(export_dict, f, indent=2)
        
        print("✅ Also saved to 'landmark_groups.json'")
    
    def handle_keyboard_naming(self, key):
        """Handle keyboard input during naming mode"""
        if key == 13:  # ENTER key
            self.finish_naming_mode(confirm=True)
            return True
        elif key == 27:  # ESC key
            self.finish_naming_mode(confirm=False)
            return True
        elif key == 8:  # BACKSPACE
            self.name_input = self.name_input[:-1]
        elif 32 <= key <= 126:  # Printable ASCII characters
            if len(self.name_input) < 20:  # Limit name length
                self.name_input += chr(key)
        return False
    
    def run(self):
        """Main run loop"""
        if not self.initialize_mediapipe():
            return
        
        if not self.initialize_camera():
            return
        
        cv2.namedWindow("Landmark Picker")
        cv2.setMouseCallback("Landmark Picker", self.mouse_callback)
        
        print("\n" + "="*70)
        print("STARTING LANDMARK PICKER")
        print("="*70)
        print("\nCOLOR SCHEME:")
        print("  WHITE: Ungrouped landmarks")
        print("  BLUE: Grouped landmarks")
        print("  YELLOW: Currently selected landmarks")
        print("\nMAXIMUM: 30 groups")
        print("\nWORKFLOW:")
        print("  1. Freeze frame (f) when face is positioned")
        print("  2. Select points (click white dots → yellow)")
        print("  3. Name group (n) and type name in window")
        print("  4. Save to group (g) → points turn blue")
        print("  5. Press 'h' to hide grouped points for fine detail")
        print("  6. Repeat for up to 30 groups")
        
        self.is_running = True
        
        while self.is_running:
            # Get frame (either live or frozen)
            if self.frame_frozen and self.frozen_frame is not None:
                frame = self.frozen_frame.copy()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                self.frozen_frame = None
            
            # Detect landmarks (only if not frozen or if we have new frame)
            if not self.frame_frozen or self.frozen_frame is None:
                self.detect_landmarks(frame)
            
            # Draw landmarks
            display_frame = self.draw_landmarks(frame)
            
            # Show frame
            cv2.imshow("Landmark Picker", display_frame)
            
            # Handle keyboard input
            if self.naming_mode:
                # In naming mode, wait for key press with delay
                key = cv2.waitKey(0) & 0xFF
                self.handle_keyboard_naming(key)
            else:
                # Normal mode
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✅ Quitting...")
                    self.is_running = False
                    break
                
                elif key == ord('f'):  # Toggle freeze frame
                    self.toggle_freeze_frame()
                
                elif key == ord('n'):  # Start naming new group
                    self.start_naming_mode()
                
                elif key == ord('g'):  # Save to current group
                    self.save_to_current_group()
                
                elif key == ord('h'):  # Toggle hide grouped points
                    self.toggle_hide_grouped_points()
                
                elif key == ord('c'):  # Clear selection
                    self.clear_selection()
                
                elif key == ord('b'):  # Toggle black background
                    self.black_background = not self.black_background
                    status = "BLACK" if self.black_background else "CAMERA"
                    print(f"  Background: {status}")
                
                elif key == ord('e'):  # Export groups
                    self.export_to_python()
                
                elif key == ord('s'):  # Toggle show all points
                    self.show_all_points = not self.show_all_points
                    status = "ON" if self.show_all_points else "OFF"
                    print(f"  Show all points: {status}")
                
                elif key == ord('d'):  # Toggle show numbers
                    self.show_numbers = not self.show_numbers
                    status = "ON" if self.show_numbers else "OFF"
                    print(f"  Show numbers: {status}")
                
                elif key == ord('l'):  # List all groups
                    print("\n📋 CURRENT GROUPS:")
                    for group_name, indices in self.groups.items():
                        print(f"  {group_name}: {len(indices)} points")
                
                elif key == ord('x'):  # Delete current group
                    if self.current_group_name in self.groups:
                        del self.groups[self.current_group_name]
                        print(f"✅ Deleted group: {self.current_group_name}")
                        # Set to next available group or first group
                        if self.groups:
                            self.current_group_name = list(self.groups.keys())[-1]
                        else:
                            self.current_group_name = "new_group"
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final export prompt
        if self.groups:
            export = input("\nExport groups before exiting? (y/n): ").strip().lower()
            if export == 'y':
                self.export_to_python()

def main():
    """Main function"""
    picker = LandmarkPicker()
    picker.run()

if __name__ == "__main__":
    main()