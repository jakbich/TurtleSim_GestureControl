

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import mediapipe as mp

class GestureControlNode(Node):
    def __init__(self):
        super().__init__('gesture_control_node')
        self.publisher_ = self.create_publisher(Twist, 'turtle1/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.gesture_control_callback)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        print("hello3")


    
    def thumb_orientation(self, hand_landmarks):
        """Simple logic to detect if the thumb is up."""
        # Assuming the hand is right hand, and landmarks[4] is the tip of the thumb.
        # Adjust the logic based on your requirements and hand orientation.
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP.value]
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP.value]

        # A basic check: if the thumb tip's y-coordinate is less (higher in the image) than the index finger tip's y-coordinate.


        region = 0.1

        if thumb_tip.y < index_tip.y and thumb_tip.x > index_tip.x - region and thumb_tip.x < index_tip.x + region:
            return 'up'
        
        if thumb_tip.y > index_tip.y and thumb_tip.x > index_tip.x - region and thumb_tip.x < index_tip.x + region:
            return 'down'
        
        if thumb_tip.x < index_tip.x:
            return 'right'
        
        else:
            return 'left'


    def gesture_control_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture image.')
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Draw the hand annotations on the frame.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Gesture recognition
                thumb_orientation = self.thumb_orientation(hand_landmarks)
                print(thumb_orientation)


                twist = Twist() 

                if thumb_orientation == 'up':
                    twist.linear.x = 0.5
                    twist.angular.z = 0.0

                elif thumb_orientation == 'down':
                    twist.linear.x = -0.5
                    twist.angular.z = 0.0
                
                elif thumb_orientation == 'right':
                    twist.linear.x = 0.0
                    twist.angular.z = -0.5
                
                elif thumb_orientation == 'left':
                    twist.linear.x = 0.0
                    twist.angular.z = 0.5


                self.publisher_.publish(twist)

        cv2.imshow('MediaPipe Gesture Recognition', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    gesture_control_node = GestureControlNode()
    rclpy.spin(gesture_control_node)
    gesture_control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
