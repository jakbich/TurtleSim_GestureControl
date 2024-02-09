

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2
import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 


class DepthEstimator(Node):
    def __init__(self, model_type="DPT_Large"):
        super().__init__('depth_publisher')

        # Midas Depth Estimator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device)
        self.midas.eval()

        
        # Log that the node has been started
        self.get_logger().info('Gesture Control Node has been started')

        # Create a subscriber to /gesture image of type image
        self.image_subscriber = self.create_subscription(Image, 'gesture_image', self.image_callback, 10)
        self.bridge = CvBridge()

        # Create a publisher to publish the depth image
        self.depth_publisher = self.create_publisher(Image, 'depth_image', 10)


    def estimate_depth(self, img, input_tensor):

        with torch.no_grad():
            prediction = self.midas(input_tensor.to(self.device))

            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        # Rescale output to original size but flip axes for visualization purposes
        return output


    def image_callback(self, msg):

        # Log that an image was received
        self.get_logger().info('Received an image')

        # Convert the image message to a cv image
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Resize
        img = cv2.resize(img, (384, 384))
        
        # Normalize to [0, 1]
        img = img.astype("float32") / 255.0
        
        # Convert to torch tensor and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (C, H, W) -> (B, C, H, W)

        # Estimate the depth of the image
        depth_estimate = self.estimate_depth(img, img_tensor)
        depth_estimate = (depth_estimate - depth_estimate.min()) / (depth_estimate.max() - depth_estimate.min()+1e-8)

        print(depth_estimate.shape)
        print(type(depth_estimate))

        # Convert the depth estimate to the message type
        depth_img_msg = self.bridge.cv2_to_imgmsg(depth_estimate, encoding="32FC1")
        self.depth_publisher.publish(depth_img_msg)
        self.get_logger().info('Published the depth image')


def main(args=None):
    rclpy.init(args=args)
    depth_estimation_node = DepthEstimator()
    rclpy.spin(depth_estimation_node)
    depth_estimation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
