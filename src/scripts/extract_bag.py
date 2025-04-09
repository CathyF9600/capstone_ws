import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

def extract_images(bag_path, depth_topic="/depth_image", color_topic="/color_image"):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    bridge = CvBridge()

    depth_images = []
    color_images = []

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    reader.set_filter({'topics': [depth_topic, color_topic]})

    while reader.has_next():
        topic, data, t = reader.read_next()
        msg_type = type_map[topic]
        if msg_type == "sensor_msgs/msg/Image":
            img_msg = deserialize_message(data, Image)
            cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

            if topic == depth_topic:
                depth_images.append(cv_img)
            elif topic == color_topic:
                color_images.append(cv_img)

    # Convert to numpy arrays
    depth_array = np.stack(depth_images) if depth_images else None
    color_array = np.stack(color_images) if color_images else None

    print(f"Depth array shape: {None if depth_array is None else depth_array.shape}")
    print(f"Color array shape: {None if color_array is None else color_array.shape}")

    return depth_array, color_array

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_images_from_rosbag2.py <path_to_rosbag>")
        exit(1)

    rclpy.init()
    depth_np, color_np = extract_images(sys.argv[1])
    rclpy.shutdown()
