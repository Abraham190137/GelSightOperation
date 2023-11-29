import os
import re
import cv2

def get_camera_id(camera_name) -> int:
    """
    Get the camera id for a camera with a given name.
    :param camera_name: The name of the camera to find. (e.g. "GelSight")
    :return: The camera id.
    """
    cam_num = None
    if os.name == 'nt':
        cam_num = find_cameras_windows(camera_name)
    else:
        for file in os.listdir("/sys/class/video4linux"):
            real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
            with open(real_file, "rt") as name_file:
                name = name_file.read().rstrip()
            if camera_name in name:
                cam_num = int(re.search("\d+$", file).group(0))
                if cv2.VideoCapture(cam_num).isOpened():
                    break
                else:
                    cam_num = None
    if cam_num is None:
        raise Exception("Camera not found!")
    else:
        print("{} {} -> {}".format("FOUND", file, name))

    return cam_num

# This is from the gelsight libary. I haven't tested it yet.
if os.name == 'nt':
    def find_cameras_windows(camera_name) -> int:
        """
        Helper function to find the camera id on Windows.
        :param camera_name: The name of the camera to find. (e.g. "GelSight")
        :return: The camera id.
        """

        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()

        # get the device name
        allcams = graph.get_input_devices() # list of camera device
        description = ""
        for cam in allcams:
            if camera_name in cam:
                description = cam
        try:
            device = graph.get_input_devices().index(description)
        except ValueError as e:
            print("Device is not in this list")
            print(graph.get_input_devices())
            import sys
            sys.exit()

        return device