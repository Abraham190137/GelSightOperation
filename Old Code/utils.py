import os
import re
import cv2

def find_cameras():
    # checks the first 10 indexes.
    index = 0
    arr = []
    if os.name == 'nt':
        raise NotImplementedError('You chose to use windows, you figure it out.')
    i = 10
    while i >= 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            command = 'v4l2-ctl -d ' + str(index) + ' --info'
            is_arducam = os.popen(command).read()
            if is_arducam.find('Arducam') != -1 or is_arducam.find('Mini') != -1:
                arr.append(index)
            cap.release()
        index += 1
        i -= 1

    return arr


def get_camera_id(camera_name):
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
                found = "FOUND!"
            else:
                found = "      "
            print("{} {} -> {}".format(found, file, name))

    return cam_num

print(get_camera_id('Gelsight Mini'))