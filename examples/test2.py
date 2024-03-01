
import time
import matplotlib.pyplot as plt
from simple_gelsight import get_camera_id, GelSightMultiprocessed, Gelsight

camera_id = get_camera_id('GelSight')
gs = GelSightMultiprocessed(camera_id)
gs.process.start()
time.sleep(1)
for i in range(10):
    frame, frame_data, depthmap, strain_x, strain_y = gs.get_next_frame()
    plt.imshow(depthmap)
    plt.show()
gs.close()