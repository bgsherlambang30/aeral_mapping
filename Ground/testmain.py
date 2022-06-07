import time
import socketserver
import os
import utils.utils as utils
from stitcher import Stitcher

HOME_PATH = 'C:\\Users\\bagas\\Documents\\TUB\\Praktikum\\LabFly\\second_test_flight\\F1\\F1'
TMP_FILE_PATH =  'data_test\\tmp'

FOCAL_LENGTH = 10.2666666667  # mm
SENSOR_WIDTH = 13.2  # mm
SENSOR_HEIGHT = 8.8  # mm

stitcher = Stitcher(HOME_PATH,
                    cam_rotation=1,
                    focal_length=FOCAL_LENGTH,
                    sensor_width=SENSOR_WIDTH,
                    sensor_height=SENSOR_HEIGHT)

while True:
    # try:
    img_path_list = utils.get_files(HOME_PATH)
    #  while len(self.ftp_in_file_list) > 0:
        #  img_path = self.ftp_in_file_list[0]
        #  img_path_list.append(img_path)
        #  self.ftp_in_file_list.pop(0)
    if len(img_path_list) > 0:
        print("map {} files".format(len(img_path_list)))
        map_file_name, top_left, bot_right = stitcher.create_map(
            "{}/tmp_map".format(TMP_FILE_PATH), img_path_list, meander=False)       
    else:
        # print("no files to map")
        time.sleep(1.)
# except:
        #    pass
    time.sleep(.1)