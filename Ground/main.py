import threading
import time
import socketserver
import os

from stitcher import Stitcher
from tile_generator import TileGenerator
from tools.ftp_server import FtpServer
from tools.tile_server import TileHttpRequestHandler

FTP_HOME_PATH = "data/home"
TMP_FILE_PATH = "data/tmp"
MIN_ZOOM_LEVEL = 14
TILE_SERVER_PORT = 8000
TILES_ROOT_PATH = "data/tiles/"

FTP_PORT = 21
FTP_USER = "user1"
FTP_PW = "peter123"

FOCAL_LENGTH = 10.2666666667  # mm
SENSOR_WIDTH = 13.2  # mm
SENSOR_HEIGHT = 8.8  # mm


class MainServer:
    def __init__(self) -> None:
        self._create_dir(FTP_HOME_PATH)
        self._create_dir(TMP_FILE_PATH)
        self._create_dir(TILES_ROOT_PATH)

        self.ftp_in_file_list = []
        self.stitcher = Stitcher(FTP_HOME_PATH,
                                 cam_rotation=0,
                                 focal_length=FOCAL_LENGTH,
                                 sensor_width=SENSOR_WIDTH,
                                 sensor_height=SENSOR_HEIGHT)
        self.gen = TileGenerator()

    @staticmethod
    def _create_dir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    def start(self):
        ser = FtpServer(FTP_HOME_PATH, port=FTP_PORT, user=FTP_USER, pw=FTP_PW)
        ser.on_receive_callback_f = self.ftp_received_file_callback
        ser.start()

        #map_t = threading.Thread(target=self.mapping_task, name="map_t")
        #map_t.setDaemon(True)
        #map_t.start()

        server_t = threading.Thread(
            target=self.serve_task, name="tile_server_t")
        server_t.setDaemon(True)
        server_t.start()

    def serve_task(self):
        handler_object = TileHttpRequestHandler
        handler_object.tile_path = TILES_ROOT_PATH

        my_server = socketserver.TCPServer(
            ("", TILE_SERVER_PORT), handler_object)
        my_server.serve_forever()

    def ftp_received_file_callback(self, file_name):
        self.ftp_in_file_list.append(file_name)

    def mapping_task(self):
        while True:
            # try:
            img_path_list = []
            while len(self.ftp_in_file_list) > 0:
                img_path = self.ftp_in_file_list[0]
                img_path_list.append(img_path)
                self.ftp_in_file_list.pop(0)
            if len(img_path_list) > 0:
                print("map {} files".format(len(img_path_list)))
                map_file_name, top_left, bot_right = self.stitcher.create_map(
                    "{}/tmp_map".format(TMP_FILE_PATH), img_path_list, meander=False)

                zoom = self.gen.generate_tiles(map_file_name,
                                               top_left,
                                               bot_right,
                                               out_folder=TILES_ROOT_PATH)
                zoom_min = self.gen.generate_zoom_levels(
                    zoom, end_zoom=MIN_ZOOM_LEVEL)
                print("created tiles for zoom {} to {}".format(zoom_min, zoom))
            else:
                # print("no files to map")
                time.sleep(1.)
            # except:
            #    pass
            time.sleep(.1)


if __name__ == '__main__':
    mapper = MainServer()
    mapper.start()
    while True:
        time.sleep(1.0)