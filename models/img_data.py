import os


class ImgData:
    def __init__(self, img, lat, lon, alt, yaw, file_path=None):
        self.img = img
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.yaw = yaw
        self.file_path = file_path

    def get_file_name(self):
        if self.file_path is None:
            return None
        return os.path.basename(self.file_path)
