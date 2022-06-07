from PIL import Image
import piexif
import pickle
import os
import cv2

def data2txt(path,filename,lat,lon,alt,hdg):
    PATH = path
    f = open(PATH + '\\' + filename + '.txt', 'a')
    f.write('|  ' + str(lat) +  '  |  ' + str(lon) + '  |  ' + str(alt) + '  |  ' + str(hdg) + '  |\n')
    f.close()

def get_img(files):
    #files = get_files(path)
    img = []
    for f in files:
        i = cv2.imread(f)
        Lat, Lon, _, Alt, Yaw = read_img_exif(f)
        Data = ImgData(i, Lat, Lon, Alt, Yaw, f)
        img.append(Data)
    return img


def get_files(path):
    ''' Funtion to get list of images

    Args:
        path : '...'
    Returns:
        files : list of all images files in a folder with format JPG
    '''
    if os.path.exists(path):
        print('Path exist')
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.JPG' in file:
                files.append(os.path.join(r, file))
            elif '.jpg' in file:
                files.append(os.path.join(r, file))
    return files

def read_img_exif(file_path):
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])
    raw = exif_dict['Exif'][piexif.ExifIFD.MakerNote]
    tags = pickle.loads(raw)
    return tags["lat"], tags["lon"], tags["alt"], tags["rel_alt"], tags["hdg"]


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










