from GPSPhoto import gpsphoto
import os
import numpy as np
import cv2 as cv
from geopy import distance
from numpy import genfromtxt
from models.img_data import ImgData
from PIL import Image
import piexif
import pickle


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


def get_img(files, scale_x=1, scale_y=1, rotate=0):
    #files = get_files(path)
    img = []
    for f in files:
        i = cv.imread(f)
        i = cv.resize(i, None, fx=float(scale_x), fy=float(
            scale_y), interpolation=cv.INTER_AREA)
        if rotate == 0:
            i = i
        if rotate == 1:
            i = np.rot90(i, 1)
        if rotate == 2:
            i = np.rot90(i, 2)
        if rotate == 3:
            i = np.rot90(i, 3)
        #Lat,Lon,_ = get_GPSdata_scalar(f)
        #Yaw,Alt = read_yaw_from_exif(f)
        Lat, Lon, _, Alt, Yaw = read_img_exif(f)
        Data = ImgData(i, Lat, Lon, Alt, Yaw, f)
        img.append(Data)
    return img


def get_img_old(path, scale_x=1, scale_y=1, rotate=0):
    files = get_files(path)
    img = []
    for f in files:
        i = cv.imread(f)
        i = cv.resize(i, None, fx=float(scale_x), fy=float(
            scale_y), interpolation=cv.INTER_AREA)
        if rotate == 0:
            i = i
        if rotate == 1:
            i = np.rot90(i, 1)
        if rotate == 2:
            i = np.rot90(i, 2)
        if rotate == 3:
            i = np.rot90(i, 3)
        img.append(i)
    return img


def get_Cali_img(files, path_cali_Mat, scale_x=1, scale_y=1, rotate=0):
    calib_data = np.load(path_cali_Mat)
    mat = calib_data["matrix"]
    dist = calib_data["distortion"]
    img = []
    for f in files:
        i = cv.imread(f)
        h,  w = i.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            mat, dist, (w, h), 1, (w, h))
        # undistort
        dst = cv.undistort(i, mat, dist, None, newcameramtx)
        x, y, w, h = roi
        ratio = w/h
        while ratio != 16/9:
            w = int(16/9*h)
            h = int(9/16 * w)
            ratio = w/h
        i = dst[y:y+h, x:x+w]
        i = cv.resize(i, None, fx=float(scale_x), fy=float(
            scale_y), interpolation=cv.INTER_AREA)
        if rotate == 0:
            i = i
        if rotate == 1:
            i = np.rot90(i, 1)
        if rotate == 2:
            i = np.rot90(i, 2)
        if rotate == 3:
            i = np.rot90(i, 3)
        #Lat,Lon,_,Alt,Yaw = get_GPSdata_Mav_scalar(Path_Mav,f)
        Lat, Lon, _, Alt, Yaw = read_img_exif(f)
        Data = ImgData(i, Lat, Lon, Alt, Yaw, f)
        img.append(Data)
    return img


def get_images(file_path_list, calib):
    # files = get_files(path)
    img_data_list = []
    for f in file_path_list:
        img = cv.imread(f)
        # add alpha channel (png)
        img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)
        if calib is not None:
            h,  w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(
                calib["matrix"], calib["distortion"], (w, h), 1, (w, h))
            # undistort
            img = cv.undistort(
                img,  calib["matrix"],  calib["distortion"], None, newcameramtx)
        Lat, Lon, _, Alt, Yaw = read_img_exif(f)
        Data = ImgData(img, Lat, Lon, Alt, Yaw, f)
        img_data_list.append(Data)
    return img_data_list


def get_GPSdata(path):
    lat = []
    lon = []
    alt = []
    files = get_files(path)
    for f in files:
        data = gpsphoto.getGPSData(f)
        lat.append(data["Latitude"])
        lon.append(data["Longitude"])
        alt.append(data["Altitude"])
    return (lat, lon, alt)


def read_img_exif(file_path):
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])
    raw = exif_dict['Exif'][piexif.ExifIFD.MakerNote]
    tags = pickle.loads(raw)
    return tags["lat"], tags["lon"], tags["alt"], tags["rel_alt"], tags["hdg"]


def get_GPSdata_scalar(f):
    data = gpsphoto.getGPSData(f)
    lat = data["Latitude"]
    lon = data["Longitude"]
    alt = data["Altitude"]
    return lat, lon, alt


def find_param(xml_str: str, param_name: str) -> str:
    start = xml_str.find(param_name)
    first_quote_pos = xml_str[start:].find("\"")
    second_quote_pos = xml_str[start + first_quote_pos + 1:].find("\"")
    value = xml_str[start+first_quote_pos +
                    1:start+first_quote_pos+second_quote_pos+1]
    return value


def read_yaw_from_exif(file_name: str) -> float:
    fd = open(file_name, "rb")
    d = fd.read()
    xmp_start = d.find(b"<x:xmpmeta")
    xmp_end = d.find(b"</x:xmpmeta")
    xmp_str = d[xmp_start:xmp_end+12]
    fd.close()

    xml_str = xmp_str.decode()

    yaw = float(find_param(xml_str, "drone-dji:FlightYawDegree"))
    yaw_gimbal = float(find_param(xml_str, "drone-dji:GimbalYawDegree"))
    alt = float(find_param(xml_str, "drone-dji:RelativeAltitude"))
    # yaw = None
    # yaw_gimbal = None
    # alt = None

    # for l in xml_str.split("\n"):
    #     if "drone-dji:FlightYawDegree" in l:
    #         found = re.findall('"([^"]*)"', l)
    #         if len(found) > 0:
    #             yaw = float(found[0])
    #     if "drone-dji:GimbalYawDegree" in l:
    #         found = re.findall('"([^"]*)"', l)
    #         if len(found) > 0:
    #             yaw_gimbal = float(found[0])
    #     if "drone-dji:RelativeAltitude" in l:
    #         found = re.findall('"([^"]*)"', l)
    #         if len(found) > 0:
    #             alt = float(found[0])
    # # yaw = yaw + (yaw_gimbal - 180.0)
    return yaw, alt


def get_GPSdata_Mav(path, path_im):
    Lat = []
    Lon = []
    Alt = []
    Alt_rel = []
    hdg = []
    files = get_files(path_im)
    # with open(path, encoding='utf8') as f:
    #     data = csv.reader(f)
    #     next(data)
    data = genfromtxt(path, delimiter=',', skip_header=1)
    for i in range(0, len(files)):
        ts = files[i]
        ts = float(ts[-18:-4])
        j = 0
        for j in range(0, len(data)):
            if data[j][0] >= ts:
                d = data[max(0, j-1)]
                break
        Lat.append(float(d[1]))
        Lon.append(float(d[2]))
        Alt.append(float(d[3]))
        Alt_rel.append(float(d[4]))
        hdg.append(float(d[5]))

    return(Lat, Lon, Alt, Alt_rel, hdg)


def get_GPSdata_Mav_scalar(path, f):
    data = genfromtxt(path, delimiter=',', skip_header=1)
    ts = float(f[-18:-4])
    j = 0
    for j in range(0, len(data)):
        if data[j][0] >= ts:
            d = data[max(0, j-1)]
            break
    Lat = (float(d[1]))
    Lon = (float(d[2]))
    Alt = (float(d[3]))
    Alt_rel = (float(d[4]))
    hdg = (float(d[5]))
    return(Lat, Lon, Alt, Alt_rel, hdg)


def get_dist_xy(lat, lon):
    f_dist_y = []
    f_dist_x = []
    idx = 0
    for idx, (o, m) in enumerate(zip(lat, lon)):
        if (idx+1 < len(lat) and idx-1 >= 0):
            p = (lat[idx-1], lon[idx-1])
            px = (lat[idx], lon[idx-1])
            py = (lat[idx-1], lon[idx])
            dx = distance.distance(p, px).m
            dy = distance.distance(p, py).m
            if lat[idx-1] < lat[idx]:
                dx = -dx
            if lon[idx-1] > lon[idx]:
                dy = -dy
            f_dist_x.append(dx)
            f_dist_y.append(dy)
        else:
            p = (lat[idx-1], lon[idx-1])
            px = (lat[idx], lon[idx-1])
            py = (lat[idx-1], lon[idx])
            dx = distance.distance(p, px).m
            dy = distance.distance(p, py).m
            if lat[idx-1] < lat[idx]:
                dx = -dx
            if lon[idx-1] > lon[idx]:
                dy = -dy
            f_dist_x.append(dx)
            f_dist_y.append(dy)
    del f_dist_x[0]
    del f_dist_y[0]
    return f_dist_x, f_dist_y


def get_dist_00(lat, lon):
    f_dist_y = []
    f_dist_x = []
    idx = 0
    for idx, (o, m) in enumerate(zip(lat, lon)):
        if (idx+1 < len(lat) and idx-1 >= 0):
            p = (lat[0], lon[0])
            px = (lat[idx], lon[0])
            py = (lat[0], lon[idx])
            dx = distance.distance(p, px).m
            dy = distance.distance(p, py).m
            if lat[0] < lat[idx]:
                dx = -dx
            if lon[0] > lon[idx]:
                dy = -dy
            #d = round(d,1)
            f_dist_x.append(dx)
            f_dist_y.append(dy)
        else:
            p = (lat[0], lon[0])
            px = (lat[idx], lon[0])
            py = (lat[0], lon[idx])
            dx = distance.distance(p, px).m
            dy = distance.distance(p, py).m
            if lat[0] < lat[idx]:
                dx = -dx
            if lon[0] > lon[idx]:
                dy = -dy
            #d = round(d,1)
            f_dist_x.append(dx)
            f_dist_y.append(dy)
    del f_dist_x[0]
    del f_dist_y[0]
    return f_dist_x, f_dist_y


def get_cmperpx(alt):
    '''
        Altitude in m
    '''
    w_px = 5472  # px
    h_px = 3648  # px
    ratio_w2alt = 124.5  # in cm/m
    ratio_h2alt = 83.6  # in cm/m
    real_w = ratio_w2alt*alt  # in cm
    real_h = ratio_h2alt*alt  # in cm
    w_cmperpx = real_w/w_px
    h_cmperpx = real_h/h_px
    cmperpx = (w_cmperpx+h_cmperpx)/2
    return cmperpx


def get_dist_in_px(lat, lon,  cmperpx, scale=1, method='00', round=1):
    if method == '00':
        f_dist_x, f_dist_y = get_dist_00(lat, lon)
    else:
        f_dist_x, f_dist_y = get_dist_xy(lat, lon)

    f_dist_px = []
    f_dist_py = []
    for x, y in zip(f_dist_x, f_dist_y):
        px_x = (x*100)/cmperpx
        px_x = px_x*scale
        if round == 1:
            px_x = round(px_x)
        f_dist_px.append(px_x)
        px_y = (y*100)/cmperpx
        px_y = px_y*scale
        if round == 1:
            px_y = round(px_y)
        f_dist_py.append(px_y)
    return f_dist_px, f_dist_py


def make_canvas(lat, lon, cmperpx):
    x1 = (min(lat), max(lon))
    x2 = (max(lat), max(lon))
    y1 = (max(lat), min(lon))
    y2 = (max(lat), max(lon))
    x = int((distance.distance(x1, x2).m)*100/cmperpx)+4000
    y = int((distance.distance(y1, y2).m)*100/cmperpx)
    canvas = np.zeros(shape=(x, y, 3))
    return canvas


def plot_images_xy(path, alt, scale=1):
    img = get_img(path, scale_x=scale, scale_y=scale)
    setofimg = list(map(lambda i: np.rot90(i, 3), img))
    Lat, Lon, h = get_GPSdata(path)
    ratio_cm_px = get_cmperpx(alt)
    dist_px, dist_py = get_dist_in_px(
        Lat, Lon, ratio_cm_px, scale, method='xy')
    dist_py.insert(0, int(0))
    dist_px.insert(0, int(0))
    blank_page = make_canvas(Lat, Lon, ratio_cm_px)
    j = 0
    v = int(500)
    a = int(0)+v
    b = int(setofimg[0].shape[0])+v
    c = int(0)+v
    d = int(setofimg[0].shape[1])+v
    while j < len(setofimg):
        im = setofimg[j]
        a = int(a+dist_px[j])
        b = int(b+dist_px[j])
        c = int(c+dist_py[j])
        d = int(d+dist_py[j])
        blank_page[a:b, c:d] = im
        j += 1
        if j != len(setofimg):
            if abs(dist_px[j]) > 100:
                while j != len(setofimg):
                    im = np.rot90(setofimg[j], 2)
                    a = int(a+dist_px[j])
                    b = int(b+dist_px[j])
                    c = int(c+dist_py[j])
                    d = int(d+dist_py[j])
                    blank_page[a+0:b+0, c:d] = im
                    j += 1
                    if j != len(setofimg):
                        if abs(dist_px[j]) > 100:
                            break
        else:
            break
    return blank_page


def get_rot_axes(hdg1, hdg2):
    rot_0 = (hdg1 - hdg2)
    cr = np.cos(np.radians(rot_0))
    sr = np.sin(np.radians(rot_0))
    rot_axes = np.float32(([cr, -sr], [sr, cr]))
    return rot_axes


class TextGen:
    def __init__ (self, path, filename, CV_im, G_im, Tot_im, time, time_CV, time_Geo):
        self.path = path
        self.filename = filename
        self.cv = CV_im
        self.G = G_im
        self.Total = Tot_im
        self.time = time
        self.t_cv = time_CV
        self.t_g = time_Geo
    def write(self):
        time_im = self.time / self.Total
        f = open(self.path +'\\' + self.filename + '.txt','w')
        Inhalt = [  'Number of images with CV Algo = ' + str(self.cv) + '\n',
                    'Number of images with Geo Algo = ' + str(self.G) + '\n',
                    'Total image = ' + str(self.Total) + '\n',
                    'Time in CV = ' + str(self.t_cv) + ' second \n',
                    'Time in Geo = ' + str(self.t_g) + ' second \n',
                    'Time needed = ' + str(self.time) + ' second \n',
                    'Time per Image = ' + str(time_im) + 'second']
        f.write(self.filename + '\n')
        f.writelines(Inhalt)
        return             

