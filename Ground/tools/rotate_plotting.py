

import cv2
import math
import os
import sys
if True:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import piexif
import pickle
from tile_generator import LatLon
from geopy.distance import geodesic
#from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt


from utils.utils import read_yaw_from_exif

HEAD_ESTIMATE_DIFF = 2


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def load_exif(file_path):
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])
    raw = exif_dict['Exif'][piexif.ExifIFD.MakerNote]
    tags = pickle.loads(raw)
    return tags


def find_pos(data, ts):
    min_delta = np.inf
    for i in range(0, len(data)):
        d = data[i]
        if abs(ts - d[0]) <= min_delta:
            min_delta = abs(ts - d[0])
        else:
            return d[1], d[2], d[4], d[5], d[6]
    return None


def check_rot_mavic(in_folder, out_folder):
    file_names = [f for f in listdir(
        in_folder) if isfile(join(in_folder, f))]
    out_csv = open("{}/out.csv".format(out_folder), "w")
    for f in file_names:
        if f.endswith(".jpg") or f.endswith(".JPG"):
            file_path = "{}/{}".format(in_folder, f)
            yaw, alt, gimb = read_yaw_from_exif(file_path)
            out_csv.write("{},{},{},{}\n".format(f, alt, yaw, gimb))
            hdg = yaw * -1.0
            #img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            #img_rot = rotate_image(img, hdg)
            #cv2.imwrite("{}/{}".format(out_folder, f), img_rot)

    out_csv.close()


def plot_plane_path(in_csv):
    ARROW_LENFTH = 1.0
    csv_data = np.genfromtxt(in_csv, delimiter=',', skip_header=1)
    offset_lat = csv_data[0][1]
    offset_lon = csv_data[0][2]
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    i = 0
    for d in csv_data:
        x = geodesic((offset_lat, offset_lon), (offset_lat, d[2])).m
        y = geodesic((offset_lat, offset_lon), (d[1], offset_lon)).m
        hdg = math.radians(d[5])
        plt.arrow(x, y, ARROW_LENFTH*math.sin(hdg), ARROW_LENFTH *
                  math.cos(hdg), length_includes_head=True, head_width=0.5, head_length=0.5)
        i += 1
    plt.gca().set_aspect('equal')
    plt.show()


def check_rot_plane(in_folder, out_folder):
    csv_data = np.genfromtxt(
        "{}/pos.csv".format(in_folder), delimiter=',', skip_header=1)
    out_csv = open("{}/pos.csv".format(out_folder), "w")
    out_csv.write("ts,lat,lon,alt,alt_rel,hdg,hdg_fixed\n")
    prev_d = csv_data[0]
    geod = Geodesic.WGS84
    for i in range(0, len(csv_data)):
        d = csv_data[i]
        next_d = csv_data[min(i+HEAD_ESTIMATE_DIFF,
                              len(csv_data)-HEAD_ESTIMATE_DIFF)]
        g = geod.Inverse(prev_d[1], prev_d[2], next_d[1], next_d[2])
        hdg = g['azi1']
        if hdg < 0:
            hdg += 360
        prev_d = csv_data[i]
        out_csv.write("{},{},{},{},{},{},{}\n".format(
            d[0], d[1], d[2], d[3], d[4], d[5], hdg
        ))
    out_csv.close()

    csv_data = np.genfromtxt(
        "{}/pos.csv".format(out_folder), delimiter=',', skip_header=1)

    file_names = [f for f in listdir(
        file_dir) if isfile(join(file_dir, f))]

    out_csv = open("{}/out.csv".format(out_folder), "w")

    for f in file_names:
        if f.endswith(".jpg") or f.endswith(".JPG"):
            file_path = "{}/{}".format(file_dir, f)
            exif = load_exif(file_path)
            # hdg = exif["hdg"]
            # yaw, alt, gimb = read_yaw_from_exif(file_path)
            # out_csv.write("{},{},{},{}\n".format(f, alt, yaw, gimb))
            lat, lon, alt, hdg, hdg_fixed = find_pos(
                csv_data, float(f.replace(".jpg", "")))
            hdg_rot = ((hdg * -1.0) + 180) % 360
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            img_rot = rotate_image(img, hdg_rot)
            cv2.imwrite("{}/{}".format(out_dir, f), img_rot)
            out_csv.write("{},{},{},{},{},{},{}\n".format(
                f, lat, lon, alt, hdg, hdg_fixed, hdg_rot))
    out_csv.close()


file_dir = "data/planeFlight_02"
#file_dir = "data/mavic"
#file_dir = "data/2021-09-03_neuruppin"
out_dir = "data/outPlane"
#out_dir = "data/0000_first_test_flight"


#check_rot_mavic(file_dir, out_dir)
#check_rot_plane(file_dir, out_dir)

plot_plane_path("{}/pos.csv".format(out_dir))

print("done")


class ImgData:
    def __init__(self, img, lat, lon, alt, yaw, file_name=None):
        self.img = img
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.yaw = yaw
        self.file_name = file_name
