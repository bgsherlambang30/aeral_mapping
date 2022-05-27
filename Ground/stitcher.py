
import utils.utils_stitching as us
import cv2 as cv
import os
import numpy as np
import utils.utils as utils
from typing import Tuple, List
from models.lat_lon import LatLon
from geopy.distance import geodesic
import time
from utils.utils import TextGen

class Stitcher:
    def __init__(self,
                 image_out_path,
                 focal_length,
                 sensor_width,
                 sensor_height,
                 cam_rotation=0,
                 scaling=1,
                 calib_path=None
                 ):
        """
        initializes the Stitcher
        note: focal_length and sensor_width/height can also be replaced by measures in m, e.g. 
        distance of the lense to a wall, and the wall width/height that is visible in the image
        Args:
            image_out_path ([type]): output path
            focal_length ([type]): focal length (real not the equivalent)
            sensor_width ([type]): sensor width
            sensor_height ([type]): sensor height
            cam_rotation (int, optional): mount offset rotation of the cam
            scaling (int, optional): not used...
            calib_path ([type], optional): path to numpy cv2 calibration matrix or None to skip calibration
        """
        self.image_out_path = image_out_path
        self.cam_rotation = cam_rotation
        self.scaling = scaling
        self.calib = None
        self.calib_path = calib_path
        if self.calib_path is not None:
            self.calib = np.load(calib_path)
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.map_canvas = np.zeros((1, 1, 4), dtype=np.uint8)
        self.ref_pos = None

    def createMapPlane(self, image_out_name, input_image_path_list, Pos_CSV, Calib, Pattern=False):
        self.Pos_CSV = Pos_CSV
        self.calib_path = Calib
        self.create_map(image_out_name, input_image_path_list, meander=Pattern)

    def createMapMavic(self,image_out_name, input_image_path_list, Pattern=True):
        self.create_map(image_out_name, input_image_path_list, meander=Pattern)

    def create_map_from_folder(self, image_out_name: str, input_image_folder_path: str, meander: bool = False) -> Tuple[str, LatLon, LatLon]:
        input_image_path_list = utils.get_files(input_image_folder_path)
        return self.create_map(image_out_name, input_image_path_list, meander=meander)

    def create_map(self, image_out_name: str, input_image_path_list: List[str], meander: bool = False) -> Tuple[str, LatLon, LatLon]:
        start_time = time.time()
        if self.calib_path is None:
            images_list = utils.get_img(
                input_image_path_list, scale_x=self.scaling, scale_y=self.scaling, rotate=self.cam_rotation)
        else:
            images_list = utils.get_Cali_img(input_image_path_list, self.calib_path,
                                             scale_x=self.scaling, scale_y=self.scaling, rotate=self.cam_rotation)
        w = int(images_list[0].img.shape[1])
        h = int(images_list[0].img.shape[0])
        c = (int(w/2), int(h/2))

        if meander is True:
            yaw_param = abs(round(images_list[0].yaw))
            j = 0
            while j < len(images_list):
                if abs(round(images_list[j].yaw)) == yaw_param or abs(round(images_list[j].yaw)) == (180-yaw_param):
                    j += 1
                else:
                    del images_list[j]

        data_0 = images_list[0]

        yaw_0 = round(data_0.yaw)
        if yaw_0 > 0:
            yaw_0 = yaw_0 - 90
        elif yaw_0 < 0:
            yaw_0 = yaw_0 + 90
        else:
            yaw_0 = yaw_0

        if self.ref_pos is None:
            height, width = data_0.img.shape[:2]
            self.ref_pos = LatLon(data_0.lat, data_0.lon)
            width_m = data_0.alt * self.sensor_width / self.focal_length
            height_m = data_0.alt * self.sensor_height / self.focal_length

            self.pix_per_m = width / width_m
            m_per_lat = geodesic((data_0.lat, data_0.lon),
                                 (data_0.lat+1, data_0.lon)).m
            m_per_lon = geodesic((data_0.lat, data_0.lon),
                                 (data_0.lat, data_0.lon+1)).m
            self.pix_per_lat = m_per_lat * self.pix_per_m
            self.pix_per_lon = m_per_lon * self.pix_per_m

        H_com = np.identity(3, dtype=float)
        H_Mat = np.identity(3, dtype=float)
        H_R = np.identity(2, dtype=float)
        H_y = 0
        H_x = 0
        base = np.zeros((h, w, 3), np.uint8)
        Rot = us.angle2Rotmatrix2D(center=c, angle=np.deg2rad(yaw_0))
        H_R = Rot[0:2, 0:2]
        H_com[0:2, 0:2] = H_R
        H_com[0][-1] = Rot[0][-1]
        H_com[1][-1] = Rot[1][-1]
        xmin, xmax, ymin, ymax = us.get_corner(base, data_0.img, H_com)
        translation = np.float32(([1, 0, -1*xmin], [0, 1, -1*ymin], [0, 0, 1]))
        H_com = np.dot(translation, H_com)
        base_img = cv.warpPerspective(
            data_0.img, H_com, (xmax-xmin, ymax-ymin))
        counter = False

        self.ref_pos.lon += min(0, xmin) / self.pix_per_lon
        self.ref_pos.lat -= min(0, ymin) / self.pix_per_lat
        k = 0
        G_Algo = 0
        CV_Algo = 0
        Time_CV = 0
        Time_Geo = 0
        #images_list = images_list
        for i in range(1, len(images_list), 1):
            print(i)
            data_now = images_list[i-1]
            data_next = images_list[i]
            rot_angle = data_next.yaw-data_now.yaw
            if (abs(rot_angle) > 2 or k > 0) and meander is False:
                st_CV = time.time() 
                H_Matrix = us.get_whole_Hmat_seq(
                    data_now.img, data_next.img, method='2DTransform')
                H_com = np.matmul(H_com, H_Matrix)
                _, _, _, theta = us.getRotMatrix_seq(H_com, c)
                if k == 1 and self.calib_path is None:
                    if round(theta) == yaw_0 or round(theta) == yaw_0+180:
                        theta = theta
                    elif round(theta) >= yaw_0-10 and theta < round(yaw_0+10):
                        theta_f = yaw_0-theta
                        Rot_Mat_cor = us.get_RotMatrix_Correction(c, theta_f)
                        H_com = np.matmul(H_com, Rot_Mat_cor)
                    elif round(theta) >= yaw_0+170 and round(theta) < yaw_0+190:
                        theta_f = (yaw_0+180)-theta
                        Rot_Mat_cor = us.get_RotMatrix_Correction(c, theta_f)
                        H_com = np.matmul(H_com, Rot_Mat_cor)
                if self.calib_path is None:
                    if k == 0 or abs(rot_angle) > 2:
                        k = 1
                    else:
                        k -= 1
                CV_Algo += 1
                dt_CV =  time.time() - st_CV
                Time_CV += dt_CV
                print(Time_CV)
            else:
                st_Geo = time.time()
                cmpx_now = data_now.alt * self.sensor_width / \
                    (self.focal_length * data_now.img.shape[0]) * 100
                cmpx_next = data_next.alt * self.sensor_width / \
                    (self.focal_length * data_next.img.shape[0]) * 100
                scaling = cmpx_next/cmpx_now
                x, y = utils.get_dist_in_px(
                    [data_now.lat, data_next.lat], [data_now.lon, data_next.lon], cmpx_now, method='xy', round=0)
                xy = [y[0], x[0]]
                y, x = xy[:]
                c = (int(data_next.img.shape[1]/2),
                     int(data_next.img.shape[0]/2))
                Rot = us.angle2Rotmatrix2D(
                    center=c, angle=np.deg2rad(rot_angle))
                Rot[0][0] = Rot[0][0]*scaling
                Rot[1][1] = Rot[1][1]*scaling
                h_x, h_y = us.rot_corr(c, yaw_0)
                if meander is True:
                    H_R = np.matmul(Rot[0:2, 0:2], H_R)
                    H_com[0:2, 0:2] = H_R
                    if round(rot_angle) < 0:
                        H_com[0][-1] += y+Rot[0][-1]+h_y
                        H_com[1][-1] += x+Rot[1][-1]+h_x
                    elif round(rot_angle) > 0:
                        H_com[0][-1] += y-Rot[0][-1]-h_y
                        H_com[1][-1] += x-Rot[1][-1]-h_x
                    else:
                        H_com[0][-1] += y-Rot[0][-1]
                        H_com[1][-1] += x-Rot[1][-1]
                else:
                    H_R = Rot[0:2, 0:2]
                    H_Mat[0:2, 0:2] = H_R
                    if round(rot_angle) < 0:
                        H_y = y+Rot[0][-1]
                        H_x = x+Rot[1][-1]
                    elif round(rot_angle) >= 0:
                        H_y = y-Rot[0][-1]
                        H_x = x-Rot[1][-1]
                    H_Mat[0][-1] = H_y
                    H_Mat[1][-1] = H_x
                    H_com = np.matmul(H_Mat, H_com)
                G_Algo += 1
                dt_Geo =  time.time() - st_Geo
                Time_Geo += dt_Geo 
                print(Time_Geo)
                k = 0
            xmin, xmax, ymin, ymax = us.get_corner(
                base_img, data_next.img, H_com)
            translation = np.float32(
                ([1, 0, -1*xmin], [0, 1, -1*ymin], [0, 0, 1]))
            H_com = np.dot(translation, H_com)
            H_com_inv = np.linalg.inv(H_com)
            next_img = cv.warpPerspective(
                data_next.img, H_com_inv, (xmax-xmin, ymax-ymin), flags=cv.WARP_INVERSE_MAP)
            (_, data_map) = cv.threshold(cv.cvtColor(
                next_img, cv.COLOR_RGB2GRAY), 0, 255, cv.THRESH_BINARY)
            base_img = cv.warpPerspective(
                base_img, translation, (xmax-xmin, ymax-ymin))
            canvas = np.zeros(
                (base_img.shape[0], base_img.shape[1], 3), np.uint8)
            can = cv.add(canvas, base_img, mask=np.bitwise_not(
                data_map), dtype=cv.CV_8U)
            base_img = cv.add(can, next_img, dtype=cv.CV_8U)



        tmp = cv.cvtColor(base_img, cv.COLOR_BGR2GRAY)
        _, alpha = cv.threshold(tmp, 0, 255, cv.THRESH_BINARY)
        b, g, r = cv.split(base_img)
        rgba = [b, g, r, alpha]
        t_base_img = cv.merge(rgba, 4)
        print('Stitch Done...')
        out_file_path = "{}/{}.png".format(self.image_out_path,
                                           image_out_name)
        if not os.path.exists(os.path.dirname(out_file_path)):
            os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        cv.imwrite(out_file_path, t_base_img)

        top_left = LatLon(self.ref_pos.lat, self.ref_pos.lon)
        bot_right = LatLon(self.ref_pos.lat + (base_img.shape[0] / self.pix_per_lat),
                           self.ref_pos.lon + (base_img.shape[1] / self.pix_per_lon))
        print("top_left: {}, {}".format(top_left.lat, top_left.lon))
        print("bot_right: {}, {}".format(bot_right.lat, bot_right.lon))

        time_need = time.time() - start_time
        text = TextGen (self.image_out_path,
                        image_out_name,
                        CV_Algo,
                        G_Algo,
                        i,
                        time_need,
                        time_CV=Time_CV,
                        time_Geo=Time_Geo)
        text.write()
        print(Time_CV,Time_Geo)
        return out_file_path, top_left, bot_right