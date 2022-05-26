from typing import Tuple
import cv2
import math
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from models.lat_lon import LatLon

from utils.tile_api_utils import deg2num, num2deg

# https://developers.planet.com/planetschool/xyz-tiles-and-slippy-maps/


class TileGenerator:

    TILE_SIZE_PX = 256
    OUT_FOLDER = "tiles"

    def __init__(self):
        pass

    def generate_tiles(self, file_name, left_top_pos: LatLon, right_button_pos: LatLon, out_folder=OUT_FOLDER) -> int:
        self.left_top_pos = left_top_pos
        self.right_button_pos = right_button_pos
        self.out_folder = out_folder
        # os.makedirs(self.out_folder, exist_ok=True)

        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        height, width = img.shape[:2]

        # find suitable zoom for image
        lat_per_pix = abs((right_button_pos.lat - left_top_pos.lat) / height)
        lon_per_pix = abs((right_button_pos.lon - left_top_pos.lon) / width)

        zoom = 25
        lat_per_pix_z, lon_per_pix_z = self.get_deg_per_pix_for_level(
            left_top_pos, zoom)
        while lat_per_pix_z < lat_per_pix or lon_per_pix_z < lon_per_pix:
            zoom -= 1
            if zoom <= 0:
                raise ValueError("could not find zoom level")
            lat_per_pix_z, lon_per_pix_z = self.get_deg_per_pix_for_level(
                self.left_top_pos, zoom)

        #zoom = self.find_next_zoom_level(self.left_button_pos, 19)
        lat_per_pix_z, lon_per_pix_z = self.get_deg_per_pix_for_level(
            self.left_top_pos, zoom)

        # scale image to next possible zoom
        new_width = int(lon_per_pix / lon_per_pix_z * width)
        new_height = int(lat_per_pix / lat_per_pix_z * height)
        scaled_img = cv2.resize(img, (new_width, new_height))

        # update deg per pixel
        lat_per_pix = abs(
            (right_button_pos.lat - left_top_pos.lat) / new_height)
        lon_per_pix = abs(
            (right_button_pos.lon - left_top_pos.lon) / new_width)

        canvas_start_tile_x, canvas_start_tile_y = deg2num(
            self.left_top_pos.lat, self.left_top_pos.lon, zoom)
        # self.print_tile_url(canvas_start_tile_x, canvas_start_tile_y, zoom)

        _lat, _lon = num2deg(canvas_start_tile_x, canvas_start_tile_y, zoom)
        canvas_start_pos = LatLon(_lat, _lon)
        pad_top = int((canvas_start_pos.lat -
                       self.left_top_pos.lat) / lat_per_pix_z)
        pad_bot = (math.ceil((new_height + pad_top) / self.TILE_SIZE_PX)
                   * self.TILE_SIZE_PX) - new_height - pad_top
        pad_left = int((self.left_top_pos.lon-canvas_start_pos.lon
                        ) / lon_per_pix_z)
        pad_right = (math.ceil((new_width + pad_left) / self.TILE_SIZE_PX)
                     * self.TILE_SIZE_PX) - new_width - pad_left
        scaled_img = cv2.copyMakeBorder(
            scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)
        canvas_height, canvas_width = scaled_img.shape[:2]

        cv2.imwrite("dataOut/tmp_test.png", scaled_img)

        tile_dir = "{}/{}".format(self.out_folder, zoom)
        os.makedirs(tile_dir, exist_ok=True)

        for i_y in range(0, int(canvas_height / self.TILE_SIZE_PX)):
            for i_x in range(0, int(canvas_width / self.TILE_SIZE_PX)):
                x = i_x * self.TILE_SIZE_PX
                y = i_y * self.TILE_SIZE_PX
                crop_img = scaled_img[y:y +
                                      self.TILE_SIZE_PX, x:x+self.TILE_SIZE_PX]
                cv2.imwrite("{}/{}_{}.png".format(tile_dir,
                            canvas_start_tile_x+i_x, canvas_start_tile_y+i_y), crop_img)
        return zoom

    def generate_zoom_levels(self, start_zoom: int, end_zoom: int = 10):
        prev_zoom = start_zoom
        zoom = start_zoom - 1
        img_dummy = np.zeros(
            (self.TILE_SIZE_PX, self.TILE_SIZE_PX, 4), dtype=np.uint8)
        while zoom >= end_zoom:
            prev_tile_dir = "{}/{}".format(self.out_folder, prev_zoom)
            tile_names = [f for f in listdir(
                prev_tile_dir) if isfile(join(prev_tile_dir, f))]
            tile_dir = "{}/{}".format(self.out_folder, zoom)
            os.makedirs(tile_dir, exist_ok=True)
            while len(tile_names) > 0:
                parts = tile_names[0].replace(".png", "").split("_")
                # make the number be even (-1 if not)
                x_start = int(int(parts[0]) / 2) * 2
                y_start = int(int(parts[1]) / 2) * 2

                imgs = []
                for x in range(x_start, x_start+2):
                    for y in range(y_start, y_start+2):
                        tile_name = "{}_{}.png".format(x, y)
                        if tile_name in tile_names:
                            imgs.append(cv2.imread(
                                "{}/{}".format(prev_tile_dir, tile_name), cv2.IMREAD_UNCHANGED))
                            tile_names.remove(tile_name)
                        else:
                            imgs.append(img_dummy)
                vert1 = np.concatenate((imgs[0], imgs[1]), axis=0)
                vert2 = np.concatenate((imgs[2], imgs[3]), axis=0)
                new_tile = np.concatenate((vert1, vert2), axis=1)
                new_tile = cv2.resize(
                    new_tile, (self.TILE_SIZE_PX, self.TILE_SIZE_PX))
                lat, lon = num2deg(x_start, y_start, prev_zoom)
                # to prevent rounding errors move position slightly into the tile
                new_x, new_y = deg2num(
                    lat - 0.00000001, lon + 0.00000001, zoom)
                cv2.imwrite("{}/{}_{}.png".format(tile_dir,
                            new_x, new_y), new_tile)
            prev_zoom = zoom
            zoom -= 1
        return zoom + 1

    def get_deg_per_pix_for_level(self, left_button_pos: LatLon, zoom: int) -> Tuple[float, float]:
        x, y = deg2num(left_button_pos.lat, left_button_pos.lon, zoom)
        lat_start, lon_start = num2deg(x, y, zoom)
        lat_next, lon_next = num2deg(x+1, y+1, zoom)
        lat_per_pix_z = (lat_next - lat_start) / self.TILE_SIZE_PX
        lon_per_pix_z = (lon_next - lon_start) / self.TILE_SIZE_PX
        return abs(lat_per_pix_z), abs(lon_per_pix_z)

    def get_closest_tile_lat_lon(self, pos: LatLon, zoom: int, x_floor=True, y_floor=True) -> Tuple[float, float]:
        x, y = deg2num(pos.lat, pos.lon, zoom)
        if not x_floor:
            x += 1
        if not y_floor:
            y += 1
        return num2deg(x, y, zoom)

    @staticmethod
    def print_tile_url(x, y, zoom):
        print("https://c.tile.openstreetmap.org/{}/{}/{}.png".format(zoom, x, y))


if __name__ == '__main__':
    file_name = "data/photo_2021-12-06_12-30-31.png"
    # file_name = "data/test01.png"

    # start: button left
    # stop: top right
    # lat -> north -> height -> y
    # lon -> east -> width -> x

    gen = TileGenerator()
    zoom = gen.generate_tiles(file_name,
                              LatLon(52.945179264164544, 12.781487703323366),
                              LatLon(52.94320740335978, 12.785328626632692))
    gen.generate_zoom_levels(zoom, end_zoom=15)

    print("done")