
import os
import sys
if True:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from os.path import isfile, join
from os import listdir
import piexif
import utils.utils as utils
import pickle
import math

in_folder = "data/mavic"
out_folder = "data/mavicSmall"

# divider for width and height
FACTOR = 4

ROTATE = 90.0

img = Image.open("data/mavicUnitTest/DJI_0552.JPG")
exif_dict = piexif.load(img.info['exif'])
raw = exif_dict['Exif'][piexif.ExifIFD.MakerNote]
tags = pickle.loads(raw)

tags["hdg"] = (tags["hdg"]+ROTATE) % 360


print("pos: {}, {}, {} m (rel: {} m)".format(
    tags["lat"], tags["lon"], tags["alt"], tags["rel_alt"]))
print("heading: {}".format(tags["hdg"]))

data = pickle.dumps(tags)
exif_ifd = {piexif.ExifIFD.MakerNote: data}
# exif[piexif.ExifIFD.MakerNote] = data
exif_dict = {"0th": {}, "Exif": exif_ifd, "1st": {},
             "thumbnail": None, "GPS": {}}
exif_dat = piexif.dump(exif_dict)

img = img.rotate(ROTATE, expand=True)
img.save("data/mavicUnitTestMod/DJI_0552.JPG", 'JPEG',
         exif=exif_dat)
