
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

in_folder = "data/FV-18.08.21/4. Versuch (WO)"
out_folder = "data/FV-18.08.21/4. Versuch (WO)Small"

# divider for width and height
FACTOR = 4

if not os.path.exists(out_folder):
    os.makedirs(out_folder, exist_ok=True)

file_names = [f for f in listdir(
    in_folder) if isfile(join(in_folder, f))]

i = 0
for f in file_names:
    print("{}/{}".format(i, len(file_names)))
    image = Image.open("{}/{}".format(in_folder, f))
    exif = image.info['exif']
    width, height = image.size
    image = image.resize(
        (int(width/FACTOR), int(height/FACTOR)), Image.ANTIALIAS)

    lat, lon, alt = utils.get_GPSdata_scalar("{}/{}".format(in_folder, f))
    hdg, rel_alt = utils.read_yaw_from_exif("{}/{}".format(in_folder, f))

    tags = {'ts': 0.0,
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'rel_alt': rel_alt,
            'hdg': hdg}
    data = pickle.dumps(tags)
    exif_ifd = {piexif.ExifIFD.MakerNote: data}
    #exif[piexif.ExifIFD.MakerNote] = data
    exif_dict = {"0th": {}, "Exif": exif_ifd, "1st": {},
                 "thumbnail": None, "GPS": {}}
    exif_dat = piexif.dump(exif_dict)

    image.save("{}/{}".format(out_folder, f), 'JPEG',
               exif=exif_dat)

    img = Image.open("{}/{}".format(out_folder, f))
    exif_dict = piexif.load(img.info['exif'])
    raw = exif_dict['Exif'][piexif.ExifIFD.MakerNote]
    tags = pickle.loads(raw)

    print("pos: {}, {}, {} m (rel: {} m)".format(
        tags["lat"], tags["lon"], tags["alt"], tags["rel_alt"]))
    print("heading: {}".format(tags["hdg"]))

    i += 1
