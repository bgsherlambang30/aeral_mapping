from PIL import Image
import piexif
import pickle

img = Image.open('home/1637691998.098.jpg')
exif_dict = piexif.load(img.info['exif'])
raw = exif_dict['Exif'][piexif.ExifIFD.MakerNote]
tags = pickle.loads(raw)

print("pos: {}, {}, {} m (rel: {} m)".format(
    tags["lat"], tags["lon"], tags["alt"], tags["rel_alt"]))
print("heading: {}".format(tags["hdg"]))