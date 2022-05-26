import math


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


if __name__ == "__main__":
    zoom = 12
    x, y = deg2num(52.515845, 13.343883, 12)
    print("x: {:d} x: {:d} zoom: {:d}".format(x, y, zoom))
    print(
        "https://c.tile.openstreetmap.org/{:d}/{:d}/{:d}.png".format(zoom, x, y))

    lat, lon = num2deg(x, y, zoom)
    lat1, lon1 = num2deg(x+1, y+1, zoom)
    print("lat: {:f}, lon: {:f}".format(lat, lon))
    print("size: {:f} x {:f} deg".format(lat1-lat, lon1-lon))
