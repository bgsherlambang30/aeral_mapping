import time
import datetime
import os
import io
import threading
from random import randrange
import cv2
import pymavlink.mavutil as mavutil
import ftplib
from PIL import Image
import piexif
import pickle

from picamera.array import PiRGBArray
from picamera import PiCamera


class Sat:

    FRAME_RATE = 1

    new_img_list = []

    def __init__(self):
        self.log_folder_name = "log/{:s}".format(str(randrange(9999)).zfill(4))
        self.last_frame_ts = 0
        self.last_mav_ts = 0
        self.last_gps_pos = GpsPos(-1.0, -1.0, -1.0, -1.0, -1.0)
        self.csv_file_name = None
        self.last_csv_sync_ts = 0
        try:
            os.mkdir("log")
        except:
            pass
        try:
            os.mkdir(self.log_folder_name)
        except:
            pass

    def start(self):
        mav_t = threading.Thread(target=self.mav_thread, name="mav_t")
        mav_t.setDaemon(True)
        mav_t.start()

        cam_t = threading.Thread(target=self.picam_thread, name="cam_t")
        cam_t.setDaemon(True)
        cam_t.start()

        ftp_t = threading.Thread(target=self.ftp_thread, name="ftp_t")
        ftp_t.setDaemon(True)
        ftp_t.start()

    def mav_thread(self):
        while True:
            mav1 = None
            try:
                # "/dev/ttyS0"
                mav1 = mavutil.mavlink_connection(
                    "/dev/ttyAMA0", baud=115200, dialect="ardupilotmega", autoreconnect=True)
                print("wait for HB")
                mav1.wait_heartbeat()
                self.last_mav_ts = time.time()
                print("Heartbeat from system (system %u component %u)" %
                      (mav1.target_system, mav1.target_component))
                mav1.mav.request_data_stream_send(mav1.target_system, mav1.target_component,
                                                  mavutil.mavlink.MAV_DATA_STREAM_POSITION, 5, 1)

                # f = open(self.log_folder_name + "/" + datetime.datetime.now().strftime(
                #    "%Y-%m-%d_%H-%M-%S.csv"), "w")
                self.csv_file_name = self.log_folder_name + "/pos.csv"
                f = open(self.csv_file_name, "w")
                f.write("ts,lat,lon,alt,alt_rel,hdg\n")
                while True:
                    # https://mavlink.io/en/messages/common.html#GLOBAL_POSITION_INT
                    # todo: do this unblocking and break on timeout
                    msg = mav1.recv_match(blocking=False)
                    if msg != None:
                        self.last_mav_ts = time.time()
                        if msg.name == "GLOBAL_POSITION_INT":
                            self.last_gps_pos = GpsPos(msg.lat*10e-8,
                                                       msg.lon*10e-8,
                                                       msg.alt*10e-4,
                                                       msg.relative_alt*10e-4,
                                                       msg.hdg*10e-3)
                            """
                            f.write("{:.3f},{:.7f},{:.7f},{:.3f},{:.3f},{:.1f}\n".format(
                                    time.time(),
                                    msg.lat*10e-8,
                                    msg.lon*10e-8,
                                    msg.alt*10e-4,
                                    msg.relative_alt*10e-4,
                                    msg.hdg*10e-3))
                            """
                            f.write(str(self.last_gps_pos))
                            f.flush()
                            print("got pos")
                        else:
                            # print(msg.name)
                            pass
                    else:
                        if time.time() - self.last_mav_ts > 3.0:
                            print("mav timeout, reconnect...")
                            break
                        time.sleep(0.1)
            except Exception as ex:
                print(ex)
            if mav1 != None:
                try:
                    mav1.close()
                except:
                    pass
            time.sleep(1.0)

    def webcam_thread(self):
        cap = cv2.VideoCapture(0)
        # cap.set(3, 1920)
        # cap.set(4, 1080)
        cap.set(3, 1280)
        cap.set(4, 720)
        while True:
            ret, frame = cap.read()
            if(time.time() - self.last_frame_ts >= Sat.FRAME_RATE):
                self.last_frame_ts = time.time()
                if not ret:
                    print("no frame")
                    time.sleep(1)
                    continue
                print("got frame")
                file_name = '{:s}/{:.3f}.jpg'.format(
                    self.log_folder_name, time.time())
                cv2.imwrite(file_name, frame)
                self.new_img_list.append(file_name)
        cv2.destroyAllWindows()
        cap.release()

    def picam_thread(self):
        camera = None
        while True:
            try:
                print("connect to piCam")
                camera = PiCamera()
                camera.resolution = (1920, 1080)
                camera.framerate = 32
                rawCapture = PiRGBArray(camera, size=(1920, 1080))
                # allow the camera to warmup
                time.sleep(0.1)
                for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                    image = frame.array
                    if(time.time() - self.last_frame_ts >= Sat.FRAME_RATE):
                        if image is None:
                            print("no frame")
                            if time.time() - self.last_frame_ts > 5.0:
                                break
                            time.sleep(1)
                        else:
                            print("got frame")
                            self.last_frame_ts = time.time()
                            file_name = '{:s}/{:.3f}.jpg'.format(
                                self.log_folder_name, time.time())
                            cv2.imwrite(file_name, image)
                            # edit exif
                            im = Image.open(file_name)
                            tags = {'ts': self.last_gps_pos.ts,
                                    'lat': self.last_gps_pos.lat,
                                    'lon': self.last_gps_pos.lon,
                                    'alt': self.last_gps_pos.alt,
                                    'rel_alt': self.last_gps_pos.rel_alt,
                                    'hdg': self.last_gps_pos.hdg}
                            data = pickle.dumps(tags)
                            exif_ifd = {piexif.ExifIFD.MakerNote: data}
                            exif_dict = {"0th": {}, "Exif": exif_ifd, "1st": {},
                                         "thumbnail": None, "GPS": {}}
                            exif_dat = piexif.dump(exif_dict)
                            im.save(file_name,  exif=exif_dat)
                            self.new_img_list.append(file_name)
                    key = cv2.waitKey(1) & 0xFF
                    rawCapture.truncate(0)
                    if key == ord("q"):
                        break
            except:
                pass
            try:
                camera.close()
            except:
                pass
            time.sleep(1.0)

    def ftp_thread(self):
        while True:
            session = None
            try:
                session = ftplib.FTP(
                    '192.168.1.27', 'user1', 'peter123', timeout=5.0)
                while True:
                    if len(self.new_img_list) > 0:
                        img_name = self.new_img_list[0]
                        file = open(img_name, 'rb')
                        code = session.storbinary('STOR {:s}'.format(
                            os.path.basename(img_name)), file)
                        file.close()
                        if '226' in code:
                            self.new_img_list.pop(0)
                        else:
                            print("ftp: upload failed, reconnect...")
                            break
                    else:
                        time.sleep(0.1)
                    if time.time() - self.last_csv_sync_ts > 30.0 and self.csv_file_name is not None:
                        file = open(self.csv_file_name, 'rb')
                        code = session.storbinary('STOR {:s}'.format(
                            os.path.basename(self.csv_file_name)), file)
                        file.close()
                        self.last_csv_sync_ts = time.time()
            except:
                print("ftp login failed")
            if session != None:
                try:
                    session.quit()
                except:
                    pass
            time.sleep(1.0)


class GpsPos:

    def __init__(self, lat, lon, alt, rel_alt, hdg):
        self.ts = time.time()
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.rel_alt = rel_alt
        self.hdg = hdg

    def __str__(self):
        return "{:.3f},{:.7f},{:.7f},{:.3f},{:.3f},{:.1f}\n".format(
            self.ts,
            self.lat,
            self.lon,
            self.alt,
            self.rel_alt,
            self.hdg)


if __name__ == '__main__':
    sat = Sat()
    sat.start()

    while True:
        time.sleep(1.0)