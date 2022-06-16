from stitcher import Stitcher

path_MavLink  = 'C:\\Users\\bagas\\Documents\\TUB\\Praktikum\\LabFly\\second_test_flight\\planeFlight_02\\pos.csv'
path_cali     ='C:/Users/bagas/Documents/TUB/Praktikum/LabFly/CameraCalib/calib_gopro.npz'

path_im_plane  = 'C:\\Users\\bagas\\Documents\\TUB\\Praktikum\\LabFly\\second_test_flight\\F1'
path_im_mavic = 'C:\\Users\\bagas\\Documents\\TUB\\Praktikum\\LabFly\\FV-18.08.21\\2. Versuch\\SortKlein'
path_im_mavic2 = 'C:\\Users\\bagas\\Documents\\TUB\\Praktikum\\LabFly\\2021-11-24_neuruppin\\100MEDIA\\sort_w'


IMG_NAME = 'Auswertung'
PATH_OUT = 'Assesment\\data_assesment'
FOCAL_LENGTH = 10.2666666667  # mm
SENSOR_WIDTH = 13.2  # mm
SENSOR_HEIGHT = 8.8  # mm

map = Stitcher( image_out_path = PATH_OUT,
                focal_length = FOCAL_LENGTH,
                sensor_width = SENSOR_WIDTH,
                sensor_height= SENSOR_HEIGHT,
                cam_rotation = 1)

map.create_map_from_folder( image_out_name = IMG_NAME,
                            input_image_folder_path = path_im_plane)