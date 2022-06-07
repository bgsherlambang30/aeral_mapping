from tkinter import Label
from turtle import width
import matplotlib.pyplot as plt
from datatxtGen import data2txt, get_img, get_files
import numpy as np
import os

Curr_dir = os.getcwd()

PATH_IM = 'C:\\Users\\bagas\\Documents\\TUB\\Praktikum\\LabFly\\second_test_flight\\F1'
PATH_TXT = Curr_dir + '\\Assesment\\data_assesment'

filename = 'test'

images_path_list = get_files(PATH_IM)
images_list = get_img(images_path_list)
if os.path.isfile(PATH_TXT + '\\' + filename + '.txt'):
    os.remove(PATH_TXT + '\\' + filename + '.txt')
f = open(PATH_TXT + '\\' + filename + '.txt', 'a')
f.write('|  Lat  |  Lon  |  Alt  |  Heading  |\n')
f.close()
Lat_list = []
Lon_list = []
Hdg_list = []
Alt_list = []
i = 0
for img in images_list:
    data2txt(PATH_TXT, filename, round(img.lat, 7), round(img.lon, 7), round(img.alt, 3) ,round(img.yaw, 3))
    Lat_list.append(img.lat)
    Lon_list.append(img.lon)
    Hdg_list.append(img.yaw+180)
    Alt_list.append(img.alt)


#creating plot Lat Lon
fig1 = plt.figure(1)
plt.quiver(Lon_list, Lat_list, np.cos(np.radians(Hdg_list)), np.sin(np.radians(Hdg_list)), label = 'Bildaufnahme', width = 0.005, color='r')
plt.xlabel('Längengrad')
plt.ylabel('Breitengrad')
plt.legend(loc = 'lower left')
fig1.savefig(PATH_TXT + '\\' + filename + '_LatLon.jpg')
plt.close(fig1)

#creating plot Alt
fig2 = plt.figure(2)
plt.plot(Alt_list,'ro')
plt.ylabel('Höhe über dem Boden [m]')
fig2.savefig(PATH_TXT + '\\' +  filename + '_Altitude.jpg')
plt.close(fig2)