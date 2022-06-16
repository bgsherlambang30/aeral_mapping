from tkinter import Label
from turtle import width
import matplotlib.pyplot as plt
from datatxtGen import data2txt, get_img, get_files, get_overlap
import numpy as np
import os

Curr_dir = os.getcwd()

PATH_IM = 'C:\\Users\\bagas\\Documents\\TUB\\Praktikum\\LabFly\\second_test_flight\\F1'
PATH_TXT = Curr_dir + '\\Assesment\\data_assesment'

filename = 'Auswertung'

images_path_list = get_files(PATH_IM)
images_list = get_img(images_path_list)
if os.path.isfile(PATH_TXT + '\\' + filename + '.txt'):
    os.remove(PATH_TXT + '\\' + filename + '.txt')
f = open(PATH_TXT + '\\' + filename + '.txt', 'a')
f.write('|  Lat  |  Lon  |  Alt  |  Heading  |  OV  |  Distance  |\n')
f.close()
Lat_list = []
Lon_list = []
Hdg_list = []
Alt_list = []
OV_list = []
Dist_rout = []
i = 0
Lat_prev = 0
Lon_prev = 0
for img in images_list:
    Lat_list.append(img.lat)
    Lon_list.append(img.lon)
    Hdg_list.append(img.yaw)
    Alt_list.append(img.alt)
    p1 = (Lat_prev,Lon_prev)
    p2 = (img.lat,img.lon)
    pr = (52.9438215,12.7850094)
    OV,Dist = get_overlap(p1,p2,img.alt)
    _, DR = get_overlap(pr,p2,img.alt)
    Dist_rout.append(DR)
    OV_list.append(OV)
    Lat_prev = img.lat
    Lon_prev = img.lon
    data2txt(PATH_TXT, filename, round(img.lat, 7), round(img.lon, 7), round(img.alt, 3) ,round(img.yaw, 3),OV,Dist)

print(np.mean(OV_list[1:-1]),np.mean(Alt_list),np.mean(Dist_rout))
#creating plot Lat Lon
fig1 = plt.figure(figsize=(12,12))
plt.rcParams.update({'font.size': 20})
plt.quiver(Lon_list, Lat_list, np.cos(np.radians(Hdg_list)), np.sin(np.radians(Hdg_list)), label = 'geografische Position der Bildaufnahme', width = 0.005, color='r')
plt.xlabel('Längengrad')
plt.ylabel('Breitengrad')
plt.legend(loc = 'lower left')
fig1.savefig(PATH_TXT + '\\' + filename + '_LatLon.jpg')
fig1.savefig(PATH_TXT + '\\' + filename + '_LatLon.pgf')
plt.close(fig1)

# #creating plot Alt
# fig2 = plt.figure(2)
# plt.plot(Alt_list,'ro')
# plt.xlabel('Anzahl der Bilder')
# plt.ylabel('Höhe über dem Boden [m]')
# fig2.savefig(PATH_TXT + '\\' +  filename + '_Altitude.jpg')
# fig2.savefig(PATH_TXT + '\\' +  filename + '_Altitude.pgf')
# plt.close(fig2)