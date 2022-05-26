
class TextGen:
    def __init__ (self, path, filename, CV_im, G_im, Tot_im, time):
        self.path = path
        self.filename = filename
        self.cv = CV_im
        self.G = G_im
        self.Total = Tot_im
        self.time = time

    def write(self):
        time_im = self.time / self.Total
        f = open(self.path +'\\' + self.filename + '.txt','w')
        Inhalt = [  'Number of images with CV Algo = ' + str(self.cv) + '\n',
                    'Number of images with Geo Algo = ' + str(self.G) + '\n',
                    'Total image = ' + str(self.Total) + '\n',
                    'Time needed = ' + str(self.time) + ' second \n'
                    'Time per Image = ' + str(time_im) + 'second']
        f.write(self.filename + '\n')
        f.writelines(Inhalt)
        return             

