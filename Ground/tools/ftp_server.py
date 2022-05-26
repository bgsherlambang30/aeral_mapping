
import os
import threading
import time
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
from pyftpdlib.authorizers import DummyAuthorizer


class MyHandler(FTPHandler):

    on_receive_callback = None

    def on_connect(self):
        print("%s:%s connected" % (self.remote_ip, self.remote_port))

    def on_disconnect(self):
        # do something when client disconnects
        print("disconnected")

    def on_login(self, username):
        # do something when user login
        pass

    def on_logout(self, username):
        # do something when user logs out
        pass

    def on_file_sent(self, file):
        # do something when a file has been sent
        pass

    def on_file_received(self, file):
        # do something when a file has been received
        if self.on_receive_callback != None:
            try:
                self.on_receive_callback(file)
            except:
                pass
        # print("got file: " + str(file))

    def on_incomplete_file_sent(self, file):
        # do something when a file is partially sent
        pass

    def on_incomplete_file_received(self, file):
        # remove partially uploaded files
        os.remove(file)


class FtpServer:

    def __init__(self, home_path="home", port=21, user="user1", pw="peter123"):
        self.home_path = home_path
        self.port = port
        self.user = user
        self.pw = pw
        self.rec_count = 0
        self.on_receive_callback_f = self.default_on_receive_callback

    def start(self):
        ftp_t = threading.Thread(target=self.start_server_thread, name="ftp_t")
        ftp_t.setDaemon(True)
        ftp_t.start()

    def default_on_receive_callback(self, file_name):
        self.rec_count += 1
        print("received: {} - {}".format(self.rec_count, file_name))

    def start_server_thread(self):
        try:
            os.mkdir(self.home_path)
        except:
            pass
        authorizer = DummyAuthorizer()
        authorizer.add_user(self.user, self.pw,
                            homedir=self.home_path, perm='elradfmwMT')
        # authorizer.add_anonymous(homedir='.')

        handler = MyHandler
        handler.on_receive_callback = self.on_receive_callback_f
        handler.authorizer = authorizer
        server = FTPServer(('0.0.0.0', self.port), handler)
        server.serve_forever()


if __name__ == "__main__":
    ser = FtpServer(home_path="ftpHome", port=22, user="user1", pw="password1")
    ser.start()

    while True:
        time.sleep(1.0)