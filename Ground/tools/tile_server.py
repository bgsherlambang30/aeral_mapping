import http.server
import socketserver
import os
import threading

# if True a all transparent tile is sent, if no image exists for the requested location
# enabling this is necessary for some map-libraries that are not great in handling 4xx responses
RESPOND_WITH_DUMMY = True


class TileHttpRequestHandler(http.server.SimpleHTTPRequestHandler):

    tile_path = "tiles"

    def do_GET(self):
        parts = self.path.split("/")
        if len(parts) == 4:
            file_path = "{}/{}/{}_{}".format(
                self.tile_path,
                parts[1], parts[2], parts[3])
            if not file_path.endswith(".png"):
                file_path += ".png"
            if os.path.isfile(file_path):
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                #self.path = file_path
                f = open(file_path, 'rb')
                self.wfile.write(f.read())
                f.close()
            if RESPOND_WITH_DUMMY:
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                f = open("tools/img/dummy.png", 'rb')
                self.wfile.write(f.read())
                f.close()
            else:
                self.send_error(404, "File not found")
            return
        # self.send_response(400)
        self.send_error(400, "Bad request")
        # self.wfile.write(bytes("bad request", "utf8"))
        return


if __name__ == '__main__':
    TILE_SERVER_PORT = 8000
    TILES_ROOT_PATH = "dataOut/tiles/"
    handler_object = TileHttpRequestHandler
    handler_object.tile_path = TILES_ROOT_PATH

    my_server = socketserver.TCPServer(
        ("", TILE_SERVER_PORT), handler_object)
    my_server.serve_forever()
