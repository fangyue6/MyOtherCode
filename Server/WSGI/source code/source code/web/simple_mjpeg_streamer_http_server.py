#!/usr/bin/python
'''
        Author: Igor Maculan - n3wtron@gmail.com
        A Simple mjpg stream http server
'''
import cv2
import Image
import threading
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn
import StringIO
import time
import qrtools
import serial
import MySQLdb
import datetime
import cgi
import random
import string

capture = None
qr = qrtools.QR()
sr = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1)
db = MySQLdb.connect('localhost', 'root', 'root', 'homedb')
cursor = db.cursor()

def passwd_generator():
    s = string.ascii_letters + string.digits
    l = len(s)
    passwd_l = 10
    return ''.join([s[random.randrange(0, l-1)] for x in range(passwd_l)])


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    rc, img = capture.read()
                    if not rc:
                        continue
                    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    jpg = Image.fromarray(imgRGB)
                    tmpFile = StringIO.StringIO()
                    jpg.save(tmpFile,'JPEG')
                    if qr.decode(tmpFile):
                        print(qr.data)
                        ''' Verify QR code '''
                        username, passwd = qr.data.trim().split(',')
                        cursor.execute('select * from user where username="' + qr.username + '"')
                        data = cursor.fetchone()
                        if not data:
                            continue
                        d = datetime.datetime.now() - data[3]
                        if d < datetime.delta(seconds=10):
                            if passwd == data[2]:
                                ''' Open lock via arduino'''
                                if sr.isOpen():
                                    sr.write(b'1')
                    self.wfile.write("--jpgboundary")
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(tmpFile.len))
                    self.end_headers()
                    jpg.save(self.wfile,'JPEG')
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    break
            return
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            with open('front.html', 'r') as f:
                self.wfile.write(f.read());
            # self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img class="img-responsive" src="http://192.168.1.201:8800/cam.mjpg"/>')
            # self.wfile.write('</body></html>')
            with open('end.html', 'r') as f:
                self.wfile.write(f.read())
            return

    def do_POST(self):
        if self.path == '/authenticate':
            ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
            if ctype == 'multipart/form-data':
                postvars = cgi.parse_multipart(self.rfile, pdict)
            elif ctype == 'application/x-www-form-urlencoded':
                length = int(self.headers.getheader('content-length'))
                postvars = cgi.parse_qs(self.rfile.read(length), keep_blank_values=1)
            else:
                postvars = {}
            if postvars:
                username = postvars['username'][0]
                print("username=" + username)
                cursor.execute('select * from user where username=%s', (username, ))
                d = cursor.fetchone()
                if d:
                    new_passwd = passwd_generator()
                    print("new passwd=" + new_passwd)
                    # cursor.execute('select * from user')
                    cursor.execute('update user set passwd=%s, last_update=%s where username=%s', (new_passwd, datetime.datetime.now(), username))
                    db.commit()
                    if cursor:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(username + ',' + new_passwd)
                        print("response OK!")
            return
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<htm><body>Hello world</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def main():
    global capture
    global qr
    global sr
    global cursor
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960);
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 544);
    capture.set(cv2.CAP_PROP_SATURATION,0.2);
    global img
    try:
        server = ThreadedHTTPServer(('192.168.1.201', 8800), CamHandler)
        print "server started"
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()
        cursor.close()
        db.close()

if __name__ == '__main__':
    main()
