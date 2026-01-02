import time
from picamera2 import Picamera2
import cv2 as cv
from flask import Flask, Response
from threading import Thread

class Camera:
    def __init__(self, htmlConfig, size=(480, 480), fps=60,
                 host="0.0.0.0", port=5000, route="/raw_feed"):

        self.camera = Picamera2()
        self.size = size
        self.fps = fps

        self.port = port
        self.host = host
        self.htmlConfig = htmlConfig
        self.route = route

        self.config = self.camera.create_preview_configuration(main={"size": self.size})
        self.camera.configure(self.config)
        self.camera.start()

        self.app = Flask(__name__)

        self.app.add_url_rule(self.route,"raw_feed", lambda: Response(self.generateFrame(),mimetype="multipart/x-mixed-replace; boundary=frame",))
        self.app.add_url_rule("/raw","index", lambda: self.htmlConfig)
        self.thread = Thread(target=self.startFeed, daemon=True)
        self.thread.start()

    def generateFrame(self):
        delay = 1 / self.fps
        while True:
            frame = self.camera.capture_array()

            ret, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue

            frameBytes = buffer.tobytes()

            yield (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frameBytes)).encode() + b"\r\n\r\n"
                + frameBytes +
                b"\r\n"
            )

            elapsed = time.time() - time.time()
            time.sleep(max(0, delay - elapsed))

    def startFeed(self):
        self.app.run(host=self.host, port=self.port, threaded=True)
