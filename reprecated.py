from flask import Flask, Response
from picamera2 import Picamera2
import cv2 as cv
import threading
import time
import numpy as np

class Camera:
    def __init__(self, coordinates, lock, captureSize=(480, 480), fps=25, host='0.0.0.0', port=5000):
        self.captureSize = captureSize
        self.fps = fps
        self.host = host
        self.port = port
        self.coordinates = coordinates
        self.lock = lock
        self.prevCircle = None

        self.camera = None
        self.app = Flask(__name__)
        self._serverStarted = threading.Event()
        self.cameraThread = None

    @staticmethod
    def dist(x1, y1, x2, y2):
        return np.hypot(int(x2) - int(x1), int(y2) - int(y1))

    def _generateFrames(self):
        delay = 1 / self.fps

        while True:
            start_time = time.time()
            frame = self.camera.capture_array()

            if frame is None or frame.size == 0:
                continue

            ret, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' +buffer.tobytes() +b'\r\n')

            elapsed = time.time() - start_time
            time.sleep(max(0, delay - elapsed))

    def startStream(self):

        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(main={"size": self.captureSize})
        self.camera.configure(config)
        self.camera.start()

        @self.app.route('/')
        def index():
            return '<html><body><h1>Camera Feed</h1><img src="/video_feed"></body></html>'

        @self.app.route('/video_feed')
        def videoFeed():
            return Response(self._generateFrames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        def runApp():
            self._serverStarted.set()
            self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

        self.cameraThread = threading.Thread(target=runApp, daemon=True)
        self.cameraThread.start()
        self._serverStarted.wait()
        print(f"Camera stream running on http://{self.host}:{self.port}")
        return self.cameraThread, self.camera


def runCameraThread(coordinates, lock, captureSize=(480, 480), fps=25, host='0.0.0.0', port=5000):

    streamer = Camera(coordinates, lock, captureSize=captureSize, fps=fps, host=host, port=port)
    thread, cam = streamer.startStream()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting camera thread...")
