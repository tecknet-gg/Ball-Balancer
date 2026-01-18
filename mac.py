from flask import Flask, Response, jsonify
from collections import deque
import requests
import threading
import cv2
import numpy as np
import time

# Ball colour
hsvLow = np.array([130, 40, 40])
hsvHigh = np.array([179, 255, 220])

app = Flask(__name__)
url = "http://192.168.3.58:5000/raw_feed"

ballCenter = None
relCenter = None
lastKnownCenter = None
missingFrames = 0
emaCenter = None
emaAlpha = 0.25
isLocked = False
velocity = np.array([0.0, 0.0])
lastFrameTime = time.time()

latestRaw = None
latestProcessed = None
latestBlurred = None
frameLock = threading.Lock()

vertBar = [60, 10, 650, 750]
horizBar = [0, 190, 750, 320]
searchWindow = None


def processFrame(frame):
    global ballCenter, lastKnownCenter, missingFrames, emaCenter, isLocked, velocity, lastFrameTime, relCenter, latestBlurred, searchWindow
    currentTime = time.time()
    deltaTime = currentTime - lastFrameTime
    lastFrameTime = currentTime

    if frame is None:
        return frame

    height, width = frame.shape[:2]

    maskRoi = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(maskRoi, (vertBar[0], vertBar[1]), (vertBar[0] + vertBar[2], vertBar[1] + vertBar[3]), 255, -1)
    cv2.rectangle(maskRoi, (horizBar[0], horizBar[1]), (horizBar[0] + horizBar[2], horizBar[1] + horizBar[3]), 255, -1)

    if isLocked and ballCenter is not None:
        winSize = 400
        x1 = max(0, ballCenter[0] - winSize // 2)
        y1 = max(0, ballCenter[1] - winSize // 2)
        x2 = min(width, ballCenter[0] + winSize // 2)
        y2 = min(height, ballCenter[1] + winSize // 2)

        searchMask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(searchMask, (x1, y1), (x2, y2), 255, -1)
        maskRoi = cv2.bitwise_and(maskRoi, searchMask)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    maskedGray = cv2.bitwise_and(grayFrame, maskRoi)
    blurredFrame = cv2.medianBlur(maskedGray, 3)

    _, encodedBlurred = cv2.imencode(".jpg", blurredFrame)
    with frameLock:
        latestBlurred = encodedBlurred.tobytes()

    detectedCircles = cv2.HoughCircles(blurredFrame, cv2.HOUGH_GRADIENT, 1.2, 60,param1=50, param2=18, minRadius=80, maxRadius=100)
    detectedThisFrame = False

    if detectedCircles is not None:
        detectedCircles = np.uint16(np.around(detectedCircles))
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for circle in detectedCircles[0, :]:
            centerX, centerY, radius = circle
            if 0 <= centerY < height and 0 <= centerX < width:
                sampleRadius = 5
                sy1, sy2 = max(0, centerY - sampleRadius), min(height, centerY + sampleRadius)
                sx1, sx2 = max(0, centerX - sampleRadius), min(width, centerX + sampleRadius)

                if sy2 > sy1 and sx2 > sx1:
                    sampleRegion = hsvFrame[sy1:sy2, sx1:sx2]
                    orangeMask = cv2.inRange(sampleRegion, hsvLow, hsvHigh)
                    orangePercent = np.count_nonzero(orangeMask) / orangeMask.size

                    if orangePercent > 0.2:
                        newCenter = np.array([centerX, centerY], dtype=float)
                        detectedThisFrame = True

                        if lastKnownCenter is not None and deltaTime > 0:
                            diffVector = newCenter - np.array(lastKnownCenter)
                            distanceMoved = np.linalg.norm(diffVector)
                            instantVelocity = diffVector / deltaTime if distanceMoved >= 2.5 else np.array([0.0, 0.0])
                            velocity = (0.90 * velocity) + (0.10 * instantVelocity)

                        if emaCenter is None:
                            emaCenter = newCenter
                        else:
                            currentSpeed = np.linalg.norm(velocity)
                            dynamicAlpha = np.clip(0.35 + (currentSpeed / 1000), 0.35, 0.8)
                            emaCenter = (dynamicAlpha * newCenter) + ((1 - dynamicAlpha) * emaCenter)

                        ballCenter = (int(emaCenter[0]), int(emaCenter[1]))
                        lastKnownCenter = (int(newCenter[0]), int(newCenter[1]))
                        missingFrames, isLocked = 0, True
                        cv2.circle(frame, (centerX, centerY), radius, (0, 255, 0), 2)
                        break

    if not detectedThisFrame:
        missingFrames += 1
        if isLocked and missingFrames > 1:
            isLocked = False
            ballCenter = None

        cv2.putText(frame, f"Searching Field ({missingFrames})", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if missingFrames > 5:
            emaCenter, velocity = None, np.array([0.0, 0.0])

    cv2.rectangle(frame, (vertBar[0], vertBar[1]), (vertBar[0] + vertBar[2], vertBar[1] + vertBar[3]), (255, 255, 255), 1)
    cv2.rectangle(frame, (horizBar[0], horizBar[1]), (horizBar[0] + horizBar[2], horizBar[1] + horizBar[3]), (255, 255, 255), 1)

    if emaCenter is not None:
        cv2.drawMarker(frame, (int(emaCenter[0]), int(emaCenter[1])), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
        relCenter = (float(emaCenter[0] - width / 2), float(-(emaCenter[1] - height / 2)))
    return frame

@app.route("/coordinates")
def coordinatesNew():
    global lastFrameTime, relCenter, velocity, missingFrames
    with frameLock:
        center, vList, tLast, missed = relCenter, velocity.tolist(), lastFrameTime, missingFrames
    if center is None: return jsonify({"x": 0, "y": 0, "detected": False})
    tSince = time.time() - tLast
    posX = round(float(center[0] + (vList[0] * tSince)), 2) if missed > 0 else round(float(center[0]), 2)
    posY = round(float(center[1] + (vList[1] * tSince)), 2) if missed > 0 else round(float(center[1]), 2)
    return jsonify({"x": posX, "y": posY, "detected": True, "mode": "extrapolated" if missed > 0 else "actual", "velocity": vList})

def fetchRaw():
    global latestRaw, latestProcessed
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        buffer = b""
        boundary = b"--frame"
        for chunk in r.iter_content(chunk_size=4096):
            if not chunk: continue
            buffer += chunk
            while True:
                start = buffer.find(boundary)
                if start == -1: break
                headerEnd = buffer.find(b"\r\n\r\n", start)
                if headerEnd == -1: break
                headers = buffer[start:headerEnd].decode(errors="ignore")
                dataStart = headerEnd + 4
                contentLength = None
                for line in headers.split("\r\n"):
                    if "Content-Length" in line:
                        contentLength = int(line.split(":")[1].strip())
                if contentLength is None or len(buffer) < dataStart + contentLength: break
                jpg = buffer[dataStart:dataStart + contentLength]
                buffer = buffer[dataStart + contentLength:]
                with frameLock:
                    latestRaw = jpg
                npImg = np.frombuffer(jpg, dtype=np.uint8)
                img = cv2.imdecode(npImg, cv2.IMREAD_COLOR)
                if img is not None:
                    processed = processFrame(img)
                    _, enc = cv2.imencode(".jpg", processed)
                    with frameLock:
                        latestProcessed = enc.tobytes()

def mjpegGenerator(source):
    while True:
        with frameLock:
            frame = source()
        if frame is not None:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                   + frame + b"\r\n")
        time.sleep(0.01)

@app.route("/blurred")
def blurredProxy():
    return Response(mjpegGenerator(lambda: latestBlurred), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/rawproxy")
def rawProxy():
    return Response(mjpegGenerator(lambda: latestRaw), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/processed")
def processedProxy():
    return Response(mjpegGenerator(lambda: latestProcessed), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    threading.Thread(target=fetchRaw, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, threaded=True)