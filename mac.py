from flask import Flask, Response, jsonify
import requests
import threading
import cv2
import numpy as np
import time
import json
import socket

hsvLow = np.array([0, 0, 50])
hsvHigh = np.array([40, 255, 120])

threshold = 0.4

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
targetIP = "192.168.3.65"
port = 5005

app = Flask(__name__)
url = "http://pizero2.local:5000/raw_feed"

ballCenter = None
relCenter = None
lastKnownCenter = None
missingFrames = 0
emaCenter = None
lastRawCenter = None
emaAlpha = 0.4
isLocked = False
velocity = np.array([0.0, 0.0])
lastFrameTime = time.time()

latestRaw = None
latestProcessed = None
latestBlurred = None
frameLock = threading.Lock()

vertBar = [60, 10, 650, 750]
horizBar = [0, 190, 750, 320]


def processFrame(frame):
    global ballCenter, lastKnownCenter, missingFrames, emaCenter, isLocked, velocity, lastFrameTime, relCenter, latestBlurred, lastRawCenter
    currentTime = time.time()
    deltaTime = currentTime - lastFrameTime
    lastFrameTime = currentTime
    height, width = frame.shape[:2]

    maskRoi = np.zeros((height, width), dtype=np.uint8)  # static roi to define area to ignore
    cv2.rectangle(maskRoi, (vertBar[0], vertBar[1]), (vertBar[0] + vertBar[2], vertBar[1] + vertBar[3]), 255, -1)
    cv2.rectangle(maskRoi, (horizBar[0], horizBar[1]), (horizBar[0] + horizBar[2], horizBar[1] + horizBar[3]), 255, -1)

    if isLocked and ballCenter is not None:
        winSize = 500
        x1, y1 = max(0, ballCenter[0] - winSize // 2), max(0, ballCenter[1] - winSize // 2)
        x2, y2 = min(width, ballCenter[0] + winSize // 2), min(height, ballCenter[1] + winSize // 2)
        searchMask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(searchMask, (x1, y1), (x2, y2), 255, -1)
        maskRoi = cv2.bitwise_and(maskRoi, searchMask)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colourMask = cv2.inRange(hsvFrame, hsvLow, hsvHigh)
    kernel = np.ones((7, 7), np.uint8)
    colourMask = cv2.morphologyEx(colourMask, cv2.MORPH_CLOSE, kernel)
    colourMask = cv2.dilate(colourMask, kernel, iterations=1)

    combinedMask = cv2.bitwise_and(maskRoi, colourMask)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    maskedGray = cv2.bitwise_and(grayFrame, combinedMask)
    blurredFrame = cv2.GaussianBlur(maskedGray, (9, 9), 0)

    _, encodedBlurred = cv2.imencode(".jpg", blurredFrame)
    with frameLock:
        latestBlurred = encodedBlurred.tobytes()

    detectedCircles = cv2.HoughCircles(blurredFrame, cv2.HOUGH_GRADIENT, 1.2, 60, param1=50, param2=15, minRadius=75,
                                       maxRadius=100)
    detectedThisFrame = False

    if detectedCircles is not None:
        detectedCircles = np.uint16(np.around(detectedCircles))
        for circle in detectedCircles[0, :]:
            centerX, centerY, radius = circle
            if 0 <= centerY < height and 0 <= centerX < width:
                sampleRadius = 20
                sy1, sy2 = max(0, centerY - sampleRadius), min(height, centerY + sampleRadius)
                sx1, sx2 = max(0, centerX - sampleRadius), min(width, centerX + sampleRadius)
                if sy2 > sy1 and sx2 > sx1:
                    sampleRegion = hsvFrame[sy1:sy2, sx1:sx2]
                    avgHsv = [round(x, 1) for x in cv2.mean(sampleRegion)[:3]]
                    orangeMask = cv2.inRange(sampleRegion, hsvLow, hsvHigh)
                    orangePercent = np.count_nonzero(orangeMask) / orangeMask.size

                    if orangePercent > threshold:
                        print(f"Lock - HSV: {avgHsv} Match: {orangePercent * 100:.1f}%")
                        newCenter = np.array([centerX, centerY], dtype=float)
                        detectedThisFrame = True

                        if lastRawCenter is not None and deltaTime > 0:
                            rawDiff = newCenter - lastRawCenter
                            distMoved = np.linalg.norm(rawDiff)

                            if distMoved < 4.0:
                                rawDiff = np.array([0.0, 0.0])

                            instantVelocity = rawDiff / deltaTime
                            instantSpeed = np.linalg.norm(instantVelocity)

                            currentSpeed = np.linalg.norm(velocity)
                            if (instantSpeed > 25.0) or (currentSpeed > 10.0 and instantSpeed > 5.0):
                                velocity = (0.92 * velocity) + (0.08 * instantVelocity)
                            else:
                                velocity = velocity * 0.5
                                if np.linalg.norm(velocity) < 1.0:
                                    velocity = np.array([0.0, 0.0])

                        lastRawCenter = newCenter.copy()

                        if emaCenter is None:
                            emaCenter = newCenter
                        else:
                            currentSpeed = np.linalg.norm(velocity)
                            dynamicAlpha = np.clip(0.15 + (currentSpeed / 2000), 0.15, 0.6)
                            emaCenter = (dynamicAlpha * newCenter) + ((1 - dynamicAlpha) * emaCenter)

                        ballCenter = (int(emaCenter[0]), int(emaCenter[1]))
                        lastKnownCenter = (int(newCenter[0]), int(newCenter[1]))
                        missingFrames, isLocked = 0, True
                        cv2.circle(frame, (centerX, centerY), radius, (0, 255, 0), 2)
                        break

    if not detectedThisFrame:
        missingFrames += 1
        lastRawCenter = None
        if isLocked and missingFrames > 3:
            isLocked = False
            ballCenter = None
        if missingFrames > 5:
            emaCenter, velocity = None, np.array([0.0, 0.0])

    if emaCenter is not None:
        posX = float(emaCenter[0] - width / 2)
        posY = float(-(emaCenter[1] - height / 2))
        relCenter = (posX, posY)

        outputVel = velocity.tolist()
        if np.linalg.norm(velocity) < 5.0:
            outputVel = [0.0, 0.0]

        try:
            packet = json.dumps({
                "x": round(posX, 2),
                "y": round(posY, 2),
                "det": True,
                "vel": [round(v, 2) for v in outputVel]
            }).encode()
            sock.sendto(packet, (targetIP, port))
        except Exception as e:
            print(f"UDP Error: {e}")
    else:
        relCenter = None
        try:
            packet = json.dumps({"det": False}).encode()
            sock.sendto(packet, (targetIP, port))
        except:
            pass

    cv2.rectangle(frame, (vertBar[0], vertBar[1]), (vertBar[0] + vertBar[2], vertBar[1] + vertBar[3]), (255, 255, 255),
                  1)  # annotations
    cv2.rectangle(frame, (horizBar[0], horizBar[1]), (horizBar[0] + horizBar[2], horizBar[1] + horizBar[3]),
                  (255, 255, 255), 1)
    if emaCenter is not None:
        cv2.drawMarker(frame, (int(emaCenter[0]), int(emaCenter[1])), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
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