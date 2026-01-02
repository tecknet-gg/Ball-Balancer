from flask import Flask, Response, jsonify
from collections import deque
import requests
import threading
import cv2
import numpy as np
import time

hLow, sLow, vLow = 120, 10, 180    # lower bounds
hHigh, sHigh, vHigh = 160, 80, 255  # upper bounds

hsvLow = np.array([hLow, sLow, vLow])
hsvHigh = np.array([hHigh, sHigh, vHigh])

app = Flask(__name__) #flask stuff
url = "http://pizero2.local:5000/raw_feed"


ballCenter = None #defined globally so it can be read by threads
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
frameLock = threading.Lock()

vertBar = [60, 10, 650, 750]
horizBar = [0, 190, 750, 320]


def processFrame(frame):
    global ballCenter, lastKnownCenter, missingFrames, emaCenter, isLocked, velocity, lastFrameTime, relCenter
    currentTime = time.time()
    dt = currentTime - lastFrameTime
    lastFrameTime = currentTime
    if frame is None: return frame
    h, w = frame.shape[:2]
    maskRoi = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(maskRoi, (vertBar[0], vertBar[1]), (vertBar[0] + vertBar[2], vertBar[1] + vertBar[3]), 255, -1)
    cv2.rectangle(maskRoi, (horizBar[0], horizBar[1]), (horizBar[0] + horizBar[2], horizBar[1] + horizBar[3]), 255, -1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    maskedGray = cv2.bitwise_and(gray, maskRoi)
    blurred = cv2.medianBlur(maskedGray, 3)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 60, param1=50, param2=22, minRadius=80, maxRadius=120)
    detectedThisFrame = False
    if circles is not None:
        circles = np.uint16(np.around(circles))
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for circle in circles[0, :]:
            cx, cy, r = circle
            if 0 <= cy < h and 0 <= cx < w:
                sampleColor = hsvFrame[cy, cx] #swatching centre of deteced thing
                isWhite = (hsvLow[0] <= sampleColor[0] <= hsvHigh[0] and hsvLow[1] <= sampleColor[1] <= hsvHigh[1] and hsvLow[2] <= sampleColor[2] <= hsvHigh[2])
                if not isWhite:
                    newCenter = np.array([cx, cy], dtype=float)
                    detectedThisFrame = True
                    if lastKnownCenter is not None and dt > 0:
                        diff = newCenter - np.array(lastKnownCenter)
                        distMoved = np.linalg.norm(diff)
                        instantV = diff / dt if distMoved >= 2.5 else np.array([0.0, 0.0])
                        newCenter = newCenter if distMoved >= 2.5 else np.array(lastKnownCenter)
                        velocity = (0.95 * velocity) + (0.05 * instantV)
                        if np.linalg.norm(velocity) < 7.0: velocity = np.array([0.0, 0.0])
                    if emaCenter is None: emaCenter = newCenter
                    else:
                        speed = np.linalg.norm(velocity)
                        dynamicAlpha = np.clip(0.30 + (speed / 1000), 0.30, 0.8)
                        emaCenter = (dynamicAlpha * newCenter) + ((1 - dynamicAlpha) * emaCenter)
                    newEmaX, newEmaY = int(emaCenter[0]), int(emaCenter[1])
                    if ballCenter is None or (abs(newEmaX - ballCenter[0]) >= 1 or abs(newEmaY - ballCenter[1]) >= 1): ballCenter = (newEmaX, newEmaY)
                    lastKnownCenter = (int(newCenter[0]), int(newCenter[1]))
                    missingFrames, isLocked = 0, True
                    cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
                    break
    if not detectedThisFrame:
        missingFrames += 1
        cv2.putText(frame,f"No Ball", (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),2,cv2.LINE_AA)
        if missingFrames > 10: isLocked, ballCenter, emaCenter, velocity = False, None, None, np.array([0.0, 0.0])
    cv2.rectangle(frame, (vertBar[0], vertBar[1]), (vertBar[0] + vertBar[2], vertBar[1] + vertBar[3]), (255, 255, 255), 1)
    cv2.rectangle(frame, (horizBar[0], horizBar[1]), (horizBar[0] + horizBar[2], horizBar[1] + horizBar[3]), (255, 255, 255), 1)
    cv2.drawMarker(frame, (int(w/2), int(h/2)), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
    if emaCenter is not None:
        speed = np.linalg.norm(velocity)
        cv2.drawMarker(frame, (int(emaCenter[0]), int(emaCenter[1])), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(frame,f"Detected Ball", (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,f" Speed: {speed}", (0, 700),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),2,cv2.LINE_AA)

        relCenter = (float(emaCenter[0] - w / 2), float(-(emaCenter[1] - h / 2)))
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
    print("Connecting to:", url)
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

                if contentLength is None:
                    buffer = buffer[dataStart:]
                    break
                if len(buffer) < dataStart + contentLength:
                    break

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
            yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                    + frame +
                    b"\r\n")
        time.sleep(0.01)

@app.route("/rawproxy")
def rawProxy():
    return Response(mjpegGenerator(lambda: latestRaw), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/processed")
def processedProxy():
    return Response(mjpegGenerator(lambda: latestProcessed), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/coordinatesold")
def coordinatesProxy():
    with frameLock:
        center = relCenter
    if center is None:
        return jsonify({"x": 0, "y": 0, "detected": False})
    return jsonify({"x": int(center[0]), "y": int(center[1]), "detected": True})


if __name__ == "__main__":
    threading.Thread(target=fetchRaw, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, threaded=True)