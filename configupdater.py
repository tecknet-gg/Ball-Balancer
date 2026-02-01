import socket
import json
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
targetIP = "192.168.3.65"
port = 5006

while True:
    try:
        with open('liveconfig.json', 'r') as f:
            data = json.load(f)

        packet = json.dumps({
            "kx": data.get("kx", [1.0, 0.01, 0.2]),
            "ky": data.get("ky", [1.0, 0.01, 0.2]),
            "dt": data.get("dt", 0.05),
            "servo1offset": data.get("servo1offset", 0.0),
            "servo2offset": data.get("servo2offset", 0.0),
            "servo3offset": data.get("servo3offset", 0.0)
        }).encode()

        sock.sendto(packet, (targetIP, port))
    except Exception as e:
        print(f"UDP Error: {e}")

    time.sleep(0.1)
