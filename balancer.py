import time
import os
import json
import board
import busio
from adafruit_pca9685 import PCA9685
from servo import Servo
from imu import IMU
import math
import threading
from pid import PID
from flask import Flask, request, jsonify
import logging
from camera import Camera
from callibrate import callibrate
import requests
from kinematics import Kinematics

class Balancer:
    def __init__(self, channel1 = 13, channel2 = 14, channel3 = 15):
        print("Starting balancer...")
        self.pwm = PCA9685(busio.I2C(board.SCL, board.SDA))
        self.pwm.frequency = 50

        self.kinematics = Kinematics()

        self.restingRoll = 0
        self.restingPitch = 0

        self.servoOffsets = []
        with open("config.json", "r") as f:
            config = json.load(f)

        self.servoOffsets = [config[f"servo{i+1}offset"] for i in range(3)]
        print(self.servoOffsets)

        self.htmlconfig = config.get("htmlConfig", "<html><body><h1>Camera Feed</h1></body></html>")

        self.servo1 = Servo(channel1, self.servoOffsets[0], self.pwm) #setting default offsets
        self.servo2 = Servo(channel2, self.servoOffsets[1], self.pwm)
        self.servo3 = Servo(channel3, self.servoOffsets[2], self.pwm)
        self.servos = [self.servo1, self.servo2, self.servo3]

        self.setAngles([60,60,60])
        time.sleep(1)
        self.home()

        self.imu = IMU()
        orientation = self.imu.getOrientation()
        print(f"Initial Roll: {orientation[0]:.2f}, Pitch: {orientation[1]:.2f}")

        self.lock = threading.Lock()
        with self.lock:
            self.coordinates = None

        self.camera = Camera(self.htmlconfig, (720, 720), 120)

    def manualCallibrate(self): #callibrate manually using callibrate.py
        while True:
            result = callibrate(self.servoOffsets)
            if result == False:
                break
            else:
                self.servoOffsets = result
                self.servo1.updateOffset(self.servoOffsets[0])
                self.servo2.updateOffset(self.servoOffsets[1])
                self.servo3.updateOffset(self.servoOffsets[2])
                self.home()

    def testKinematics(self, roll, pitch): #testing the maths
        angles = self.kinematics.calculate(roll,pitch)
        print(f"Angles: {angles}")
        self.setAngles(angles)

    def setAngles(self, angles):
        servos = [self.servo1, self.servo2, self.servo3]
        for servo, angle in zip(servos, angles): #associate each servo with each angle
            servo.setAngle(angle)

    def setAngle(self, servoNumber, angle):
        servo = [self.servo1, self.servo2, self.servo3][servoNumber - 1]
        servo.setAngle(angle)

    def sweepServo(self, servoNumber, startAngle, endAngle, stepAngle, toStart=False, delay=0.01): #ooh default parameters
        servo = [self.servo1, self.servo2, self.servo3][servoNumber - 1]
        servo.sweep(startAngle, endAngle, stepAngle, toStart, delay)

    def sweepAll(self, startAngle, endAngle, stepAngle, toStart=False, delay=0.01): #should probaly wrap setAll and sweepAll into the single arguement functions
        self.sweepServo(1, startAngle, endAngle, stepAngle, toStart, delay)
        self.sweepServo(2, startAngle, endAngle, stepAngle, toStart, delay)
        self.sweepServo(3, startAngle, endAngle, stepAngle, toStart, delay)

    def home(self,delay=1): #going to the mathematical home position
        angles = self.kinematics.calculate(0,0)
        theta1, theta2, theta3 = angles
        theta1 + self.servo1.offset
        theta2 + self.servo2.offset
        theta3 + self.servo3.offset
        angles = [theta1, theta2, theta3]
        self.setAngles(angles)
        time.sleep(1)

    def manualControl(self): #send manual control commands -> move and refactor into callibrate.py?
        while True:
            motor = int(input("Enter motor number (1-3): "))
            angle = int(input("Enter angle (-180 to 180): "))
            self.setAngle(motor, angle)

    def waveMotion(self, duration=10, steps=100, amplitude=45, offsets=[0, 0, 0]): #sine wave :]
        startTime = time.time()
        servos = [self.servo1, self.servo2, self.servo3]
        while time.time() - startTime < duration:
            for i in range(steps):
                t = i / steps * 2 * math.pi
                angles = [
                    offsets[0] + amplitude * math.sin(t),
                    offsets[1] + amplitude * math.sin(t + 2 * math.pi / 3),
                    offsets[2] + amplitude * math.sin(t + 4 * math.pi / 3)
                ]
                self.setAngles(angles)
                time.sleep(0.02)

    def autoCallibrate(self): #gradient-descent(-ish) homing. doesn't really take into account IMU inaccuracies
        print("Starting Auto-Calibration...")
        pitchTolerance = 0.3
        rollTolerance = 0.3
        stepNormal = 0.5
        stepSmall = 0.25
        stepLarge = 2.0
        while True:
            roll, pitch = self.imu.getOrientation()
            if roll != 0 and pitch != 0:
                print("IMU secured")
                break
        while True:
            roll, pitch = self.imu.getOrientation()
            errorR = abs(roll - self.restingRoll)
            errorP = abs(pitch - self.restingPitch)
            step = stepSmall if (errorR < 1 and errorP < 1) else stepNormal #dynamicStep = min(0.5,0+error*0.1) or something similar for more dynamic step size
            step = stepLarge if errorR > 4 else stepNormal
            print(f"Roll: {roll:.2f}, Pitch: {pitch:.2f} Step: {step:.2f}")
            if roll < self.restingRoll + rollTolerance and roll > self.restingRoll - rollTolerance and pitch < self.restingPitch + pitchTolerance and pitch > self.restingPitch - pitchTolerance:
                print("Calibration Complete!")
                print(f"Servo Offsets: {self.servo1.offset, self.servo2.offset, self.servo3.offset}")
                break
            if pitch > 0:
                self.servo1.updateOffset(self.servo1.offset - step) #primary actuator set as servo 1
            elif pitch < 0:
                self.servo1.updateOffset(self.servo1.offset + step)

            if roll > 0:
                self.servo3.updateOffset(self.servo3.offset - step)
                self.servo2.updateOffset(self.servo2.offset + step)
            elif roll < 0:
                self.servo3.updateOffset(self.servo3.offset + step)
                self.servo2.updateOffset(self.servo2.offset - step)

            self.home(0.1)

    def startListener(self, url="http://Jeevans-Mac.local:5001/coordinates", pollInterval=0.05): #host name of processing computer
        def listener():
            print("Starting listener...")
            while True:
                try:
                    r = requests.get(url, timeout=0.2)
                    if r.ok:
                        data = r.json()
                        with self.lock:
                            self.coordinates = data
                except Exception as e:
                    with self.lock:
                        self.coordinates = None
                time.sleep(pollInterval)
        thread = threading.Thread(target=listener, daemon=True)
        thread.start()

    def orientation(self): #heper functions
        while True:
            print(self.imu.getOrientation())

    def idle(self):
        while True:
            time.sleep(0.01)
            print("Idling")

    def balance(self,hz=20): #main control loop
        self.startListener()
        delay = 1 / hz
        while True:
            startTime = time.time()
            with self.lock:
                coordinates = self.coordinates
            print("Coordinates:", coordinates)
            time.sleep(max(0,delay-((time.time()-startTime))))


if __name__ == "__main__":
    balancer = Balancer()
    balancer.home()
    balancer.autoCallibrate() #9 0 5
    balancer.balance()