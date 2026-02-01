import time
import json

class Servo:
    def __init__(self, channel, offset, pwm):
        self.channel = channel
        self.offset = offset
        self.pwm = pwm

        self.FREQ = 50
        self.MIN_PULSE = 0.5
        self.MAX_PULSE = 2.5
        self.MAX_ANGLE = 270
        self.PERIOD = 1000.0 / self.FREQ

        #self.rangeMax = 15 #i forgot what i added these in for
        #self.rangeMin = 10

    def angleToCount(self, angle):
        pulse = self.MIN_PULSE + (angle / self.MAX_ANGLE) * (self.MAX_PULSE - self.MIN_PULSE) #we love linear interpolation
        count12bit = int((pulse / self.PERIOD) * 4096) #native 12bit count for PCA9685 (C++)
        count16bit = count12bit * 16 #convert count to 16bit for Adafruit PWM library (API is in 16bit)
        return count16bit

    def setAngle(self, angle):
        finalAngle = 135 + angle + self.offset
        finalAngle = max(0, min(270, finalAngle))
        finalAngle = self.inputSanitation(finalAngle)
        count = self.angleToCount(finalAngle)
        self.pwm.channels[self.channel].duty_cycle = count

    def sweep(self,startAngle,endAngle, stepAngle, toStart = False, delay=0.01):
        for angle in range(startAngle,endAngle+(1 if stepAngle>0 else -1),stepAngle):
            self.setAngle(angle)
            time.sleep(delay)
        if toStart:
            for angle in range(endAngle,startAngle+(1 if stepAngle>0 else -1),-stepAngle):
                self.setAngle(angle)
                time.sleep(delay)

    def inputSanitation(self, angle):
        return max(0, min(270, angle))
    def updateOffset(self, offset):
        self.offset = offset

