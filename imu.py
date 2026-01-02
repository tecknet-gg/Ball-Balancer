import time
import math
import mpu6050


class IMU:
    def __init__(self):
        self.imu = mpu6050.mpu6050(0x68)

    def getAccel(self):
        acceleration = self.imu.get_accel_data()
        return acceleration

    def getGyro(self):
        gyro = self.imu.get_gyro_data()
        return gyro

    def getRawData(self):
        return self.getAccel(), self.getGyro()

    def getOrientation(self):
        samples = 100
        sumRoll = 0
        sumPitch = 0

        for i in range(samples):
            accel = self.getAccel()
            sumPitch += math.degrees(math.atan2(accel['x'], abs(accel['z'])))
            sumRoll += math.degrees(math.atan2(accel['y'], abs(accel['z'])))
            time.sleep(0.005)

        roll = sumRoll / samples
        pitch = sumPitch / samples
        return roll, pitch

if __name__ == "__main__":
    imu = IMU()
    while True:
        roll, pitch = imu.getOrientation()
        print(f"Filtered Roll: {roll:.2f}, Filtered Pitch: {pitch:.2f}")
        time.sleep(1)