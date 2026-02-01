import math

class Kinematics:
    def __init__(self, L1=95.0, L2=95.0, heightTarget=155.0, platformRadius=120.0, baseRadius=67.0, offset=20.0):
        self.lServo = L1  #servo arm
        self.lRod = L2  #tie rod arm
        self.platformRadius = platformRadius
        self.baseRadius = baseRadius
        self.offset = offset  #to account for platform mounting offset

        self.heightTarget = heightTarget #target height
        self.maxTiltLimit = 15.0 #restricting angular motion

        print(f"Kinematics initialized.")

    def getPoseVector(self, rollDeg, pitchDeg):
        rollRad = math.radians(max(min(rollDeg, self.maxTiltLimit), -self.maxTiltLimit)) #clamping values
        pitchRad = math.radians(max(min(pitchDeg, self.maxTiltLimit), -self.maxTiltLimit))

        alpha = math.sin(pitchRad)
        beta = -math.sin(rollRad)
        gamma = math.cos(pitchRad) * math.cos(rollRad)

        return alpha, beta, gamma

    def solveTopCoords(self, alpha, beta, gamma):
        effectiveHeight = self.heightTarget - self.offset

        d1 = math.sqrt(4 * gamma ** 2 + (alpha - math.sqrt(3) * beta) ** 2)
        d2 = math.sqrt(gamma ** 2 + alpha ** 2)
        d3 = math.sqrt(4 * gamma ** 2 + (alpha + math.sqrt(3) * beta) ** 2)


        a1 = [-(self.platformRadius * gamma) / d1, (math.sqrt(3) * self.platformRadius * gamma) / d1, effectiveHeight + ((alpha - math.sqrt(3) * beta) * self.platformRadius) / d1]
        a2 = [(self.platformRadius * gamma) / d2, 0, effectiveHeight - (self.platformRadius * alpha) / d2]
        a3 = [-(self.platformRadius * gamma) / d3, -(math.sqrt(3) * self.platformRadius * gamma) / d3, effectiveHeight + ((alpha + math.sqrt(3) * beta) * self.platformRadius) / d3]

        return a1, a2, a3

    def solveElbowAngles(self, nodes):
        thetas = []
        for i, node in enumerate(nodes):
            u = math.hypot(node[0], node[1])
            v = node[2]
            p = u - self.baseRadius
            q = v
            distSq = p ** 2 + q ** 2
            dist = math.sqrt(distSq)

            if dist > (self.lServo + self.lRod) or dist < abs(self.lServo - self.lRod): #checks if reachable
                print(f"Leg {i + 1} out of reach.c")
                return [None, None, None]

            phiBase = math.atan2(q, p) #angle calculation
            cosVal = (self.lServo ** 2 + distSq - self.lRod ** 2) / (2 * self.lServo * dist)
            phiElbow = math.acos(max(-1.0, min(1.0, cosVal)))

            angle = math.degrees(phiBase - phiElbow)
            thetas.append(round(angle, 2))

        return thetas

    def calculate(self, rollDeg, pitchDeg):
        alpha, beta, gamma = self.getPoseVector(rollDeg, pitchDeg)
        nodes = self.solveTopCoords(alpha, beta, gamma)
        return self.solveElbowAngles(nodes)


if __name__ == "__main__":
    kinematics = Kinematics(L1=95.0, L2=95.0, heightTarget=140.0, platformRadius=120.0, baseRadius=67.0)
    while True:
        roll = float(input("Enter Roll: "))
        pitch = float(input("Enter Pitch: "))
        angles = kinematics.calculate(roll,pitch)
        print(f"Angles: {angles}")