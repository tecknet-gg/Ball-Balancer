class PID:
    def __init__(self):
        self.kx = [0,0,0] #kp, ki, kd
        self.ky = [0,0,0]

        self.xSign = 1
        self.ySign = 1

        self.xTarget = 0.0
        self.yTarget = 0.0

        self.integralX = 0.0
        self.integralY = 0.0

        self.dt = 0.05


    def setKx(self, kx):
        self.kx = kx

    def setKy(self, ky):
        self.ky = ky

    def setXTarget(self, xTarget):
        self.xTarget = xTarget

    def setYTarget(self, yTarget):
        self.yTarget = yTarget

    def setDt(self, dt):
        self.dt = dt

    def setXSign(self, sign):
        self.xSign = sign

    def setYSign(self, sign):
        self.ySign = sign

    def calculatePose(self, x, y, velX, velY):
        errorX = x - self.xTarget
        errorY = y - self.yTarget

        self.integralX += errorX * self.dt
        self.integralX = max(-2.0, min(2.0, self.integralX))

        self.integralY += errorY * self.dt
        self.integralY = max(-2.0, min(2.0, self.integralY))

        derivativeX = -velX
        derivativeY = -velY

        rawPoseX = self.xSign * (self.kx[0] * errorX + self.kx[1] * self.integralX + self.kx[2] * derivativeX)
        rawPoseY = self.ySign * (self.ky[0] * errorY + self.ky[1] * self.integralY + self.ky[2] * derivativeY)

        MAXTILT = 15.0
        poseX = rawPoseX * MAXTILT
        poseY = rawPoseY * MAXTILT

        return poseX, poseY
