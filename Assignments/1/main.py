import numpy as np
import matplotlib.pyplot as plt

eps = 10e-9


def generatePoints():
    x = np.random.uniform(0, 45 + eps, 50)
    y = np.random.uniform(0, 100 + eps, 50)

    x2 = np.random.uniform(55, 100 + eps, 50)
    y2 = np.random.uniform(0, 100 + eps, 50)

    x = np.concatenate((x, x2), axis=0)

    y = np.concatenate((y, y2), axis=0)

    ps = np.column_stack((x, y))

    return ps


def generateABC():
    ps = np.random.uniform(-100, 100, 3)
    return ps


def addLabels(pts):
    for i in range(0, 50):
        plt.text(pts[i][0], pts[i][1], "-1")
    for i in range(50, 100):
        plt.text(pts[i][0], pts[i][1], "1")


def drawLine(prms):

    m = -prms[0] / prms[1]
    n = -prms[2] / prms[1]

    x = np.linspace(0, 102, 100)
    y = m * x + n
    plt.plot(x, y, "-g")


def draw(pts, prms, it):
    plt.plot(pts[:50, 0], pts[:50, 1], "bo", label="-1")
    plt.plot(pts[50:, 0], pts[50:, 1], "ro", label="1")

    drawLine(prms)

    addLabels(pts)

    plt.title(f"Correctly classified = {fitness(pts, prms)} found at iteration {it}")

    plt.gca().set_xlim([-25, 125])
    plt.gca().set_ylim([-25, 125])
    plt.show()


def classify(point, prms):
    if np.dot(np.array([point[0], point[1], 1]), prms) >= 0:
        return 1
    return 0


def fitness(pts, prms):
    minus1 = []
    for i in range(0, 50):
        minus1.append(classify(pts[i], prms))

    plus1 = []
    for i in range(50, 100):
        plus1.append(classify(pts[i], prms))

    return minus1.count(0) + plus1.count(1)


def hillClimbingBest(pts, prms):
    maxVal = fitness(pts, prms)
    local = False
    closer = np.copy(prms)

    while not local:
        local = False
        initMax = maxVal

        for i in range(prms.size):
            for j in np.arange(-1, 1+eps, 0.5):
                if j == 0:
                    continue

                prms[i] += j

                tmp = fitness(pts, prms)
                if tmp > maxVal:
                    maxVal = tmp
                    closer = np.copy(prms)
                prms[i] -= j

        if initMax == maxVal:
            local = True
        else:
            prms = closer

    return maxVal, prms


def iteratedHillClimbing():
    points = generatePoints()
    finalValue = 0
    minIteration = 0
    finalParams = None

    for iteration in range(5000):
        if iteration % 1000 == 0:
            print(f"Iteration: {iteration}")

        params = generateABC()

        maxCandidate, paramsCandidate = hillClimbingBest(points, params)

        if maxCandidate > finalValue:
            finalValue = maxCandidate
            finalParams = np.copy(paramsCandidate)
            minIteration = iteration

    print(f"Correctly classified = {finalValue}\nLine = {finalParams[0]} * x + {finalParams[1]} * y + {finalParams[2]}\nAt Iteration = {minIteration}")
    draw(points, finalParams, minIteration)


if __name__ == '__main__':
    iteratedHillClimbing()
