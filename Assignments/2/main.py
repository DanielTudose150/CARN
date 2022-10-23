import pickle, gzip, numpy as np
import random
import time


def activation(value):
    if value > 0:
        return 1
    return 0


ARRAY_LENGTH = 784
LEARNING_RATE = 0.03


def train():
    print("Train")
    with gzip.open('mnist.pkl.gz', 'rb') as fd:
        train_set, _, _ = pickle.load(fd, encoding='latin')

        allClassified = False
        ITERATIONS = 25

        weights = np.random.randn(ARRAY_LENGTH, 10)  # (784, 10) matrix
        biases = np.random.randn(1, 10)  # (1, 10) matrix
        t = np.identity(10)  # (10, 10) identity matrix
        wronglyClassifiedCounter = 0

        lastPercentage = None
        lastD = None
        lastWcc = None

        while (not allClassified) and ITERATIONS > 0:
            failedClassifications = [0] * 10  # frequency array to count wrongly classified elements
            allClassified = True

            data = train_set[0]  # data array
            labels = train_set[1]  # labels array
            permutation = [i for i in range(len(data))]
            random.shuffle(permutation)

            for index in permutation:
                x = np.resize(data[index], (1, ARRAY_LENGTH))  # input
                label = labels[index]  # input label

                z = np.dot(x, weights) + biases
                maxActivated = np.argmax(z)
                output = [activation(z[0, i]) for i in range(z.size)]
                tDifference = np.resize(t[label] - output, (1, 10))  # (1,10)

                xT = x.transpose()  # (1,784) -> (784, 1)
                weights += xT.dot(tDifference) * LEARNING_RATE  # (784, 1) * (1, 10) -> (784, 10)
                biases += tDifference * LEARNING_RATE

                if output[maxActivated] != t[label, maxActivated]:
                    failedClassifications[label] += 1
                    wronglyClassifiedCounter += 1
                    allClassified = False

            percentage = wronglyClassifiedCounter * 100 / len(data)
            print(
                f"Iteration={ITERATIONS}:\n\tWrongly classified elements = {wronglyClassifiedCounter} || "
                f"{percentage}%\n\tDistribution = {failedClassifications}")
            lastWcc = wronglyClassifiedCounter
            lastD = failedClassifications
            wronglyClassifiedCounter = 0
            ITERATIONS -= 1
            lastPercentage = percentage

        accuracy = 100 - lastPercentage
        print(f"Accuracy = {accuracy}%")
        writeTestData(lastWcc, lastPercentage, lastD, accuracy)
        writeParams(weights, biases)


def writeTestData(wcc, p, d, acc):
    with open("train.txt", "w") as fd:
        fd.write(f"Wrongly classified elements = {wcc}\n"
                 f"Percentage of wrongly classified elements = {p}%\n"
                 f"Accuracy = {acc}%\n"
                 f"Distribution = {d}")
        fd.close()


def writeValidData(wcc, p, d, acc):
    with open("validate.txt", "w") as fd:
        fd.write(f"Wrongly classified elements = {wcc}\n"
                 f"Percentage of wrongly classified elements = {p}%\n"
                 f"Accuracy = {acc}%\n"
                 f"Distribution = {d}")
        fd.close()


def writeParams(w, b):
    wFile = open("weights.txt", "w")
    for row in w:
        np.savetxt(wFile, row)
    wFile.close()

    bFile = open("biases.txt", "w")
    for row in b:
        np.savetxt(bFile, row)
    bFile.close()


def readParams():
    ws = None
    bs = None
    with open("weights.txt", "r") as wFile:
        ws = np.loadtxt(wFile)
        ws = np.resize(ws, (784, 10))
        wFile.close()

    with open("biases.txt", "r") as bFile:
        bs = np.loadtxt(bFile)
        bs = np.resize(bs, (1, 10))
        bFile.close()
    return ws, bs


def validate():
    print("Validate")
    with gzip.open("mnist.pkl.gz", "rb") as fd:
        _, valid_set, _ = pickle.load(fd, encoding='latin')

        weights, biases = readParams()
        t = np.identity(10)
        wronglyClassifiedCounter = 0
        failedClassifications = [0] * 10
        data = valid_set[0]
        labels = valid_set[1]

        for index in range(len(data)):
            x = np.resize(data[index], (1, ARRAY_LENGTH))
            label = labels[index]
            z = np.dot(x, weights) + biases
            maxActivated = np.argmax(z)
            output = [activation(z[0, i]) for i in range(z.size)]

            if output[maxActivated] != t[label, maxActivated]:
                failedClassifications[label] += 1
                wronglyClassifiedCounter += 1

        percentage = wronglyClassifiedCounter * 100 / len(data)
        accuracy = 100 - percentage
        print(
            f"\tWrongly classified elements = {wronglyClassifiedCounter} || "
            f"{percentage}%\n\t"
            f"Accuracy = {accuracy}%\n\t"
            f"Distribution = {failedClassifications}")
        writeValidData(wronglyClassifiedCounter, percentage, failedClassifications, accuracy)


def test():
    print("Test")
    with gzip.open("mnist.pkl.gz", "rb") as fd:
        _, _, test_set = pickle.load(fd, encoding='latin')

        weights, biases = readParams()
        t = np.identity(10)
        wronglyClassifiedCounter = 0
        failedClassifications = [0] * 10
        data = test_set[0]
        labels = test_set[1]

        for index in range(len(data)):
            x = np.resize(data[index], (1, ARRAY_LENGTH))
            label = labels[index]
            z = np.dot(x, weights) + biases
            maxActivated = np.argmax(z)
            output = [activation(z[0, i]) for i in range(z.size)]

            if output[maxActivated] != t[label, maxActivated]:
                failedClassifications[label] += 1
                wronglyClassifiedCounter += 1

        percentage = wronglyClassifiedCounter * 100 / len(data)
        print(
            f"\tWrongly classified elements = {wronglyClassifiedCounter} || "
            f"{percentage}%\n\t"
            f"Accuracy = {100 - percentage}%\n\t"
            f"Distribution = {failedClassifications}")


def main():
    print("Insert number for operation:", "\n\t1. Train", "\n\t2. Validate", "\n\t3. Test\n\t4. Train and Validate")
    option = int(input())

    if option == 1:
        train()
    elif option == 2:
        validate()
    elif option == 3:
        test()
    elif option == 4:
        print("Train and Validate")
        train()
        validate()
    else:
        print("Incorrect input")


if __name__ == '__main__':
    main()
