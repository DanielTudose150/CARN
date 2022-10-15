import numpy as np

def prim(x):
    if x < 2:
        return False
    if x % 2 == 0 and x != 2:
        return False
    for i in range(3, x, 2):
        if i * i > x:
            break
        if x % i == 0:
            return False
    return True

def ex1():
    number = int(input())
    res = prim(number)
    print(res)

    file1 = open("Latin-Lipsum.txt", "r")
    lines = file1.readlines()

    wordsList = []
    for line in lines:
        line = line.replace(". ", " ")
        line = line.replace(", ", " ")
        words = line.split()
        wordsList += words

    file1.close()
    wordsList.sort()
    print(wordsList)

    mat = [
        [1, 2, 3, 4],
        [11, 12, 13, 14],
        [21, 22, 23, 24]
    ]
    v = [2, -5, 7, -10]

    # mat(3x4) v(4,1) => (3,1)
    res = []
    for line in mat:
        r = 0
        for i in range(len(line)):
            r += line[i] * v[i]
        res += [r]

    print(res)

if __name__ == '__main__':
    mat = np.array([
       [1, 2, 3, 4],
       [11, 12, 13, 14],
       [21, 22, 23, 24]
    ])
    v = np.array([2, -5, 7, 10])

    # 1. a
    print(mat[:2, 2:])
    # 1.b
    print(v[3:, ])

    # 2. a
    a = np.random.random(5)
    b = np.random.random(5)

    print("\n\n\n\n")
    print(a)
    print(b)

    if np.sum(a) >= np.sum(b):
        print("a")
    else:
        print("b")

    vsum = np.add(a, b)
    vprod = np.multiply(a, b)
    vdot = np.dot(a, b)

    print(f"Sum = {vsum}\nVectorial Prod = {vprod}\nDot = {vdot}")

    a = np.sqrt(a)
    b = np.sqrt(b)
    print(a)
    print(b)


    # 3
    print("\n\n\n")
    mat = np.random.random((5, 5))
    print("Normal mat")
    print(mat)
    print("\n\nTransposed Mat")
    print(mat.T)

    matinv = np.linalg.inv(mat)
    print(f"Inverse of mat\n{matinv}")

    matdet = np.linalg.det(mat)
    print(f"Determinant of mat = {matdet}")

    # 4
    v = np.random.random(5)
    print(v)

    vdot = np.dot(mat, v)
    print(vdot)

    vdot2 = np.dot(v, mat)
    print(vdot2)

    a = np.array([1, 2])
    b = np.array([a[0], a[1], 3])

    print(a)
    a = b
    print(a)
    print(a)

