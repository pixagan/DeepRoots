import matplotlib.pyplot as plt

class Line:

    def __init__(self, x, y, interval=1):
        self.x = x
        self.y = y
        self.interval = 1

    def plot(self):
        plt.plot(self.x, self.y)
        plt.show()


