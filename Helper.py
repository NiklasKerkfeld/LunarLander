import time
import matplotlib
import matplotlib.pyplot as plt


import numpy as np
LOG = "Log/log.txt"


def prettyTime(t):
    sec = int(t % 60)
    t = int((t - sec) / 60)
    min = t % 60
    t = int((t- min) / 60)
    h = t % 60
    return f"{h}h:{min}min:{sec}sec"


def fill(string, value, length):
    if len(string) <= length:
        return string + value * (length - len(string))
    return string


class LoadingBar:
    def __init__(self):
        self.start = time.time()
        self.last = self.start
        self.step_times = []

    def __call__(self, step, all, scores, random, version, loss, gamma, rolling=100):
        self.step_times.append(time.time() - self.last)
        self.last = time.time()
        avg_time = np.round(np.average(self.step_times[-rolling:]), decimals=2) if self.step_times else '-'
        needed = time.time() - self.start
        approx_left = (all - step) * avg_time
        procent = int(step/all * 1000)
        avg_loss = np.round(np.average(loss[-rolling:]), decimals=4) if loss else '-'
        average = np.round(np.average(scores[-rolling:]), decimals=2) if scores else '-'
        max = np.max(scores[-rolling:]) if scores else '-'
        min = np.min(scores[-rolling:]) if scores else '-'
        print("\r", end="")
        print(f"{' ' if (procent / 10) < 10 else ''}{procent / 10} % |{'>'* int(procent / 20)}{'-'* int((1000 - procent) / 20)}| {avg_time}s/it, {prettyTime(needed)}-->{prettyTime(approx_left)}, loss: {avg_loss} average: {average}, max: {max}, min: {min}, random {np.round(random, 4)}, gamma: {np.round(1-gamma, 5)}, version: {version}", end="")


class Log:
    def __init__(self, path=LOG):
        self.path = path
        with open(self.path, 'w') as f:
            f.write("")

    def __call__(self, x_train, y_train):
        lines = []
        for x, y in zip(x_train, y_train):
            lines.append(f"state: {x}, target: {y}")
        with open(self.path, 'a') as f:
            f.writelines('\n'.join(lines))

    def log(self, text):
        lines = []
        for x in text:
            lines.append(f"{x}\n")
        with open(self.path, 'a') as f:
            f.writelines('\n'.join(lines))

def plot(values, names, rolling=10):
    avg_values = []
    avg_names = []
    for value, name in zip(values, names):
        avg_values.append([np.average(value[max(0, x-rolling):x]) for x in range(1, len(value))])
        avg_names.append(name + "-avg")

    fig, ax = plt.subplots(ncols=len(values))
    for x, row in enumerate(ax):
        row.plot(values[x], label=names[x])
        row.plot(avg_values[x], label=avg_names[x])
        row.legend(loc='upper left')

    plt.show()


if __name__ == '__main__':
    dummy = [np.random.randint(1, 100, 100) for _ in range(2)]
    names = ["sample", "random"]

    plot(dummy, names)