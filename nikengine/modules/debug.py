# this file is for debugging purposes only

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
prevln = []
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'pink', 'brown', 'gray', 'cyan']


# non blocking plotting function to plot two lists
def plot(x, y, markers, xlabel='x', ylabel='y', title='plot'):
    global prevln
    for ln in prevln:
        ln.remove()
    prevln = []

    prevln.append(ax.plot(x, y, color='blue')[0])
    for marker in markers:
        prevln.append(ax.plot(marker[0], marker[1], 'o', color='red')[0])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.canvas.draw()
    plt.pause(0.000001)


def plot_poses_2d(poses: list):
    global prevln
    for ln in prevln:
        ln.remove()
    prevln = []

    for i, pose in enumerate(poses):
        for vec in pose.body_pos:
            prevln.append(ax.plot(vec[0], vec[1], 'o', color=colors[i % len(colors)])[0])

    fig.canvas.draw()
    plt.pause(0.000001)
