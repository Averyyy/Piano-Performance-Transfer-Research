import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *
import pprint as p
from main_with_simulation import *

ref_pitch = 69
ref_vel = 60
var_pitch = 63

def save_plot(filename, randrange, testround):
    pse_records = pd.read_csv(filename) ## pse_records 的目录 ** needed to change
    # plt.show()
    std = format(np.std(pse_records.var_pse_v, ddof=1), '.2f')
    plt.title(f'Rand Range = {randrange}, Test Round={testround}, std = {std}')
    pse_60 = pse_records.query(f'ref_p == {ref_pitch} and ref_v == {ref_vel} and var_p == {var_pitch}')
    pse_60.var_pse_v.hist()
    plt.savefig(f'simulator_files/results/range{randrange}_test{testround}.png', format = 'png', dpi = 300)
    plt.clf()

def findstd(filename):
    pse_records = pd.read_csv(filename)  ## pse_records 的目录 ** needed to change
    # plt.show()
    stddd = np.std(pse_records.var_pse_v, ddof=1)
    return stddd

def threedplot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z, c= data[:,0], data[:,1], data[:,2], data[:,3]
    ax.set_xlabel('Test Round')
    ax.set_ylabel('Random Range')
    ax.set_zlabel('Simulation Time')

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot(), s=4)
    fig.colorbar(img)
    plt.show()

def get_data():
    data = []
    for t in TEST_ROUND:
        for r in RND_RANGE:
            for time in TIMES:
                filename = f'simulator_files/pse_{r}_simulator{t}_{time}.csv'
                data.append([t, r, time, findstd(filename)])

    return np.asarray(data)

def plotmain():
    data = get_data()
    threedplot(data)

def findbest():
    data = get_data()
    loss = []
    for i in data:
        loss.append((pow(i[2]*i[0], 2) + pow(2, 10 * i[3])))
        # loss.append(pow(i[2]*i[0], 10) + pow(2, 75*i[3]))

        # loss.append(i[3])
    loss = np.asarray(loss)
    # idx = np.argpartition(loss, 50)
    idx = loss.argsort()[:50]
    # result = []
    for i in range(50):
        # result.append([data[idx[i]], loss[idx[i]]])
        print(data[idx[i]])
    # p.pprint(result)
    # data = np.hstack(data, np.asarray(loss).reshape(np.shape(loss)[0], 1))

findbest()






# get_data()