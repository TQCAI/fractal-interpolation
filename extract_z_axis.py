import pylab as plt
import pandas as pd
import os
import numpy as np
from fif import G, FIF
from shapely.geometry import *
from scipy.interpolate import interp1d


def process(U, th=.05):
    X = U.copy()
    size = X[:, 1].size
    print(size)
    # 设置处理条件
    if size <= 6:
        print('return ')
        return X
    # 特别处理第一个点
    X[0, 1] = (X[1, 1] + X[2, 1]) / 2
    for i in range(1, size - 1):
        if abs(X[i, 1] - X[i - 1, 1]) > th:
            # print('process')
            X[i, 1] = (X[i - 1, 1] + X[i + 1, 1]) / 2
    # X[:,1]+=max(U[:,1])-max(X[:,1])
    X[:, 1] += U[0, 1] - X[0, 1]
    # X[:,0]-=0.05
    return X


def compress(X, x):
    '''将X[0]压缩到x的大小'''
    minus = X[:, 0].max() - x.max()
    n = len(X[:, 0])
    d = (minus * 2) / (n ** 2 - n)
    cur_minus = d
    # 批量左移
    for i in range(1, n):
        X[i:, 0] -= cur_minus
        cur_minus += d
    return X


def elevate(X, U):
    pass
    endpoint = []
    ans = []
    X_list = [Point(xpt) for xpt in X]

    for xpt in X_list:
        for upt in U:
            if Point(upt).distance(Point(xpt)) < 0.01:
                endpoint.append(xpt)
    print(endpoint)

    for i, xpt in enumerate(X_list):
        if xpt in endpoint:
            index = endpoint.index(xpt)
            next_pt = X_list[i + 1]
            if index + 1 < len(endpoint):

                next_endpoint = endpoint[index + 1]
                line = LineString([xpt, next_endpoint])
                vline = LineString([next_pt, Point(next_pt.x, next_pt.y + 1)])
                inter = line.intersection(vline)
                up = inter.distance(next_pt)
                print(up)
                # up = xpt.y - next_pt.y
            else:
                up = xpt.y - next_pt.y
            ans.append(xpt)
        else:
            ans.append(Point(xpt.x, xpt.y + up))
    ans_a = np.zeros((len(ans), 2), 'float')
    for i, xpt in enumerate(ans):
        ans_a[i, 0] = xpt.x
        ans_a[i, 1] = xpt.y
    return ans_a


def fractal_interpolation(z):
    max_z = np.max(z)
    len_z = (len(z))
    z /= max_z
    x = np.arange(len(z)) / len_z
    U = np.vstack((x, z))
    U = U.T
    X = G(U, 0.05, balance=0)
    # X=elevate(X,U)
    X = process(X, .03)
    X = compress(X, x)
    # X = FIF( X, 0.005, balance=0 )
    # plt.plot(np.linspace(0,len(x),len(X[:, 1])), X[:, 1], '.-')
    z *= max_z
    x *= len_z
    X[:, 1] *= max_z
    X[:, 0] *= len_z
    plt.close()
    plt.plot(X[:, 0], X[:, 1], 'b.-', label='Fractal interpolation')
    plt.plot(x, z, 'r.-', label='Slice sample')
    plt.legend(loc='best')
    plt.xlabel('Z')
    plt.ylabel('X')


def getZ(zBound=100, xSclice=0):
    data = []
    for i in range(442, 1005 + 1):  # 442 500
        name = "csv2/%04d.csv" % i
        if os.path.exists(name):
            df = pd.read_csv(name)
            data.append(np.array(df.Y))
    z = []
    for line in data[:zBound]:
        z.append(line[xSclice])
    z = np.array(z)
    return z


def combine(z):
    max_z = np.max(z)
    len_z = (len(z))
    z /= max_z
    x = np.arange(len(z)) / len_z
    U = np.vstack((x, z))
    U = U.T
    X = G(U, 0.05, balance=0)
    # X=elevate(X,U)
    X = process(X, .03)
    X = compress(X, x)
    # X = FIF( X, 0.005, balance=0 )
    # plt.plot(np.linspace(0,len(x),len(X[:, 1])), X[:, 1], '.-')
    z *= max_z
    x *= len_z
    X[:, 1] *= max_z
    X[:, 0] *= len_z
    plt.close()
    f = interp1d(x, z, kind='quadratic')
    xp = np.linspace(x.min(), x.max(), 100)
    zp = f(xp)
    plt.plot(X[:, 0], X[:, 1], 'y-', label='Fractal interpolation')
    plt.plot(x, z, 'r.-', label='Slice sample')
    plt.plot(xp, zp, 'g-', label='Cubic spline interpolation')

    plt.legend(loc='best')
    plt.xlabel('Z')
    plt.ylabel('X')


def combine_demo():
    '''
    绘制不同的插值算法的比较(在不同的样本点数目下)
    :return:
    '''
    li = [5, 10, 15, 20]
    for i in li:
        z = getZ(i + 1)
        combine(z)
        plt.title(f'{i} layers')
        plt.savefig(f'combine-{i}.png')


def fractal_interpolation_demo():
    '''绘制不同的样本点下分形插值'''
    li = [5, 10, 15, 20]
    for i in li:
        z = getZ(i + 1)
        fractal_interpolation(z)
        plt.title(f'{i} layers')
        plt.savefig(f'fractal-{i}.png')


def raw_sample():
    '''
    绘制一个采样切面
    :return:
    '''
    z = getZ(None)
    x = np.arange(len(z))
    plt.plot(x, z, 'r', label='Slice sample')
    plt.legend(loc='best')
    plt.xlabel('Z')
    plt.ylabel('X')
    plt.title('Y = 0')
    plt.show()


def getFractal(z, x0):
    max_z = np.max(z)
    len_z = (len(z))
    z /= max_z
    x = np.arange(len(z)) / len_z
    U = np.vstack((x, z))
    U = U.T
    X = G(U, 0.05, balance=0)
    # X=elevate(X,U)
    X = process(X, .03)
    X = compress(X, x)
    z *= max_z
    x *= len_z
    X[:, 1] *= max_z
    X[:, 0] *= len_z
    X[:, 0] += x0
    return X[:, 0],X[:, 1]


def segmentFractal(z,ox,l,d,title):
    v = l // d
    x = np.zeros((0,))
    y = np.zeros((0,))
    for i in range(d):
        zz = z[i * v:(i + 1) * v]
        x0 = i * v
        xx,yy=getFractal(zz,x0)
        x=np.concatenate((x,xx))
        y=np.concatenate((y,yy))
    plt.close()
    plt.plot(ox, z, 'y-', label='Slice sample')
    plt.plot(x, y, 'b', label='Fractal interpolation')
    # plt.scatter(x, y)
    plt.legend(loc='best')
    plt.xlabel('Z')
    plt.ylabel('X')
    plt.title(title)
    # plt.show()

def drawSegmentFractal():
    '''
    绘制分段分形插值, 并且保存成图片
    :return:
    '''
    z = getZ(36)
    ox=np.arange(len(z))
    for i in [2,4,6,9]:
        segmentFractal(z,ox,36,i,f'It consists of {i} sections, each with {36//i} points.')
        plt.savefig(f'{i}-{36//i}.png')


if __name__ == '__main__':
    fractal_interpolation_demo()
    # combine_demo()
    # raw_sample()
    # drawSegmentFractal()
