import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from extract_z_axis import getZ


def cubic_interpolation(z):
    x=np.arange(len(z))
    f=interp1d(x, z, kind='cubic')
    xp = np.linspace(x.min(), x.max(), 100)
    zp=f(xp)
    plt.close()
    plt.plot(xp,zp, 'b.-', label='Cubic spline interpolation')
    plt.plot(x, z, 'r.-', label='Slice sample')
    plt.legend(loc='best')
    plt.xlabel('Z')
    plt.ylabel('X')

def cubic_interpolation_demo():
    li = [5, 10, 20]
    for i in li:
        z = getZ(i + 1)
        cubic_interpolation(z)
        plt.title(f'{i} layers')
        plt.savefig(f'cubic-{i}.png')

def quadratic_interpolation(z):
    x=np.arange(len(z))
    f=interp1d(x, z, kind='quadratic')
    xp = np.linspace(x.min(), x.max(), 100)
    zp=f(xp)
    plt.close()
    plt.plot(xp,zp, 'b.-', label='quadratic spline interpolation')
    plt.plot(x, z, 'r.-', label='Slice sample')
    plt.legend(loc='best')
    plt.xlabel('Z')
    plt.ylabel('X')

def quadratic_interpolation_demo():
    '''
    绘制不同的样本点下三阶样条插值
    :return:
    '''
    li = [5, 10, 15,20]
    for i in li:
        z = getZ(i + 1)
        quadratic_interpolation(z)
        plt.title(f'{i} layers')
        plt.savefig(f'quadratic-{i}.png')

if __name__ == '__main__':
    quadratic_interpolation_demo()