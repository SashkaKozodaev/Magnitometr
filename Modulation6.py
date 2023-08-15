from qutip import Qobj, Options, mesolve
import qutip as q
import numpy as np
import scipy
from math import pi, exp, sin, cos, sqrt
import matplotlib.pyplot as plt
from scipy import integrate
import time
from progress.bar import IncrementalBar

def Plotter(res):
    bra = Qobj([[1, 0]])
    ket = Qobj([[1], [0]])
    solve = bra * (res.states * ket)
    points_to_plot = []

    for s in solve: points_to_plot.append(s.data.data.real[0])
    return (points_to_plot)

const = 1.e6

Sx,Sy,Sz = [q.jmat(0.5, p) for p in 'xyz']
Wrabi = 2*pi*0.5*1e6 / const
#t= np.linspace(0, 1e-2, 10000) * const     #время измерений
koef_time = 10
#t= np.linspace(0, 2*1e-2, 20000*koef_time) * const     #время измерений
t= np.linspace(0, 2*1e-3, 20000*koef_time) * const     #время измерений

dt = 1e-2 / 10000 * const

rho0 = Qobj(np.array([[1., 0.], [0., 0.]]))#начальная матрица плотности
s = 0.001 #1e-2  #контраст
gamma_Ti2 = 0.45
gamma_tau = 77/3
g = 1. / 2 * 1e-3

L1 = Sz/sqrt(gamma_Ti2/2)
L2_gen = lambda s: q.destroy(2) * sqrt(s*gamma_tau)
L2 = q.destroy(2) * sqrt(s*gamma_tau)
L3 = q.create(2) * sqrt(g/2)
L4 = q.destroy(2) * sqrt(g/2)
L5 = Sz* 1j * sqrt(g)
c_ops = [L1, L2, L3, L4, L5]
c_ops_gen = lambda s: [L1, L2_gen(s), L3, L4, L5]

fm = 0.5 * 1e3 / const
Fm = 2*pi*fm
deep = 2*pi*0.5/sqrt(2) * 1e6 / const
dB = 1e-3 * sqrt(5)

H = [Sz*dB - Sx*Wrabi, [Sz*deep, 'sin(Fm*t)']]
H_gen = lambda dB ,Wrabi, deep: [Sz*dB - Sx*Wrabi, [Sz*deep, 'sin(Fm*t)']]

def Evolution(dB_arg, Wrabi_arg, deep_arg, s_arg, Fm_arg):
    H = H_gen(dB_arg, Wrabi_arg, deep_arg)
    res = q.mesolve(rho0 = rho0, H=H, tlist=t, c_ops=c_ops_gen(s_arg), args={'Fm' : Fm_arg}, options=Options(nsteps=1e4, atol=1e-5))#q.Options(method='bdf',rhs_reuse=True))
    return(res)

Fm = 2*pi*( 0.125 * 1e3) / const
dB = 0.01
deep = 2*pi
rho00 = Plotter(Evolution(dB, Wrabi, deep, s, Fm))
'''
fig, ax = plt.subplots()
ax.plot(t[2000-1 : 6000-1], rho00[2000-1 : 6000-1])
ax.set_xlabel('t, мкс')
ax.set_ylabel('ms = 0')
plt.show()
'''



#fm_arr = np.array([ 0.5*1e2, 1e2, 0.5*1e3, 1e3]) / const
#fm_arr = np.array([ 1e3, 0.5*1e4, 1e4, 0.5*1e5, 1e5, 0.5*1e6, 1e6]) / const
fm_arr = np.array([ 0.5*1e6 ]) / const
Fm_arr = 2 * pi * fm_arr

deep_arr = 2 * pi * np.linspace(0.1, 1, 5)
dB_arr = [-0.01, 0.01]
#dB_arr = np.linspace(-0.01, 0.01, 20)
#s_arr = [0.001, 0.0009, 0.0008, 0.0007, 0.006, 0.0005]
#s_arr =[0.0001, 0.0005, 0.001, 0.005, 0.01]
s_arr = [0.001]

bar = IncrementalBar('Counter', max = len(s_arr) * len(Fm_arr) * len(deep_arr) * len(dB_arr))
for s in s_arr:
    dSignal_dB_arr = []
    phase_arr = []
    for Fm in Fm_arr:
        t_start = koef_time*int(2*pi / Fm)
        t_finish = koef_time*int(2*pi * 5 / Fm)

        if t_start<1000:
            t_start += koef_time * int(10 * pi / Fm)
            t_finish += koef_time * int(10 * pi / Fm)

        norm = 1./(t_finish - t_start)
        dSignal_dB_max_search = []
        phase_max_search = []

        for deep in deep_arr:
            Signal = []
            for dB in dB_arr:
                rho00 = Plotter(Evolution(dB, Wrabi, deep, s, Fm))
                t_delta = 1
                #while abs( rho00[t_start + t_delta] - rho00[t_finish + t_delta] ) > 2*1e-5:   t_delta += 1
                #if t_delta != 0: t_delta +=2
                #while rho00[t_finish] >= rho00[t_finish + t_delta] and rho00[t_finish] - rho00[t_finish + t_delta] > 1e-4 : t_delta += 1
                if rho00[t_start] > rho00[t_start + t_delta]:
                    while rho00[t_start] > rho00[t_start + t_delta]:    t_delta += 1
                if t_delta == 1: t_delta = 0

                plt.plot(t[t_start + t_delta : t_finish + t_delta], rho00[t_start + t_delta : t_finish + t_delta])

                x = t[t_start : t_finish ]
                sig_mod = np.sin(x * Fm)
                rho = rho00[t_start + t_delta : t_finish + t_delta]
                y = sig_mod * rho

                sig = norm * integrate.cumtrapz(x, y, initial = 0)

                Signal.append(sig[-1])
                bar.next()

            #plt.plot(dB_arr, Signal)
            #plt.show()
            #print(Signal)


            dSignal_dB = abs( (Signal[-1] - Signal[0]) / (dB_arr[-1] - dB_arr[0]) )
            dSignal_dB_max_search.append(dSignal_dB)

            phase_max_search.append(t_delta)

        dSignal_dB_max = np.max(dSignal_dB_max_search)
        dSignal_dB_arr.append(dSignal_dB_max)

        phase_max = phase_max_search[np.argmax(dSignal_dB_max_search)]
        phase_arr.append(phase_max)
        #print('s = ', s, ', Fm = ', Fm, 'depp = ', deep_arr[np.argmax(dSignal_dB_max_search)]/2/pi)

        #plt.xlabel('t, мкс')
        #plt.ylabel('ms = 0')
        #plt.suptitle('Fm = ' + str(Fm/2/pi))
        plt.show()

    plt.plot(Fm_arr, dSignal_dB_arr)
    #plt.legend(s)
bar.finish()


plt.xlabel('Fm, MHz')
plt.ylabel('dSignal/dB')
plt.xscale('log')
plt.suptitle('Wrabi/2pi = ' + str(Wrabi/2/pi) + 'MHz , deep/2pi = ' + str(deep_arr[0]/2/pi) + ' - ' + str(deep_arr[-1]/2/pi) + ' MHz')

plt.show()

plt.plot(Fm_arr, phase_arr)
plt.xlabel('Fm, MHz')
plt.ylabel('Phase')
plt.xscale('log')
plt.suptitle('Wrabi/2pi = ' + str(Wrabi/2/pi) + 'MHz , deep/2pi = ' + str(deep_arr[0]/2/pi) + ' - ' + str(deep_arr[-1]/2/pi) + ' MHz')

plt.show()

'''
fig, ax = plt.subplots()
ax.plot(Fm_arr, dSignal_dB_arr)
ax.set_xlabel('Fm, MHz')
ax.set_ylabel('dSignal/dB')
ax.set_xscale('log')
plt.suptitle('s = ' + str(s) + ',Wrabi = ' + str(Wrabi) + 'MHz , deep/2pi = 0.1 - 1 MHz')


fig, ax = plt.subplots()
ax.plot(Fm_arr, dSignal_dB_arr)
ax.set_xlabel('Fm, MHz')
ax.set_ylabel('dSignal/dB')
plt.suptitle('s = ' + str(s) + ',Wrabi = ' + str(Wrabi) + 'MHz , deep/2pi = 0.1 - 1 MHz')
plt.show()
'''

