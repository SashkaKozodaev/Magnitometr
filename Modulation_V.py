import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import qutip as q



#Time in us
#RWA applied
Sx,Sy,Sz = [q.jmat(0.5,p) for p in 'xyz']
pol_op = q.Qobj([[0.,1.],[0.,0.]])
print(Sx, Sy, Sz)
t_opt_pol = 200.
rabi_f = 0.010

c_ops_gen = lambda Tpol : [np.sqrt(2)*Sx/np.sqrt(500.),Sz/np.sqrt(1.e4),pol_op/np.sqrt(Tpol)]
c_ops = [np.sqrt(2)*Sx/np.sqrt(500.),Sz/np.sqrt(1.e4),pol_op/np.sqrt(t_opt_pol)]
Hconst_gen = lambda Wrabi, detune : Wrabi*Sx+Sz*detune
Wrabi = 2*np.pi*rabi_f
H = [Wrabi*Sx] #['Amod*sin(Wmod*t)',Sz]

ts = np.linspace(0.,1000.,1000)
s0 = q.steadystate(Wrabi*Sx,c_ops)
print(s0)
#s0 = np.array([[0.5, 0.], [0., 0.5]])


# Hstep = [[Wrabi*Sx,'cos(wmod*t)'],[Wrabi*Sy,'sin(wmod*t)']]

fmods = np.linspace(0.,rabi_f*10,100)
Szs = []
allSzs = []
#plt.figure()
for fmod in fmods:
    #break
    wmod=2*np.pi*fmod
    Hstep = [Wrabi*Sx+Sz*wmod]
    #res = q.mesolve(rho0=s0,H=Hstep,tlist=ts,c_ops=c_ops,args={'wmod':wmod},options=q.Options(method='bdf',rhs_reuse=True))
    res = q.mesolve(rho0=s0,H=Hstep,tlist=ts,c_ops=c_ops,options=q.Options(method='bdf',rhs_reuse=False))
    sz_value = q.expect(Sz,res.states)
    allSzs.append(sz_value)
    Szs.append(sz_value[-1])
    plt.plot(ts,sz_value)
#plt.plot(fmods,Szs)
plt.show()

def get_odmr(freqs,Wrabi,T_pol=100.):
    c_ops = c_ops_gen(T_pol)
    return [ q.expect(Sz,q.steadystate(Hconst_gen(Wrabi, 2*np.pi*f),c_op_list=c_ops))
             for f in freqs
           ]
freqs = np.linspace(-5.e-1,5.e-1,500)
for Wrabi in np.logspace(-3,0,5):
    plt.plot(freqs,get_odmr(freqs,Wrabi,10.),label='Fr = {0:.3f} MHz'.format(Wrabi/2./np.pi))
plt.ylabel('Sz')
plt.xlabel('Detuning, MHz')
#plt.xscale('log')
plt.legend()
plt.suptitle('ODMR with Tpol = 100us')
plt.show()
