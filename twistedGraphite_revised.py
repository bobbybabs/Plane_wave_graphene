import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt
from ase.dft.kpoints import bandpath
from tqdm import tqdm
import scipy as scp
from scipy.special import assoc_laguerre,gammaln,eval_genlaguerre
import math

pi = np.pi
m_s_angstrom_to_mev = 0.006582  # m/s/angstrom*hbar to mev
hbar_e_angstrom2_to_tesla = 65821.2  # hbar/e/angstrom^2 in Teslas

a0 = 2.46
vf = 6.582  # eV Ang
waa = 0.09  # eV
wab = 0.117

w1 = 1000 * waa
w2 = 1000 * wab

d=1.42
kD=4*np.pi/(3*np.sqrt(3)*d)
vel  = 0.92e6

s0 = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

T = np.zeros((3, 2, 2), dtype=complex)

T[0] = waa * s0 + wab * sx
T[1] = waa * s0 + wab * (np.cos(2 * np.pi / 3) * sx + np.sin(2 * np.pi / 3) * sy)
T[2] = waa * s0 + wab * (np.cos(4 * np.pi / 3) * sx + np.sin(4 * np.pi / 3) * sy)

points = {
    "K": [-1 / 3, -1 / 3, 0],
    "G": [0, 0, 0],
    "A": [-1 / 3, 2 / 3, 0],
    "B": [1 / 3, 1 / 3, 0],
}

def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class TwistGraphite_Bzero(object):
    def __init__(self, theta=1.1) -> None:
        self.am = a0 / (theta / 180 * pi)
        self.thetadeg = theta
        self.icell = (
            4
            * pi
            / (3 * self.am)
            * np.array(([[sqrt(3) / 2, 1 / 2], [sqrt(3) / 2, -1 / 2]]))
        )
        self.cell = 2 * np.pi * np.linalg.inv(self.icell).T

    def init_mesh(self, ecut=1, N=100, epsilon=1e-6):

        g1 = np.array([0, 4 * pi / (3 * self.am)])
        g2 = rot(+2 * np.pi / 3).dot(g1)
        g3 = rot(+4 * np.pi / 3).dot(g1)

        indx, G = [], []

        for i in range(-N, N + 1):
            for j in range(-N, N + 1):
                G.append(i * self.icell[0] + j * self.icell[1])
                indx.append([i, j])

        G = np.array(G)
        indx = np.array(indx)

        ekin = vf * np.linalg.norm(G, axis=-1)

        indx = indx[ekin < ecut]
        G = G[ekin < ecut]

        dG = G[:, None] - G[None, :]
        self.g_indx = []
        for g in [g1, g2, g3]:
            self.g_indx.append(
                np.where(np.linalg.norm(dG + g[None, None, :], axis=-1) < epsilon)
            )

        self.indx = indx.T
        self.G = G
        self.N = len(G)
        self.g = [g1, g2, g3]

        self.M = self.indx.max() - self.indx.min() + 1

        self.r = np.array(np.meshgrid(np.fft.fftfreq(self.M), np.fft.fftfreq(self.M)))
        self.r = self.r.transpose((1, 2, 0)).dot(self.cell)
        self.r = self.r.transpose((2, 0, 1))


    def plot_mesh(self):

        plt.figure()
        plt.scatter(self.G[:, 0], self.G[:, 1])
        for c, g in zip(["black", "red", "blue"], self.g):
            plt.quiver(0, 0, g[0], g[1], color=c, scale=1, scale_units="xy")
        plt.axis("equal")
        plt.show()

    def hamiltonian_kinetic(self, k=[0, 0]):
        kG = k + self.G
        H = np.zeros((self.N, self.N, 2, 2), dtype=complex)

        for i in range(self.N):
            H[i, i] += vf * kG[i, 0] * sx
            H[i, i] += vf * kG[i, 1] * sy

        return H
    
    def potential_deltas(self, kz=0):
        H = np.zeros((self.N, self.N, 2, 2), dtype=complex)

        for j, indx in enumerate(self.g_indx):
            for a, b in zip(*indx):
                H[a, b] = T[j] * np.exp(2j * pi * kz)
                H[b, a] = H[a, b].T.conj()

        return H
    
    def hamiltonian(self, Hkin = None, Hdelta = None, k=[0, 0], kz=0):

        # --------------Create Hkin matrix----------------
        if(Hkin is None):
            Hkin = self.hamiltonian_kinetic(k)

        # --------------Calculate the potential Delta--------------
        if(Hdelta is None):
            Hdelta = self.potential_deltas(kz)

        H = Hkin + Hdelta

        H = H.transpose((0, 2, 1, 3)).reshape((2 * self.N, 2 * self.N))

        return H

    def plot_bandstructure(self, path="KGBAK", npoints=120, kz=0, ax=None):

        a = 2.46  # lattice constant in angstroms
        c = 15.0  # vacuum spacing in the z direction for a monolayer
        cell = [[a, 0, 0], [-a / 2, np.sqrt(3) / 2 * a, 0], [0, 0, c]]

        band_path = bandpath(
            path=path, cell=cell, special_points=points, npoints=npoints
        )

        x, x_ticks, x_labels = band_path.get_linear_kpoint_axis()

        qpts = band_path.kpts[:, :2]
        kpts = qpts.dot(self.icell)

        e = []
        Hdelta = self.potential_deltas(kz)

        for kpt in tqdm(kpts):
            e.append(np.linalg.eigvalsh(self.hamiltonian(k=kpt, Hdelta = Hdelta, kz=kz)))
        e = np.array(e)

        if ax is None:
            plt.figure(figsize=(5, 4))
            ax = plt.gca()
        ax.plot(x, e)
        ax.set_ylim([-0.1, 0.1])
        ax.axhline(0, linestyle="--")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        for x_ in x_ticks[1:-1]:
            ax.axvline(x_, linestyle="--")
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylabel("Energy [eV]")
        plt.show()

    def get_energ_inplane(self, path="KGBAK", npoints=120, kz=0):
        a = 2.46  # lattice constant in angstroms
        c = 15.0  # vacuum spacing in the z direction for a monolayer
        cell = [[a, 0, 0], [-a / 2, np.sqrt(3) / 2 * a, 0], [0, 0, c]]

        band_path = bandpath(
            path=path, cell=cell, special_points=points, npoints=npoints
        )

        x, x_ticks, x_labels = band_path.get_linear_kpoint_axis()

        qpts = band_path.kpts[:, :2]
        kpts = qpts.dot(self.icell)

        e = []
        Hdelta = self.potential_deltas(kz)

        for kpt in tqdm(kpts):
            e.append(np.linalg.eigvalsh(self.hamiltonian(k=kpt, Hdelta = Hdelta, kz=kz)))
        e = np.array(e)
        
        return e,x,x_ticks,x_labels

    def fft(self, A):
        B = np.fft.fft2(A, axes=(0, 1))
        B = B[self.indx[0], self.indx[1]]
        return B.flatten()

    def ifft(self, A):
        A = A.reshape((self.N, 2))
        B = np.zeros((self.M, self.M, 2), dtype=complex)
        B[self.indx[0], self.indx[1]] = A
        return np.fft.ifft2(B, axes=(0, 1))
    
    def zakPhase(self, kz, k=[0, 0]):
        #self.N is half the size of the Hamiltonian so since we are in semimetal--> number of occ states
        phiOccTot = np.ones((self.N,self.N))

        # --------------Create Hkin matrix once only (kz independent)----------------
        H = self.hamiltonian_kinetic(k)

        h1 = self.hamiltonian(Hkin = H, k=k, kz=kz[0])
        _,u1=np.linalg.eigh(h1)

        for i in range(1, len(kz)):
            h2 = self.hamiltonian(Hkin = H, k=k, kz=kz[i])
            _, u2 = np.linalg.eigh(h2)
            tmp = u1[:,:self.N].conj().T @ u2[:,:self.N]
            phiOccTot = phiOccTot @ tmp
            h1 = h2.copy()
            u1 = u2.copy()

        zakphases, _ = np.linalg.eig(phiOccTot)
        zakphase = abs(np.angle(np.prod(zakphases)))
        return zakphase, zakphases
    
    def energy_kz(self,npoints,kx = 0,ky = 0,giveE = False):
        self.kz = np.linspace(-0.5,0.5,npoints)
        e = []
        for i,kz in tqdm(enumerate(self.kz)):
            h = self.hamiltonian(k = [kx,ky],kz = kz)
            ei = np.linalg.eigvalsh(h)
            e.append(ei)
        
        e = np.array(e)
        self.ekz = e.copy()

        if giveE:
          return e,self.kz

    def occupation(self,mu):
        #before this caclulate energy_kz
        kb = 8.617333262e-5#eV/K
        T = 4 #K
        beta = 1/kb/T
        n = np.sum(scp.special.expit(-beta* (self.ekz - mu) - 0.5))
        return n
    
    def rhonk(self,mu):
        #kb = 8.617333262e-5#eV/K
        kb = 1# we use this riht now else rho is 0
        T = 4 #K
        beta = 1/kb/T

        rhonk= beta * np.power(4 * np.cosh(beta * (self.ekz - mu)/2),-2) 
        return rhonk     

    
    

    
class TwistGraphite_LL(object):
    """
    ------------------------Description of parameters---------------------------------
    :param vel: Dirac velocity in m/s.
    :param d: Distance between graphene atoms, in Angstrom.
    :param theta: Moire angle, in radians.
    :param sublat_pot: Optional: add potential between the A and B lattice cites.
    :param layer_pot: Optional: add potential between the Layers.
    :param kx: Moire Brillouin zone kx.
    :param ky: Moire Brillouin zone ky.
    :param kz: vertical kz path in units of 2pi
    :param p,q: Flux per unit cell=phi_0 * q/p.
    :param max_l: maximal energy level to include in the Hamiltonian.
    ----------------------------------------------------------------------------------
    """

    def __init__(self, theta=1.1) -> None:
       self.theta = np.deg2rad(theta)
       self.ktheta = kD * 2 * np.sin(self.theta/2)
       self.thetadeg = np.copy(theta)
       
    def fnm(self,zx, zy, n, m):
        z2 = zx ** 2 + zy ** 2
        if n >= m:
            return np.exp((gammaln(m + 1) - gammaln(n + 1)) / 2 +
                        np.log(-zx + 1j * zy) * (n - m) -
                        z2 / 2) * eval_genlaguerre(m, n - m, z2)
        else:
            return np.exp((gammaln(n + 1) - gammaln(m + 1)) / 2 +
                        np.log(zx + 1j * zy) * (m - n) -
                        z2 / 2) * eval_genlaguerre(n, m - n, z2)

    def hamiltonian(self,p: int,q: int,kz: float,max_l = 10,kx = 0,ky =0 ,sublat_pot = 0, layer_pot = 0):
        
        h = np.zeros((q * (max_l + 1) * 2, q * (max_l + 1) * 2), dtype=np.complex128)
        #replaced p with q

        #new bfield
        b = np.sqrt(3) / 8 / pi * self.ktheta ** 2 * q / abs(p)#need to convert it to tesla with habr e converter above, here we have hbar/e = 1

        mag_l = b ** (-1 / 2)
    
        wc = np.sqrt(2) * vel / mag_l * m_s_angstrom_to_mev

        #We now need to include the kz dependance
        phi = 2 * pi / 3
        exphi = np.exp(1j * phi)


        #-------------------- These are the T values as in the BM model just slightly changed -------------------
        # T1 = np.array([[w1, w2], [w2, w1]], dtype=np.complex128)* np.exp(2j * pi * kz)
        # T2 = np.array([[w1 / exphi, w2], [w2 * exphi, w1 / exphi]])* np.exp(2j * pi * kz)
        # T3 = np.array([[w1 * exphi, w2], [w2 / exphi, w1 * exphi]])* np.exp(2j * pi * kz)

        # ---------------------- let me try something with our oroginal T---------------AS in the  analysis 

        T1 = np.array([[w1, w2], [w2, w1]], dtype=np.complex128)* np.exp(2j * pi * kz)
        T2 = np.array([[w1 , w2 / exphi], [w2 * exphi, w1]])* np.exp(2j * pi * kz)
        T3 = np.array([[w1 , w2 / (exphi**2)], [w2 * (exphi**2), w1]])* np.exp(2j * pi * kz)

        q1 = self.ktheta * np.array([0, -1])
        q2 = self.ktheta * np.array([np.sqrt(3), 1]) / 2
        q3 = self.ktheta * np.array([-np.sqrt(3), 1]) / 2

        z1 = q1 * mag_l / np.sqrt(2)
        z2 = q2 * mag_l / np.sqrt(2)
        z3 = q3 * mag_l / np.sqrt(2)

        
        delta =  4 * pi * p / q / self.ktheta  # 2 * pi * q / p / self.ktheta
        y0 = kx * mag_l ** 2
        #--> new versions changed p to q in the reshape
        ind = np.reshape(np.arange(q * (max_l + 1) * 2), (q, (max_l + 1), 2))
        
        #----------- WE take out the layer for loops since there are no explicit dependance---------------
        # diagonal term
        for level in range(max_l):
            for y in range(q):
                # landau level energies
                h[ind[y, level, 1], ind[y, level + 1, 0]] = -wc * np.exp(
                    1j * (-1) * self.theta / 2) * np.sqrt(level + 1.)

                # sublattice and layer potential
                # we dont have layer potential in our case --> zero here
                for sublat in range(2):
                    ii = ind[y, level, sublat]
                    h[ii, ii] = (-1) ** sublat * sublat_pot -1 * layer_pot

        for y in range(q):
            h[ind[y, max_l, 1], ind[y, max_l, 1]] = 1e4
        
        #I changed y in range 1,..,p to y in range 1,..,q

        # off-diagonal term
        # m,n are level indices
        # alpha,beta are sublattice indices
        # j is the guiding center coordinate
        for m in range(max_l + 1):
            for n in range(max_l + 1):
                for beta in range(2):
                    for alpha in range(2):
                        for j in range(q):
                            # top scattering
                            h[ind[j, m, beta], ind[j, n, alpha]] += \
                                T1[alpha, beta] * self.fnm(z1[0], z1[1], n, m) * np.exp(
                                    -1j * self.ktheta * y0 - 4j * pi * p * j / q)
                            jp = np.mod(j + 1, q)
                            jm = np.mod(j - 1, q)
                            # right scattering
                            h[ind[j, m, beta], ind[jp, n, alpha]] += \
                                T2[alpha, beta] * self.fnm(z2[0], z2[1], n, m) * np.exp(
                                    1j * ky * delta + 1j / 2 * self.ktheta * y0 + 1j * pi * p / q * (2 * j - 1))
                            # left scattering
                            h[ind[j, m, beta], ind[jm, n, alpha]] += \
                                T3[alpha, beta] * self.fnm(z3[0], z3[1], n, m) * np.exp(
                                    -1j * ky * delta + 1j / 2 * self.ktheta * y0 + 1j * pi * p / q * (2 * j + 1))


        #----------- I changed q/p to p/q to make sense with the rest + I changed the for loop from 1...p to 1...q to again make sens with the derivations
        h = h + np.transpose(np.conj(h))
        argnan = np.argwhere(np.isnan(h))
        if argnan.size > 0:
            raise ValueError('Nan value caught in h')
        return ind, h
    
    def plot_kz(self,kx,ky,kz,p,q,max_l = 10):
        e=[]
        for k in tqdm(kz):
            _,h= self.hamiltonian(p = p,q = q,kz = k,max_l = max_l,kx = kx,ky = ky,sublat_pot = 0, layer_pot = 0)
            e.append(np.linalg.eigvalsh(h))
        
        e = np.array(e)

        plt.rcParams['text.usetex'] = True
        plt.figure()
        plt.plot(kz,e)
        plt.ylim([-300,300])
        plt.xlim([np.min(kz),np.max(kz)])
        plt.ylabel('Energy [meV]')
        plt.xlabel('$k_z$')
        plt.title(rf'Bands along $k_z$ path with $\theta = {self.thetadeg}^{{\circ}}$ and {max_l} LL with $p/q = ${p}/{q}')
        plt.show()

    def zakPhase(self,p,q,kz,max_l = 10,kx=0, ky = 0):

        _,h1 = self.hamiltonian(p,q,0,max_l = max_l, kx =kx,ky = ky)
        _,u1=np.linalg.eigh(h1)
        #for now we jsut take half the number of states we will to change depending on Fermi level
        # it might alos change at each kz value so to be continued...
        N = int(len(h1)/2)

        phiOccTot = np.ones((N,N))
        for i in tqdm(range(1,len(kz))):
            _,h2 = self.hamiltonian(p,q, kz=kz[i],max_l = max_l, kx =kx,ky = ky)
            _,u2 = np.linalg.eigh(h2)
            tmp = u1[:,:N].conj().T @ u2[:,:N]
            h1 = h2.copy()
            u1 = u2.copy()
            phiOccTot = phiOccTot @ tmp
        
        zakphases, _ = np.linalg.eig(phiOccTot)
        zakphase = abs(np.angle(np.prod(zakphases)))
        return zakphase,zakphases
    
    def energy_kz(self,npoints,p,q,max_l,giveE = False):
        self.kz = np.linspace(-0.5,0.5,npoints)
        e = []
        for i,kz in tqdm(enumerate(self.kz)):
            _,h = self.hamiltonian(p = p,q = q,kz = kz,max_l =max_l)
            ei = np.linalg.eigvalsh(h)
            e.append(ei)
        
        e = np.array(e)
        self.ekz = e.copy()

        if giveE:
          return e,self.kz



            