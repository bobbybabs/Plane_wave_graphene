import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt
from ase.dft.kpoints import bandpath
from tqdm import tqdm

a0 = 2.46
vf = 6.582  # eV Ang
waa = 0.09  # eV
wab = 0.117

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


class TwistGraphite(object):
    def __init__(self, theta=1.1) -> None:
        self.am = a0 / (theta / 180 * pi)
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

    def hamiltonian(self, k=[0, 0], kz=0):

        kG = k + self.G
        H = np.zeros((self.N, self.N, 2, 2), dtype=complex)

        # --------------Add Hkin matrix----------------
        for i in range(self.N):
            H[i, i] += vf * kG[i, 0] * sx
            H[i, i] += vf * kG[i, 1] * sy

        # #    #--------------Cacluating the potential Delta--------------
        for j, indx in enumerate(self.g_indx):
            for a, b in zip(*indx):
                H[a, b] = T[j] * np.exp(2j * pi * kz)
                H[b, a] = H[a, b].T.conj()

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
        for kpt in tqdm(kpts):
            e.append(np.linalg.eigvalsh(self.hamiltonian(kpt, kz=kz)))
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

    def fft(self, A):
        B = np.fft.fft2(A, axes=(0, 1))
        B = B[self.indx[0], self.indx[1]]
        return B.flatten()

    def ifft(self, A):
        A = A.reshape((self.N, 2))
        B = np.zeros((self.M, self.M, 2), dtype=complex)
        B[self.indx[0], self.indx[1]] = A
        return np.fft.ifft2(B, axes=(0, 1))
