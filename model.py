import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt
from ase.dft.kpoints import bandpath
from tqdm import tqdm

a0 = 2.46
vf = 6.582  # eV Ang
waa = 0.09  # eV
wab = 0.117

t = 2 * vf / np.sqrt(3) / a0
print(t)

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
    def __init__(self, theta=1.1, q=1) -> None:
        self.am = a0 / (theta / 180 * pi)
        self.icell_init = (
            4
            * pi
            / (3 * self.am)
            * np.array(([[sqrt(3) / 2, 1 / 2], [sqrt(3) / 2, -1 / 2]]))
        )
        self.cell_init = 2 * np.pi * np.linalg.inv(self.icell_init).T

        self.cell = np.array([q * self.cell_init[0], self.cell_init[1]])
        self.icell = 2 * np.pi * np.linalg.inv(self.cell).T

        self.q = q

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

        self.gm = g2 / self.q
        self.gm_indx = []
        for i in range(100):
            indx = np.where(
                np.linalg.norm(dG + (i + 1) * self.gm[None, None, :], axis=-1) < epsilon
            )
            if len(indx[0]) == 0:
                break
            else:
                self.gm_indx.append(indx)

        self.M = self.indx.max() - self.indx.min() + 1
        print(self.M)
        self.r = np.array(
            np.meshgrid(np.fft.fftfreq(self.M), np.fft.fftfreq(self.q * self.M))
        )
        self.r = self.r.transpose((1, 2, 0))
        self.r = self.r.transpose((2, 0, 1))

    def plot_mesh(self, c=None):

        plt.figure()
        plt.scatter(
            self.G[:, 0],
            self.G[:, 1],
            c=c,
            cmap="coolwarm",
        )
        if c is not None:
            plt.colorbar()
        else:
            for c, g in zip(["black", "red", "blue"], self.g):
                plt.quiver(0, 0, g[0], g[1], color=c, scale=1, scale_units="xy")

            plt.quiver(
                0,
                0,
                self.gm[0][0],
                self.gm[0][1],
                color="purple",
                scale=1,
                scale_units="xy",
            )
        plt.axis("equal")
        plt.show()

    def hamiltonian(self, k=[0, 0], kz=0, p=1):

        kG = k + self.G
        H = np.zeros((self.N, self.N, 2, 2), dtype=complex)

        # --------------Add Hkin matrix----------------
        for i in range(self.N):
            H[i, i] += vf * (kG[i, 0] * sx + kG[i, 1] * sy)

        # #    #--------------Cacluating the potential Delta--------------
        # for j, indx in enumerate(self.g_indx):
        #     for a, b in zip(*indx):
        #         h = T[j] * np.exp(2j * pi * kz)
        #         H[a, b] += h
        #         H[b, a] += h.T.conj()
        for j, indx in enumerate(self.gm_indx):
            for a, b in zip(*indx):
                h = p / (j + 1) * vf * (self.gm[0] * sx + self.gm[1] * sy).T.conj()
                H[a, b] += h
                H[b, a] += h.T.conj()

        H = H.transpose((0, 2, 1, 3)).reshape((2 * self.N, 2 * self.N))
        return H

    def plot_bandstructure(self, path="KGBAK", npoints=120, kz=0, ax=None, p=0):

        c = 15.0  # vacuum spacing in the z direction for a monolayer
        cell = [
            [self.cell[0, 0], self.cell[0, 1], 0],
            [self.cell[1, 0], self.cell[1, 1], 0],
            [0, 0, c],
        ]

        band_path = bandpath(
            path=path, cell=cell, special_points=points, npoints=npoints
        )

        x, x_ticks, x_labels = band_path.get_linear_kpoint_axis()

        qpts = band_path.kpts[:, :2]
        kpts = qpts.dot(self.icell)

        e = []
        for kpt in tqdm(kpts):
            e.append(np.linalg.eigvalsh(self.hamiltonian(kpt, kz=kz, p=p)))
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
        return B.flatten() / (self.M) ** 2

    def ifft(self, A):
        A = A.reshape((self.N, 2))
        B = np.zeros((self.M, self.M, 2), dtype=complex)
        B[self.indx[0], self.indx[1]] = A
        return np.fft.ifft2(B, axes=(0, 1))
