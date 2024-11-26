import numpy as np
from matplotlib.widgets import Button, Slider
from scipy import sparse as sp
from scipy import special
from scipy.sparse import linalg as lsp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

"""Système d'unité"""
hbar = 1  # Constante de Planck réduite
mu = 1  # Masse de l'électron
metre = 6.67e-9  # en unités naturelles (=Longueur d'un mètre dans notre système d'unité)
g = 6.67e-24  # en unités naturelles (= valeur de g dans notre système d'unité)


def Kinetic1D(N, Dx, mu=1):
    rx = (-1 / (2 * mu)) * (1 / Dx ** 2)  # Constante utile pour la résolution

    main_diag = np.full(N, -2 * rx)  # Main diagonal (central points)
    off_diag = np.full(N - 1, rx)  # Off-diagonal (neighboring points)

    # Create a sparse 1D derivative
    return sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N))


def Hamiltonian2D(Dx, Dy, V, mu=1):
    Nx, Ny = V.shape

    # Create a sparse 1D Laplacian
    Tx = Kinetic1D(Nx, Dx, mu)
    Ty = Kinetic1D(Ny, Dy, mu)

    # Construct the 2D Laplacian using the Kronecker product of the 1D Laplacian with the identity matrix
    return sp.kron(Tx, np.eye(Ny)) + sp.kron(np.eye(Nx), Ty) + sp.diags(V.flatten(), offsets=0)


def Get_Propagator_Chebyshev_Expansion(Psi, Hz, a):
    """
    This fucntion advances Psi in time units of delta t by
    the Chebyshev expansion of the propagator.
    """
    g0, g1 = Psi, Hz @ Psi
    Psi = (float(a[0]) * g0) + (float(a[1]) * g1)

    for jt in range(2, len(a)):
        g0, g1 = g1, g0 + 2 * (Hz @ g1)  # reccurence of Chebyshev polynomials
        Psi += float(a[jt]) * g1  # accumulation step

    return Psi


def Get_Chebyshev_Coefficients(Nc, Nc_max, amin, r):
    a = np.zeros(Nc_max, float)
    C = 2
    a[0] = special.jv(0, r)
    k = np.arange(1, Nc)
    a[k] = C * special.jv(k, r)

    while a[Nc - 1] > amin:
        a[Nc] = C * special.jv(Nc, r)
        Nc += 1

        if Nc > Nc_max:
            print('The argument of the Chebyshev exp. is too large')
            break
    return a[:Nc], Nc


# Fonction principale, simulation dans boîte carrée
def simulation(psi0, V, Dx, Dy, Nt=200, mu=1, skip=5):
    """
    psi0: fonction d'onde initiale, taille Lx*Ly, (Nx-2)*(Ny-2) points
    V : potentiel, taille (NxN) points
    Dx: Pas spatial
    Dy: Pas spatial
    Nt: nombre d'itérations
    mu: Masse de l'électron
    """
    Nx, Ny = V.shape

    # Résolution du sysème Ax=My à chaque itération
    density = []  # Densité de proba à chaque itération
    density.append(np.abs(psi0) ** 2)  # Stockage de density0 dans la liste

    # Résolution via Cranck Nichoslon
    Dt = (Dx / 2) ** 2 + (Dy / 2) ** 2

    Emax = ((np.pi / Dx)**2 + (np.pi / Dy)**2) / (2 * mu) + np.max(V)
    Emin = np.min(V)
    Eshift = (Emax + Emin) / 2
    H = Hamiltonian2D(Dx, Dy, V - Eshift, mu)
    # H = Hamiltonian2D(Dx, Dy, V, mu)

    psi_vect = psi0.flatten()
    Energy0 = np.real(np.conj(psi_vect) @ H @ psi_vect) / np.real(np.conj(psi_vect) @ psi_vect)
    print(f"Initial WP energy: {Energy0:.5f} (potential min/max {np.min(V):.2f}/{np.max(V):.2f})")

    # E = sp.eye(H.shape[0], dtype=H.dtype)
    # Asp, Bsp = E + 1j * Dt / 2 * H, E - 1j * Dt / 2 * H


    #   Bessel coefficients for Chebyshev expansion
    Nc_max = 10000  # The maximum number of iterations
    amin = 5e-7  # The accuracy of the Chebyshev decomposition
    r = (skip*Dt) * (Emax - Emin) / 2
    z = -1j * (skip*Dt) / r
    Nc = int(max(2, r))  # nombre de paremtre Chebyshev de devloppement
    a, Nc = Get_Chebyshev_Coefficients(Nc, Nc_max, amin, r)
    Hz = H * z
    for i in range(1, Nt + 1):
        # psi_vect = lsp.spsolve(Asp, Bsp @ psi_vect)

        if i % skip == 0:
            psi_vect = Get_Propagator_Chebyshev_Expansion(psi_vect, Hz, a)
            print(f"{i} / {Nt}")  # Pour voir la progression
            # Ajout aux listes
            psi2 = np.abs(psi_vect.reshape((Nx, Ny))) ** 2
            density.append(psi2)

    print("\nSimulation complete")
    return density


# Paramètres de la simulation
Lx = 6
Ly = 6
Dx = 0.03
Dy = 0.03
Nx = int(Lx / Dx) + 1
Ny = int(Ly / Dy) + 1
# Nt = 1500
dt = 0.01
Nt = 50
skip = int(dt / ((Dx / 2) ** 2 + (Dy / 2) ** 2))  # save data after "skip" time points (skip = 1 - save all time points)
t_max = Nt * dt
Nt0 = Nt * skip

"""Fonction d'onde initiale"""
sigmax = 0.4  # Largeur du paquet d'onde initial
sigmay = 0.4  # Largeur du paquet d'onde initial
x0 = Lx / 2
y0 = Ly / 2

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

"""Potentiel"""
g = 100  # Valeur de g pour la simulation, arbitraire
V = g * Y

print(f"V shape = {V.shape}")
Energy0 = 1 / (2 * mu) * (1 / sigmax ** 2 + 1 / sigmay ** 2) / 2 + g * y0
print(f"Initial WP energy (analytical): {Energy0:.5f} (potential min/max {np.min(V):.2f}/{np.max(V):.2f})")

"""Simulation"""
psi0 = np.exp(-1 / 2 * (((X - x0) / sigmax) ** 2 + ((Y - y0) / sigmay) ** 2))  # Fonction d'onde initiale
psi0 /= np.real(np.sqrt(np.conj(psi0.flatten()) @ psi0.flatten()))
density = simulation(psi0, V, Dx, Dx, Nt0, mu, skip=skip)

"""Visualisation de la fonction d'onde initiale et du potentiel"""
fig = plt.figure(figsize=(14, 6))
ax = (fig.add_subplot(121), fig.add_subplot(122, projection='3d'))  # Important: projection='3d'

# Densité de probabilité
cax1 = ax[0].imshow(density[0], extent=(0, Lx, 0, Ly), origin='lower', cmap='magma')
ax[0].set_title('Densité de probabilité')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
fig.colorbar(cax1, ax=ax[0])

# Potentiel
# cax2 = ax[1].imshow(V, extent=(0, Lx, 0, Ly), origin='lower', cmap='binary')
Norm = 0.1 * (np.max(V) - np.min(V)) / np.max(density[0])
cax2 = ax[1].plot_surface(X, Y, V, cmap='viridis', edgecolor='none')
Z = Energy0 + Norm * density[0]
cax3 = ax[1].plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
# cax3 = ax[1].plot_surface(X, Y, Z,
#                           facecolors=None,  # Transparent surface
#                           edgecolor=plt.cm.viridis(Z.flatten() / np.max(Z)),
#                           rstride=1, cstride=1)
ax[1].set_title('Potentiel')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
fig.colorbar(cax2, ax=ax[1])

plt.tight_layout()
# plt.show()


"""Animation"""
# Parameters for animation
pause = False
# Flag to prevent recursion during slider update
animation_is_playing = True

# fig = plt.figure(figsize=(9, 9))
# ax = fig.add_subplot(111, xlim=(0, Lx), ylim=(0, Ly))
# img = ax.imshow(density[0], extent=[0, Lx, 0, Ly], origin='lower', cmap="magma", vmin=0, zorder=1)  # Here the modulus of the 2D wave function shall be represented.
img = cax1
img2 = cax3


# Fonction pour l'animation
def animate(i):
    slider.set_val(i)  # Update slider position to match the current frame


# Function to toggle play/pause
def toggle_play(event):
    global pause, anim, animation_is_playing
    if pause:
        animation_is_playing = False
        anim.pause()  # Pause the animation
        play_button.label.set_text("►")
    else:
        animation_is_playing = True
        anim.resume()  # Pause the animation
        play_button.label.set_text("❚❚")
    pause = not pause


# Function to update frame with slider
def slider_update(val):
    global pause, anim, img2, animation_is_playing
    i = int(val)
    Z = density[i]
    img.set_data(Z)
    img.set_zorder(1)
    img.set_clim(vmin=0, vmax=np.max(Z))

    img2.remove()  # Remove the old surface
    Z = Energy0 + Norm * Z
    img2 = ax[1].plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # img2 = ax[1].plot_surface(X, Y, Energy0 + Norm * Z,
    #                           facecolors=None,  # Transparent surface
    #                           edgecolor=plt.cm.viridis(Z.flatten() / np.max(Z)),
    #                           rstride=1, cstride=1)

    fig.canvas.draw_idle()


pos = ax[0].get_position()
ax[0].set_position((pos.x0, pos.y0 + 0.1, pos.width, pos.height - 0.1))
ax_play = plt.axes([pos.x0, 0.05, 0.1 * (pos.x1 - pos.x0), 0.05])  # Play/Pause button position
ax_slider = plt.axes([pos.x0 + 0.15 * (pos.x1 - pos.x0), 0.05, 0.8 * (pos.x1 - pos.x0), 0.05])  # Slider position

play_button = Button(ax_play, "❚❚")  # Create Play/Pause button
play_button.on_clicked(toggle_play)

slider = Slider(ax_slider, "", 0, len(density) - 1, valinit=0, valstep=1)
slider.on_changed(slider_update)

anim = FuncAnimation(fig, animate, interval=100, frames=range(len(density)), blit=False, repeat=True)

# Sauvegarder l'animation en MP4
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
anim.save("pesanteur_electron100.mp4", writer=writer)

plt.show()
