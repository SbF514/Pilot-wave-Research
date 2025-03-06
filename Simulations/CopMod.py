import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
N = 256  # Number of points in each spatial dimension
L = 20.0  # Physical length of each spatial dimension
dx = L / N  # Spatial step size
x = np.linspace(-L / 2, L / 2, N)  # x-axis grid
y = np.linspace(-L / 2, L / 2, N)  # y-axis grid
X, Y = np.meshgrid(x, y)  # Create 2D spatial grid
dt = 0.01  # Time step
total_time = 2.0  # Total time of the simulation
steps = int(total_time / dt)  # Number of time steps

# Initial Gaussian wave packet parameters
x0, y0 = 0.0, 0.0  # Initial position of the wave packet (centered)
k0x, k0y = 3.0, 3.0  # Initial momentum components in x and y directions
sigma = 1.0  # Width of the Gaussian packet

# Create initial 2D Gaussian wave packet
psi = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2)) * np.exp(1j * (k0x * X + k0y * Y))
psi /= np.sqrt(np.sum(np.abs(psi) ** 2))  # Normalize wave function

# Precompute the Fourier transform of the kinetic energy operator in 2D
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Frequency domain for x
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Frequency domain for y
KX, KY = np.meshgrid(kx, ky)  # Create 2D frequency grid
kinetic_operator = np.exp(-1j * (KX ** 2 + KY ** 2) * dt / 2)  # Kinetic energy operator in Fourier space


# Function to propagate wave packet for one time step, with boundary enforcement
def propagate_with_boundaries(psi):
    # Apply Fourier transform to go to momentum space
    psi_k = np.fft.fft2(psi)
    # Apply kinetic energy operator in momentum space
    psi_k *= kinetic_operator
    # Inverse Fourier transform to go back to position space
    psi = np.fft.ifft2(psi_k)
    # Apply boundary conditions to make psi zero at the edges (infinite potential well)
    psi[0, :] = psi[-1, :] = 0  # Set psi to zero at top and bottom edges
    psi[:, 0] = psi[:, -1] = 0  # Set psi to zero at left and right edges
    return psi


# Visualization setup
fig, ax = plt.subplots()
prob_density = ax.imshow(np.abs(psi) ** 2, extent=(-L / 2, L / 2, -L / 2, L / 2), origin='lower', cmap='viridis')
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Time Evolution of Free Electron in 2D Infinite Potential Well")
colorbar = fig.colorbar(prob_density, ax=ax, label="Probability Density |ψ(x, y)|²")


# Animation function
def animate(i):
    global psi
    psi = propagate_with_boundaries(psi)  # Update wave function with boundary condition

    # Update probability density data and color limits dynamically
    density = np.abs(psi) ** 2
    prob_density.set_data(density)
    prob_density.set_clim(vmin=density.min(), vmax=density.max())  # Adjust color limits

    return [prob_density]


# Run the animation
ani = FuncAnimation(fig, animate, frames=steps, interval=20, blit=True)
plt.show()
