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
k0x, k0y = 1.0, 1.0  # Initial momentum components in x and y directions
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


# Function to compute the phase gradient (guiding equation)
def get_velocity(psi):
    # Calculate the gradient of the phase (S = arg(psi))
    phase = np.angle(psi)
    # Compute gradient in x and y directions
    grad_x, grad_y = np.gradient(phase, dx)
    # Compute velocity field (hbar/m * gradient of the phase)
    velocity_x = grad_x
    velocity_y = grad_y
    return velocity_x, velocity_y


# Initial position of the particle (chosen randomly within the box)
particle_pos = np.array([0.1, 0.0])  # Start at (L/4, L/4)

# Visualization setup
fig, ax = plt.subplots()
ax.set_xlim(-L / 2, L / 2)
ax.set_ylim(-L / 2, L / 2)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Bohmian Mechanics: Particle and Wave Function Evolution")

# Particle plot
particle_dot, = ax.plot([], [], 'ro', markersize=5)

# Wave function plot (for visualization of the wave evolution)
prob_density = ax.imshow(np.abs(psi) ** 2, extent=(-L / 2, L / 2, -L / 2, L / 2), origin='lower', cmap='viridis')
fig.colorbar(prob_density, ax=ax, label="Probability Density |ψ(x, y)|²")


# Animation function
def animate(i):
    global psi, particle_pos

    # Propagate the wave function and apply boundary conditions
    psi = propagate_with_boundaries(psi)

    # Calculate the velocity from the guiding equation
    velocity_x, velocity_y = get_velocity(psi)

    # Update the particle position using the guiding equation
    # Interpolate the velocity at the particle's current position
    px, py = particle_pos
    i_x, i_y = int((px + L / 2) / dx), int((py + L / 2) / dx)

    # Get velocity components at the particle's current position
    vel_x = velocity_x[i_x, i_y]
    vel_y = velocity_y[i_x, i_y]

    # Update the particle's position using the velocity field
    particle_pos += np.array([vel_x, vel_y]) * dt

    # Correct the particle position update for set_data
    particle_dot.set_data([particle_pos[0]], [particle_pos[1]])  # Use lists for x and y coordinates

    # Update the probability density plot
    prob_density.set_data(np.abs(psi) ** 2)
    prob_density.set_clim(vmin=(np.abs(psi) ** 2).min(), vmax=(np.abs(psi) ** 2).max())  # Adjust color limits


    return [particle_dot, prob_density]


# Run the animation
ani = FuncAnimation(fig, animate, frames=steps, interval=20, blit=True)
plt.show()
