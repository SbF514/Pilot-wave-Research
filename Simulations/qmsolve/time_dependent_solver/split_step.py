import numpy as np
from .method import Method
import time
from ..util.constants import hbar, Å, femtoseconds
from ..particle_system import SingleParticle, TwoParticles
import progressbar
import random

"""
Split-operator method for the Schrödinger equation.
References:
https://www.algorithm-archive.org/contents/
split-operator_method/split-operator_method.html
https://en.wikipedia.org/wiki/Split-step_method
Original implementation:
https://github.com/marl0ny/split-operator-simulations
"""

class SplitStep(Method):
    def __init__(self, simulation):

        self.simulation = simulation
        self.H = simulation.H
        self.simulation.Vmin = np.amin(self.H.Vgrid)
        self.simulation.Vmax = np.amax(self.H.Vgrid)
        self.H.particle_system.compute_momentum_space(self.H)
        self.p2 = self.H.particle_system.p2


    def run(self, initial_wavefunction, total_time, dt, store_steps = 1):

        self.simulation.store_steps = store_steps
        dt_store = total_time/store_steps
        self.simulation.total_time = total_time

        Nt = int(np.round(total_time / dt))
        Nt_per_store_step = int(np.round(dt_store / dt))
        self.simulation.Nt_per_store_step = Nt_per_store_step

        #time/dt and dt_store/dt must be integers. Otherwise dt is rounded to match that the Nt_per_store_stepdivisions are integers
        self.simulation.dt = dt_store/Nt_per_store_step

        if isinstance(self.simulation.H.particle_system ,SingleParticle):
            Ψ = np.zeros((store_steps + 1, *([self.H.N] *self.H.ndim )), dtype = np.complex128)

        elif isinstance(self.simulation.H.particle_system,TwoParticles):
            Ψ = np.zeros((store_steps + 1, *([self.H.N] * 2)), dtype = np.complex128)

        Ψ[0] = np.array(initial_wavefunction(self.H.particle_system))

        # Initialize 5 Q trajectories with different starting positions
        #Q_list = [[random.gauss(0, 0.5)] for i in range(10)]  # Using gauss with mean=0, std=0.5 to get values mostly between -1 and 1
        #Q_list = [[i/5-1] for i in range(10)]
        m = self.H.particle_system.m

        Ur = np.exp(-0.5j*(self.simulation.dt/hbar)*np.array(self.H.Vgrid))
        Uk = np.exp(-0.5j*(self.simulation.dt/(m*hbar))*self.p2)

        t0 = time.time()
        bar = progressbar.ProgressBar()
        for i in bar(range(store_steps)):
            tmp = np.copy(Ψ[i])
            for j in range(Nt_per_store_step):
                c = np.fft.fftshift(np.fft.fftn(Ur*tmp))
                tmp = Ur*np.fft.ifftn(np.fft.ifftshift(Uk*c))
            Ψ[i+1] = tmp

            # Process all Q trajectories
            #for q_idx in range(10):
                # Convert Q position to index
                #idx = int((Q_list[q_idx][i] + 15*Å) * 500/(30*Å))
                #idx = max(0, min(499, idx))
                
                # Get wavefunction value at current position
                #psi = Ψ[i][idx]
                
                # Calculate gradient using finite difference
                #if idx < 499:
                    #dpsi = Ψ[i][idx+1] - psi
                #else:
                    #dpsi = psi - Ψ[i][idx-1]
                
                # Calculate velocity using Bohmian guidance equation
                #v = (hbar/m) * np.imag(dpsi/psi)
                
                # Update position
                #Q_list[q_idx].append(Q_list[q_idx][i] + v * self.simulation.dt * 7)

        print("Took", time.time() - t0)

        # Store all Q trajectories in simulation
        #for q_idx in range(10):
            #setattr(self.simulation, f'Q{q_idx}', Q_list[q_idx])

        self.simulation.Ψ = Ψ
        self.simulation.Ψmax = np.amax(np.abs(Ψ))




class SplitStepCupy(Method):
    def __init__(self, simulation):

        self.simulation = simulation
        self.H = simulation.H
        self.simulation.Vmin = np.amin(self.H.Vgrid)
        self.simulation.Vmax = np.amax(self.H.Vgrid)

        self.H.particle_system.compute_momentum_space(self.H)
        self.p2 = self.H.particle_system.p2


    def run(self, initial_wavefunction, total_time, dt, store_steps = 1):

        import cupy as cp

        self.p2 = cp.array(self.p2)
        self.simulation.store_steps = store_steps
        dt_store = total_time/store_steps
        self.simulation.total_time = total_time

        Nt = int(np.round(total_time / dt))
        Nt_per_store_step = int(np.round(dt_store / dt))
        self.simulation.Nt_per_store_step = Nt_per_store_step

        #time/dt and dt_store/dt must be integers. Otherwise dt is rounded to match that the Nt_per_store_stepdivisions are integers
        self.simulation.dt = dt_store/Nt_per_store_step


        Ψ = cp.zeros((store_steps + 1, *([self.H.N] *self.H.ndim )), dtype = cp.complex128)
        Ψ[0] = cp.array(initial_wavefunction(self.H.particle_system))



        m = self.H.particle_system.m


        Ur = cp.exp(-0.5j*(self.simulation.dt/hbar)*cp.array(self.H.Vgrid))
        Uk = cp.exp(-0.5j*(self.simulation.dt/(m*hbar))*self.p2)

        t0 = time.time()
        bar = progressbar.ProgressBar()
        for i in bar(range(store_steps)):
            tmp = cp.copy(Ψ[i])
            for j in range(Nt_per_store_step):
                c = cp.fft.fftshift(cp.fft.fftn(Ur*tmp))
                tmp = Ur*cp.fft.ifftn( cp.fft.ifftshift(Uk*c))
            Ψ[i+1] = tmp
        print("Took", time.time() - t0)

        self.simulation.Ψ = Ψ.get()
        self.simulation.Ψmax = np.amax(np.abs(self.simulation.Ψ ))


