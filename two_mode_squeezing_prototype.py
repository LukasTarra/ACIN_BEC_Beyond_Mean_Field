"""import packages"""
import time
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, vmap, random # , profiler
import jax.debug as jdebug
from jax.lax import scan
from functools import partial
import pdb
# import jax.scipy.optimize as jopt
from jaxopt import ScipyBoundedMinimize
# broader shell output before linebreaks for debugging 
np.set_printoptions(linewidth=300, edgeitems=10)

# Timer class for simple runtime checking
class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.perf_counter()
    
    def stop(self):
        self.end_time = time.perf_counter()
        return self.elapsed()
    
    def elapsed(self):
        if self.start_time is None:
            raise ValueError("Timer not started")
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time

# simulation results class for plotting & analytics
class SimResults:
    def __init__(self, t_array, traj, var_0, J_array):
        """
        Initialize the simulation results container.
        
        Parameters:
        t_array (numpy.ndarray): Array of time steps
        traj (numpy.ndarray): Array of shape (num_trajectories, num_time_steps, 2) 
                              containing the simulation trajectories for [n, Phi].
        var_0 : tuple of initial variances:
        - var_0_n (float): Initial variance of n.
        - var_0_Phi (float): Initial variance of Phi.
        """
        var_0_n, var_0_Phi = var_0

        self.t_array = t_array
        self.traj = traj
        self.var_0_n = var_0_n
        self.var_0_Phi = var_0_Phi
        self.num_trajectories, self.num_time_steps, _ = traj.shape
        self.J_array = J_array

    def plot_trajectories(self, N=20, plotting_sample_step=2, plot_mean=True):
        """
        Plot sample trajectories along with the mean trajectory.
        
        Parameters:
        N (int): Number of sample trajectories to plot.
        plotting_sample_step (int): Plot every nth time sample.
        plot_mean (bool): Whether to plot the mean trajectory.
        """
        
        # Select random sample indices
        sample_indices = np.random.choice(self.num_trajectories, min(N, self.num_trajectories), replace=False)
        
        # Calculate means
        mean_n = np.mean(self.traj[:, :, 0], axis=0)
        mean_phi = np.mean(self.traj[:, :, 1], axis=0)
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        # Plot n trajectories
        for i in sample_indices:
            ax1.plot(self.t_array[::plotting_sample_step], 
                     self.traj[i, ::plotting_sample_step, 0], 
                     alpha=0.15, color='orange')
        if plot_mean:
            ax1.plot(self.t_array[::plotting_sample_step], 
                     mean_n[::plotting_sample_step], 
                     label='Mean n', linewidth=1.2, color="blue")
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('n')
        ax1.set_title('Trajectories of n')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        # Plot Phi trajectories
        for i in sample_indices:
            ax2.plot(self.t_array[::plotting_sample_step], 
                     self.traj[i, ::plotting_sample_step, 1], 
                     alpha=0.15, color='orange')
        if plot_mean:
            ax2.plot(self.t_array[::plotting_sample_step], 
                     mean_phi[::plotting_sample_step], 
                     label='Mean Phi', linewidth=1.2, color="blue")
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Phi')
        ax2.set_title('Trajectories of Phi')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        ax3.plot(self.t_array[::plotting_sample_step],self.J_array[::plotting_sample_step], linewidth=1.2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('J(t)')
        ax3.set_title('Input trajectory')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_variances(self, plotting_sample_step=2):
        """
        Plot the variances of n and Phi over time.
        
        Parameters:
        plotting_sample_step (int): Plot every nth time sample.
        """
        
        # Calculate variances over trajectories
        var_n = np.var(self.traj[:, :, 0], axis=0)
        var_phi = np.var(self.traj[:, :, 1], axis=0)
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        # Plot variance of n
        ax1.plot(self.t_array[::plotting_sample_step], 
                 var_n[::plotting_sample_step], 
                 linewidth=1.2, color="blue")
        ax1.axhline(y=self.var_0_n, color='red', linestyle='--', label=f'Initial variance: {self.var_0_n:.2e}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Variance of n')
        ax1.set_title('Variance of n over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot variance of Phi
        ax2.plot(self.t_array[::plotting_sample_step], 
                 var_phi[::plotting_sample_step], 
                 linewidth=1.2, color="blue")
        ax2.axhline(y=self.var_0_Phi, color='red', linestyle='--', label=f'Initial variance: {self.var_0_Phi:.2e}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Variance of Phi')
        ax2.set_title('Variance of Phi over Time')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(self.t_array[::plotting_sample_step],self.J_array[::plotting_sample_step], linewidth=1.2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('J(t)')
        ax3.set_title('Input trajectory')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

    
@jit
def Josephson_step(x, J, U, dt, params):
    """
    Perform one RK2 step for the system of ODEs:
    dn/dt = -(2*J/hbar)*sqrt(1-n^2)*sin(Phi)
    dPhi/dt = (NU/hbar)*n + (2*J/hbar)*(n/sqrt(1-n^2))*cos(Phi) + Delta_E
    
    Inputs:
    x: [n, Phi] state vector
    J, U: tunnel coupling inputs depending on left & right wavefunction
    params: system parameters (hbar, N, Delta_E)
    
    Returns:
    x_next: time increment of x
    """
    # Load parameters
    hbar = params['hbar']
    N = params['N']
    Delta_E = params['E']
    two_J_by_hbar = (2*J/hbar)
    
    # # Avoid division by zero
    # if jnp.sqrt(1 - n**2) < 1e-10:
    #     return x

    # define rhs of the Josephson ODE
    def rhs(x):
        n, Phi = x
        sqrt_1_minus_n2 = jnp.sqrt(1 - n**2)
        dn_dt = -two_J_by_hbar * sqrt_1_minus_n2 * jnp.sin(Phi)
        dPhi_dt = (N*U/hbar) * n + two_J_by_hbar * (n/sqrt_1_minus_n2) * jnp.cos(Phi) + Delta_E
    
        return jnp.array([dn_dt, dPhi_dt])

    # # perform on RK2 step with the rhs
    # k1 = dt * rhs(x)
    # k2 = dt * rhs(x + k1)
    
    # return x + 0.5* (k1 + k2)

    # perform RK4 step
    k1 = dt * rhs(x)
    k2 = dt * rhs(x + 0.5 * k1)
    k3 = dt * rhs(x + 0.5 * k2)
    k4 = dt * rhs(x + k3)
    
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6
    
# @jit
def simulate_single_traj(J_traj, U_traj, x_0, dt, params):
    """Simulate one trajectory of the Josephson system for one initial state."""
    
    def step_fn(carry, inputs):
        x = carry
        J, U = inputs
        
        x_next = Josephson_step(x, J, U, dt, params)
        return x_next, x_next

    inputs = (J_traj, U_traj)
    _, traj = scan(step_fn, x_0, inputs)
    
    return traj

sim_forward_vmap = (vmap(simulate_single_traj, in_axes=(None, None, 0, None, None), out_axes=0))

#cost function for traj optimization
@jit
def traj_cost_function(input_array, traj, t_array, cost_weights, input_base, f_p):
    c_n, c_J_deriv, c_J_abs = cost_weights
    T = t_array[-1]
    delta_t = 1/(4*jnp.pi*f_p)
    var_n_traj = jnp.var(traj[:, :, 0], axis=0)
    # n_traj = traj[:, :, 0]
    # cost_n = c_n/T * jnp.mean(jnp.trapezoid(n_traj**2, x=t_array, axis=1) , axis=0)
    cost_n = c_n/T * jnp.trapezoid(var_n_traj, x=t_array)
    # cost_n = c_n * jnp.min(var_n_traj)
    # cost_J_deriv = c_J_deriv/T * (delta_t/input_base)**2 * jnp.trapezoid(jnp.gradient(input_array,t_array)**2, x=t_array)
    # cost_J_abs = c_J_abs * jnp.mean(input_array)/input_base
    
    return cost_n

@jit
def input_guess_fun(t, input_amplitude, input_frequency, input_phase, input_offset, base_parameters):
    base_amplitude, base_frequency, base_phase, base_offset = base_parameters
    return jnp.array(input_amplitude*base_amplitude* jnp.sin(2*jnp.pi*input_frequency*base_frequency*t + input_phase*base_phase) + input_offset*base_offset)

def sim_and_cost_guess_fun(input_choice, input_amplitude, input_frequency, input_phase, input_offset, base_parameters, cost_weights, input_base, f_p, other_input_array,  x_0_array, t_array, dt, params):
    if input_choice == "J":
        J_array = input_guess_fun(t_array, input_amplitude, input_frequency, input_phase, input_offset, base_parameters)
        U_array = other_input_array
    elif input_choice == "U":
        J_array = other_input_array
        U_array = input_guess_fun(t_array, input_amplitude, input_frequency, input_phase, input_offset, base_parameters)
    else:
        raise ValueError("input choice must be either J or U.")
    traj = sim_forward_vmap(J_array, U_array, x_0_array, dt, params)

    if input_choice == "J":
        cost = traj_cost_function(J_array, traj, t_array, cost_weights, input_base, f_p)
    elif input_choice == "U":
        cost = traj_cost_function(U_array, traj, t_array, cost_weights, input_base, f_p)
    else:
        raise ValueError("input choice must be either J or U.")
    return cost

def optimize_J_guess_fun_BFGS(initial_opt_var, base_parameters, cost_weights, J_base, f_p, U_array, x_0_array, t_array, dt, params):
    
    def wrapper(opt_var):
        amp, freq, phase, offset = opt_var
        return sim_and_cost_guess_fun("J", amp, freq, phase, offset, base_parameters, cost_weights, J_base, f_p, U_array, x_0_array, t_array, dt, params)

    # result = jopt.minimize(wrapper, initial_opt_var, method='BFGS')
    lbfgsb = ScipyBoundedMinimize(fun=wrapper, method="slsqp", maxiter=1000)
    lower_bounds = jnp.array([0, 0.1, -1, 0.5])
    upper_bounds = jnp.array([0.5, 2.2, 1, 3])
    bounds = (lower_bounds, upper_bounds)
    lbfgsb_sol = lbfgsb.run(initial_opt_var, bounds=bounds)
    
    # lbfgsb = LBFGSB(fun=wrapper, maxiter=100, verbose=True)
    # lower_bounds = jnp.array([0, -jnp.pi, 0.5])
    # upper_bounds = jnp.array([0.5, jnp.pi, 1])
    # bounds = (lower_bounds, upper_bounds)
    # lbfgsb_sol = lbfgsb.run(initial_opt_var, bounds=bounds)
    
    return lbfgsb_sol.params

# def optimize_U_guess_fun_BFGS(initial_opt_var, cost_weights, U_base, f_p, J_array, x_0_array, t_array, dt, params):
    
#     def wrapper(opt_var):
#         amp, freq, phase, offset = opt_var
#         return sim_and_cost_guess_fun("U", amp, freq, phase, offset, cost_weights, U_base, f_p, J_array, x_0_array, t_array, dt, params)

#     # # result = jopt.minimize(wrapper, initial_opt_var, method='BFGS')
#     # lbfgsb = ScipyBoundedMinimize(fun=wrapper, method="slsqp", maxiter=500)
#     # lower_bounds = jnp.array([0, -jnp.pi, 0.3])
#     # upper_bounds = jnp.array([0.3, jnp.pi, 1])
#     # bounds = (lower_bounds, upper_bounds)
#     # lbfgsb_sol = lbfgsb.run(initial_opt_var, bounds=bounds)

#     lbfgsb = LBFGSB(fun=wrapper, maxiter=100, verbose=True)
#     lower_bounds = jnp.array([0, -jnp.pi, 0.3])
#     upper_bounds = jnp.array([0.3, jnp.pi, 1])
#     bounds = (lower_bounds, upper_bounds)
#     lbfgsb_sol = lbfgsb.run(initial_opt_var, bounds=bounds)
    
#     return lbfgsb_sol.params

@jit
def costate_step(x, J, U, dt, params):
    """
    Perform one RK4 step for the costates of the optimality conditions.
    
    Inputs:
    
    Returns:
    """
    # Load parameters
    hbar = params['hbar']
    N = params['N']
    Delta_E = params['E']
    two_J_by_hbar = (2*J/hbar)

    # define rhs of the optimality condition ODE
    def rhs(x):
        lambda_n, lambda_Phi = x
        dlambda_n_dt = -c_n/(num_traj*T) *2*(1-1/num_traj)*(n-n_avg) - 2*J/hbar*n/jnp.sqrt(1-n**2)*jnp.sin(Phi)*lambda_n - 1/hbar*(N*U+2*J/((1-n**2)**(3/2))*jnp.cos(Phi))*lambda_Phi
    
        return jnp.array([dlambda_n_dt, dlambda_Phi_dt])

    # # perform on RK2 step with the rhs
    # k1 = dt * rhs(x)
    # k2 = dt * rhs(x + k1)
    
    # return x + 0.5* (k1 + k2)

    # perform RK4 step
    k1 = dt * rhs(x)
    k2 = dt * rhs(x + 0.5 * k1)
    k3 = dt * rhs(x + 0.5 * k2)
    k4 = dt * rhs(x + k3)
    
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6
    
# @jit
def simulate_costate_traj(J_traj, U_traj, x_0, dt, params):
    """Simulate one trajectory of the Josephson system for one initial state."""
    
    def step_fn(carry, inputs):
        x = carry
        J, U = inputs
        
        x_next = costate_step(x, J, U, dt, params)
        return x_next, x_next

    inputs = (J_traj, U_traj)
    _, traj = scan(step_fn, x_0, inputs)
    
    return traj

sim_costate_vmap = (vmap(simulate_costate_traj, in_axes=(None, None, 0, None, None), out_axes=0))


if __name__ == "__main__":

    # Define system parameters
    params = {
        # 'hbar': 6.626e-34 / (2*jnp.pi), # Planck (Js)
        'hbar': 1, # set to 1 by virtue of the other params
        'N': 3500,
        'E': 0
    }

    #Load further fundamental parameters
    J_baseline = 1*    41 # Hz
    U_baseline = 0.33 # Hz
    Lambda_baseline = U_baseline*params['N'] / (2*J_baseline)

    # Define variances for Gaussian distribution of initial states
    sqrt_1_plus_Lambda = jnp.sqrt(1+   1*Lambda_baseline)
    # sqrt_1_plus_Lambda = jnp.sqrt(1+ Lambda_baseline)
    variance_0_n =     1*  1/(sqrt_1_plus_Lambda*params['N'])
    variance_0_Phi =   1*  sqrt_1_plus_Lambda / params['N']

    # Generate num_trajectories random initial states (n, Phi) with Gaussian distribution
    rng = random.PRNGKey(1)  # For reproducibility
    num_trajectories = 500
    # Sample from Gaussian distributions for n and Phi
    n_samples = random.normal(rng, (num_trajectories,)) * np.sqrt(variance_0_n)
    Phi_samples = random.normal(rng, (num_trajectories,)) * np.sqrt(variance_0_Phi)
    
    # Simulation settings
    Phi_0_mean = jnp.mean(Phi_samples)
    plasma_frequency = 2*J_baseline/(params['hbar']*2*jnp.pi) * jnp.sqrt(Phi_0_mean + Lambda_baseline)
    # t_final = 3/plasma_frequency
    t_final = 50e-3
    num_steps = int(500)
    dt = t_final / num_steps
    time_array = jnp.linspace(0, t_final, num_steps)

    # Ensure that |n| < 1 to avoid numerical issues
    n_samples = jnp.clip(n_samples, -0.99, 0.99)
    
    # Stack into initial state vectors
    x_0_array = jnp.column_stack([n_samples, Phi_samples])
    
    # Define dummy trajectories for J and U (can be replaced with actual time-series)
    J_traj = jnp.ones(num_steps) * J_baseline
    U_traj = jnp.ones(num_steps) * U_baseline

    # optimize the trajectory for J
    initial_opt_var = jnp.array([0.1, 2, 0.5, 1])
    opt_base_parameters = J_baseline, plasma_frequency, jnp.pi, J_baseline
    # initial_opt_var = jnp.array([0.2*U_baseline, 1.25*jnp.pi/2, 0.8*U_baseline])
    c_params = 1000, 0, 0
    opt_variables_out = np.array(optimize_J_guess_fun_BFGS(initial_opt_var, opt_base_parameters, c_params, J_baseline, plasma_frequency, U_traj, x_0_array, time_array, dt, params))
    J_traj = input_guess_fun(time_array, opt_variables_out[0], opt_variables_out[1], opt_variables_out[2], opt_variables_out[3], opt_base_parameters)
    
    # run simulation with the optimized trajectory
    # Time the simulation
    timer = Timer()
    timer.start()
    # Run the simulation
    trajectories = sim_forward_vmap(J_traj, U_traj, x_0_array, dt, params)
    trajectories = np.array(trajectories)
    timer.stop()
    print(f"Simulation completed in {timer.elapsed():.4f} seconds")

    print("Initial relative parameters:")
    print(np.array(initial_opt_var))    
    print("Optimized relative parameters:")
    print(opt_variables_out)
    print("Resulting cost value:")
    print(np.array(traj_cost_function(J_traj, trajectories, time_array, c_params, opt_base_parameters, plasma_frequency)))

    
    # create SimResults object
    sim_results = SimResults(time_array, trajectories, (variance_0_n, variance_0_Phi), J_traj )
    # plot trajectories and variances
    sim_results.plot_trajectories(N=num_trajectories, plotting_sample_step=1, plot_mean=True)
    sim_results.plot_variances(plotting_sample_step=1)



    
