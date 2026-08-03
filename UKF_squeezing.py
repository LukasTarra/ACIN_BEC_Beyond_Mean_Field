
"""import packages"""
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, jacobian # , profiler
import jax.debug as jdebug
from jax.lax import scan, fori_loop
from quadax import cumulative_trapezoid
from functools import partial
import pdb
# import jax.scipy.optimize as jopt
from jaxopt import Broyden
from jaxopt import AndersonAcceleration
# broader shell output before linebreaks for debugging 
np.set_printoptions(linewidth=300, edgeitems=10)
import os 

from utilities import Timer

@jit
def cov_step(x, eta, U, dt, params):
    
    # Load parameters
    hbar = params['hbar']
    N = params['N']
    
    # define rhs of the Josephson ODE
    def rhs(x):
        q_1, q_2, q_3, J = x
    
        return jnp.array([-2 * jnp.sin(jnp.sqrt(2) * q_2 * q_1 ** (-0.1e1 / 0.2e1)) * jnp.sqrt(-2 * q_1 + 1) * jnp.sqrt(2) * jnp.sqrt(q_1) / hbar * J, -(-2 * q_1 + 1) ** (-0.1e1 / 0.2e1) * (-2 * J * jnp.cos(jnp.sqrt(2) * q_2 * q_1 ** (-0.1e1 / 0.2e1)) * q_1 ** (0.3e1 / 0.2e1) - 2 * J * (q_1 - 0.1e1 / 0.2e1) * jnp.sqrt(2) * q_2 * jnp.sin(jnp.sqrt(2) * q_2 * q_1 ** (-0.1e1 / 0.2e1)) + (J * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1) * jnp.sqrt(2) * jnp.sqrt(q_1) * jnp.sin(jnp.sqrt(2) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1)) - N * q_1 ** (0.3e1 / 0.2e1) * U) * jnp.sqrt(-2 * q_1 + 1)) * q_1 ** (-0.1e1 / 0.2e1) / hbar , -4 * (jnp.cos(jnp.sqrt(2) * q_2 * q_1 ** (-0.1e1 / 0.2e1)) * jnp.sqrt(-2 * q_1 + 1) * J - (q_1 - 0.1e1 / 0.2e1) * U * N) * q_2 / hbar / (2 * q_1 - 1) , eta])

    # perform RK4 step
    k1 = dt * rhs(x)
    k2 = dt * rhs(x + 0.5 * k1)
    k3 = dt * rhs(x + 0.5 * k2)
    k4 = dt * rhs(x + k3)
    
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6
    
# @jit
def simulate_state_traj(eta_traj, U_traj, x_0, dt, params):
    """Simulate one trajectory of the cov system for one initial state."""
    
    def step_fn(carry, inputs):
        x = carry
        eta, U = inputs
        
        x_next = cov_step(x, eta, U, dt, params)
        return x_next, x_next

    inputs = (eta_traj, U_traj)
    _, traj = scan(step_fn, x_0, inputs)
    
    return traj

# sim_forward_vmap = (vmap(simulate_single_traj, in_axes=(None, None, 0, None, None), out_axes=0))


# might cut n_avg from the input as this becomes more settled
@jit
def costate_step(x, U, q_1, q_2, q_3, J, J_base, c_J, c_B, dt, params):
    
    # Load parameters
    hbar = params['hbar']
    N = params['N']

    # define rhs of the optimality condition ODE
    def rhs(x):
        lambda_1, lambda_2, lambda_3, lambda_J = x
    
        return jnp.array([-2 * (-2 * J * (q_1 - 0.1e1 / 0.2e1) ** 2 * q_2 ** 2 * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1) * lambda_2 * jnp.cos(jnp.sqrt(2) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1)) - jnp.sqrt(2) * J * (q_1 - 0.1e1 / 0.2e1) ** 2 * q_2 ** 2 * lambda_2 * jnp.sin(jnp.sqrt(2) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1)) + jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1) * (2 * J * ((2 * q_2 * lambda_1 - lambda_2 / 2) * q_1 ** 3 + (q_2 ** 2 * lambda_2 + (lambda_3 - 2 * lambda_1) * q_2 + lambda_2 / 2) * q_1 ** 2 + (-q_2 ** 2 * lambda_2 + q_2 * lambda_1 / 2) * q_1 + q_2 ** 2 * lambda_2 / 4) * jnp.sqrt(-2 * q_1 + 1) * jnp.cos(jnp.sqrt(2) * q_2 * q_1 ** (-0.1e1 / 0.2e1)) + jnp.sqrt(2) * J * ((-2 * q_2 ** 2 * lambda_3 - lambda_1 / 2) * q_1 ** (0.3e1 / 0.2e1) + (-q_2 * lambda_2 + 3 * lambda_1) * q_1 ** (0.5e1 / 0.2e1) - 4 * q_1 ** (0.7e1 / 0.2e1) * lambda_1 + jnp.sqrt(q_1) * q_2 * (lambda_3 * q_2 + lambda_2 / 4)) * jnp.sqrt(-2 * q_1 + 1) * jnp.sin(jnp.sqrt(2) * q_2 * q_1 ** (-0.1e1 / 0.2e1)) + 2 * (q_1 - 0.1e1 / 0.2e1) ** 2 * U * q_1 ** 2 * N * lambda_2)) * ((q_1 * q_3 - q_2 ** 2) / q_1) ** (-0.1e1 / 0.2e1) / q_1 ** 2 / hbar / (2 * q_1 - 1) ** 2, 4 * ((q_1 * q_3 - q_2 ** 2) / q_1) ** (-0.1e1 / 0.2e1) * (-J * jnp.cos(jnp.sqrt(2) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1)) * jnp.sqrt(-2 * q_1 + 1) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1) * q_2 * lambda_2 / 2 - J * jnp.sqrt(-2 * q_1 + 1) * jnp.sqrt(2) * jnp.sin(jnp.sqrt(2) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1)) * q_2 * lambda_2 / 4 + (-J * (2 * lambda_1 * q_1 ** 2 + (q_2 * lambda_2 - lambda_1 + lambda_3) * q_1 - q_2 * lambda_2 / 2) * jnp.cos(jnp.sqrt(2) * q_2 * q_1 ** (-0.1e1 / 0.2e1)) + J * jnp.sqrt(2) * jnp.sqrt(q_1) * (lambda_3 * q_2 + lambda_2 / 4) * jnp.sin(jnp.sqrt(2) * q_2 * q_1 ** (-0.1e1 / 0.2e1)) - N * jnp.sqrt(-2 * q_1 + 1) * U * q_1 * lambda_3 / 2) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1)) * (-2 * q_1 + 1) ** (-0.1e1 / 0.2e1) / q_1 / hbar, lambda_2 / hbar * J * (jnp.sqrt(2) * jnp.sin(jnp.sqrt(2) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1)) + 2 * jnp.cos(jnp.sqrt(2) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1)) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1)) * ((q_1 * q_3 - q_2 ** 2) / q_1) ** (-0.1e1 / 0.2e1) / 2, (-2 * J * J_base ** 2 * (2 * lambda_3 * jnp.sqrt(q_1) * q_2 + lambda_2 * q_1 ** (0.3e1 / 0.2e1)) * jnp.cos(jnp.sqrt(2) * q_2 * q_1 ** (-0.1e1 / 0.2e1)) - 2 * (2 * q_1 * lambda_1 + q_2 * lambda_2) * J * jnp.sqrt(2) * (q_1 - 0.1e1 / 0.2e1) * J_base ** 2 * jnp.sin(jnp.sqrt(2) * q_2 * q_1 ** (-0.1e1 / 0.2e1)) + (J * J_base ** 2 * jnp.sqrt(2) * jnp.sin(jnp.sqrt(2) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1)) * jnp.sqrt((q_1 * q_3 - q_2 ** 2) / q_1) * lambda_2 - (c_J * J ** 2 - c_J * J * J_base - c_B * J_base ** 2 / 2) * hbar) * jnp.sqrt(-2 * q_1 + 1) * jnp.sqrt(q_1)) * q_1 ** (-0.1e1 / 0.2e1) * (-2 * q_1 + 1) ** (-0.1e1 / 0.2e1) / J_base ** 2 / J / hbar])

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
def simulate_costate_traj(U_traj, q_1_traj, q_2_traj, q_3_traj, J_traj, x_0, J_base, c_J, c_B, dt, params):
    """Simulate one trajectory of the Josephson system for one initial state."""
    
    def step_fn(carry, inputs):
        x = carry
        U, q_1, q_2, q_3, J = inputs
        
        x_next = costate_step(x, U, q_1, q_2, q_3, J, J_base, c_J, c_B, dt, params)
        return x_next, x_next

    inputs = (U_traj, q_1_traj, q_2_traj, q_3_traj, J_traj)
    _, traj = scan(step_fn, x_0, inputs)
    
    return traj

# sim_costate_vmap = (vmap(simulate_costate_traj, in_axes=(None, None, 0, None, 0, 0, None, None, None, None), out_axes=0))


@jit
def J_array_iteration_w_deriv_cost(eta_in_array, U_array, x_0, dt, T, c_eta, c_n, c_J, c_B, J_base, params):
    
    # simulate the Josephson system foward for n and Phi
    traj = simulate_state_traj(eta_in_array, U_array, x_0, dt, params)
    q_1_traj = traj[:,0]
    q_2_traj = traj[:,1]
    q_3_traj = traj[:,2]
    J_traj = traj[:,3]

    lambda_0_array = jnp.array([c_n,0,0,0])
    
    # simulate the costate system backward, starting from lambda_0_array
    lambda_traj_rev = simulate_costate_traj(U_array, q_1_traj, q_2_traj, q_3_traj, J_traj, lambda_0_array, J_base, c_J, c_B, dt, params)
    lambda_traj = lambda_traj_rev[::-1, :]
    eta_out_array = - (J_base/T)**2/c_eta * lambda_traj[:,3]

    return eta_out_array


def run_Newton_J_descent(eta_init_array, U_array, x_0, dt, T, c_eta, c_n, c_J, c_B, J_base, params, iteration_fun, max_iter=2):
    """Iterate the J_array iteration for max_iter iterations."""

    @jit
    def f(val):
        return iteration_fun(val, U_array, x_0, dt, T, c_eta, c_n, c_J, c_B, J_base, params) - val
    
    f_Jacobian = jacobian(f)

    def Newton_step(i, val):

        # tmp = jnp.linalg.solve( f_Jacobian(val), f(val) )
        # jdabug.breakpoint()
        
        return val - jnp.linalg.solve( f_Jacobian(val), f(val) )

    eta_out_array = fori_loop(0, max_iter, Newton_step, eta_init_array)
    
    return eta_out_array


def run_FixedPoint_J_descent(eta_init_array, U_array, x_0, dt, T, c_eta, c_n, c_J, c_B, J_base, params, iteration_fun):
    """Iterate the J_array iteration for max_iter iterations."""

    # @jit
    # def f(val):
    #     return iteration_fun(val, U_array, x_0, dt, T, c_eta, c_n, c_J, c_B, J_base, params) - val
    
    # Configure history_size (m) based on your problem
    # Higher m = faster convergence but more memory and potential instability
    accel = AndersonAcceleration(iteration_fun)
    eta_out_array = accel.run(eta_init_array, U_array, x_0, dt, T, c_eta, c_n, c_J, c_B, J_base, params)  # initial_p is your initial guess
    
    return eta_out_array.params


def run_Broyden_J_descent(eta_init_array, U_array, x_0, dt, T, c_eta, c_n, c_J, c_B, J_base, params, iteration_fun):
    """Iterate the J_array iteration for max_iter iterations."""

    broyden = Broyden(fun=lambda x, U_array, x_0, dt, T, c_eta, c_n, c_J, c_B, J_base, params: iteration_fun(x, U_array, x_0, dt, T, c_eta, c_n, c_J, c_B, J_base, params) - x)
    eta_out_array = broyden.run(eta_init_array, U_array, x_0, dt, T, c_eta, c_n, c_J, c_B, J_base, params)  # initial_p is your initial guess
    
    return eta_out_array.params


if __name__ == "__main__":

    # Define system parameters
    params = {
        # 'hbar': 6.626e-34 / (2*jnp.pi), # Planck (Js)
        'hbar': 1, # set to 1 by virtue of the other params
        'N': 3500,
        # 'N': 3351,
        'E': 0
    }

    #Load further fundamental parameters
    # J_baseline = 610 # Hz
    # J_baseline = 297.6522 # Hz
    # J_baseline = 41 # Hz 
    U_baseline = 0.33 # Hz

    # infer J baseline from plasma freq measurement
    f_p_exp = 324/2 #Hz, 0.52 trap
    # f_p_exp = 155/2 #Hz, 0.53 trap
    cos_Phi_avg = 1
    J_baseline = np.sqrt( (U_baseline*params["N"]/4/cos_Phi_avg)**2 + (params["hbar"]*np.pi*f_p_exp)**2/cos_Phi_avg ) - U_baseline*params["N"]/4/cos_Phi_avg
    print(f"J_base: {J_baseline}")

    # Define variances for Gaussian distribution of initial states
    Lambda_baseline = U_baseline*params['N'] / (2*J_baseline)
    sqrt_1_plus_Lambda = jnp.sqrt(1+   1*Lambda_baseline)
    # sqrt_1_plus_Lambda = jnp.sqrt(1+ Lambda_baseline)
    variance_0_n =     1.1*  1/(sqrt_1_plus_Lambda*params['N'])
    variance_0_Phi =   1.1*  sqrt_1_plus_Lambda / params['N']
    covariance_0_nPhi = 0

    # print the resulting plasma frequency
    plasma_frequency = 2*J_baseline/(params['hbar']*2*jnp.pi) * jnp.sqrt(1 + Lambda_baseline)
    print(f"plasma frequency: {plasma_frequency} Hz")

    # time window for pre-simulation -> yield experimental init state x_start
    phase_offset_exp = 0.14 #rad; read off from experiment
    t_pre = phase_offset_exp / (4*np.pi*plasma_frequency)
    num_steps_pre = int(100)
    dt_pre = t_pre / num_steps_pre
    time_array_pre = jnp.linspace(0, t_pre, num_steps_pre)
    # Stack into initial state vectors
    x_0 = jnp.array([variance_0_n,covariance_0_nPhi,variance_0_Phi,J_baseline])
    # Define dummy trajectories for J and U (can be replaced with actual time-series)
    eta_traj_pre = jnp.zeros(num_steps_pre)
    U_traj_pre = jnp.ones(num_steps_pre) * U_baseline
    # simulate the pre-time window
    trajectories_pre = simulate_state_traj(eta_traj_pre, U_traj_pre, x_0, dt_pre, params)
    trajectories_pre = np.array(trajectories_pre)
    # compute x_start
    x_start = trajectories_pre[-1,:]

    # define tiem window for optimization
    t_final = 2/plasma_frequency
    # t_final = 0.005
    num_steps = int(500)
    dt = t_final / num_steps
    time_array = jnp.linspace(0, t_final, num_steps)
    
    # Define dummy trajectories for J and U (can be replaced with actual time-series)
    eta_traj = jnp.zeros(num_steps)
    U_traj = jnp.ones(num_steps) * U_baseline

    # # hard-coded ansatz function
    # mod_frequency = 2*plasma_frequency
    # time_array_for_ansatz = jnp.linspace(0, t_final, num_steps+1)
    # J_ansatz = 0.2*J_baseline* jnp.sin(2*jnp.pi*mod_frequency*time_array_for_ansatz)
    # # compute eta_ansatz as derivative of J_ansatz
    # eta_ansatz = jnp.diff(J_ansatz) / jnp.diff(time_array_for_ansatz)
    # eta_traj = eta_ansatz

    # GOOD CHOICE FOR 0.52 TRAP and T=3.5/f_p
    # eta_traj = run_FixedPoint_J_descent(eta_traj, U_traj, x_0, dt, t_final, 0.8, 85., 3., 0, J_baseline, params, J_array_iteration_w_deriv_cost)
    # eta_traj = run_Newton_J_descent(eta_traj, U_traj, x_0, dt, t_final, 0.5, 150., 0, 0, J_baseline, params, J_array_iteration_w_deriv_cost , 2)

    eta_traj = run_FixedPoint_J_descent(eta_traj, U_traj, x_0, dt, t_final, 0.8, 150., 3., 0, J_baseline, params, J_array_iteration_w_deriv_cost)
    eta_traj = run_Newton_J_descent(eta_traj, U_traj, x_0, dt, t_final, 0.5, 750., 0, 0, J_baseline, params, J_array_iteration_w_deriv_cost , 8)
    
    # run simulation with the optimized trajectory
    # Time the simulation
    timer = Timer()
    timer.start()
    # Run the simulation
    trajectories = simulate_state_traj(eta_traj, U_traj, x_start, dt, params)
    trajectories = np.array(trajectories)
    _ = timer.stop()

    plt.figure(figsize=(8,10))
    time_array_np = np.array(time_array)
    # Plot q_1 (column 0)
    plt.subplot(4, 1, 1)
    plt.plot(time_array_np, trajectories[:, 0], label='q_1 (variance_n)')
    plt.axhline(y=variance_0_n, color='black', linestyle='--', label='Initial variance')
    plt.ylabel('q_1')
    plt.grid(True)
    plt.legend(loc='best')
    # Plot q_3 (column 2)
    plt.subplot(4, 1, 2)
    plt.plot(time_array_np, trajectories[:, 2], label='q_3 (variance_Phi)', color='orange')
    plt.axhline(y=variance_0_Phi, color='black', linestyle='--', label='Initial variance')
    plt.ylabel('q_3')
    plt.grid(True)
    plt.legend(loc='best')
    # Plot J (column 3)
    plt.subplot(4, 1, 3)
    plt.plot(time_array_np, trajectories[:, 3], label='J (Josephson Energy)', color='green')
    plt.ylabel('J (Hz)')
    plt.grid(True)
    plt.legend(loc='best')
    # Plot q_1 * q_3
    plt.subplot(4, 1, 4)
    plt.plot(time_array_np, trajectories[:, 0] * trajectories[:, 2], label='q_1 * q_3', color='red')
    plt.axhline(y=variance_0_n * variance_0_Phi, color='black', linestyle='--', label='Initial product variance')
    plt.ylabel('q_1 * q_3')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    

    # # to my coding asstistant: store time_array_np and J_array as columns to a csv file  
    # J_array = trajectories[:, 3]
    # var_n   = trajectories[:, 0]
    # var_Phi = trajectories[:, 2]
    # np.savetxt('variances_J-deriv-cost_T=2-by-fp_0p52-trap.csv', np.column_stack((time_array_np, var_n, var_Phi, J_array)), delimiter=',', header='times,varn,varPhirad,JHz', comments='')

    
