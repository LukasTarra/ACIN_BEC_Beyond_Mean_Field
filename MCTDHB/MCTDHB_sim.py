"""
Full MCTDHB Simulation Implementation based on Grond et al. (2009)

This implementation uses the full Multi-Configurational Time-Dependent Hartree for Bosons (MCTDHB)
formalism with coupled orbital and coefficient dynamics.

Key Features:
- 1D spatial grid with finite differences for spatial derivatives
- Coupled orbital dynamics (phi_g, phi_e) with coefficient evolution (C_k)
- Self-consistent time evolution with projector enforcement
- Double-well potential V(x,t)
- JAX-based for GPU acceleration and automatic differentiation
- Functional/procedural style

Based on:
- Grond et al. (2009): "Many-particle tunneling dynamics in a double-well potential"
- grond09b.pdf: MCTDHB theory and implementation

The MCTDHB equations (Grond et al., Eq. 16-20):
- Orbital evolution: i*hbar d(phi_k)/dt = P [h phi_k + Sum_{j,l,m} rho_{kjlm} W_{jlm} phi_j* phi_l phi_m]
- Coefficient evolution: i*hbar dC/dt = H C

Notation:
- phi_g(x,t), phi_e(x,t): Ground and excited orbital wavefunctions
- C_k(t): Configuration coefficients for |k, N-k> states
- N: Total particle number
- U0: Interaction strength (contact interaction)
"""

import time
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad
from jax.lax import scan, fori_loop
from functools import partial
import jax.debug as jdebug
import os
import csv
from typing import Dict, Tuple, NamedTuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utilities import Timer


# =============================================================================
# Data Structures
# =============================================================================

class MCTDHBState(NamedTuple):
    """
    Complete MCTDHB state including orbitals and configuration coefficients.
    """
    phi_g: jnp.ndarray  # Shape: (num_grid_points,)
    phi_e: jnp.ndarray  # Shape: (num_grid_points,)
    config_coeffs: jnp.ndarray  # Shape: (N+1,)


class MCTDHBParams(NamedTuple):
    """
    MCTDHB simulation parameters.
    """
    N: int
    hbar: float
    m: float
    U0: float
    
    # Spatial grid
    x_min: float
    x_max: float
    num_grid_points: int
    
    # Double-well potential
    V0: float
    a: float
    omega: float
    
    # Time stepping
    dt: float
    num_steps: int
    
    @property
    def dx(self) -> float:
        """Grid spacing."""
        return (self.x_max - self.x_min) / (self.num_grid_points - 1)
    
    @property
    def x_grid(self) -> jnp.ndarray:
        """Spatial grid array."""
        return jnp.linspace(self.x_min, self.x_max, self.num_grid_points)


class MCTDHB_Results:
    """Container for MCTDHB simulation results."""
    
    def __init__(self, t_array: jnp.ndarray, params: MCTDHBParams):
        self.t_array = jnp.array(t_array)
        self.num_time_steps = len(t_array)
        self.params = params
        
        # Storage for results
        self.phi_g_trajectory = None
        self.phi_e_trajectory = None
        self.config_coeffs_trajectory = None
        
        # Observables
        self.n_expect = None
        self.n_variance = None
        self.squeezing_parameter = None
        self.phi_expect = None
        self.energy_expect = None
        self.orbital_overlap = None
        
        # Control fields
        self.A_array = None
        self.U_array = None
    
    def plot_dynamics(self, plotting_sample_step: int = 2):
        """Plot MCTDHB dynamics observables."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plots")
            return
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 14))
        
        if self.n_expect is not None:
            axes[0].plot(self.t_array[::plotting_sample_step], 
                         self.n_expect[::plotting_sample_step], 'b-', linewidth=1.2)
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('n')
            axes[0].set_title('Population Imbalance')
            axes[0].grid(True)
        
        if self.n_variance is not None:
            axes[1].plot(self.t_array[::plotting_sample_step], 
                         self.n_variance[::plotting_sample_step], 'g-', linewidth=1.2)
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Var(n)')
            axes[1].set_title('Variance of Population Imbalance')
            axes[1].grid(True)
        
        if self.squeezing_parameter is not None:
            axes[2].plot(self.t_array[::plotting_sample_step], 
                         self.squeezing_parameter[::plotting_sample_step], 'r-', linewidth=1.2)
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Squeezing Parameter')
            axes[2].set_title('Number Squeezing Parameter')
            axes[2].grid(True)
        
        if self.orbital_overlap is not None:
            axes[3].plot(self.t_array[::plotting_sample_step], 
                         jnp.abs(self.orbital_overlap[::plotting_sample_step]), 'p-', linewidth=1.2)
            axes[3].set_xlabel('Time (s)')
            axes[3].set_ylabel('|<phi_g|phi_e>|')
            axes[3].set_title('Orbital Overlap')
            axes[3].grid(True)
        
        if self.A_array is not None:
            axes[4].plot(self.t_array[::plotting_sample_step], 
                         self.A_array[::plotting_sample_step], 'o-', linewidth=1.2)
            axes[4].set_xlabel('Time (s)')
            axes[4].set_ylabel('J(t) (Hz)')
            axes[4].set_title('Tunnel Coupling Control')
            axes[4].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_to_csv(self, filename: str = "mctdhb_results.csv", directory: str = "./"):
        """Save MCTDHB simulation results to CSV file."""
        filepath = os.path.join(directory, filename)
        
        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['t (s)', 'J (Hz)', 'U (Hz)', 'n_expect', 'n_variance', 
                      'squeezing_param', 'orbital_overlap']
            writer.writerow(header)
            
            for i in range(self.num_time_steps):
                row = [
                    float(self.t_array[i]),
                    float(self.A_array[i]) if self.A_array is not None else 0.0,
                    float(self.U_array[i]) if self.U_array is not None else 0.0,
                    float(self.n_expect[i]) if self.n_expect is not None else 0.0,
                    float(self.n_variance[i]) if self.n_variance is not None else 0.0,
                    float(self.squeezing_parameter[i]) if self.squeezing_parameter is not None else 0.0,
                    float(jnp.abs(self.orbital_overlap[i])) if self.orbital_overlap is not None else 0.0
                ]
                writer.writerow(row)
        
        print(f"MCTDHB results saved to: {filepath}")


# =============================================================================
# Spatial Grid and Finite Difference Operators
# =============================================================================

# @partial(jit, static_argnums=(1,))
def create_spatial_grid(x_min: float, x_max: float, num_grid_points: int) -> Tuple[jnp.ndarray, float]:
    """Create 1D spatial grid with uniform spacing."""
    x_grid = jnp.linspace(x_min, x_max, num_grid_points)
    dx = (x_max - x_min) / (num_grid_points - 1)
    return x_grid, dx


# @partial(jit, static_argnums=(1, 2))
def second_derivative_fd(f: jnp.ndarray, dx: float, num_grid_points: int) -> jnp.ndarray:
    """
    Compute second derivative using central finite differences.
    f''(x_i) ~= (f_{i+1} - 2*f_i + f_{i-1}) / dx^2
    Boundary conditions: f=0 at boundaries (hard wall).
    """
    d2f = jnp.zeros_like(f)
    interior = jnp.arange(1, num_grid_points - 1)
    d2f = d2f.at[interior].set(
        (f[interior + 1] - 2 * f[interior] + f[interior - 1]) / (dx * dx)
    )
    d2f = d2f.at[0].set((f[1] - 2 * f[0]) / (dx * dx))
    d2f = d2f.at[-1].set((-2 * f[-1] + f[-2]) / (dx * dx))
    return d2f


# @partial(jit, static_argnums=(1, 2))
def first_derivative_fd(f: jnp.ndarray, dx: float, num_grid_points: int) -> jnp.ndarray:
    """Compute first derivative using central finite differences."""
    df = jnp.zeros_like(f)
    interior = jnp.arange(1, num_grid_points - 1)
    df = df.at[interior].set((f[interior + 1] - f[interior - 1]) / (2 * dx))
    df = df.at[0].set((f[1] - f[0]) / dx)
    df = df.at[-1].set((f[-1] - f[-2]) / dx)
    return df


# @partial(jit, static_argnums=(1, 2))
def trapz_integrate(f: jnp.ndarray, dx: float, num_grid_points: int) -> jnp.ndarray:
    """Numerical integration using trapezoidal rule."""
    interior_sum = jnp.sum(f[1:num_grid_points - 1])
    return dx * (0.5 * f[0] + interior_sum + 0.5 * f[num_grid_points - 1])


# @partial(jit, static_argnums=(2, 3))
def inner_product(f: jnp.ndarray, g: jnp.ndarray, dx: float, num_grid_points: int) -> jnp.ndarray:
    """Compute inner product <f|g> = integral f*(x) g(x) dx."""
    integrand = jnp.conj(f) * g
    return trapz_integrate(integrand, dx, num_grid_points)


# =============================================================================
# Double-Well Potential
# =============================================================================

# TODO: Insert the actual potential coeffs

# @jit
def a_2_potential(A: float) -> float:
    return 1
# @jit
def a_4_potential(A: float) -> float:
    return 1

# @jit
def double_well_potential_with_control(x_grid_squared: jnp.ndarray, x_grid_fourth: jnp.ndarray, A: float) -> jnp.ndarray:
    """Double-well potential with control input A determining coefficients."""
    term_quadratic = a_2_potential(A) * x_grid_squared
    term_quartic = a_4_potential(A) * x_grid_fourth
    return term_quadratic + term_quartic


# =============================================================================
# Single-Particle Hamiltonian
# =============================================================================

# @partial(jit, static_argnums=(5))
def single_particle_hamiltonian(phi: jnp.ndarray,
                                 V: jnp.ndarray, hbar: float, m: float,
                                 dx: float, num_grid_points: int) -> jnp.ndarray:
    """Apply single-particle Hamiltonian: h = -hbar^2/(2m) d^2/dx^2 + V(x)"""
    d2phi = second_derivative_fd(phi, dx, num_grid_points)
    kinetic = -(hbar**2 / (2 * m)) * d2phi
    potential = V * phi
    return kinetic + potential


# @partial(jit, static_argnums=(6))
def hamiltonian_matrix_element(phi_i: jnp.ndarray, phi_j: jnp.ndarray, V: jnp.ndarray,
                               hbar: float, m: float,
                               dx: float, num_grid_points: int) -> jnp.ndarray:
    """Compute matrix element <phi_i|h|phi_j>."""
    h_phi_j = single_particle_hamiltonian(phi_j, V, hbar, m, dx, num_grid_points)
    return inner_product(phi_i, h_phi_j, dx, num_grid_points)


# =============================================================================
# Projector for Orthonormality
# =============================================================================

# @partial(jit, static_argnums=(3, 4))
def projector_P(f: jnp.ndarray, phi_g: jnp.ndarray, phi_e: jnp.ndarray,
                dx: float, num_grid_points: int) -> jnp.ndarray:
    """
    Apply projector P = 1 - |phi_g><phi_g| - |phi_e><phi_e|.
    P[f] = f - phi_g * integral(phi_g* f dx) - phi_e * integral(phi_e* f dx)
    """
    overlap_g = inner_product(phi_g, f, dx, num_grid_points)
    overlap_e = inner_product(phi_e, f, dx, num_grid_points)
    projected = f - phi_g * overlap_g - phi_e * overlap_e
    return projected


# @partial(jit, static_argnums=(2, 3))
def gram_schmidt_orthonormalize(phi_g: jnp.ndarray, phi_e: jnp.ndarray,
                                 dx: float, num_grid_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Orthonormalize orbitals using Gram-Schmidt procedure."""
    norm_g = jnp.sqrt(jnp.abs(inner_product(phi_g, phi_g, dx, num_grid_points)))
    phi_g_norm = phi_g / jnp.where(norm_g > 1e-10, norm_g, 1.0)
    
    overlap = inner_product(phi_g_norm, phi_e, dx, num_grid_points)
    phi_e_ortho = phi_e - phi_g_norm * overlap
    
    norm_e = jnp.sqrt(jnp.abs(inner_product(phi_e_ortho, phi_e_ortho, dx, num_grid_points)))
    phi_e_norm = phi_e_ortho / jnp.where(norm_e > 1e-10, norm_e, 1.0)
    
    return phi_g_norm, phi_e_norm


# =====================================================
# Reduced Density Matrices and Non-linear Coefficients
# =====================================================

# @partial(jit, static_argnums=(1,))
def compute_reduced_densities(config_coeffs: jnp.ndarray, N: int) -> Dict[str, jnp.ndarray]:
    """
    Compute reduced density matrix elements from configuration coefficients.
    For two-mode MCTDHB with configurations |k, N-k>.
    """
    num_configs = N + 1
    k_values = jnp.arange(num_configs, dtype=jnp.float32)
    
    probs = jnp.abs(config_coeffs)**2
    
    # One-body densities
    rho_gg = jnp.sum(probs * k_values)
    rho_ee = jnp.sum(probs * (N - k_values))
    
    # Two-body densities (diagonal)
    rho_gggg = jnp.sum(probs * k_values * (k_values - 1))
    rho_eeee = jnp.sum(probs * (N - k_values) * (N - k_values - 1))
    rho_gege = jnp.sum(probs * k_values * (N - k_values))
    
    # Two-body exchange term (off-diagonal)
    k_lower = k_values[:-1]
    k_upper = k_values[1:]
    matrix_elem = jnp.sqrt((k_upper) * (N - k_lower)) * jnp.sqrt(k_lower * (N - k_upper + 1))
    rho_ggee = jnp.sum(jnp.conj(config_coeffs[:-1]) * config_coeffs[1:] * matrix_elem)
    
    return {
        'rho_gg': rho_gg, 'rho_ee': rho_ee,
        'rho_gggg': rho_gggg, 'rho_eeee': rho_eeee,
        'rho_gege': rho_gege, 'rho_ggee': rho_ggee
    }

# @partial(jit, static_argnums=(2,))
def compute_nonlinear_coefficients(densities: Dict[str, jnp.ndarray], 
                                    U0: float, N: int) -> Dict[str, jnp.ndarray]:
    """Compute non-linear coefficients for orbital evolution (Grond Eq. 17)."""
    rho_gg = densities['rho_gg']
    rho_ee = densities['rho_ee']
    rho_gggg = densities['rho_gggg']
    rho_eeee = densities['rho_eeee']
    rho_gege = densities['rho_gege']
    rho_ggee = densities['rho_ggee']
    
    rho_gg_safe = jnp.where(rho_gg > 1e-10, rho_gg, 1.0)
    rho_ee_safe = jnp.where(rho_ee > 1e-10, rho_ee, 1.0)
    
    f_gg = U0 * rho_gggg / rho_gg_safe
    f_ge = 2 * U0 * rho_gege / rho_gg_safe
    f_tilde_g = U0 * rho_ggee / rho_gg_safe
    
    f_ee = U0 * rho_eeee / rho_ee_safe
    f_eg = 2 * U0 * rho_gege / rho_ee_safe
    f_tilde_e = U0 * jnp.conj(rho_ggee) / rho_ee_safe
    
    return {
        'f_gg': f_gg, 'f_ge': f_ge, 'f_tilde_g': f_tilde_g,
        'f_ee': f_ee, 'f_eg': f_eg, 'f_tilde_e': f_tilde_e
    }


# =============================================================================
# Orbital Evolution (Grond Eq. 16)
# =============================================================================

# @partial(jit, static_argnums=(7))
def compute_orbital_rhs(phi_g: jnp.ndarray, phi_e: jnp.ndarray, V: jnp.ndarray,
                        nonlinear_coeffs: Dict[str, jnp.ndarray],
                        hbar: float, m: float, dx: float, 
                        num_grid_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute right-hand side of orbital evolution equations (Grond Eq. 16).
    i*hbar d(phi_g)/dt = P [h*phi_g + (f_gg*|phi_g|^2 + f_ge*|phi_e|^2)*phi_g + f_tilde_g * phi_g* phi_e^2]
    i*hbar d(phi_e)/dt = P [h*phi_e + (f_eg*|phi_g|^2 + f_ee*|phi_e|^2)*phi_e + f_tilde_e * phi_e* phi_g^2]
    """
    f_gg = nonlinear_coeffs['f_gg']
    f_ge = nonlinear_coeffs['f_ge']
    f_tilde_g = nonlinear_coeffs['f_tilde_g']
    f_ee = nonlinear_coeffs['f_ee']
    f_eg = nonlinear_coeffs['f_eg']
    f_tilde_e = nonlinear_coeffs['f_tilde_e']
    
    n_g = jnp.abs(phi_g)**2
    n_e = jnp.abs(phi_e)**2
    
    # Ground orbital RHS
    h_phi_g = single_particle_hamiltonian(phi_g, V, hbar, m, dx, num_grid_points)
    mean_field_g = (f_gg * n_g + f_ge * n_e) * phi_g
    exchange_g = f_tilde_g * jnp.conj(phi_g) * phi_e**2
    rhs_g_unprojected = h_phi_g + mean_field_g + exchange_g
    
    # Excited orbital RHS
    h_phi_e = single_particle_hamiltonian(phi_e, V, hbar, m, dx, num_grid_points)
    mean_field_e = (f_eg * n_g + f_ee * n_e) * phi_e
    exchange_e = f_tilde_e * jnp.conj(phi_e) * phi_g**2
    rhs_e_unprojected = h_phi_e + mean_field_e + exchange_e
    
    # Apply projector
    rhs_g = projector_P(rhs_g_unprojected, phi_g, phi_e, dx, num_grid_points)
    rhs_e = projector_P(rhs_e_unprojected, phi_g, phi_e, dx, num_grid_points)
    
    # Time derivative: d(phi)/dt = -i * rhs / hbar
    dphi_g_dt = -1j * rhs_g / hbar
    dphi_e_dt = -1j * rhs_e / hbar
    
    return dphi_g_dt, dphi_e_dt


# @partial(jit, static_argnums=(7))
def evolve_orbitals(phi_g: jnp.ndarray, phi_e: jnp.ndarray, V: jnp.ndarray,
                    nonlinear_coeffs: Dict[str, jnp.ndarray],
                    hbar: float, m: float, dx: float,
                    num_grid_points: int, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evolve orbitals using first-order Euler with normalization."""
    dphi_g_dt, dphi_e_dt = compute_orbital_rhs(
        phi_g, phi_e, V, nonlinear_coeffs,
        hbar, m, dx, num_grid_points
    )
    
    phi_g_new = phi_g + dt * dphi_g_dt
    phi_e_new = phi_e + dt * dphi_e_dt
    
    # # Normalize and orthonormalize
    # norm_g = jnp.sqrt(jnp.abs(inner_product(phi_g_new, phi_g_new, dx, num_grid_points)))
    # norm_e = jnp.sqrt(jnp.abs(inner_product(phi_e_new, phi_e_new, dx, num_grid_points)))
    
    # phi_g_new = phi_g_new / jnp.where(norm_g > 1e-10, norm_g, 1.0)
    # phi_e_new = phi_e_new / jnp.where(norm_e > 1e-10, norm_e, 1.0)
    
    phi_g_new, phi_e_new = gram_schmidt_orthonormalize(
        phi_g_new, phi_e_new, dx, num_grid_points
    )
    
    return phi_g_new, phi_e_new


# =============================================================================
# Two-Body Matrix Elements (Grond Eq. 18)
# =============================================================================

# @partial(jit, static_argnums=(5))
def compute_two_body_integral(phi_i: jnp.ndarray, phi_j: jnp.ndarray,
                               phi_k: jnp.ndarray, phi_l: jnp.ndarray,
                               dx: float, num_grid_points: int) -> jnp.ndarray:
    """
    Compute two-body matrix element W_{ijkl} = U0 * integral dx phi_i* phi_j* phi_k phi_l.
    For contact interaction g delta(x-x'), this simplifies to U0 * integral phi_i* phi_j* phi_k phi_l dx.
    """
    integrand = jnp.conj(phi_i) * jnp.conj(phi_j) * phi_k * phi_l
    return trapz_integrate(integrand, dx, num_grid_points)


# @partial(jit, static_argnums=(4))
def compute_all_two_body_matrix_elements(phi_g: jnp.ndarray, phi_e: jnp.ndarray, U0: float, dx: float, num_grid_points: int) -> Dict[str, jnp.ndarray]:
    """
    Compute all relevant two-body matrix elements for two-mode system.
    W_ijkl is pairwise-symmetric: W_ijkl = W_jikl = W_ijlk.
    This function computes all 16 combinations exhaustively using symmetry.
    """
    # Compute unique integrals based on symmetry pairs (i,j) and (k,l)
    # Unique pairs for (i,j): (g,g), (g,e), (e,e)
    # Unique pairs for (k,l): (g,g), (g,e), (e,e)
    # Total unique integrals: 9

    # Pair (g,g) for first two indices
    I_gg_gg = U0 * compute_two_body_integral(phi_g, phi_g, phi_g, phi_g, dx, num_grid_points)
    I_gg_ge = U0 * compute_two_body_integral(phi_g, phi_g, phi_g, phi_e, dx, num_grid_points)
    I_gg_ee = U0 * compute_two_body_integral(phi_g, phi_g, phi_e, phi_e, dx, num_grid_points)

    # Pair (g,e) for first two indices (symmetric to (e,g))
    I_ge_gg = U0 * compute_two_body_integral(phi_g, phi_e, phi_g, phi_g, dx, num_grid_points)
    I_ge_ge = U0 * compute_two_body_integral(phi_g, phi_e, phi_g, phi_e, dx, num_grid_points)
    I_ge_ee = U0 * compute_two_body_integral(phi_g, phi_e, phi_e, phi_e, dx, num_grid_points)

    # Pair (e,e) for first two indices
    I_ee_gg = U0 * compute_two_body_integral(phi_e, phi_e, phi_g, phi_g, dx, num_grid_points)
    I_ee_ge = U0 * compute_two_body_integral(phi_e, phi_e, phi_g, phi_e, dx, num_grid_points)
    I_ee_ee = U0 * compute_two_body_integral(phi_e, phi_e, phi_e, phi_e, dx, num_grid_points)
    
    # Map all 16 combinations to the unique integrals based on symmetry W_ijkl = W_jikl = W_ijlk
    return {
        'W_gggg': I_gg_gg,
        'W_ggge': I_gg_ge,
        'W_ggee': I_gg_ee,
        'W_geee': I_ge_ee,
        'W_eggg': I_ge_gg,
        'W_egeg': I_ge_ge,
        'W_eegg': I_ee_gg,
        'W_eeeg': I_ee_ge,
        'W_eeee': I_ee_ee
    }
    # all other combinations are unused and thus omitted


# =============================================================================
# Tunnel Coupling Omega (Grond Eq. 19)
# =============================================================================

# @partial(jit, static_argnums=(6))
def compute_tunnel_coupling(phi_g: jnp.ndarray, phi_e: jnp.ndarray, V: jnp.ndarray,
                             hbar: float, m: float,
                             dx: float, num_grid_points: int) -> jnp.ndarray:
    """
    Compute tunnel coupling Omega = <phi_e|h|phi_e> - <phi_g|h|phi_g>.
    This is the energy difference between the two orbitals.
    """
    h_gg = hamiltonian_matrix_element(phi_g, phi_g, V, hbar, m, dx, num_grid_points)
    h_ee = hamiltonian_matrix_element(phi_e, phi_e, V, hbar, m, dx, num_grid_points)
    
    # Omega is the tunnel coupling (energy splitting)
    Omega = jnp.real(h_ee - h_gg)
    
    return Omega


# =============================================================================
# Coefficient Evolution (Grond Eq. 19-20)
# =============================================================================

# # @partial(jit, static_argnums=(2,))
# def construct_two_mode_hamiltonian(Omega: float, two_body_W: Dict[str, jnp.ndarray],
#                                     N: int) -> jnp.ndarray:
#     """
#     Construct the Hamiltonian matrix for coefficient evolution.
    
#     H = -Omega J_x + (1/2) Sum' a^dagger_k a^dagger_q a_l a_m W_{kqlm},
#     where J_x = (1/2)(a^dagger_g a_g - a^dagger_e a_e) is the population imbalance operator.
    
#     Hence, a configuration |k, N-k> =: |k> is mapped to imbalance + interation term:
#     imbalance term = -Omega*(k-N/2)*|k>
#     interaction term = sqrt(k*(k-1)*(N-k+1)*(N-k+2))*W_eegg*|k-2>
#     + ( 2*(N-k)*sqrt(k*(N-k+1))*W_eeeg + 2*(k-1)*sqrt(k*(N-k+1))*W_eggg )*|k-1>
#     + ( k*(k-1)*W_gggg + (N-k)*(N-k-1)*W_eeee + 4*k*(N-k)*W_egeg )*|k>
#     + ( 2*(N-k-1)*sqrt((N-k)*(k+1))*W_geee + 2*k*sqrt((N-k)*(k+1))*W_ggge )*|k+1>
#     + sqrt( (N-k)*(N-k-1)*(k+1)*(k+2) )*W_ggee*|k+2>
#     """
#     num_configs = N + 1
#     k_values = jnp.arange(num_configs, dtype=jnp.float32)
    
#     # Initialize Hamiltonian matrix
#     H_mat = jnp.zeros((num_configs, num_configs), dtype=jnp.complex128)
    
#     # Extract interaction coefficients
#     W_gggg = two_body_W['W_gggg']
#     W_eeee = two_body_W['W_eeee']
#     W_egeg = two_body_W['W_egeg']
#     W_eegg = two_body_W['W_eegg']
#     W_eeeg = two_body_W['W_eeeg']
#     W_eggg = two_body_W['W_eggg']
#     W_geee = two_body_W['W_geee']
#     W_ggge = two_body_W['W_ggge']
#     W_ggee = two_body_W['W_ggee']
    
#     # --- Diagonal Terms (k -> k) ---
#     # Imbalance term: -Omega * (k - N/2)
#     imbalance = -Omega * (k_values - N / 2.0)
    
#     # Interaction diagonal: k*(k-1)*W_gggg + (N-k)*(N-k-1)*W_eeee + 4*k*(N-k)*W_egeg
#     interaction_diag = (
#         k_values * (k_values - 1) * W_gggg +
#         (N - k_values) * (N - k_values - 1) * W_eeee +
#         4 * k_values * (N - k_values) * W_egeg
#     )
    
#     diag_total = imbalance + interaction_diag
#     H_mat = H_mat.at[jnp.arange(num_configs), jnp.arange(num_configs)].set(diag_total)
    
#     # --- Lower Diagonal 1 (k -> k-1) ---
#     # Valid for k >= 1
#     k_lower1 = jnp.arange(1, num_configs)
#     row_lower1 = k_lower1 - 1
    
#     # Term: 2*(N-k)*sqrt(k*(N-k+1))*W_eeeg + 2*(k-1)*sqrt(k*(N-k+1))*W_eggg
#     sqrt_fact_lower1 = jnp.sqrt(k_lower1 * (N - k_lower1 + 1))
#     term1_lower1 = 2 * (N - k_lower1) * sqrt_fact_lower1 * W_eeeg
#     term2_lower1 = 2 * (k_lower1 - 1) * sqrt_fact_lower1 * W_eggg
#     val_lower1 = term1_lower1 + term2_lower1
    
#     H_mat = H_mat.at[row_lower1, k_lower1].set(val_lower1)
    
#     # --- Lower Diagonal 2 (k -> k-2) ---
#     # Valid for k >= 2
#     k_lower2 = jnp.arange(2, num_configs)
#     row_lower2 = k_lower2 - 2
    
#     # Term: sqrt(k*(k-1)*(N-k+1)*(N-k+2))*W_eegg
#     sqrt_fact_lower2 = jnp.sqrt(
#         k_lower2 * (k_lower2 - 1) * (N - k_lower2 + 1) * (N - k_lower2 + 2)
#     )
#     val_lower2 = sqrt_fact_lower2 * W_eegg
    
#     H_mat = H_mat.at[row_lower2, k_lower2].set(val_lower2)
    
#     # --- Upper Diagonal 1 (k -> k+1) ---
#     # Valid for k <= N-1
#     k_upper1 = jnp.arange(0, num_configs - 1)
#     row_upper1 = k_upper1 + 1
    
#     # Term: 2*(N-k-1)*sqrt((N-k)*(k+1))*W_geee + 2*k*sqrt((N-k)*(k+1))*W_ggge
#     sqrt_fact_upper1 = jnp.sqrt((N - k_upper1) * (k_upper1 + 1))
#     term1_upper1 = 2 * (N - k_upper1 - 1) * sqrt_fact_upper1 * W_geee
#     term2_upper1 = 2 * k_upper1 * sqrt_fact_upper1 * W_ggge
#     val_upper1 = term1_upper1 + term2_upper1
    
#     H_mat = H_mat.at[row_upper1, k_upper1].set(val_upper1)
    
#     # --- Upper Diagonal 2 (k -> k+2) ---
#     # Valid for k <= N-2
#     k_upper2 = jnp.arange(0, num_configs - 2)
#     row_upper2 = k_upper2 + 2
    
#     # Term: sqrt((N-k)*(N-k-1)*(k+1)*(k+2))*W_ggee
#     sqrt_fact_upper2 = jnp.sqrt(
#         (N - k_upper2) * (N - k_upper2 - 1) * (k_upper2 + 1) * (k_upper2 + 2)
#     )
#     val_upper2 = sqrt_fact_upper2 * W_ggee
    
#     H_mat = H_mat.at[row_upper2, k_upper2].set(val_upper2)
    
#     return H_mat


# # @jit
# def evolve_coefficients(config_coeffs: jnp.ndarray, H_mat: jnp.ndarray,
#                          dt: float, hbar: float) -> jnp.ndarray:
#     """
#     Evolve configuration coefficients: i*hbar dC/dt = H C
    
#     Using first-order Euler: C(t+dt) = C(t) - i H C dt / hbar
#     """
#     dC_dt = -1j * H_mat @ config_coeffs / hbar
#     new_coeffs = config_coeffs + dt * dC_dt
    
#     # Normalize
#     norm = jnp.linalg.norm(new_coeffs)
#     new_coeffs = new_coeffs / jnp.where(norm > 1e-10, norm, 1.0)
    
#     return new_coeffs


# @partial(jit, static_argnums=(1,))
def construct_banded_hamiltonian(Omega: float, two_body_W: Dict[str, jnp.ndarray],
                                  N: int) -> Dict[str, jnp.ndarray]:
    """
    Construct banded Hamiltonian as separate diagonal arrays.
    Returns dict with 'diag', 'lower1', 'lower2', 'upper1', 'upper2'.
    """
    num_configs = N + 1
    k_values = jnp.arange(num_configs, dtype=jnp.float32)
    
    # Extract interaction coefficients
    W_gggg = two_body_W['W_gggg']
    W_eeee = two_body_W['W_eeee']
    W_egeg = two_body_W['W_egeg']
    W_eegg = two_body_W['W_eegg']
    W_eeeg = two_body_W['W_eeeg']
    W_eggg = two_body_W['W_eggg']
    W_geee = two_body_W['W_geee']
    W_ggge = two_body_W['W_ggge']
    W_ggee = two_body_W['W_ggee']
    
    # --- Diagonal (k -> k) ---
    imbalance = -Omega * (k_values - N / 2.0)
    interaction_diag = (
        k_values * (k_values - 1) * W_gggg +
        (N - k_values) * (N - k_values - 1) * W_eeee +
        4 * k_values * (N - k_values) * W_egeg
    )
    diag = imbalance + interaction_diag
    
    # --- Lower Diagonal 1 (k -> k-1), length N ---
    k_lower1 = jnp.arange(1, num_configs, dtype=jnp.float32)
    sqrt_fact_lower1 = jnp.sqrt(k_lower1 * (N - k_lower1 + 1))
    lower1 = 2 * (N - k_lower1) * sqrt_fact_lower1 * W_eeeg + \
             2 * (k_lower1 - 1) * sqrt_fact_lower1 * W_eggg
    
    # --- Lower Diagonal 2 (k -> k-2), length N-1 ---
    k_lower2 = jnp.arange(2, num_configs, dtype=jnp.float32)
    sqrt_fact_lower2 = jnp.sqrt(k_lower2 * (k_lower2 - 1) * (N - k_lower2 + 1) * (N - k_lower2 + 2))
    lower2 = sqrt_fact_lower2 * W_eegg
    
    # --- Upper Diagonal 1 (k -> k+1), length N ---
    k_upper1 = jnp.arange(0, num_configs - 1, dtype=jnp.float32)
    sqrt_fact_upper1 = jnp.sqrt((N - k_upper1) * (k_upper1 + 1))
    upper1 = 2 * (N - k_upper1 - 1) * sqrt_fact_upper1 * W_geee + \
             2 * k_upper1 * sqrt_fact_upper1 * W_ggge
    
    # --- Upper Diagonal 2 (k -> k+2), length N-1 ---
    k_upper2 = jnp.arange(0, num_configs - 2, dtype=jnp.float32)
    sqrt_fact_upper2 = jnp.sqrt((N - k_upper2) * (N - k_upper2 - 1) * (k_upper2 + 1) * (k_upper2 + 2))
    upper2 = sqrt_fact_upper2 * W_ggee
    
    return {
        'diag': diag,
        'lower1': lower1,
        'lower2': lower2,
        'upper1': upper1,
        'upper2': upper2,
    }


# @jit
def banded_hamiltonian_action(C: jnp.ndarray, H_banded: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Compute H @ C for pentadiagonal H without forming dense matrix.
    O(N) operations instead of O(N^2).
    """
    diag = H_banded['diag']
    lower1 = H_banded['lower1']
    lower2 = H_banded['lower2']
    upper1 = H_banded['upper1']
    upper2 = H_banded['upper2']
    
    result = diag * C
    
    # Lower diagonals
    result = result.at[1:].add(lower1 * C[:-1])
    result = result.at[2:].add(lower2 * C[:-2])
    
    # Upper diagonals (note: upper1[i] connects i to i+1)
    result = result.at[:-1].add(upper1 * C[1:])
    result = result.at[:-2].add(upper2 * C[2:])
    
    return result


# @jit
def evolve_coefficients_banded(config_coeffs: jnp.ndarray, H_banded: Dict[str, jnp.ndarray],
                                dt: float, hbar: float) -> jnp.ndarray:
    """
    Evolve coefficients using banded Hamiltonian.
    """
    HC = banded_hamiltonian_action(config_coeffs, H_banded)
    dC_dt = -1j * HC / hbar
    new_coeffs = config_coeffs + dt * dC_dt
    
    norm = jnp.linalg.norm(new_coeffs)
    new_coeffs = new_coeffs / jnp.where(norm > 1e-10, norm, 1.0)
    
    return new_coeffs


# =============================================================================
# Self-Consistent Time Step
# =============================================================================

# @jit
def mctdhb_single_step(state: MCTDHBState, params: MCTDHBParams,
                       V: jnp.ndarray) -> MCTDHBState:
    """
    Perform one self-consistent MCTDHB time step.
    
    a. Compute reduced densities from current C(t)
    b. Compute non-linear coefficients f from densities
    c. Evolve orbitals phi_g, phi_e using orbital PDE
    d. Compute Omega and W_{kqlm} from new orbitals
    e. Evolve coefficients C using coefficient equation
    f. Ensure orbital orthonormality
    """
    phi_g, phi_e, config_coeffs = state
    N = params.N
    hbar = params.hbar
    m = params.m
    U0 = params.U0
    dt = params.dt
    dx = params.dx
    num_grid_points = params.num_grid_points
    
    # Step a: Compute reduced densities from coefficients
    densities = compute_reduced_densities(config_coeffs, N)
    
    # Step b: Compute non-linear coefficients
    nonlinear_coeffs = compute_nonlinear_coefficients(densities, U0, N)
    
    # Step c: Evolve orbitals
    phi_g_new, phi_e_new = evolve_orbitals(
        phi_g, phi_e, V, nonlinear_coeffs,
        hbar, m, dx, num_grid_points, dt
    )
    
    # Step d: Compute matrix elements from new orbitals
    Omega = compute_tunnel_coupling(
        phi_g_new, phi_e_new, V, hbar, m, dx, num_grid_points
    )
    
    two_body_W = compute_all_two_body_matrix_elements(
        phi_g_new, phi_e_new, U0, dx, num_grid_points
    )
    
    # Step e: Construct Hamiltonian (banded version) and evolve coefficients
    H_mat = construct_banded_hamiltonian(Omega, two_body_W, N)
    config_coeffs_new = evolve_coefficients_banded(config_coeffs, H_mat, dt, hbar)
    
    # Step f: Orbitals already orthonormalized in evolve_orbitals
    
    return MCTDHBState(phi_g_new, phi_e_new, config_coeffs_new)


# =============================================================================
# Initial Conditions
# =============================================================================

def initialize_orbitals_harmonic_oscillator(params: MCTDHBParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Initialize orbitals as ground and first excited states of harmonic oscillator.
    
    phi_0(x) = (m*omega/(pi*hbar))^(1/4) exp(-m*omega x^2 / (2*hbar))
    phi_1(x) = (m*omega/(pi*hbar))^(1/4) sqrt(2*m*omega/hbar) x exp(-m*omega x^2 / (2*hbar))
    
    These are then adapted to the double-well potential.
    """
    x_grid = params.x_grid
    hbar = params.hbar
    m = params.m
    omega = params.omega
    
    # Harmonic oscillator length
    a_ho = jnp.sqrt(hbar / (m * omega))
    
    # Ground state
    norm_factor = (1.0 / (jnp.pi * a_ho**2))**0.25
    phi_g = norm_factor * jnp.exp(-x_grid**2 / (2 * a_ho**2))
    
    # First excited state
    phi_e = norm_factor * jnp.sqrt(2) * (x_grid / a_ho) * jnp.exp(-x_grid**2 / (2 * a_ho**2))
    
    # Normalize
    dx = params.dx
    num_grid_points = params.num_grid_points
    
    norm_g = jnp.sqrt(jnp.abs(inner_product(phi_g, phi_g, dx, num_grid_points)))
    norm_e = jnp.sqrt(jnp.abs(inner_product(phi_e, phi_e, dx, num_grid_points)))
    
    phi_g = phi_g / norm_g
    phi_e = phi_e / norm_e
    
    # Orthonormalize
    phi_g, phi_e = gram_schmidt_orthonormalize(phi_g, phi_e, dx, num_grid_points)
    
    return phi_g, phi_e


def initialize_coherent_state_two_mode(N: int, initial_n: float = 0.0, 
                                        initial_phase: float = 0.0) -> jnp.ndarray:
    """
    Initialize configuration coefficients for coherent state in two-mode system.
    
    Configurations are |k, N-k> where k=0,...,N.
    Coherent state has binomial distribution centered at k = N/2.
    """
    num_configs = N + 1
    k_values = jnp.arange(num_configs, dtype=jnp.float32)
    
    # Coherent state: binomial distribution
    # For large N, use normal approximation
    mean = N / 2.0
    std = jnp.sqrt(N / 4.0)
    
    binomial_coeffs = jnp.exp(-0.5 * ((k_values - mean) / std)**2)
    
    # Normalize
    binomial_coeffs = binomial_coeffs / jnp.linalg.norm(binomial_coeffs)
    
    # Add phase gradient for initial population imbalance
    phases = initial_phase * (k_values - N / 2)
    config_coeffs = binomial_coeffs * jnp.exp(1j * phases)
    
    # Normalize again
    config_coeffs = config_coeffs / jnp.linalg.norm(config_coeffs)
    
    return config_coeffs


def mctdhb_initialize(params: MCTDHBParams, initial_n: float = 0.0,
                      initial_phase: float = 0.0) -> MCTDHBState:
    """
    Initialize complete MCTDHB state.
    
    Parameters:
        params: MCTDHB parameters
        initial_n: Initial population imbalance (-1 to 1)
        initial_phase: Initial relative phase
    
    Returns:
        Initial MCTDHBState
    """
    # Initialize orbitals
    phi_g, phi_e = initialize_orbitals_harmonic_oscillator(params)
    
    # Initialize coefficients
    config_coeffs = initialize_coherent_state_two_mode(
        params.N, initial_n, initial_phase
    )
    
    return MCTDHBState(phi_g, phi_e, config_coeffs)


# =============================================================================
# Observables Computation
# =============================================================================

def compute_observables(state: MCTDHBState, params: MCTDHBParams,
                        V: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute physical observables from MCTDHB state.
    
    Returns dictionary with:
    - n_expect: Population imbalance
    - n_variance: Variance of population imbalance
    - squeezing_parameter: N * Var(n)
    - phi_expect: Relative phase
    - energy_expect: Total energy
    - orbital_overlap: <phi_g|phi_e>
    """
    phi_g, phi_e, config_coeffs = state
    N = params.N
    hbar = params.hbar
    m = params.m
    dx = params.dx
    num_grid_points = params.num_grid_points
    x_grid = params.x_grid
    
    num_configs = N + 1
    k_values = jnp.arange(num_configs, dtype=jnp.float32)
    
    # Population imbalance from coefficients
    probs = jnp.abs(config_coeffs)**2
    n_values = (2 * k_values - N) / N  # Normalized to [-1, 1]
    
    n_expect = jnp.sum(probs * n_values)
    n_squared_expect = jnp.sum(probs * n_values**2)
    n_variance = n_squared_expect - n_expect**2
    
    # Squeezing parameter
    squeezing_parameter = N * n_variance
    
    # Relative phase from coherence
    if num_configs > 1:
        weights = jnp.sqrt((N - k_values[:-1]) * (k_values[1:]))
        rho_01 = jnp.sum(jnp.conj(config_coeffs[:-1]) * config_coeffs[1:] * weights)
        phi_expect = jnp.angle(rho_01)
    else:
        phi_expect = 0.0
    
    # Orbital overlap
    orbital_overlap = inner_product(phi_g, phi_e, dx, num_grid_points)
    
    # Energy expectation
    # E = <Psi|H|Psi> = Sum_k |C_k|^2 E_k + orbital contributions
    densities = compute_reduced_densities(config_coeffs, N)
    
    # Single-particle energy
    h_gg = hamiltonian_matrix_element(phi_g, phi_g, V, hbar, m, dx, num_grid_points)
    h_ee = hamiltonian_matrix_element(phi_e, phi_e, V, hbar, m, dx, num_grid_points)
    
    single_particle_energy = densities['rho_gg'] * jnp.real(h_gg) + densities['rho_ee'] * jnp.real(h_ee)
    
    # Interaction energy
    two_body_W = compute_all_two_body_matrix_elements(
        phi_g, phi_e, params.U0, dx, num_grid_points
    )
    
    interaction_energy = (
        0.5 * two_body_W['W_gggg'] * densities['rho_gggg'] +
        0.5 * two_body_W['W_eeee'] * densities['rho_eeee'] +
        two_body_W['W_ggee'] * densities['rho_gege']
    )
    
    energy_expect = single_particle_energy + jnp.real(interaction_energy)
    
    return {
        'n_expect': n_expect,
        'n_variance': n_variance,
        'squeezing_parameter': squeezing_parameter,
        'phi_expect': phi_expect,
        'energy_expect': energy_expect,
        'orbital_overlap': orbital_overlap
    }


# =============================================================================
# Full Simulation
# =============================================================================

def mctdhb_simulate(A_array: jnp.ndarray, params: MCTDHBParams,
                    initial_state: MCTDHBState) -> MCTDHB_Results:
    """
    Run full MCTDHB simulation with time-dependent control.
    
    Parameters:
        J_array: Tunnel coupling control over time
        U_array: Interaction strength control over time
        params: MCTDHB parameters
        initial_state: Initial state (optional, will initialize if not provided)
    
    Returns:
        MCTDHB_Results object with simulation results
    """
    num_steps = len(A_array)
    
    # Create time array
    dt = params.dt
    t_array = jnp.arange(num_steps) * dt

    #Compute powers of x for the potential V(x)
    x_grid_squared = params.x_grid**2
    x_grid_fourth  = params.x_grid**4
    
    # Storage for trajectories
    phi_g_traj = []
    phi_e_traj = []
    config_coeffs_traj = []
    observables_list = []
    
    # Current state
    state = initial_state

    # Run simulation
    for step in range(num_steps):
        # Get control values for this step
        A_control = A_array[step] if step < len(A_array) else A_array[-1]
        
        # Compute potential with control
        V = double_well_potential_with_control(x_grid_squared, x_grid_fourth, A_control)
        
        # Store current state
        phi_g_traj.append(state.phi_g)
        phi_e_traj.append(state.phi_e)
        config_coeffs_traj.append(state.config_coeffs)
        
        # Compute observables
        obs = compute_observables(state, params, V)
        observables_list.append(obs)

        # Perform time step
        state = mctdhb_single_step(state, params, V)
        
    # Create results object
    results = MCTDHB_Results(t_array, params)
    
    # Store trajectories
    results.phi_g_trajectory = jnp.array(phi_g_traj)
    results.phi_e_trajectory = jnp.array(phi_e_traj)
    results.config_coeffs_trajectory = jnp.array(config_coeffs_traj)
    
    # Store observables
    results.n_expect = jnp.array([obs['n_expect'] for obs in observables_list])
    results.n_variance = jnp.array([obs['n_variance'] for obs in observables_list])
    results.squeezing_parameter = jnp.array([obs['squeezing_parameter'] for obs in observables_list])
    results.phi_expect = jnp.array([obs['phi_expect'] for obs in observables_list])
    results.energy_expect = jnp.array([obs['energy_expect'] for obs in observables_list])
    results.orbital_overlap = jnp.array([obs['orbital_overlap'] for obs in observables_list])
    
    # Store control fields
    results.A_array = A_array
    
    return results


# =============================================================================
# Demonstration and Testing
# =============================================================================

def run_mctdhb_demo():
    """Run a demonstration MCTDHB simulation."""
    print("=" * 60)
    print("Full MCTDHB Simulation Demo (Grond et al. 2009)")
    print("=" * 60)
    
    # System parameters
    params = MCTDHBParams(
        N=5000,           # Particle number (smaller for demo)
        hbar=1.0,        # Reduced Planck constant
        m=1.0,           # Particle mass
        U0=0.1,          # Interaction strength
        
        # Spatial grid
        x_min=-10.0,
        x_max=10.0,
        num_grid_points=256,
        
        # Double-well potential
        V0=5.0,          # Barrier height
        a=1.0,           # Barrier width
        omega=0.5,       # Trap frequency
        
        # Time stepping
        dt=0.01,
        num_steps=30
    )
    
    print(f"Particle number N = {params.N}")
    print(f"Grid points = {params.num_grid_points}")
    print(f"Grid spacing dx = {params.dx:.4f}")
    print(f"Time step dt = {params.dt}")
    print(f"Total time = {params.dt * params.num_steps:.2f}")
    
    # Control fields (constant for demo)
    A_array = jnp.ones(params.num_steps) * 1.0
    
    # Initialize state
    print("\nInitializing MCTDHB state...")
    initial_state = mctdhb_initialize(params, initial_n=0.0, initial_phase=0.0)
    
    # Verify orbital orthonormality
    dx = params.dx
    num_grid_points = params.num_grid_points
    
    norm_g = jnp.sqrt(jnp.abs(inner_product(initial_state.phi_g, initial_state.phi_g, dx, num_grid_points)))
    norm_e = jnp.sqrt(jnp.abs(inner_product(initial_state.phi_e, initial_state.phi_e, dx, num_grid_points)))
    overlap = inner_product(initial_state.phi_g, initial_state.phi_e, dx, num_grid_points)
    
    print(f"Initial orbital norms: |phi_g| = {norm_g:.6f}, |phi_e| = {norm_e:.6f}")
    print(f"Initial orbital overlap: <phi_g|phi_e> = {overlap:.6e}")
    
    # Run simulation
    print("\nRunning MCTDHB simulation...")
    timer = Timer()
    timer.start()
    
    results = mctdhb_simulate(A_array, params, initial_state)
    
    elapsed = timer.stop()
    print(f"Simulation completed in {elapsed:.2f} seconds")
    
    # Print final results
    print("\nFinal Results:")
    print(f"  Population imbalance: n = {results.n_expect[-1]:.6f}")
    print(f"  Variance: Var(n) = {results.n_variance[-1]:.6e}")
    print(f"  Squeezing parameter: xi^2 = {results.squeezing_parameter[-1]:.6e}")
    print(f"  Energy: E = {results.energy_expect[-1]:.6f}")
    print(f"  Orbital overlap: |<phi_g|phi_e>| = {jnp.abs(results.orbital_overlap[-1]):.6e}")
    
    # Plot results
    print("\nGenerating plots...")
    results.plot_dynamics()
    
    # # Save results
    # results.save_to_csv("mctdhb_demo_results.csv")
    
    return results


if __name__ == "__main__":
    print("Full MCTDHB Simulation based on Grond et al. (2009)")
    print()
    
    # # Run component tests first
    # test_mctdhb_components()
    
    # print()
    
    # Run demo simulation
    results = run_mctdhb_demo()
    
    print("\nSimulation complete!")
