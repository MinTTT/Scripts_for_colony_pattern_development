# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# %%
# Built-in/Generic Imports
import shutil
import copy
import numpy as np  # Or any other
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm
import sciplot as splt
import json
import warnings
import multiprocessing
from colonyPhaseFieldUtilities import *

# from numba import jit

splt.whitegrid()

ndarray = np.ndarray  # type: object

from numba import jit, njit, prange

gpu = True

if gpu:
    from torch.fft import ifft2 as tor_ifft2, fft2 as tor_fft2
    from scipy.fft import fftfreq
    import torch

    # mem_used = get_gpu_memory()
    # selected_gup_index = np.argmin(np.array(mem_used))
    #
    # device = f"cuda:{selected_gup_index}"
    # print(f'Using GPU: {device}')
else:
    # if platform.system() == 'Windows':
    #     from scipy.fft import fftfreq, fft2, ifft2
    # else:
    #     from pyfftw.interfaces.scipy_fftpack import ifft2, fft2, fftfreq
    #     import pyfftw
    #     cpus = multiprocessing.cpu_count()
    #     pyfftw.config.NUM_THREADS = cpus
    from scipy.fft import fftfreq, fft2, ifft2, set_workers

    set_workers(multiprocessing.cpu_count())
    device = None

np.seterr(divide='ignore', invalid='ignore')


@njit(fastmath=True, cache=True)
def compute_curv_2d_cart_iso(phi: np.ndarray, dr: float, dz: float,
                             phi_min: float = 0.1):
    """
    Calculate the curve of phi. - div( grad phi / |grad phi|)
    Parameters
    ----------
    phi :
    dr :
    dz :

    Returns
    -------

    """
    # discrete method 1
    par_phi_r, par_phi_z = grad_2d_cart_iso(phi, dr, dz, 0)
    abs_grad_phi = np.sqrt(par_phi_r ** 2 + par_phi_z ** 2)
    update_box = abs_grad_phi > phi_min ** 2
    ud_i = np.nonzero(update_box)
    # ud_0 = ud_i[0]
    # ud_1 = ud_i[1]

    norm_par_phi_r = np.zeros(phi.shape, dtype=phi.dtype)
    norm_par_phi_z = np.zeros(phi.shape, dtype=phi.dtype)
    phi_r_update = get_value_2d(par_phi_r, ud_i) / get_value_2d(abs_grad_phi, ud_i)
    norm_par_phi_r = change_value_2d(norm_par_phi_r, phi_r_update, ud_i)
    # norm_par_phi_r[ud_i] = par_phi_r[ud_i] / abs_grad_phi[ud_i]
    phi_z_update = get_value_2d(par_phi_z, ud_i) / get_value_2d(abs_grad_phi, ud_i)
    norm_par_phi_z = change_value_2d(norm_par_phi_z, phi_z_update, ud_i)
    # norm_par_phi_z[ud_i] = par_phi_z[ud_i] / abs_grad_phi[ud_i]

    curv = - div_U_2d_cart_iso(norm_par_phi_r, norm_par_phi_z, dr, dz)

    return curv


@njit(fastmath=True, cache=True)
def compute_WChi(chi: np.ndarray, Adh: float, delta: float, g: float, z: np.ndarray, zs: float):
    """
    Interation potential between biofilm and substrate.

    W(Chi) = - 2. * Adh * free_energy_G(chi) / delta + g * Chi(z, zs, delta)

    Parameters
    ----------z
    chi : np.ndarray
        substrate field
    Adh :
        substrate adhesion coefficient
    delta :

    g :
    z :
    zs

    Returns
    -------

    """
    epsilon = 2. * delta
    adh_z = z + epsilon  # shift_repulsive_potential
    z_rep = zs

    w_chi = - 2. * Adh * free_energy_G(chi) / delta + g * compute_Chi(adh_z, z_rep, delta) / 2
    return w_chi


@njit(fastmath=True, cache=True)
def compute_Chi(z: np.ndarray, zs: float = 0., delta: float = .1) -> np.ndarray:
    """
    statistic phase field Chi(z). Chi = 0 if substrate.

    Parameters
    ----------
    z : np.ndarray
        z coordinate.
    zs : float
        shift the field z position.
    delta :
        void-substrate interfacial layer.
    Returns
    -------

    """
    chi = .5 - .5 * np.tanh(3 * (z - zs) / delta)
    return chi


@njit(fastmath=True, cache=True)
def compute_F_fric(ur, uz, chi, xis, delta):
    """

    Parameters
    ----------
    ur : np.ndarray
        velocity field, r
    uz : np.ndarray
        velocity field, z
    chi :  np.ndarray
        substrate field
    xis :  float
        friction coefficient
    delta : float
        width of the substrate
    Returns
    -------

    """
    F_fric_r = - xis * chi / delta * ur
    F_fric_z = - xis * chi / delta * uz
    return F_fric_r, F_fric_z


@njit(fastmath=True, cache=True)
def compute_F_incomp_3d_cyl(phi: np.ndarray, Wchi, r, dr, dz, epsilon, gamma):
    """
    Compute the Forces, including F_{ten} and F_{bs}

    Parameters
    ----------
    phi : np.ndarray
    Wchi : np.ndarray
    r : np.ndarray
    dr : float
    dz : float
    epsilon : float
    gamma : float

    Returns
    -------

    """

    F_ten_r, F_ten_z = compute_F_ten(phi, gamma, epsilon, r, dr, dz)
    F_adh_r, F_adh_z = compute_F_adh(phi, Wchi, dr, dz)
    F_r = F_ten_r + F_adh_r
    F_z = F_ten_z + F_adh_z
    return F_r, F_z


@njit(fastmath=True, cache=True)
def compute_F_adh(phi, Wchi, dr, dz):
    """ Calculate the biofilm-substrate adhesion force.
    $F_{bs} =  4 \phi (2 - 3 \phi + \phi^2) W(\chi) \nabla \phi$

    Parameters
    ----------
    phi :
    Wchi :
    dr :
    dz :

    Returns
    -------

    """
    Fadh = 4 * phi * (2. - 3. * phi + phi * phi) * Wchi
    d_phi_r = dfdr_2d_cart_iso(phi, dr, 0)
    d_phi_z = dfdz_2d_cart_iso(phi, dz)
    F_adh_r = Fadh * d_phi_r
    F_adh_z = Fadh * d_phi_z
    return F_adh_r, F_adh_z


@njit(fastmath=True, cache=True)
def compute_F_ten(phi: np.ndarray, gamma: float, epsilon: float,
                  dr: float, dz: float):
    """
    compute surface tension of the colony membrane.
    $F_{ten} = \gamma (\epsilon \Delta \phi - G^{\prime}(\phi)/\epsilon) \nabla \phi$
    Parameters
    ----------
    phi :
    gamma :
    epsilon :
    r :
    dr :
    dz :

    Returns
    -------

    """
    dG = get_dG(phi)
    F_ten = gamma * (dG / epsilon - epsilon * lap_2d_cart_iso(phi, dr, dz, case=0))

    d_phi_r, d_phi_z = grad_2d_cart_iso(phi, dr, dz, case=0)
    F_ten_r = F_ten * d_phi_r
    F_ten_z = F_ten * d_phi_z
    return F_ten_r, F_ten_z


@njit(fastmath=True, cache=True)
def compute_pressure(rho0, rhoi, BulkMod, *args):
    """

    Parameters
    ----------
    rho0 : np.ndarray
    rhoi : float
    BulkMod : float

    Returns
    -------

    """

    if 'lin' in args:
        pp = (rho0 / rhoi - 1.) * BulkMod
    else:
        pp = rho0 / rhoi * BulkMod

    return pp


# @jit(nopython=True, fastmath=True, cache=True)
def phisolver_2d_cart(phi0, lapphi, ur0, uz0, dr, dz, dt, Gamma, epsilon, *args):
    """
    phi_n - phi_1

    phi = phi0 + dt * (- ur0 * dphir - uz0 * dphiz) + dt * multiplier

    Parameters
    ----------
    phi0 :
    lapphi :
    ur0 :
    uz0 :
    dr :
    dz :
    dt :
    Gamma :
    epsilon : epsilon
    args :

    Returns
    -------

    """

    dG = get_dG(phi0)
    ten1 = lapphi * epsilon
    ten2 = dG / epsilon
    ten = ten1 - ten2

    if 'nocurv' in args:
        cterm = np.zeros(phi0.shape)
    else:
        kappa = compute_curv_2d_cart_iso(phi0, dr, dz, 0.001)
        par_phi_r, par_phi_z = grad_2d_cart_iso(phi0, dr, dz, 0)
        abs_grad_phi = np.sqrt(par_phi_r ** 2 + par_phi_z ** 2)
        cterm = kappa * epsilon * abs_grad_phi

    dphir = dfdr_2d_cart_iso(phi0, dr, 0)
    dphiz = dfdz_2d_cart_iso(phi0, dz)

    multiplier = Gamma * (ten + cterm)
    phi_p = phi0 - dt * (ur0 * dphir + uz0 * dphiz) + dt * multiplier

    return phi_p


def rhosolver_2d_cart(rho0, phi0, phi, ur0, uz0, dt,
                      dr, dz, Dc, kg: ndarray, lambda_, rhoi):
    """

    hi_t = -u*grad(phi)+Gamma*(...)
    (phi*rho)_t = RHS

    phi^new - phi^old = dt*[ -u*grad(phi)+Gamma*(...) ]^old

    rhoa^new = rhoa^old - dt/phi^old*[ -u*grad(phi)+Gamma*(...) ]^old + dt/phi^old*RHS^old

    rhoa^new = (2*phi^old-phi^new) / phi^old * rhoa^old + dt/ phi^o * RHS^o

    Parameters
    ----------
    rho0 :  np.ndarray
    phi0 : np.ndarray
    phi : np.ndarray
    ur0 : np.ndarray
    uz0 : np.ndarray
    dt : float
    dr : float
    dz : float
    Dc : float
        cell diffusion speed.
    lambda_: float
    kg: nd.ndarray
        growth rate field

    Returns
    -------

    """
    phi_box = phi > .05
    kg[~phi_box] = 0.

    f_rho = kg * phi0 * rho0

    Dc_term = div_f_grad_g_2d_cart_iso(Dc * phi0, rho0, dr, dz)
    uterm = - div4_2d_cart_iso(phi0, rho0, ur0, uz0, dr, dz)
    rho_rhs = Dc_term + f_rho + uterm

    # uterm = - div_fu_2d_cart_iso(rho0 * phi0, ur0, uz0, dr, dz)
    # rho_rhs = f_rho + uterm + Dc_term
    prod1 = (2. * phi0 - phi) * rho0 + dt * rho_rhs
    rho_p = np.zeros(rho0.shape)
    rho_p[phi_box] = prod1[phi_box] / phi0[phi_box]
    # rho_p = np.zeros(rho0.shape)
    # rho_p[phi_box] = rho0[phi_box] + (dt * rho_rhs)[phi_box]

    return rho_p


def velocitysolver_dacy(phi0, ur0, uz0, Fr, Fz, xis, chi, delta, nu0, kg, pressure,
                        dr, dz, lambda_,
                        xitrick, nutrick, xi, kp, kp2,
                        error_limit, max_steps,
                        device='cuda:0', verbose=False):
    """
        Solve the ODEs of velocity.
        Parameters
        ----------
        phi0 : ndarray
        ur0 :
        uz0 :
        Fr :
        Fz :
        xis :
        chi :
        delta :
        nu0 : Tuple[float, float]
        kg:
        dr :
        dz :
        lambda_ :
        xitrick :
        nutrick :
        xi :
        kp :
        kp2 :
        error_limit :
        max_steps :
        verbose :

        Returns
        -------

        """

    phi_box = phi0 > lambda_  # type: ndarray
    # error_box = phi0 > .1

    # determining the iterate boundary
    r_mask = np.any(phi_box, axis=0)
    z_mask = np.any(phi_box, axis=1)
    r_index = np.arange(phi_box.shape[1])
    z_index = np.arange(phi_box.shape[0])
    r_bound = phi_box.shape[1] - 1
    z_bound = phi_box.shape[0] - 1
    colony_r_index = r_index[r_mask]
    colony_z_index = z_index[z_mask]
    cr_index = colony_r_index.max() + 5
    cz_t_index = colony_z_index.max() + 25  # colony top edge index
    cz_b_index = colony_z_index.min() - 10  # colony bottom edge index
    if cr_index > r_bound:
        cr_index = r_bound
    if cz_b_index < 0:
        cz_b_index = 0
    if cz_t_index > z_bound:
        cz_t_index = z_bound
    if cr_index % 2 == 0:
        colony_r_index -= 1
    if (cz_t_index - cz_b_index) % 2 != 0:
        cz_t_index -= 1
    # get sub-matrix fields = (phi0, ur0, uz0, Fr, Fz, kg, chi, pressure)
    phi0_sub = phi0[cz_b_index:cz_t_index, 0:cr_index].astype(np.float32)
    phi_box_sub = phi_box[cz_b_index:cz_t_index, 0:cr_index].astype(bool)
    ur0_sub = ur0[cz_b_index:cz_t_index, 0:cr_index].astype(np.float32)
    uz0_sub = uz0[cz_b_index:cz_t_index, 0:cr_index].astype(np.float32)
    Fr_sub = Fr[cz_b_index:cz_t_index, 0:cr_index].astype(np.float32)
    Fz_sub = Fz[cz_b_index:cz_t_index, 0:cr_index].astype(np.float32)
    kg_sub = kg[cz_b_index:cz_t_index, 0:cr_index].astype(np.float32)
    chi_sub = chi[cz_b_index:cz_t_index, 0:cr_index].astype(np.float32)
    pressure_sub = pressure[cz_b_index:cz_t_index, 0:cr_index].astype(np.float32)
    kappa_r = 2 * np.pi * fftfreq(2 * (cr_index - 1), dr).astype(np.float32)  # FFT wave number
    kappa_z = 2 * np.pi * fftfreq(phi0_sub.shape[0], dz).astype(np.float32)
    kp_sub = np.array(np.meshgrid(kappa_z, kappa_r, indexing='ij'))
    kp2_sub = np.sum(kp_sub * kp_sub, axis=0, dtype=np.float32)
    if gpu:
        kp2_sub = torch.from_numpy(kp2_sub).to(device)

    args = (ur0_sub, uz0_sub, Fr_sub, Fz_sub, chi_sub, xis, delta, nu0, phi0_sub,
            phi_box_sub, pressure_sub, xitrick, kg_sub, dr, dz, kp2_sub, lambda_,
            error_limit, max_steps, device)
    (ur0_sub, uz0_sub, F_all_r_sub, F_all_z_sub, F_fric_r_sub, F_fric_z_sub, pressure_sub,
     it_steps, max_err, it_steps) = update_velocity(args)
    ur0[cz_b_index:cz_t_index, 0:cr_index] = ur0_sub
    uz0[cz_b_index:cz_t_index, 0:cr_index] = uz0_sub
    F_all_r = np.zeros(phi0.shape)
    F_all_z = np.zeros(phi0.shape)
    F_fric_r = np.zeros(phi0.shape)
    F_fric_z = np.zeros(phi0.shape)
    F_all_r[cz_b_index:cz_t_index, 0:cr_index] = F_all_r_sub
    F_all_z[cz_b_index:cz_t_index, 0:cr_index] = F_all_z_sub
    F_fric_r[cz_b_index:cz_t_index, 0:cr_index] = F_fric_r_sub
    F_fric_z[cz_b_index:cz_t_index, 0:cr_index] = F_fric_z_sub
    pressure[cz_b_index:cz_t_index, 0:cr_index] = pressure_sub

    if verbose is False:  # if no verbose, only record the convergence info.
        iter_list = [it_steps]
        error_max_list = [max_err]
        ur_rhs = - nu0[0] * phi0 * ur0 + F_all_r - dfdr_2d_cart_iso(pressure, dr, 0)
        uz_rhs = - nu0[1] * phi0 * uz0 + F_all_z - dfdz_2d_cart_iso(pressure, dz)
        max_dist = np.max(np.sqrt(ur_rhs * ur_rhs + uz_rhs * uz_rhs))
        error_dist_list = [max_dist]

    verbose_rcd = {'iter_list': iter_list,
                   'error_max_list': error_max_list,
                   'error_dist_list': error_dist_list}
    return ur0, uz0, F_all_r, F_all_z, F_fric_r, F_fric_z, pressure, verbose_rcd, xitrick, nutrick


def update_velocity(args):
    (ur0, uz0, Fr, Fz, chi, xis, delta, nu0, phi0, phi_box, pressure, xitrick, kg, dr, dz, kp2, lambda_,
     error_limit, max_steps, device) = args
    it_steps = 0
    max_err = error_limit * 1.1
    kg_ext = func_ext(kg, 0)
    phi0_ext = func_ext(phi0, 0)

    phi0_ext_box = phi0_ext > lambda_
    kg_ext = kg_ext * phi0_ext
    kg_ext[~phi0_ext_box] = 0.
    F_all_r = np.zeros(ur0.shape)
    F_all_z = np.zeros(uz0.shape)
    F_fric_r = np.zeros(ur0.shape)
    F_fric_z = np.zeros(uz0.shape)
    while (max_err > error_limit) and (it_steps <= max_steps):
        # ddU_r, ddU_z = div_f_grad_u_2d_cart_iso(phi0, ur0, uz0, dr, dz, 0)
        # ddU_r, ddU_z = div_stress_imcomp_cart_iso(phi0, ur0, uz0, kg, dr, dz)  # simplified stress tensor
        # note no pressure field
        # calculate Friction between substrate and colony
        F_fric_r, F_fric_z = compute_F_fric(ur0, uz0, chi, xis, delta)
        # F_fric_z = np.zeros(F_fric_z.shape)  # no friction in z axis

        F_all_r = Fr + F_fric_r
        F_all_z = Fz + F_fric_z  # no friction in z axis

        ur_rhs = - nu0[0] * phi0 * ur0 + F_all_r - .0 * dfdr_2d_cart_iso(pressure, dr, 0)
        uz_rhs = - nu0[1] * phi0 * uz0 + F_all_z - .0 * dfdz_2d_cart_iso(pressure, dz)

        # compute us
        ur_rhs_ext = func_ext(ur_rhs, 1)
        uz_rhs_ext = func_ext(uz_rhs, 0)
        ur0_ext = func_ext(ur0, 1)
        uz0_ext = func_ext(uz0, 0)

        urs_ext = ur_rhs_ext / xitrick + ur0_ext
        uzs_ext = uz_rhs_ext / xitrick + uz0_ext

        # compute pressure
        poisson_rhs = (- kg_ext +
                       dfdr_2d_cart_iso(urs_ext, dr, 1)
                       + dfdz_2d_cart_iso(uzs_ext, dz)) * xitrick

        if gpu:
            poisson_rhs_hat = tor_fft2(torch.from_numpy(poisson_rhs.astype(np.float32)).to(device))
        else:
            poisson_rhs_hat = fft2(poisson_rhs.astype(np.float32))

        PHI_ext_hat = - poisson_rhs_hat / kp2
        PHI_ext_hat[0, 0] = 0.  # let average pressure to 0

        if gpu:
            PHI_ext = tor_ifft2(PHI_ext_hat).real
            PHI_ext = PHI_ext.cpu().numpy()
        else:
            PHI_ext = ifft2(PHI_ext_hat).real  # 27486.97s/it

        PHI = func_red(PHI_ext)
        pressure = PHI + .0 * pressure
        urs = func_red(urs_ext)
        uzs = func_red(uzs_ext)
        urp = urs - dfdr_2d_cart_iso(PHI, dr, 0) / xitrick
        uzp = uzs - dfdz_2d_cart_iso(PHI, dz) / xitrick
        urp[~phi_box] = 0.
        uzp[~phi_box] = 0.

        error_r = np.abs(urp - ur0)
        error_z = np.abs(uzp - uz0)
        max_err_p = np.max((error_r / np.max(np.abs(urp)),
                            error_z / np.max(np.abs(uzp))))

        max_err = max_err_p

        # update velocity field.
        ur0 = urp
        uz0 = uzp
        it_steps += 1
        if it_steps >= max_steps:
            warnings.warn('Maximum iteration loops reached!')

    return ur0, uz0, F_all_r, F_all_z, F_fric_r, F_fric_z, pressure, it_steps, max_err, it_steps


def concsolver(c_0: ndarray, c_1: ndarray, c2_0: ndarray, c2_1: ndarray,
               phi: ndarray, phi0: ndarray, chi: ndarray, zs, r, z,
               lambda1: float, lambda2: float, K1: float, K2: float, f1: float, f2: float,
               c1: float, c2: float, Dc: float,
               dt: float, dr: float, dz: float, N_c: int = 1):
    """
    Solve the nutrient concentrations in colony and substrate.
    Parameters
    ----------
    c_0 :
    c_1 :
    c2_0 :
    c2_1 :
    phi :
    phi0 :
    chi :
    zs :
    r :
    z :
    lambda1 :
    K1 :
    c1 :
    c2 :
    Dc :
    dt :
    dr :
    dz :

    Returns
    -------

    """

    # 1. determine boundary, gamma12: colony-substrate, gamma02: sub-air, gamma01: colony-air
    zs_index = np.argmin(np.abs(z[:, 0] - zs))
    sc_interphase = phi0 * chi * ((chi > 1e-3) + (phi > 1e-3))
    max_inter_section = np.max(sc_interphase)
    inter_section_mask = sc_interphase > (max_inter_section * .9)
    sc_interphase_index_i = np.argmax(sc_interphase, axis=0)
    mask_index = inter_section_mask[sc_interphase_index_i, np.arange(phi.shape[-1])]

    sc_interphase_index_i = sc_interphase_index_i[mask_index]
    sc_interphase_index_i = int(np.mean(sc_interphase_index_i))
    if sc_interphase_index_i < zs_index:
        gamma12_z_i = sc_interphase_index_i
        # print(f"zs:{zs_index}  gamma12_z_i:{gamma12_z_i}")
    else:
        gamma12_z_i = zs_index
    # gamma12_z_i = zs_index
    gamma12_r_i = np.nonzero(phi0[gamma12_z_i, ...] >= 1e-2)[0].max()

    # substrate field
    phi_box = phi > 1e-2
    phi_box_bound = find_boundary(phi_box)
    omega_0 = np.ones(c_0.shape)
    omega_0[gamma12_z_i + 1:, ...] = 0.
    # colony field
    omega_1 = np.ones(c_0.shape)
    omega_1[~phi_box] = 0.
    omega_1[:gamma12_z_i, ...] = 0.  # colony

    # 2. update nutrient conc
    # c_0/c_1 : nutrients 1 in colony/substrate,
    # c2_0/c2_1: nutrients 2 in colony/substrate,

    c_0 = c_0 * omega_0  # make sure that the air have no nutrients.
    c2_0 = c2_0 * omega_0
    theta_n1 = KM(c_0, K1)
    theta_n2 = KM(c2_0, K2)

    c1_consump = - f1 * lambda1 * theta_n1  # carbon consumption =: rho_DW / Da_carbon *  kg / Y, Y is yield factor
    # Dc_term = div_f_grad_g_2d_cart_iso(Dc * omega_1, c_1, dr, dz)
    Dc_term = lap_2d_cart_iso(Dc * c_1 * phi0, dr, dz, case=0)
    c1_1_rhs = Dc_term + c1_consump * phi0
    prod1 = (2. * phi0 - phi) * c_1 + dt * c1_1_rhs
    c_1_p = np.zeros(c_1.shape)
    c_1_p[phi_box] = prod1[phi_box] / phi0[phi_box]
    c_1_p = c_1_p * omega_1
    # c_0_rhs = div_f_grad_g_2d_cart_iso(Dc * omega_0, c_0, dr, dz)
    c_0_rhs = lap_2d_cart_iso(Dc * omega_0 * c_0, dr, dz, case=0)
    c_0_p = c_0 + dt * c_0_rhs

    c2_consump = - f2 * lambda2 * theta_n2 * (1. - theta_n1)
    Dc2_term = lap_2d_cart_iso(Dc * c2_1 * phi0, dr, dz, case=0)
    c2_1_rhs = Dc2_term + c2_consump * phi0
    prod2 = (2. * phi0 - phi) * c2_1 + dt * c2_1_rhs
    c2_1_p = np.zeros(c2_1.shape)
    c2_1_p[phi_box] = prod2[phi_box] / phi0[phi_box]
    c2_1_p = c2_1_p * omega_1
    c2_0_rhs = lap_2d_cart_iso(Dc * omega_0 * c2_0, dr, dz, case=0)
    c2_0_p = c2_0 + dt * c2_0_rhs

    # 2. boundary condition.
    # 2.1 substrate condition
    # c1
    c_0_p[0, :] = c1  # button
    c_0_p[:gamma12_z_i + 1, -1] = c1  # left boundary
    c_0_p[gamma12_z_i, gamma12_r_i + 1:] = c_0_p[gamma12_z_i - 1, gamma12_r_i + 1:]  # gamma 02 sub-air
    # c2
    c2_0_p[0, :] = c2  # button
    c2_0_p[:gamma12_z_i + 1, -1] = c2  # left boundary
    c2_0_p[gamma12_z_i, gamma12_r_i + 1:] = c2_0_p[gamma12_z_i - 1, gamma12_r_i + 1:]  # gamma 02 sub-air

    # 2.2 colony field.
    phi_box_bound[:gamma12_z_i, :] = False
    # c1
    c_1_p[phi_box_bound] = 0.
    c_1_p[gamma12_z_i, :gamma12_r_i + 1] = (c_1_p[gamma12_z_i + 1, :gamma12_r_i + 1] +
                                            c_0_p[gamma12_z_i - 1, :gamma12_r_i + 1]) / 2.  # gamma 12
    c_0_p[gamma12_z_i, :gamma12_r_i + 1] = c_1_p[gamma12_z_i, :gamma12_r_i + 1]  # gamma 12
    # c2
    c2_1_p[phi_box_bound] = 0.
    c2_1_p[gamma12_z_i, :gamma12_r_i + 1] = (c2_1_p[gamma12_z_i + 1, :gamma12_r_i + 1] +
                                             c2_0_p[gamma12_z_i - 1, :gamma12_r_i + 1]) / 2.  # gamma 12
    c2_0_p[gamma12_z_i, :gamma12_r_i + 1] = c2_1_p[gamma12_z_i, :gamma12_r_i + 1]  # gamma 12

    # magic
    c_0_p[c_0_p > c1] = c1
    c_1_p[c_1_p > c1] = c1
    c_0_p[c_0_p < 0.] = 0.
    c_1_p[c_1_p < 0.] = 0.
    c2_0_p[c_0_p > c2] = c2
    c2_1_p[c_1_p > c2] = c2
    c2_0_p[c_0_p < 0.] = 0.
    c2_1_p[c_1_p < 0.] = 0.
    # update field
    c_0 = c_0_p
    c_1 = c_1_p
    c2_0 = c2_0_p
    c2_1 = c2_1_p

    # 3. calculate local growth rate kg
    c1_all = np.empty(c_1.shape, dtype=c1.dtype)
    c1_all[:gamma12_z_i, ...] = c_0_p[:gamma12_z_i, ...]
    c1_all[gamma12_z_i:, ...] = c_1_p[gamma12_z_i:, ...]
    c2_all = np.empty(c2_1.shape, dtype=c1.dtype)
    c2_all[:gamma12_z_i, ...] = c2_0_p[:gamma12_z_i, ...]
    c2_all[gamma12_z_i:, ...] = c2_1_p[gamma12_z_i:, ...]
    kg = diauxie_local_gr(c1_all, c2_all, K1, K2, lambda1, lambda2)

    return c_0_p, c_1_p, c2_0_p, c2_1_p, kg, c1_all, c2_all, gamma12_z_i


def concsolver2(c_0: ndarray, c_1: ndarray, c2_0: ndarray, c2_1: ndarray,
                phi: ndarray, phi0: ndarray, chi: ndarray, zs, r, z,
                lambda1: float, lambda2: float, K1: float, K2: float, f1: float, f2: float,
                c1: float, c2: float, Dc: float,
                dt: float, dr: float, dz: float, N_c: int = 1):
    """
    Solve the nutrient concentrations in colony and substrate.

    # c_0/c_1 : nutrients 1 in substrate/colony,
    # c2_0/c2_1: nutrients 2 in substrate/colony,

    Parameters
    ----------
    c_0 :
    c_1 :
    c2_0 :
    c2_1 :
    phi :
    phi0 :
    chi :
    zs :
    r :
    z :
    lambda1 :
    K1 :
    c1 :
    c2 :
    Dc :
    dt :
    dr :
    dz :

    Returns
    -------

    """

    # 1. determine boundary, gamma12: colony-substrate, gamma02: sub-air, gamma01: colony-air
    zs_index = np.argmin(np.abs(z[:, 0] - zs))
    sc_interphase = phi0 * chi * ((chi > 1e-3) + (phi > 1e-3))
    max_inter_section = np.max(sc_interphase)
    inter_section_mask = sc_interphase > (max_inter_section * .9)
    sc_interphase_index_i = np.argmax(sc_interphase, axis=0)
    mask_index = inter_section_mask[sc_interphase_index_i, np.arange(phi.shape[-1])]

    sc_interphase_index_i = sc_interphase_index_i[mask_index]
    sc_interphase_index_i = int(np.mean(sc_interphase_index_i))
    if sc_interphase_index_i < zs_index:
        gamma12_z_i = sc_interphase_index_i
        # print(f"zs:{zs_index}  gamma12_z_i:{gamma12_z_i}")
    else:
        gamma12_z_i = zs_index
    # gamma12_z_i = zs_index
    gamma12_r_i = np.nonzero(phi0[gamma12_z_i, ...] >= 1e-2)[0].max()

    phi_box = phi > 1e-2
    phi_box_bound = find_boundary(phi_box)
    # substrate field
    omega_0 = np.ones(c_0.shape)
    omega_0[gamma12_z_i + 1:, ...] = 0.
    # colony field
    omega_1 = np.ones(c_1.shape)
    omega_1[~phi_box] = 0.
    omega_1[:gamma12_z_i, ...] = 0.  # colony

    DcPhi0 = Dc * phi0
    dt = dt / N_c
    for _ in range(N_c):
        # 2. update nutrient conc
        # c_0/c_1 : nutrients 1 in substrate/colony,
        # c2_0/c2_1: nutrients 2 in substrate/colony,

        c_0 = c_0 * omega_0  # make sure that the air have no nutrients.
        c2_0 = c2_0 * omega_0
        c_1 = c_1 * omega_1
        c2_1 = c2_1 * omega_1
        theta_n1 = KM(c_1, K1)
        theta_n2 = KM(c2_1, K2)

        c1_consump = - f1 * lambda1 * theta_n1  # carbon consumption =: rho_DW / Da_carbon *  kg / Y, Y is yield factor
        Dc_term = lap_2d_cart_iso(DcPhi0 * c_1, dr, dz, case=0)
        c1_1_rhs = Dc_term + c1_consump * phi0
        c_1_p = c_1 + dt * c1_1_rhs

        c_0_rhs = lap_2d_cart_iso(Dc * omega_0 * c_0, dr, dz, case=0)
        c_0_p = c_0 + dt * c_0_rhs

        c2_consump = - f2 * lambda2 * theta_n2 * (1. - theta_n1)
        Dc2_term = lap_2d_cart_iso(DcPhi0 * c2_1, dr, dz, case=0)
        c2_1_rhs = Dc2_term + c2_consump * phi0

        c2_1_p = c2_1 + dt * c2_1_rhs
        c2_0_rhs = lap_2d_cart_iso(Dc * omega_0 * c2_0, dr, dz, case=0)
        c2_0_p = c2_0 + dt * c2_0_rhs

        # 2. boundary condition.
        # 2.1 substrate condition
        # c1
        c_0_p[0, :] = c1  # button
        c_0_p[:gamma12_z_i + 1, -1] = c1  # left boundary
        c_0_p[gamma12_z_i, gamma12_r_i + 1:] = c_0_p[gamma12_z_i - 1, gamma12_r_i + 1:]  # gamma 02 sub-air
        # c2
        c2_0_p[0, :] = c2  # button
        c2_0_p[:gamma12_z_i + 1, -1] = c2  # left boundary
        c2_0_p[gamma12_z_i, gamma12_r_i + 1:] = c2_0_p[gamma12_z_i - 1, gamma12_r_i + 1:]  # gamma 02 sub-air

        # 2.2 colony field.
        phi_box_bound[:gamma12_z_i, :] = False
        # c1
        c_1_p[phi_box_bound] = 0.
        c_1_p[gamma12_z_i, :gamma12_r_i + 1] = (c_1_p[gamma12_z_i + 1, :gamma12_r_i + 1] +
                                                c_0_p[gamma12_z_i - 1, :gamma12_r_i + 1]) / 2.  # gamma 12
        c_0_p[gamma12_z_i, :gamma12_r_i + 1] = c_1_p[gamma12_z_i, :gamma12_r_i + 1]  # gamma 12
        # c2
        c2_1_p[phi_box_bound] = 0.
        c2_1_p[gamma12_z_i, :gamma12_r_i + 1] = (c2_1_p[gamma12_z_i + 1, :gamma12_r_i + 1] +
                                                 c2_0_p[gamma12_z_i - 1, :gamma12_r_i + 1]) / 2.  # gamma 12
        c2_0_p[gamma12_z_i, :gamma12_r_i + 1] = c2_1_p[gamma12_z_i, :gamma12_r_i + 1]  # gamma 12

        # magic
        c_0_p[c_0_p > c1] = c1
        c_1_p[c_1_p > c1] = c1
        c_0_p[c_0_p < 0.] = 0.
        c_1_p[c_1_p < 0.] = 0.
        c2_0_p[c_0_p > c2] = c2
        c2_1_p[c_1_p > c2] = c2
        c2_0_p[c_0_p < 0.] = 0.
        c2_1_p[c_1_p < 0.] = 0.
        # update field
        c_0 = c_0_p
        c_1 = c_1_p
        c2_0 = c2_0_p
        c2_1 = c2_1_p

    # 3. calculate local growth rate kg
    c1_all = np.empty(c_1.shape, dtype=c1.dtype)
    c1_all[:gamma12_z_i, ...] = c_0_p[:gamma12_z_i, ...]
    c1_all[gamma12_z_i:, ...] = c_1_p[gamma12_z_i:, ...]
    c2_all = np.empty(c2_1.shape, dtype=c1.dtype)
    c2_all[:gamma12_z_i, ...] = c2_0_p[:gamma12_z_i, ...]
    c2_all[gamma12_z_i:, ...] = c2_1_p[gamma12_z_i:, ...]
    kg = diauxie_local_gr(c1_all, c2_all, K1, K2, lambda1, lambda2)

    return c_0_p, c_1_p, c2_0_p, c2_1_p, kg, c1_all, c2_all, gamma12_z_i


def concsolver3(c_0: ndarray, c_1: ndarray, c2_0: ndarray, c2_1: ndarray,
                phi: ndarray, phi0: ndarray, chi: ndarray, zs: float, r, z,
                lambda1: float, lambda2: float, K1: float, K2: float, f1: float, f2: float,
                c1: float, c2: float, Dc: float,
                dt: float, dr: float, dz: float, N_c: int = 1):
    """
    Solve the nutrient concentrations in colony and substrate.
    Note: This function is going to solve the steady state solution of Nutrients.

    # c_0/c_1 : nutrients 1 in substrate/colony,
    # c2_0/c2_1: nutrients 2 in substrate/colony,

    Parameters
    ----------
    c_0 :
    c_1 :
    c2_0 :
    c2_1 :
    phi :
    phi0 :
    chi :
    zs :
    r :
    z :
    lambda1 :
    K1 :
    c1 :
    c2 :
    Dc :
    dt :
    dr :
    dz :

    Returns
    -------

    """

    # 0. down sampling the field
    r_high = r.copy()
    z_high = z.copy()
    sparse = int(5. / dr)
    c_0 = get_low_res(c_0, sparse)
    c_1 = get_low_res(c_1, sparse)
    c2_0 = get_low_res(c2_0, sparse)
    c2_1 = get_low_res(c2_1, sparse)
    phi = get_low_res(phi, sparse)
    phi0 = get_low_res(phi0, sparse)
    chi = get_low_res(chi, sparse)
    r = get_low_res(r, sparse)
    z = get_low_res(z, sparse)
    r_axis = r[0, :]
    z_axis = z[:, 0]

    # 1. determine boundary, gamma12: colony-substrate, gamma02: sub-air, gamma01: colony-air
    zs_index = np.argmin(np.abs(z[:, 0] - zs))
    sc_interphase = (phi0 * chi *
                     ((chi > 1e-3).astype(phi.dtype) + (phi > 1e-3).astype(phi.dtype)))
    max_interface_section = np.max(sc_interphase)
    inter_section_mask = sc_interphase > (max_interface_section * .9)
    sc_interphase_index_i = np.argmax(sc_interphase, axis=0)
    mask_index = inter_section_mask[sc_interphase_index_i, np.arange(phi.shape[-1])]

    sc_interphase_index_i = sc_interphase_index_i[mask_index]
    sc_interphase_index_i = int(np.mean(sc_interphase_index_i))
    if sc_interphase_index_i < zs_index:
        gamma12_z_i = sc_interphase_index_i
        # print(f"zs:{zs_index}  gamma12_z_i:{gamma12_z_i}")
    else:
        gamma12_z_i = zs_index
    gamma12_r_i = np.nonzero(phi0[gamma12_z_i, ...] >= 1e-2)[0].max()

    phi_box = phi > 1e-4
    phi_box_bound = find_boundary(phi_box)
    # substrate field
    omega_0 = np.ones(c_0.shape)
    omega_0[gamma12_z_i + 1:, ...] = 0.
    # colony field
    omega_1 = np.ones(c_1.shape)
    omega_1[~phi_box] = 0.
    omega_1[:gamma12_z_i, ...] = 0.

    v_box = phi_box.astype(phi.dtype)
    Dc_V = Dc * v_box
    dt = dt / N_c
    nutrients_pars = (N_c, phi0, c_0, c_1, c2_0, c2_1, omega_0, omega_1, K1,
                      K2, f1, f2, lambda1, lambda2, Dc_V, Dc, dr, dz, dt, c1, c2,
                      gamma12_z_i, gamma12_r_i, phi_box_bound)

    c_0_p, c_1_p, c2_0_p, c2_1_p, it_num, err_1, err_2 = update_nutrients(nutrients_pars)
    # print(iterN)
    # 3. calculate interpolate the nutrients fields.
    c1_all = np.empty(c_0_p.shape, dtype=c_0_p.dtype)
    c1_all[:gamma12_z_i, ...] = c_0_p[:gamma12_z_i, ...]
    c1_all[gamma12_z_i:, ...] = c_1_p[gamma12_z_i:, ...]
    c2_all = np.empty(c2_0_p.shape, dtype=c2_0_p.dtype)
    c2_all[:gamma12_z_i, ...] = c2_0_p[:gamma12_z_i, ...]
    c2_all[gamma12_z_i:, ...] = c2_1_p[gamma12_z_i:, ...]

    c_0_p = interpolate_mat(r_axis, z_axis, c_0_p, r_high, z_high)
    c_1_p = interpolate_mat(r_axis, z_axis, c_1_p, r_high, z_high)
    c2_0_p = interpolate_mat(r_axis, z_axis, c2_0_p, r_high, z_high)
    c2_1_p = interpolate_mat(r_axis, z_axis, c2_1_p, r_high, z_high)
    c1_all = interpolate_mat(r_axis, z_axis, c1_all, r_high, z_high)
    c2_all = interpolate_mat(r_axis, z_axis, c2_all, r_high, z_high)
    gamma12_z_i = gamma12_z_i * sparse
    # 4. calculate kg
    kg = diauxie_local_gr(c1_all, c2_all, K1, K2, lambda1, lambda2)

    return c_0_p, c_1_p, c2_0_p, c2_1_p, kg, c1_all, c2_all, gamma12_z_i, (it_num, err_1, err_2)


def concsolver4(c_0: ndarray, c_1: ndarray, c2_0: ndarray, c2_1: ndarray,
                phi: ndarray, phi0: ndarray, chi: ndarray, zs: float, r, z,
                lambda1: float, lambda2: float,
                K1: float, K2: float, f1: float, f2: float, p2: float,
                c1: float, c2: float, Dc: float,
                dt: float, dr: float, dz: float, N_c: int = 1):
    """
    Solve the nutrient concentrations in colony and substrate. (Cross feeding case)
    Note: This function is going to solve the steady state solution of Nutrients.

    # c_0/c_1 : nutrients 1 in substrate/colony,
    # c2_0/c2_1: nutrients 2 in substrate/colony,

    Parameters
    ----------
    c_0 :
    c_1 :
    c2_0 :
    c2_1 :
    phi :
    phi0 :
    chi :
    zs :
    r :
    z :
    lambda1 :
    K1 :
    c1 :
    c2 :
    Dc :
    dt :
    dr :
    dz :

    Returns
    -------
    tuple:
        c_0_p, c_1_p, c2_0_p, c2_1_p, kg, c1_all, c2_all, gamma12_z_i, (it_num, err_1, err_2)
    """

    # 0. down sampling the field
    r_high = r.copy()
    z_high = z.copy()
    sparse = int(6. / dr)
    if sparse <= 0:
        sparse = 1
    c_0 = get_low_res(c_0, sparse)
    c_1 = get_low_res(c_1, sparse)
    c2_0 = get_low_res(c2_0, sparse)
    c2_1 = get_low_res(c2_1, sparse)
    # phi = get_low_res(phi, sparse)
    phi0 = get_low_res(phi0, sparse)
    chi = get_low_res(chi, sparse)
    r = get_low_res(r, sparse)
    z = get_low_res(z, sparse)
    r_axis = r[0, :]
    z_axis = z[:, 0]
    dr = r_axis[1] - r_axis[0]
    dz = r_axis[1] - r_axis[0]
    dt = dt / N_c
    # 1. determine boundary, gamma12: colony-substrate, gamma02: sub-air, gamma01: colony-air
    zs_index = np.argmin(np.abs(z[:, 0] - zs))
    sc_interphase = (phi0 * chi *
                     ((chi > 1e-3).astype(phi0.dtype) + (phi0 > 1e-3).astype(phi0.dtype)))
    max_interface_section = np.max(sc_interphase)
    inter_section_mask = sc_interphase > (max_interface_section * .9)
    sc_interphase_index_i = np.argmax(sc_interphase, axis=0)
    mask_index = inter_section_mask[sc_interphase_index_i, np.arange(phi0.shape[-1])]

    sc_interphase_index_i = sc_interphase_index_i[mask_index]
    sc_interphase_index_i = int(np.mean(sc_interphase_index_i))
    if sc_interphase_index_i < zs_index:
        gamma12_z_i = sc_interphase_index_i
        # print(f"zs:{zs_index}  gamma12_z_i:{gamma12_z_i}")
    else:
        gamma12_z_i = zs_index
    gamma12_r_i = np.nonzero(phi0[gamma12_z_i, ...] >= 1e-2)[0].max()

    phi_box = phi0 > 1e-3
    phi_box_bound = find_boundary(phi_box)
    # substrate field
    omega_0 = np.ones(c_0.shape)
    omega_0[gamma12_z_i + 1:, ...] = 0.
    # colony field
    omega_1 = np.ones(c_1.shape)
    omega_1[~phi_box] = 0.
    omega_1[:gamma12_z_i, ...] = 0.
    v_box = phi_box.astype(phi0.dtype)
    Dc_V = Dc * v_box
    # Dc_V= Dc * phi0
    nutrients_pars = (N_c, phi0, c_0, c_1, c2_0, c2_1, omega_0, omega_1, K1,
                      K2, f1, f2, p2, lambda1, lambda2, Dc_V, Dc, dr, dz, dt, c1, c2,
                      gamma12_z_i, gamma12_r_i, phi_box_bound)

    c_0_p, c_1_p, c2_0_p, c2_1_p, it_num, err_1, err_2 = update_nutrients_CrossFeeding(nutrients_pars)
    # print(iterN)
    # 3. calculate interpolate the nutrients fields.
    c1_all = np.empty(c_0_p.shape, dtype=c_0_p.dtype)
    c1_all[:gamma12_z_i, ...] = c_0_p[:gamma12_z_i, ...]
    c1_all[gamma12_z_i:, ...] = c_1_p[gamma12_z_i:, ...]
    c2_all = np.empty(c2_0_p.shape, dtype=c2_0_p.dtype)
    c2_all[:gamma12_z_i, ...] = c2_0_p[:gamma12_z_i, ...]
    c2_all[gamma12_z_i:, ...] = c2_1_p[gamma12_z_i:, ...]

    c_0_p = interpolate_mat(r_axis, z_axis, c_0_p, r_high, z_high)
    c_1_p = interpolate_mat(r_axis, z_axis, c_1_p, r_high, z_high)
    c2_0_p = interpolate_mat(r_axis, z_axis, c2_0_p, r_high, z_high)
    c2_1_p = interpolate_mat(r_axis, z_axis, c2_1_p, r_high, z_high)
    c1_all = interpolate_mat(r_axis, z_axis, c1_all, r_high, z_high)
    c2_all = interpolate_mat(r_axis, z_axis, c2_all, r_high, z_high)
    gamma12_z_i = gamma12_z_i * sparse
    # 4. calculate kg
    kg = diauxie_local_gr(c1_all, c2_all, K1, K2, lambda1, lambda2)

    return c_0_p, c_1_p, c2_0_p, c2_1_p, kg, c1_all, c2_all, gamma12_z_i, (it_num, err_1, err_2)


@njit(fastmath=True, cache=True)
def update_nutrients_CrossFeeding(nutrients_pars):
    """
    (N_c, phi0, c_0, c_1, c2_0, c2_1, omega_0, omega_1, K1,
     K2, f1, f2, p2, lambda1, lambda2, Dc_V, Dc, dr, dz, dt, c1, c2,
     gamma12_z_i, gamma12_r_i, phi_box_bound) = nutrients_pars
    Parameters
    ----------
    nutrients_pars :

    Returns
    -------

    """
    (N_c, phi0, c_0, c_1, c2_0, c2_1, omega_0, omega_1, K1,
     K2, f1, f2, p2, lambda1, lambda2, Dc_V, Dc, dr, dz, dt, c1, c2,
     gamma12_z_i, gamma12_r_i, phi_box_bound) = nutrients_pars

    # max_err_c1 = 1.
    # max_err_c2 = 1.
    # it_n = 0
    # while (max_err_c1 > 1e-3) or (max_err_c2 > 1e-3):
    #     it_n += 1
    #     if it_n%10000 == 0:
    #         print(max_err_c1)
    it_n = 0
    while it_n <= N_c:
        # 2. update nutrient conc
        # c_0/c_1 : nutrients 1 in substrate/colony,
        # c2_0/c2_1: nutrients 2 in substrate/colony,

        theta_n1 = KM(c_1, K1)
        theta_n2 = KM(c2_1, K2)

        lambda1_loc = lambda1 * theta_n1
        lambda2_loc = lambda2 * theta_n2
        # 2.a Nutrient 1, Colony
        c1_consumption = - f1 * lambda1_loc * theta_n1  # carbon consumption =: rho_DW / Da_carbon *  kg / Y, Y is yield factor
        # Dc_term = lap_2d_cart_iso(Dc_V * c_1, dr, dz, case=0)
        Dc_term = div_f_div_g_consv(Dc_V * phi0, c_1, dr, dz)
        c1_1_rhs = Dc_term + c1_consumption * phi0
        c_1_p = c_1 + dt * c1_1_rhs
        # 2.b Nutrient 1, Substrate
        # c_0_rhs = lap_2d_cart_iso(Dc * omega_0 * c_0, dr, dz, case=0)
        c_0_rhs = div_f_div_g_consv(Dc * omega_0, c_0, dr, dz)
        c_0_p = c_0 + dt * c_0_rhs
        # 2.c Nutrient 2, Colony
        c2_consumption = - f2 * lambda2_loc * (1. - theta_n1)
        c2_produce = p2 * lambda1_loc * theta_n1
        # Dc2_term = lap_2d_cart_iso(Dc_V * c2_1, dr, dz, case=0)
        Dc2_term = div_f_div_g_consv(Dc_V * phi0, c2_1, dr, dz)
        c2_1_rhs = Dc2_term + (c2_consumption + c2_produce) * phi0
        c2_1_p = c2_1 + dt * c2_1_rhs
        # 2.d Nutrient 2, Substrate
        # c2_0_rhs = lap_2d_cart_iso(Dc * omega_0 * c2_0, dr, dz, case=0)
        c2_0_rhs = div_f_div_g_consv(Dc * omega_0, c2_0, dr, dz)
        c2_0_p = c2_0 + dt * c2_0_rhs

        # 2. boundary condition.
        # 2.1 substrate condition
        # c1
        c_0_p[0, :] = c1  # button
        c_0_p[:gamma12_z_i + 1, -1] = c1  # left boundary
        c_0_p[gamma12_z_i, gamma12_r_i + 1:] = c_0_p[gamma12_z_i - 1, gamma12_r_i + 1:]  # gamma 02 sub-air
        # c2
        c2_0_p[0, :] = c2_0_p[1, :]  # button
        c2_0_p[:gamma12_z_i + 1, -1] = c2_0_p[:gamma12_z_i + 1, -2]  # left boundary
        c2_0_p[gamma12_z_i, gamma12_r_i + 1:] = c2_0_p[gamma12_z_i - 1, gamma12_r_i + 1:]  # gamma 02 sub-air

        # 2.2 colony field.
        phi_box_bound[:gamma12_z_i, :] = False
        # c1
        c_1_p = mask_change_value(c_1_p, 0., phi_box_bound)
        c_1_p[gamma12_z_i, :gamma12_r_i + 1] = (c_1_p[gamma12_z_i + 1, :gamma12_r_i + 1] +
                                                c_0_p[gamma12_z_i - 1, :gamma12_r_i + 1]) / 2.  # gamma 12
        c_0_p[gamma12_z_i, :gamma12_r_i + 1] = c_1_p[gamma12_z_i, :gamma12_r_i + 1]  # gamma 12
        # c2
        c2_1_p = mask_change_value(c2_1_p, 0., phi_box_bound)
        c2_1_p[gamma12_z_i, :gamma12_r_i + 1] = (c2_1_p[gamma12_z_i + 1, :gamma12_r_i + 1] +
                                                 c2_0_p[gamma12_z_i - 1, :gamma12_r_i + 1]) / 2.  # gamma 12
        c2_0_p[gamma12_z_i, :gamma12_r_i + 1] = c2_1_p[gamma12_z_i, :gamma12_r_i + 1]  # gamma 12

        c_0_p = c_0_p * omega_0  # make sure that the air have no nutrients.
        c2_0_p = c2_0_p * omega_0
        c_1_p = c_1_p * omega_1
        c2_1_p = c2_1_p * omega_1

        # update field
        c_0 = c_0_p
        c_1 = c_1_p
        c2_0 = c2_0_p
        c2_1 = c2_1_p
        it_n = it_n + 1

        # convergence ?
        error_c1_1 = np.abs(c_0_p - c_0)
        error_c1_2 = np.abs(c_1_p - c_1)
        error_c2_1 = np.abs(c2_0_p - c2_0)
        error_c2_2 = np.abs(c2_1_p - c2_1)
        max_err_c1_1 = np.max(error_c1_1 / np.max(np.abs(c_0)))
        max_err_c1_2 = np.max(error_c1_2 / np.max(np.abs(c_1)))
        max_err_c2_1 = np.max(error_c2_1 / np.max(np.abs(c2_0)))
        max_err_c2_2 = np.max(error_c2_2 / np.max(np.abs(c2_1)))
        # max_err_p = np.max((max_err_c1, max_err_c2))
        error_torrlence = 1e-9
        if (max_err_c1_2 < error_torrlence) and (max_err_c2_2 < error_torrlence):
            # print(f"Nutrients update: {it_n}")
            break

    # magic
    # c_0_p = mask_change_value(c_0_p, float(c1), c_0_p > c1)
    # c_1_p = mask_change_value(c_1_p, float(c1), c_1_p > c1)
    # c_0_p = mask_change_value(c_0_p, float(0.), c_0_p < 0.)
    # c_1_p = mask_change_value(c_1_p, float(0.), c_1_p < 0.)
    # c2_0_p = mask_change_value(c2_0_p, float(c2), c2_0_p > c2)
    # c2_1_p = mask_change_value(c2_1_p, float(c2), c2_1_p > c2)
    # c2_0_p = mask_change_value(c2_0_p, float(0.), c2_0_p < 0.)
    # c2_1_p = mask_change_value(c2_1_p, float(0.), c2_1_p < 0.)

    return c_0_p, c_1_p, c2_0_p, c2_1_p, it_n, max_err_c1_2, max_err_c2_2


@njit(fastmath=True, cache=True)
def update_nutrients(nutrients_pars):
    (N_c, phi0, c_0, c_1, c2_0, c2_1, omega_0, omega_1, K1,
     K2, f1, f2, lambda1, lambda2, Dc_V, Dc, dr, dz, dt, c1, c2, gamma12_z_i, gamma12_r_i,
     phi_box_bound) = nutrients_pars

    # max_err_c1 = 1.
    # max_err_c2 = 1.
    # it_n = 0
    # while (max_err_c1 > 1e-3) or (max_err_c2 > 1e-3):
    #     it_n += 1
    #     if it_n%10000 == 0:
    #         print(max_err_c1)
    for it_n in range(N_c):
        # 2. update nutrient conc
        # c_0/c_1 : nutrients 1 in substrate/colony,
        # c2_0/c2_1: nutrients 2 in substrate/colony,

        theta_n1 = KM(c_1, K1)
        theta_n2 = KM(c2_1, K2)

        c1_consump = - f1 * lambda1 * theta_n1  # carbon consumption =: rho_DW / Da_carbon *  kg / Y, Y is yield factor
        Dc_term = lap_2d_cart_iso(Dc_V * c_1, dr, dz, case=0)
        c1_1_rhs = Dc_term + c1_consump * phi0
        c_1_p = c_1 + dt * c1_1_rhs

        c_0_rhs = lap_2d_cart_iso(Dc * omega_0 * c_0, dr, dz, case=0)
        c_0_p = c_0 + dt * c_0_rhs

        c2_consump = - f2 * lambda2 * theta_n2 * (1. - theta_n1)
        Dc2_term = lap_2d_cart_iso(Dc_V * c2_1, dr, dz, case=0)
        c2_1_rhs = Dc2_term + c2_consump * phi0
        c2_1_p = c2_1 + dt * c2_1_rhs

        c2_0_rhs = lap_2d_cart_iso(Dc * omega_0 * c2_0, dr, dz, case=0)
        c2_0_p = c2_0 + dt * c2_0_rhs

        # 2. boundary condition.
        # 2.1 substrate condition
        # c1
        c_0_p[0, :] = c1  # button
        c_0_p[:gamma12_z_i + 1, -1] = c1  # left boundary
        c_0_p[gamma12_z_i, gamma12_r_i + 1:] = c_0_p[gamma12_z_i - 1, gamma12_r_i + 1:]  # gamma 02 sub-air
        # c2
        c2_0_p[0, :] = c2  # button
        c2_0_p[:gamma12_z_i + 1, -1] = c2  # left boundary
        c2_0_p[gamma12_z_i, gamma12_r_i + 1:] = c2_0_p[gamma12_z_i - 1, gamma12_r_i + 1:]  # gamma 02 sub-air

        # 2.2 colony field.
        phi_box_bound[:gamma12_z_i, :] = False
        # c1
        c_1_p = mask_change_value(c_1_p, 0., phi_box_bound)
        c_1_p[gamma12_z_i, :gamma12_r_i + 1] = (c_1_p[gamma12_z_i + 1, :gamma12_r_i + 1] +
                                                c_0_p[gamma12_z_i - 1, :gamma12_r_i + 1]) / 2.  # gamma 12
        c_0_p[gamma12_z_i, :gamma12_r_i + 1] = c_1_p[gamma12_z_i, :gamma12_r_i + 1]  # gamma 12
        # c2
        c2_1_p = mask_change_value(c2_1_p, 0., phi_box_bound)
        c2_1_p[gamma12_z_i, :gamma12_r_i + 1] = (c2_1_p[gamma12_z_i + 1, :gamma12_r_i + 1] +
                                                 c2_0_p[gamma12_z_i - 1, :gamma12_r_i + 1]) / 2.  # gamma 12
        c2_0_p[gamma12_z_i, :gamma12_r_i + 1] = c2_1_p[gamma12_z_i, :gamma12_r_i + 1]  # gamma 12

        c_0_p = c_0_p * omega_0  # make sure that the air have no nutrients.
        c2_0_p = c2_0_p * omega_0
        c_1_p = c_1_p * omega_1
        c2_1_p = c2_1_p * omega_1

        # convergence ?
        error_c1_1 = np.abs(c_1_p - c_1)
        error_c2_2 = np.abs(c2_1_p - c2_1)
        max_err_c1 = np.max(error_c1_1 / np.max(np.abs(c_1_p)))
        max_err_c2 = np.max(error_c2_2 / np.max(np.abs(c2_1_p)))
        # max_err_p = np.max((max_err_c1, max_err_c2))

        if (max_err_c1 < 1e-4) and (max_err_c2 < 1e-4):
            # print(f"Nutrients update: {it_n}")
            break
        # update field
        c_0 = c_0_p
        c_1 = c_1_p
        c2_0 = c2_0_p
        c2_1 = c2_1_p

    # magic
    c_0_p = mask_change_value(c_0_p, float(c1), c_0_p > c1)
    c_1_p = mask_change_value(c_1_p, float(c1), c_1_p > c1)
    c_0_p = mask_change_value(c_0_p, float(0.), c_0_p < 0.)
    c_1_p = mask_change_value(c_1_p, float(0.), c_1_p < 0.)
    c2_0_p = mask_change_value(c2_0_p, float(c2), c2_0_p > c2)
    c2_1_p = mask_change_value(c2_1_p, float(c2), c2_1_p > c2)
    c2_0_p = mask_change_value(c2_0_p, float(0.), c2_0_p < 0.)
    c2_1_p = mask_change_value(c2_1_p, float(0.), c2_1_p < 0.)

    return c_0, c_1, c2_0, c2_1, it_n, max_err_c1, max_err_c2


@njit()
def diauxie_local_gr(c1, c2, k1, k2, lambda1, lambda2):
    """
    $$
    \lambda = \lambda_1 \theta_1 + \lambda_2 (1 - \theta_1)\\
    \lambda_1 = \lambda_1^{\mathrm{max}} \cdot \theta_1 \\
    \lambda_2 = \lambda_2^{\mathrm{max}}  \cdot \theta_2
    $$
    Parameters
    ----------
    c1 :
    c2 :
    k1 :
    k2 :
    lambda1 :
    lambda2 :

    Returns
    -------

    """
    theta1 = KM(c1, k1)
    theta2 = KM(c2, k2)
    lambda_1_loc = lambda1 * theta1
    lambda_2_loc = lambda2 * theta2
    local_gr = lambda_1_loc * theta1 + lambda_2_loc * (1. - theta1)
    local_gr = mask_change_value(local_gr, float(0.), local_gr < 0.)
    local_gr = mask_change_value(local_gr, float(lambda1), local_gr > lambda1)
    return local_gr


@njit()
def KM(c, k):
    theta = c / (c + k)
    return theta


# @jit(nopython=True, cache=True)
def togglesolver(green, red, kg, phi0, phi, ur, uz, gamma12_z_i,
                 dr, dz, dt, N_t: int = 1):
    """
    double tau_G = 0.015;
    double tau_R = 0.13;
    double kG = 14. / CELL_V;  // default 20.
    double kR = 10. / CELL_V;  // default 10.
    double nG = 4.;
    double nR = 2.;
    double deltaP = 0.05;
    p[0] = alphaG(gr) * hillFunc(tau_G, kR, nR, (double)x[1] / CELL_V); // O -> G
    p[1] = alphaR(gr) * hillFunc(tau_R, kG, nG, (double)x[0] / CELL_V); // O -> R
    p[2] = gr * (double)x[0]; // G -> O
    p[3] = gr * (double)x[1]; // R -> O
    Parameters
    ----------
    green : np.ndarray
        green protein conc.
    red : np.ndarray
        red protein conc.
    kg : np.ndarray
        local growth rate
    phi0 : np.ndarray
        cell phase
    phi :
    ur : np.ndarray
    uz : np.ndarray
    gamma12_z_i:
    dr :
    dz :
    dt :
    N_t:

    Returns
    -------
    r_p, g_p : tuple[np.ndarray, 2]
        (r_p, g_p)
    """
    ur0 = ur.copy()
    uz0 = uz.copy()

    phi_box = phi > .01
    phi_box_index = np.nonzero(phi_box)
    v_box = (phi > .01).astype(phi0.dtype)

    growth_box = kg * v_box

    G_reaction = alphaG(kg) * hillFunc(0.015, 10., 2., red)
    R_reaction = alphaR(kg) * hillFunc(0.13, 14., 4., green)

    divU = div_U_2d_cart_iso(ur0, uz0, dr, dz)
    divU_err = np.abs(divU - growth_box) / growth_box
    divU_error_mask = divU_err >= .10
    divU_mask = np.logical_or(divU_error_mask, divU > kg.max())
    ur0[divU_mask] = 0.  # remove speed error
    uz0[divU_mask] = 0.  # remove speed error
    v_ur = v_box * ur0
    v_uz = v_box * uz0
    v_uz[:gamma12_z_i + 1, :] = 0.

    v_green = v_box * green
    v_red = v_box * red

    v_g_phi = v_box * green * phi0
    v_r_phi = v_box * red * phi0

    dphidr = dfdr_2d_cart_iso(phi, dr, 0)
    dphidz = dfdz_2d_cart_iso(phi, dz)
    dR_dr = dfdr_2d_cart_iso(v_red, dr, 0)
    dR_dz = dfdz_2d_cart_iso(v_red, dz)
    dG_dr = dfdr_2d_cart_iso(v_green, dr, 0)
    dG_dz = dfdz_2d_cart_iso(v_green, dz)

    adv_green = ((v_green * dphidr + dG_dr * phi) * v_ur + (v_green * dphidz + dG_dz * phi) * v_uz
                 + v_g_phi * growth_box)
    adv_red = ((v_red * dphidr + dR_dr * phi) * v_ur + (v_red * dphidz + dR_dz * phi) * v_uz
               + v_r_phi * growth_box)

    g_rhs = - adv_green + v_box * G_reaction * phi0
    r_rhs = - adv_red + v_box * R_reaction * phi0

    prod_G = (2. * phi0 - phi) * v_green + dt * g_rhs
    prod_R = (2. * phi0 - phi) * v_red + dt * r_rhs

    g_p = np.zeros(green.shape)
    g_p_value = get_value_2d(prod_G, phi_box_index) / get_value_2d(phi0, phi_box_index)
    g_p = change_value_2d(g_p, g_p_value, phi_box_index)

    r_p = np.zeros(red.shape)
    r_p_value = get_value_2d(prod_R, phi_box_index) / get_value_2d(phi0, phi_box_index)
    r_p = change_value_2d(r_p, r_p_value, phi_box_index)

    red = r_p
    green = g_p
    # houndary = find_boundary(v_box)
    # boundary_index = np.nonzero(houndary)
    # g_p[boundary_index] = 0.
    # r_p[boundary_index] = 0.
    # Magic
    # zero_index_g = np.nonzero(g_p > 0)

    # # no phase field
    # g_p = dt * r_rhs + v_green
    # r_p = dt * g_rhs + v_red

    return g_p, r_p


# @jit(nopython=True, cache=True)
def togglesolver2(green, red, kg, phi0, phi, ur, uz, gamma12_z_i,
                  dr, dz, dt, N_t: int = 1):
    """
    double tau_G = 0.015;
    double tau_R = 0.13;
    double kG = 14. / CELL_V;  // default 20.
    double kR = 10. / CELL_V;  // default 10.
    double nG = 4.;
    double nR = 2.;
    double deltaP = 0.05;
    p[0] = alphaG(gr) * hillFunc(tau_G, kR, nR, (double)x[1] / CELL_V); // O -> G
    p[1] = alphaR(gr) * hillFunc(tau_R, kG, nG, (double)x[0] / CELL_V); // O -> R
    p[2] = gr * (double)x[0]; // G -> O
    p[3] = gr * (double)x[1]; // R -> O
    Parameters
    ----------
    green : np.ndarray
        green protein conc.
    red : np.ndarray
        red protein conc.
    kg : np.ndarray
        local growth rate
    phi0 : np.ndarray
        cell phase
    phi :
    ur : np.ndarray
    uz : np.ndarray
    gamma12_z_i:
    dr :
    dz :
    dt :
    N_t:

    Returns
    -------
    r_p, g_p : tuple[np.ndarray, 2]
        (r_p, g_p)
    """
    ur0 = ur.copy()
    uz0 = uz.copy()

    # phi_box = phi > .5
    # phi_box_index = np.nonzero(phi_box)
    v_box = (phi > .95).astype(phi0.dtype)
    phi_box = phi > .95
    phi_box_bound = find_boundary(phi_box)
    growth_box = v_box * kg

    dt = dt / N_t
    for _ in range(N_t):
        G_reaction = alphaG(kg) * hillFunc(0.015, 10., 2., red)
        R_reaction = alphaR(kg) * hillFunc(0.13, 14., 4., green)

        v_ur = v_box * ur0
        v_uz = v_box * uz0
        v_uz[:gamma12_z_i + 1, :] = 0.

        v_green = v_box * green
        v_red = v_box * red

        # v_g_phi = v_box * green
        # v_r_phi = v_box * red

        dR_dr = dfdr_2d_cart_iso(v_red, dr, 0)
        dR_dz = dfdz_2d_cart_iso(v_red, dz)
        dG_dr = dfdr_2d_cart_iso(v_green, dr, 0)
        dG_dz = dfdz_2d_cart_iso(v_green, dz)

        adv_green = (dG_dr * v_ur + dG_dz * v_uz) + v_green * growth_box
        adv_red = (dR_dr * v_ur + dR_dz * v_uz) + v_red * growth_box

        g_rhs = - adv_green + v_box * G_reaction
        r_rhs = - adv_red + v_box * R_reaction

        # update the toggle, no phase field
        g_p = dt * r_rhs + v_green
        r_p = dt * g_rhs + v_red
        green = g_p * v_box
        red = r_p * v_box

    green = mask_change_value(green, 0., phi_box_bound)
    green = mask_change_value(green, 0., green < 0.)
    red = mask_change_value(red, 0., phi_box_bound)
    red = mask_change_value(red, 0., green < 0.)

    return green, red


@njit
def togglesolver3(green, red, kg, phi0, phi, ur, uz, gamma12_z_i,
                  dr, dz, dt, N_t: int = 1,
                  alphaG_pars=(2.1, 33.8, 627.0, 1.6, 0.5, 6.2),
                  alphaR_pars=(2.3, 27.8, 320.99999999999994, 1.0, 0.4, 7.8),
                  hillG=(0.13, 14., 4.),
                  hillR=(0.015, 10., 2.)):
    """
    double tau_G = 0.015;
    double tau_R = 0.13;
    double kG = 14. / CELL_V;  // default 20.
    double kR = 10. / CELL_V;  // default 10.
    double nG = 4.;
    double nR = 2.;
    double deltaP = 0.05;
    p[0] = alphaG(gr) * hillFunc(tau_G, kR, nR, (double)x[1] / CELL_V); // O -> G
    p[1] = alphaR(gr) * hillFunc(tau_R, kG, nG, (double)x[0] / CELL_V); // O -> R
    p[2] = gr * (double)x[0]; // G -> O
    p[3] = gr * (double)x[1]; // R -> O
    Parameters
    ----------
    hillR :
    hillG :
    alphaR_pars :
    alphaG_pars :
    green : np.ndarray
        green protein conc.
    red : np.ndarray
        red protein conc.
    kg : np.ndarray
        local growth rate
    phi0 : np.ndarray
        cell phase
    phi :
    ur : np.ndarray
    uz : np.ndarray
    gamma12_z_i:
    dr :
    dz :
    dt :
    N_t:

    Returns
    -------
    r_p, g_p : tuple[np.ndarray, 2]
        (r_p, g_p)
    """

    # phi_box = phi > .5
    # phi_box_index = np.nonzero(phi_box)
    phi_box = phi > .1
    phi_box_index = np.nonzero(phi_box)
    # phi_box_bound = find_boundary(phi_box)
    # growth_box = v_box * kg

    D_cell = 16.

    G_reaction = alphaG(kg, alphaG_pars) * hillFunc(hillR[0], hillR[1], hillR[2], red) * phi0
    R_reaction = alphaR(kg, alphaR_pars) * hillFunc(hillG[0], hillG[1], hillG[2], green) * phi0

    # G_reaction = alphaG(kg) * hillFunc(0.015, 4., 2, red)  # 5/26/24 update
    # R_reaction = alphaR(kg) * hillFunc(0.13, 14., 4, green)  # 5/26/24 update

    Dr_PhiG, Dz_PhiG = grad_f_g_consv(phi0, green, dr, dz)
    Dr_PhiR, Dz_PhiR = grad_f_g_consv(phi0, red, dr, dz)

    Diff_PhiR = D_cell * div_f_div_g_consv(phi, red, dr, dz)
    Diff_PhiG = D_cell * div_f_div_g_consv(phi, green, dr, dz)

    RHS_G = - (ur * Dr_PhiG + uz * Dz_PhiG) + G_reaction - phi0 * green * kg + Diff_PhiG
    RHS_R = - (ur * Dr_PhiR + uz * Dz_PhiR) + R_reaction - phi0 * red * kg + Diff_PhiR

    Phi_Gp = (2. * phi0 - phi) * green + dt * RHS_G
    Phi_Rp = (2. * phi0 - phi) * red + dt * RHS_R

    Gp = np.zeros(phi.shape)
    Rp = np.zeros(phi.shape)

    Gp_value = get_value_2d(Phi_Gp, phi_box_index) / get_value_2d(phi0, phi_box_index)
    Gp = change_value_2d(Gp, Gp_value, phi_box_index)

    Rp_value = get_value_2d(Phi_Rp, phi_box_index) / get_value_2d(phi0, phi_box_index)
    Rp = change_value_2d(Rp, Rp_value, phi_box_index)

    Gp = mask_change_value(Gp, 0., Gp < 0.)
    Rp = mask_change_value(Rp, 0., Rp < 0.)

    return Gp, Rp


@njit()
def free_energy_G(phi):
    """
    Double potential energy
    Parameters
    ----------
    phi :

    Returns
    -------

    """
    return 18 * phi * phi * (1. - phi * phi)


# @jit(nopython=True, fastmath=True)
def norm_length(u, M):
    mask = M == 0
    direction = np.zeros(mask.shape)
    direction[~mask] = u[~mask] / M[~mask]
    return direction


@jit(nopython=True, fastmath=True, cache=True)
def get_dG(phi: np.ndarray) -> np.ndarray:
    """
    Prime of potential
    36. * phi * (1. - phi) * (1. - 2 * phi)
    Parameters
    ----------
    phi : np.array
        scale field phi

    Returns
    -------
    dG/dphi
        derivative G(phi), d_G/d_phi.
    """
    return 36. * phi * (1. - phi) * (1. - 2. * phi)


def savePars(save_path, paras: dict):
    """
    save parameters dir as a json file

    Parameters
    ----------
    save_path :
    paras :

    Returns
    -------

    """
    json_str = json.dumps(paras)
    dir_name = os.path.dirname(save_path)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    with open(save_path, 'w') as j_file:
        j_file.write(json_str)
    return None


class ColonyPaseField:
    dt = 1e-4
    Dc = 0.  # density diffusion
    max_time = 0 + 10 * 1e-4

    nu0 = 5.  # viscosity of the colony (liquid), default 100
    # trick = 200

    BulkMod = 100.  # BulkMod default 1000
    rhoi = 1.0

    xitrick = 400e1  # trick parameter should comparable to the magnitude of force.
    xi = 0.
    nutrick = 800e1

    epsilon = 1
    Gamma = 0.1

    gamma = 0.05  # surface tension， default 3.5
    Adh = 100.  # substrate adhesion
    g = 5.e3  # substrate penetration barrier
    xis = 10.  # substrate friction

    zs = -1.  # substrate height coordinate
    delta = 1e-2  # substrate width

    lambda1 = 1.
    lambda2 = .4
    c1 = 1.
    c2 = 1.
    f1 = 555.5556 / .5  # rho_DW / Da_carbon, Y is yield factor
    f2 = 1.39e6 / .5 * 2
    p2 = 1.39e6 / .1 * 2

    c1_init_ratio = .1

    r0 = 0.5  # initial radius
    phi0_r = 0
    phi0_z = 100

    lambda_ = 2.5e-3  # lambda_ is the threshold determining the colony region

    max_steps = 1e4  # maximum iteration steps
    error_limit = 1e-4

    m = 2 ** 7  # box number in r dir
    n = 2 ** 8  # box number in z dir
    Lr = 25
    Lz = 25
    Lz_up = 25
    time_save_step = 0.02
    N_c = 1
    N_t = 1

    alphaG_pars = None
    alphaR_pars = None
    hillG = None
    hillR = None

    def __init__(self, pars_dict: dict = None, iter_log=False):

        self.t_start = None
        self.time_now = None
        self.volume = None
        self.radius = None
        self.height = None
        self.save_dir = None
        self.sim_name = None
        self.save_length = None
        self.time_list = None
        self.save_step = None
        self.kg = None  # type: Optional[ndarray]
        self.zg = None
        self.w_chi = None
        self.chi = None  # type: Optional[ndarray]
        self.uz0 = None
        self.ur0 = None
        self.rho0 = None
        self.curv = None
        self.phi0 = None
        self.dis = None
        self.z = None
        self.r = None
        self.dz = None
        self.dr = None
        self.z_axis = None
        self.r_axis = None
        self.phi0_r = None
        self.phi0_z = None
        self.lambda_ = None  # type: Optional[float]
        self.rhoi = None  # type: Optional[float]
        self.r_max = None
        # nutrients related
        self.c_0 = None
        self.c_1 = None
        self.c_all = None
        self.C_m = None
        self.c2_0 = None
        self.c2_1 = None
        self.c2_all = None
        self.C2_m = None
        # toggle related
        self.green = None
        self.red = None
        self.green0 = None
        self.red0 = None
        self.pressure = None
        self.kappa_r = None
        self.kappa_z = None
        self.kp = None
        self.kp2 = None
        # colony shape
        self.radius = None
        self.height = None
        self.cell_v = None
        # running info
        self.nutrients_verbose = None
        self.velocity_verbose = None
        self.time_iter_nutrients = 0
        self.time_iter_velocity = 0
        self.device = 'cpu'

        if pars_dict is not None:
            self.pars_dict = pars_dict
        else:
            self.pars_dict = dict(
                dt=.5e-4,  # seconds
                max_time=2 + 2 * 1e-3,

                xitrick=10000,  # trick parameter should comparable to the magnitude of force.
                xi=0.,
                nutrick=20000,
                lambda_=0.02,  # lambda_ is the threshold determining the colony region

                nu0=100,  # viscosity of the colony (liquid), default 100. Unit: pN s / um
                BulkMod=100.0,  # BulkMod default 1000
                Dc=0,  # density diffusion  Unit: um**3 / s
                rhoi=1.0,

                epsilon=1,  # field width (liquid thickness of colony)  um
                Gamma=10,  # colony-void relaxation parameter. Unit: um/s
                gamma=3.5,  # surface tension， default 3.5 pN

                kg_max=1.6,
                zg=10,
                phi0_r=0,
                phi0_z=0,
                r0=8,  # initial radius
                zs=-8,  # substrate height coordinate, Unit: um
                delta=2,  # substrate width, suppose delta = epsilon
                Adh=100,  # substrate adhesion
                g=5e4,  # substrate penetration barrier
                xis=.2,  # substrate friction. Unit: Pa s / um
                m=2 ** 7 + 2 ** 3,  # box number in r dir
                n=2 ** 8 + 2 ** 4,  # box number in z dir
                Lr=50,
                Lz=50,

                max_steps=10000,  # maximum iteration steps
                error_limit=0.005,

                time_save_step=.01,
                save_dir=r'./sim_rets',  # r'./sim_rets',
                sim_name='Parameter_test_002')
        for par_key, par_v in self.pars_dict.items():  # load model parameters
            self.__dict__[par_key] = par_v

        self.velocity_loop_err = None
        self.iter_index = None
        if iter_log is True:
            self.iter_log = True
        else:
            self.iter_log = False

        self.plot_z_bottom = self.zs - 10.
        if self.plot_z_bottom < -self.Lz:
            self.plot_z_bottom = - self.Lz

        if gpu:
            mem_used = get_gpu_memory()
            selected_gup_index = np.argmin(np.array(mem_used))

            self.device = f"cuda:{selected_gup_index}"
            print(f'Using GPU: {self.device}')

    def save_parameters(self):
        if self.save_dir:
            savePars(os.path.join(self.save_dir, self.sim_name, 'colony_PF_pars.json'), self.pars_dict)

    def load_fields(self, file_path, index):
        """ load binary file containing fields at different time points"""
        fields_dict = load(file_path)
        self.r_axis = fields_dict['r_axis']
        self.z_axis = fields_dict['z_axis']
        self.dr = fields_dict['dr']
        self.dz = fields_dict['dz']
        self.r = fields_dict['r']
        self.z = fields_dict['z']
        self.chi = fields_dict['chi']
        self.w_chi = fields_dict['w_chi']
        self.phi0 = fields_dict['phi_save'][index]
        self.ur0 = fields_dict['u_save'][index, 0, ...]
        self.uz0 = fields_dict['u_save'][index, 1, ...]

        self.kg = fields_dict['kg_save'][index]
        self.c_0 = fields_dict['c_0_save'][index]
        self.c_1 = fields_dict['c_1_save'][index]
        self.c_all = fields_dict['c_all_save'][index]
        self.pressure = fields_dict['pressure_save'][index]

    def load_field(self, file_path):
        """Load binary file containing only single time point"""
        fields_dict = load(file_path)
        self.r_axis = fields_dict['r_axis']
        self.z_axis = fields_dict['z_axis']
        self.dr = fields_dict['dr']
        self.dz = fields_dict['dz']
        self.r = fields_dict['r']
        self.z = fields_dict['z']
        self.Lr = fields_dict['Lr']
        self.Lz = fields_dict['Lz']
        try:  # oldversion have no parameter Lz_up
            self.Lz_up = fields_dict['Lz_up']
        except KeyError:
            self.Lz_up = self.Lz
        self.m = fields_dict['m']
        self.n = fields_dict['n']
        self.zs_index = np.argmin(np.abs(colony.z[:, 0] - colony.zs))

        self.chi = fields_dict['chi']
        self.w_chi = fields_dict['w_chi']
        self.phi0 = fields_dict['phi0']
        self.ur0 = fields_dict['ur0']
        self.uz0 = fields_dict['uz0']
        self.kg = fields_dict['kg']
        self.c_0 = fields_dict['c_0']
        self.c_1 = fields_dict['c_1']
        self.c_all = fields_dict['c_all']
        self.c2_0 = fields_dict['c2_0']
        self.c2_1 = fields_dict['c2_1']
        self.c2_all = fields_dict['c2_all']
        self.green = fields_dict['green']
        self.red = fields_dict['red']
        self.pressure = fields_dict['pressure']
        self.kappa_r = fields_dict['kappa_r']
        self.kappa_z = fields_dict['kappa_z']
        self.kp = fields_dict['kp']
        self.kp2 = fields_dict['kp2']

    def expand_fields(self, expand_size, direction):
        """
            Expand/increase the field size.

            Note: expand_size should be even.
        Parameters
        ----------
        expand_size :
        direction :

        Returns
        -------

        """
        # 1. generate new axis
        if direction == 'r':
            raxis_len = len(self.r_axis)
            m_new = raxis_len + expand_size - 1
            r_axis_new = np.empty(m_new + 1)
            r_axis_new[:raxis_len] = self.r_axis
            r_axis_new[raxis_len:] = np.arange(1, expand_size + 1) * self.dr + self.r_axis[-1]
            Lr_new = m_new * self.dr
            r_new, z_new = np.meshgrid(r_axis_new, self.z_axis)
            self.Lr = Lr_new
            self.r, self.z = r_new, z_new
            self.r_axis = r_axis_new
            self.m = len(self.r_axis) - 1
            zero_flag = False
        else:  # direction == 'z'
            zaxis_len = len(self.z_axis)
            n_new = zaxis_len + expand_size - 1
            z_axis_new = np.empty(n_new + 1)
            z_axis_new[:zaxis_len] = self.z_axis
            z_axis_new[zaxis_len:] = np.arange(1, expand_size + 1) * self.dz + self.z_axis[-1]
            Lz_up_new = z_axis_new[-1]
            self.Lz_up = Lz_up_new
            r_new, z_new = np.meshgrid(self.r_axis, z_axis_new)
            self.r, self.z = r_new, z_new
            self.z_axis = z_axis_new
            self.n = len(self.z_axis) - 1
            zero_flag = True

        # 2. expand data
        self.phi0 = expand_field(self.phi0, expand_size, direction, zero_flag)
        self.w_chi = expand_field(self.w_chi, expand_size, direction, zero_flag)
        self.chi = expand_field(self.chi, expand_size, direction, zero_flag)
        self.ur0 = expand_field(self.ur0, expand_size, direction, zero_flag)
        self.uz0 = expand_field(self.uz0, expand_size, direction, zero_flag)
        self.c_0 = expand_field(self.c_0, expand_size, direction, zero_flag)
        self.c_1 = expand_field(self.c_1, expand_size, direction, zero_flag)
        self.c_all = expand_field(self.c_all, expand_size, direction, zero_flag)
        self.c2_1 = expand_field(self.c2_1, expand_size, direction, zero_flag)
        self.c2_0 = expand_field(self.c2_0, expand_size, direction, zero_flag)
        self.c2_all = expand_field(self.c2_all, expand_size, direction, zero_flag)
        self.green = expand_field(self.green, expand_size, direction, zero_flag)
        self.red = expand_field(self.red, expand_size, direction, zero_flag)
        self.kg = expand_field(self.kg, expand_size, direction, zero_flag)
        self.pressure = expand_field(self.pressure, expand_size, direction)
        # 3. generate kp field
        self.kappa_r = 2 * np.pi * fftfreq(2 * self.m, self.dr).astype(self.phi0.dtype)  # FFT wave number
        self.kappa_z = 2 * np.pi * fftfreq(len(self.z_axis), self.dz).astype(self.phi0.dtype)
        self.kp = np.array(np.meshgrid(self.kappa_z, self.kappa_r, indexing='ij'))
        self.kp2 = np.sum(self.kp * self.kp, axis=0, dtype=self.phi0.dtype)
        if gpu:

            self.kp2 = torch.from_numpy(self.kp2).to(self.device)
        return None

    def generateField(self, plot=True):
        self.max_time += self.dt

        # select a equal step
        delta_r = self.Lr / self.m
        delta_z = (self.Lz + self.Lz_up) / self.n
        delta_min = np.min((delta_z, delta_r))
        num_r_grid = int(((self.Lr / delta_min) // 2) * 2)
        self.m = num_r_grid
        self.Lr = delta_min * self.m

        # self.r_axis = np.linspace(0, self.Lr, num_r_grid + 1)
        self.r_axis = np.arange(stop=num_r_grid) * delta_min

        num_z_grid = int((((self.Lz + self.Lz_up) / delta_min) // 2) * 2) + 1
        z_length = num_z_grid * delta_min
        self.Lz = np.abs(self.Lz_up - z_length)
        # self.z_axis = np.linspace(-self.Lz, self.Lz_up, num_z_grid + 1)
        self.z_axis = np.arange(num_z_grid) * delta_min - self.Lz
        # print(f"z: {self.z_axis[1]-self.z_axis[0]}; r: {self.r_axis[1]-self.r_axis[0]}")
        self.dr = delta_min
        self.dz = delta_min

        self.r, self.z = np.meshgrid(self.r_axis, self.z_axis)
        self.pars_dict['r'] = self.r
        self.pars_dict['z'] = self.z
        self.pars_dict['dr'] = self.dr
        self.pars_dict['dz'] = self.dz

        # % initialize phi, chi, u, v
        # create init phi field
        # self.dis = np.sqrt((self.r - self.phi0_r) ** 2 +
        #                    (self.z - self.phi0_z) ** 2)
        # self.phi0 = 0.5 + 0.5 * np.tanh(3. * (self.r0 - self.dis) / self.epsilon)
        self.phi0 = create_dome_hyper(self.phi0_z, self.phi0_r, self.r_axis, self.z_axis)
        self.curv = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)

        self.rho0 = np.ones(self.r.shape) * self.phi0
        self.ur0 = np.ones(self.r.shape) * 0.
        self.uz0 = np.ones(self.r.shape) * 0.

        self.chi = compute_Chi(self.z, self.zs, self.delta)
        self.w_chi = compute_WChi(self.chi, self.Adh, self.delta, self.g, self.z, self.zs)

        # growth field
        self.kg = np.ones(self.phi0.shape) * self.lambda1
        # # growth field, I tried nonlinear growth field
        # self.kg = np.ones(self.phi0.shape) * self.kg_max
        # self.kg = self.kg / (1. + ((self.z - self.zs) / self.zg) ** 4)
        # concentration of nutrients
        self.c_0 = np.ones(self.phi0.shape) * self.c1 * self.chi
        zs_index = np.argmin(np.abs(self.z[:, 0] - self.zs))
        self.c_0[zs_index:, :] = 0.
        self.c_1 = np.ones(self.phi0.shape) * self.c1 * self.phi0 * self.c1_init_ratio
        self.c_all = np.ones(self.phi0.shape) * self.c1
        self.c2_0 = np.ones(self.phi0.shape) * self.c2 * self.chi
        zs_index = np.argmin(np.abs(self.z[:, 0] - self.zs))
        self.c2_0[zs_index:, :] = 0.
        self.c2_1 = np.ones(self.phi0.shape) * self.c2 * self.phi0
        self.c2_all = np.ones(self.phi0.shape) * self.c2
        # toggle switch
        self.green = self.green0 * (self.phi0 > 0.01)
        self.red = self.red0 * (self.phi0 > 0.01)

        # pressure
        self.pressure = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)

        # FFT wave number
        self.kappa_r = 2 * np.pi * fftfreq(2 * self.m, self.dr).astype(self.phi0.dtype)  # FFT wave number
        self.kappa_z = 2 * np.pi * fftfreq(len(self.z_axis), self.dz).astype(self.phi0.dtype)
        self.kp = np.array(np.meshgrid(self.kappa_z, self.kappa_r, indexing='ij'))
        self.kp2 = np.sum(self.kp * self.kp, axis=0, dtype=self.phi0.dtype)
        if gpu:
            self.kp2 = torch.from_numpy(self.kp2).to(self.device)

        if plot:
            # plot init growth field.
            fig_name = f"K_g_{self.sim_name}_init"
            if self.save_dir:
                filed_path = os.path.join(self.save_dir, self.sim_name)
            else:
                filed_path = None
            args_t = (np.copy(self.kg), self.Lr, self.Lz, self.Lz_up, 0, self.plot_z_bottom,
                      fig_name, filed_path)
            thred_plot_kg = Thread(target=plot_heatmap, args=args_t)
            thred_plot_kg.start()

            # plot W_chi
            fig_name = f"W_Chi_{self.sim_name}_init"
            if self.save_dir:
                filed_path = os.path.join(self.save_dir, self.sim_name)
            else:
                filed_path = None
            args_t = (np.copy(self.w_chi), self.Lr, self.Lz, self.Lz_up, 0, self.plot_z_bottom,
                      fig_name, filed_path)
            thred_plot_W_chi = Thread(target=plot_heatmap, args=args_t)
            thred_plot_W_chi.start()

            # plot Chi
            fig_name = f"Chi_{self.sim_name}_init"
            if self.save_dir:
                filed_path = os.path.join(self.save_dir, self.sim_name)
            else:
                filed_path = None
            args_t = (np.copy(self.chi), self.Lr, self.Lz, self.Lz_up, 0, self.plot_z_bottom,
                      fig_name, filed_path)
            thred_plot_chi = Thread(target=plot_heatmap, args=args_t)
            thred_plot_chi.start()

    def iterate_model(self, velocity_verbose=False, save_flag=True):

        self.save_step = int(self.time_save_step / self.dt)
        self.time_list = np.arange(0, self.max_time, step=self.dt)
        p_term_r = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        p_term_z = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        F_ten_r = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        F_ten_z = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        F_adh_r = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        F_adh_z = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        F_fric_r = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        F_fric_z = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        F_all_r = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        F_all_z = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        pp = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)
        divU = np.zeros(self.phi0.shape, dtype=self.phi0.dtype)

        save_i = 0

        iterror = ' '
        pbar = tqdm(self.time_list, unit_scale=self.dt, desc=self.sim_name, postfix=iterror,
                    bar_format="{desc}: \t {percentage:.3f}%|{bar}| {n:.3f}/{total: .3f} h [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        for ti, t_current in enumerate(pbar):

            # save images
            if (ti % self.save_step == 0) and (save_flag is True):
                self.radius, self.height = get_colony_radius_height(self.phi0, self.r, self.z, lambda_=.5)
                self.cell_v = calculate_colony_volume(self.phi0, self.r, self.z, lambda_=.5)
                # print(f"Cell radius: {radius}, height: {height}, volume: {cell_v}")
                # ================================ plot_colony
                # fig_boundary = radius * 1.1 if radius * 1.1 <= self.Lr else self.Lr
                force_sparse = 2
                # fig. 1, plot phi
                if self.save_dir:
                    filed_path = os.path.join(self.save_dir, self.sim_name)
                else:
                    filed_path = None

                # record time now
                self.time_now = t_current
                self.iter_num = ti
                # save binary data
                file_name = f"Results_{self.sim_name}_index-{'%04d' % save_i}_time-{'%.2f' % t_current}.pkl"
                self.save_results(file_name)
                self.save_running_info()
                self.time_iter_nutrients = 0
                self.time_iter_force = 0
                # self.plot_z_bottom =
                z_bottom_offset = -40

                # # fig. 3, plot velocity
                # fig_name = f"Velocity_{self.sim_name}_index-{'%04d' % save_i}_time-{'%.2f' % t_current}"
                # args_t = (self.r, self.z, np.copy(self.ur0), np.copy(self.uz0), self.plot_z_bottom + z_bottom_offset,
                #           force_sparse,
                #           fig_name, filed_path)
                # thread_plot_V = Thread(target=plot_F, args=args_t)
                # thread_plot_V.start()
                # # fig. 3, plot pressure grad
                # fig_name = f"Grad_Pressure_{self.sim_name}_index-{'%04d' % save_i}_time-{'%.2f' % t_current}"
                # args_t = (self.r, self.z, np.copy(p_term_r), np.copy(p_term_z), self.plot_z_bottom + z_bottom_offset,
                #           force_sparse,
                #           fig_name, filed_path)
                # thread_plot_P = Thread(target=plot_F, args=args_t)
                # thread_plot_P.start()
                # # fig. 4, plot Fource
                # # fig. 4.a surface tension
                # fig_name = f"Fource_ten_{self.sim_name}_index-{'%04d' % save_i}_time-{'%.2f' % t_current}"
                # args_t = (
                #     self.r, self.z, np.copy(F_ten_r), np.copy(F_ten_z), self.plot_z_bottom + z_bottom_offset,
                #     force_sparse,
                #     fig_name, filed_path)
                # thread_plot_ten = Thread(target=plot_F, args=args_t)
                # thread_plot_ten.start()
                # # fig. 4.b substrate adhesion
                # fig_name = f"Fource_adh_{self.sim_name}_index-{'%04d' % save_i}_time-{'%.2f' % t_current}"
                # args_t = (
                #     self.r, self.z, np.copy(F_adh_r), np.copy(F_adh_z), self.plot_z_bottom + z_bottom_offset,
                #     force_sparse,
                #     fig_name, filed_path)
                # thread_plot_adh = Thread(target=plot_F, args=args_t)
                # thread_plot_adh.start()
                # # fig. 4.c force composition
                # fig_name = f"Fource_comp_{self.sim_name}_index-{'%04d' % save_i}_time-{'%.2f' % t_current}"
                # args_t = (
                #     self.r, self.z, np.copy(F_all_r), np.copy(F_all_z), self.plot_z_bottom + z_bottom_offset,
                #     force_sparse,
                #     fig_name, filed_path)
                # thread_plot_adh = Thread(target=plot_F, args=args_t)
                # thread_plot_adh.start()
                # # fig. 4.d force friction
                # fig_name = f"Fource_fric_{self.sim_name}_index-{'%04d' % save_i}_time-{'%.2f' % t_current}"
                # args_t = (self.r, self.z, np.copy(F_fric_r), np.copy(F_fric_z), self.plot_z_bottom + z_bottom_offset,
                #           force_sparse,
                #           fig_name, filed_path)
                # thread_plot_adh = Thread(target=plot_F, args=args_t)
                # thread_plot_adh.start()
                # fig. 8 scaler field
                fig_name = f"A.Scaler_field_{self.sim_name}_index-{'%04d' % save_i}_time-{'%.2f' % t_current}"
                args_t = (dict(phi=np.copy(self.phi0), divU=np.copy(divU),
                               pressure=np.copy(pp), phi_box=self.phi0 > self.lambda_),
                          self.Lr, self.Lz, self.Lz_up, 0., self.plot_z_bottom + z_bottom_offset,
                          fig_name, filed_path,)
                thread_plot_colony = Thread(target=plot_4_heatmaps, args=args_t)
                thread_plot_colony.start()
                # fig. 9 nutrients field
                fig_name = f"B.Nutri_Conc_{self.sim_name}_index-{'%04d' % save_i}_time-{'%.2f' % t_current}"
                args_t = (dict(N1=np.copy(self.c_all), N2=np.copy(self.c2_all),
                               growth_rate=np.copy(self.kg), all_filed=self.phi0 + self.chi),
                          self.Lr, self.Lz, self.Lz_up, 0., self.plot_z_bottom + z_bottom_offset,
                          fig_name, filed_path,)
                thread_plot_colony = Thread(target=plot_4_heatmaps, args=args_t)
                thread_plot_colony.start()
                # fig. 10 toggle switch
                fig_name = f"C.Toggle_Conc_{self.sim_name}_index-{'%04d' % save_i}_time-{'%.2f' % t_current}"
                args_toggle = (dict(Green=np.copy(self.green), Red=np.copy(self.red),
                                    growth_rate=np.copy(self.kg * (self.phi0 > .1)),
                                    GoverR=np.log((self.green + 1) / (self.red + 1))),
                               self.Lr, self.Lz, self.Lz_up, None, self.plot_z_bottom + z_bottom_offset,
                               fig_name, filed_path,)
                thread_plot_toggle = Thread(target=plot_4_heatmaps, args=args_toggle)
                thread_plot_toggle.start()

                if gpu:  # check gpu resources.
                    mem_used = get_gpu_memory()
                    selected_gup_index = np.argmin(np.array(mem_used))

                    self.device = f"cuda:{selected_gup_index}"
                save_i += 1

            # STEP -1: if field growth
            if self.height / self.Lz_up >= 4 / 5:
                expand_num_z = int((((self.height / self.dz) * 0.25) // 2) * 2)
                self.expand_fields(expand_num_z, 'z')
            if self.radius / self.Lr >= 4 / 5:
                expand_num_r = int((((self.radius / self.dr) * 0.25) // 2) * 2)
                self.expand_fields(expand_num_r, 'r')

            # stop sumulation if radius > r_max
            if self.radius > self.r_max:
                break

            # STEP 1: determine laplacian.
            lapphi = lap_2d_cart_iso(self.phi0, self.dr, self.dz, 0)
            # STEP 2: update phase field phi
            phi_p = phisolver_2d_cart(self.phi0, lapphi, self.ur0, self.uz0,
                                      self.dr, self.dz, self.dt, self.Gamma, self.epsilon, )

            # STEP 3: update nutrient concentration

            # c_0_high = interpolate_mat(r_axis_low, z_axis_low, c_0_low, self.r, self.z)
            t_nutrients_start = time.time()
            (c_0_p, c_1_p, c2_0_p, c2_1_p, kg_p, c_all_p, c2_all_p, gamma12_z_i, self.nutrients_verbose) = \
                concsolver4(self.c_0, self.c_1, self.c2_0, self.c2_1,
                            phi_p, self.phi0, self.chi,
                            self.zs, self.r, self.z,
                            self.lambda1, self.lambda2, self.C_m, self.C2_m,
                            self.f1, self.f2, self.p2,
                            self.c1, self.c2, self.Dc,
                            self.dt, self.dr, self.dz, self.N_c)

            grad_p_r, grad_p_z = grad_2d_cart_iso(pp, self.dr, self.dz, case=0)
            p_term_r = - grad_p_r
            p_term_z = - grad_p_z
            t_nutrients_duration = time.time() - t_nutrients_start
            # STEP 4: update toggle

            green_p, red_p = togglesolver3(self.green, self.red, self.kg, self.phi0, phi_p, self.ur0, self.uz0,
                                           gamma12_z_i, self.dr, self.dz, self.dt, self.N_t,
                                           self.alphaG_pars, self.alphaR_pars,
                                           self.hillG, self.hillR)

            # STEP 5: solve velocity field u and v
            # STEP 5.a: calculate surface tension
            # Fr, Fz = compute_F_incomp_3d_cyl(phi, self.w_chi,
            #                                  self.r, self.dr, self.dz, self.epsilon, self.gamma)
            F_ten_r, F_ten_z = compute_F_ten(phi_p, self.gamma, self.epsilon, self.dr, self.dz)
            F_adh_r, F_adh_z = compute_F_adh(phi_p, self.w_chi, self.dr, self.dz)

            Fr = F_ten_r + F_adh_r  # imcompre
            Fz = F_ten_z + F_adh_z  # imcompre
            # Fr = F_ten_r # imcompre, no adh
            # Fz = F_ten_z # imcompre

            # STEP 5.b solve equation 10c
            t_velocity_start = time.time()
            (ur_p, uz_p, F_all_r, F_all_z, F_fric_r, F_fric_z, pp, self.velocity_verbose,
             self.xitrick, self.nutrick) = velocitysolver_dacy(phi_p, self.ur0, self.uz0, Fr, Fz, self.xis, self.chi,
                                                               self.delta,
                                                               self.nu0, self.kg, self.pressure,
                                                               self.dr, self.dz,
                                                               self.lambda_, self.xitrick, self.nutrick, self.xi,
                                                               self.kp, self.kp2,
                                                               self.error_limit, self.max_steps,
                                                               device=self.device,
                                                               verbose=velocity_verbose)
            t_velocity_duration = time.time() - t_velocity_start
            # # test phi growth
            # it_errors = dict(error_dist_list=[0])
            # ur_p = np.ones(self.phi0.shape) * 5
            # uz_p = np.ones(self.phi0.shape) * 15
            divU = div_U_2d_cart_iso(ur_p, uz_p, self.dr, self.dz)

            if t_current < self.t_start:
                # update toggle to sst more fast than real time-lapse
                for i in range(int(self.N_t)):
                    green_p, red_p = togglesolver3(self.green, self.red, self.kg, self.phi0, self.phi0,
                                                   self.ur0 * 0, self.ur0 * 0,
                                                   gamma12_z_i, self.dr, self.dz, self.dt, self.N_t,
                                                   self.alphaG_pars, self.alphaR_pars,
                                                   self.hillG, self.hillR
                                                   )
                    dif_G = np.abs(green_p - self.green)
                    dif_R = np.abs(red_p - self.red)
                    err_R = np.max(dif_R / np.max(self.red))
                    err_G = np.max(dif_G / np.max(self.green))
                    self.green = green_p
                    self.red = red_p
                    if (err_R < 1e-4) and (err_G < 1e-4):
                        break

            # update fields
            self.phi0 = phi_p
            self.ur0 = ur_p
            self.uz0 = uz_p
            self.c_0 = c_0_p
            self.c_1 = c_1_p
            self.c_all = c_all_p
            self.c2_1 = c2_1_p
            self.c2_0 = c2_0_p
            self.c2_all = c2_all_p
            self.green = green_p
            self.red = red_p
            self.pressure = pp
            self.kg = kg_p

            # iterror = '%.3f' % self.velocity_verbose['error_dist_list'][-1]  # update velocity error
            iterror = 'V: %.1f s; N: %.1f s' % (t_velocity_duration, t_nutrients_duration)
            pbar.set_postfix({'Time:': iterror})
            self.time_iter_velocity += t_velocity_duration
            self.time_iter_nutrients += t_nutrients_duration

    def runSim(self, velocity_verbose=False):
        self.save_parameters()
        self.generateField()
        self.iterate_model(velocity_verbose)
        # self.save_results()

    def save_results(self, filename=None):
        if self.save_dir:
            save_dir = os.path.join(self.save_dir, self.sim_name, f'{self.sim_name}_results')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dict = copy.deepcopy(self.__dict__)
            # save_pickle(save_dict, save_dir, filename, False)
            thread_save = Thread(target=save_pickle, args=(save_dict, save_dir, filename, False))
            thread_save.start()

    def save_running_info(self):
        if self.save_dir:
            save_dir = os.path.join(self.save_dir, self.sim_name, f'{self.sim_name}_results')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file_name = os.path.join(save_dir, 'running_info.log')
            if os.path.exists(save_file_name):
                with open(save_file_name, 'a') as f:
                    log = f"{time.strftime('%Y-%m-%d_%H:%M:%S')}\t Iter: {'%.3f' % self.time_now} {self.iter_num}\t" + \
                          f"Velocity: {self.velocity_verbose['iter_list'][-1]} {'%.5f' % self.velocity_verbose['error_max_list'][-1]} {'%.2f' % self.time_iter_velocity}\t" + \
                          f"Nutrients: {self.nutrients_verbose[0]} {'%.5f' % self.nutrients_verbose[1]} {'%.5f' % self.nutrients_verbose[2]} {'%.2f' % self.time_iter_nutrients}\n"
                    f.write(log)
            else:
                with open(save_file_name, 'w') as f:
                    f.write(f"{self.sim_name}\t {time.strftime('%Y-%m-%d_%H:%M:%S')}\n")


def plot_colony(phi, Lr, Lz, index: int, time: float, phi_thre: float = 0.5,
                r_bound: float = None, z_bound: float = None,
                name: Optional[str] = None, save_dir: Optional[str] = None):
    fig_colony, ax = plt.subplots(1, 1, figsize=(18, 18))
    phi_ext = func_ext(phi, 0)
    phi_ext_mask = phi_ext >= phi_thre
    phi_ext[~phi_ext_mask] = 0.
    mp1 = ax.imshow(phi_ext, origin='lower', extent=[-Lr, Lr, -Lz, Lz], cmap='jet')
    ax.set_title(f'Name: {name} Index: {index} Time: {time}')
    if r_bound is not None:
        ax.set_xlim(-r_bound, r_bound)
    if z_bound is not None:
        ax.set_ylim(-z_bound, z_bound)
    fig_colony.colorbar(mp1, ax=ax)
    if save_dir is not None:
        save_path = os.path.join(save_dir, name)
        save_fold = save_dir
        if not os.path.isdir(save_fold):
            os.makedirs(save_fold)
        fig_colony.savefig(f'{save_path}.png')
    else:
        fig_colony.show()
    plt.close(fig_colony)
    return None


def plot_heatmap(phi, Lr, Lz, Lz_up: float = None, phi_thre: float = None, z_btm: float = 0,
                 name: Optional[str] = None, save_dir: Optional[str] = None,
                 c_min: float = None, c_max: float = None, ax=None):
    if ax is None:
        fig_colony, ax = plt.subplots(1, 1, figsize=(22, 15))
    else:
        fig_colony = None
    if phi_thre is not None:
        phi_mask = phi >= phi_thre
        phi[~phi_mask] = 0.
    if (c_min is None) and (c_max is None):
        if phi.dtype in (np.float32, np.float64):
            vmax = np.nanquantile(phi, .95)
        else:
            vmax = None
        mp1 = ax.imshow(phi, origin='lower', extent=[0., Lr, -Lz, Lz_up], cmap='coolwarm', vmax=vmax)
    elif c_max is None:
        mp1 = ax.imshow(phi, origin='lower', extent=[0., Lr, -Lz, Lz_up], cmap='coolwarm', vmin=c_min)
    elif c_min is None:
        mp1 = ax.imshow(phi, origin='lower', extent=[0., Lr, -Lz, Lz_up], cmap='coolwarm', vmax=c_max)
    else:
        mp1 = ax.imshow(phi, origin='lower', extent=[0., Lr, -Lz, Lz_up], cmap='coolwarm', vmax=c_max, vmin=c_min)

    ax.set_title(name)
    ax.set_ylim(ymin=z_btm)
    plt.colorbar(mp1, ax=ax)

    if save_dir is not None:
        save_path = os.path.join(save_dir, name)
        save_fold = save_dir
        if not os.path.isdir(save_fold):
            os.makedirs(save_fold)
        fig_colony.savefig(f'{save_path}.png')
        plt.close(fig_colony)

    elif fig_colony is not None:
        fig_colony.show()
        return fig_colony, ax
    else:
        pass
    return None


def plot_4_heatmaps(phi_dict, Lr, Lz, Lz_up: float = None, phi_thre: float = None, z_btm: float = 0,
                    name: Optional[str] = None, save_dir: Optional[str] = None,
                    c_min: float = None, c_max: float = None):
    fig, axs = plt.subplots(2, 2, figsize=(22 * 4, 15 * 4))
    axs_list = axs.flatten()
    axi = 0
    for title, field in phi_dict.items():
        plot_heatmap(field, Lr, Lz, Lz_up, phi_thre, z_btm, name=title, c_min=c_min, c_max=c_max,
                     ax=axs_list[axi])
        axi += 1

    if save_dir is not None:
        save_path = os.path.join(save_dir, name)
        save_fold = save_dir
        if not os.path.isdir(save_fold):
            os.makedirs(save_fold)
        fig.savefig(f'{save_path}.png')
        plt.close(fig)
        # sleep(1)
    else:
        fig.show()
        return fig, axs

    plt.close(fig)
    # sleep(1)
    return None


def plot_F(r, z, fr, fz, z_btm: Optional[float] = None, sparse: int = 4,
           name: Optional[str] = None, save_dir: Optional[str] = None):
    f_abs = np.sqrt(fz ** 2 + fr ** 2)
    f_mask = f_abs > 1e-4
    fz_n = np.zeros(f_abs.shape)
    fr_n = np.zeros(f_abs.shape)
    fz_n[f_mask] = fz[f_mask] / f_abs[f_mask]
    fr_n[f_mask] = fr[f_mask] / f_abs[f_mask]
    fig2, ax2 = plt.subplots(1, 1, figsize=(18, 18))
    M = np.zeros(f_abs.shape)
    M[f_mask] = f_abs[f_mask]
    mp2 = ax2.quiver(r[::sparse, ::sparse], z[::sparse, ::sparse],
                     fr_n[::sparse, ::sparse], fz_n[::sparse, ::sparse],
                     M[::sparse, ::sparse], cmap='jet', scale_units='xy')

    if z_btm is not None:
        ax2.set_ylim(ymin=z_btm)
    else:
        pass
    ax2.set_xlim(xmin=0)
    ax2.set_title(name)
    fig2.colorbar(mp2, ax=ax2)

    if save_dir is not None:
        save_path = os.path.join(save_dir, name)
        save_fold = save_dir
        if not os.path.isdir(save_fold):
            os.makedirs(save_fold)
        fig2.savefig(f'{save_path}.png')
        plt.close(fig2)

    else:
        fig2.show()
        return fig2, ax2

    return None


@njit(fastmath=True)
def expand_field(data, expand_size: int, direction: str, zero_expand: bool = False):
    """
    a = np.zeros((2048,2048))
    %timeit b = expand_field(a, 512, 'z')

    Batch mark: no parallel 10.4 ms ± 366 µs per loop
                parallel    8.63 ms ± 1.75 ms per loop
    Parameters
    ----------
    data :
    expand_size :
    direction : str
    zero_expand: bool
        if False, the expanded values are that in boundary, otherwise the values are zero.
    Returns
    -------

    """
    len0 = data.shape[0]
    len1 = data.shape[1]

    if direction == 'r':
        data_new = np.empty((len0, len1 + expand_size))
        for j in prange(len1, len1 + expand_size):
            if zero_expand:
                for i in prange(len0):
                    data_new[i, j] = 0.
            else:
                for i in prange(len0):
                    data_new[i, j] = data[i, len1 - 1]
    else:
        data_new = np.empty((len0 + expand_size, len1))
        for i in prange(len0, len0 + expand_size):
            if zero_expand:
                for j in prange(len1):
                    data_new[i, j] = 0
            else:
                for j in prange(len1):
                    data_new[i, j] = data[len0 - 1, j]

    for i in prange(len0):
        for j in prange(len1):
            data_new[i, j] = data[i, j]

    return data_new


# time unit: h;
pars_model = dict(
    #=============== Simulation Parameters =========================
    dt=5e-4,  # 1e-5, hours  time step should not greater than .5e-3, it may cause instability of velocity field
    N_c=60,  # 60 is ok when dt=5e-4
    N_t=10,
    max_time=26. + 20 * 1e-3,
    t_start=0.0,
    r_max=1000,
    max_steps=5000,  # maximum iteration steps
    error_limit=0.0005,
    time_save_step=0.05,  # 1e-3,
    # =============== velocity fields =========================
    xitrick=500,  # trick parameter should comparable to the magnitude of force.
    xi=0.,
    nutrick=500,
    lambda_=1e-3,  # lambda_ is the threshold determining the colony region
    nu0=(10., 50.),  # viscosity of the colony (liquid), default 1. Unit: pN s / um  (r, z)
    BulkMod=1000.0,  # BulkMod default 1000
    Dc=110 * 3600,  # density diffusion  Unit: um**2 / s
    rhoi=1.0,
    epsilon=16.,  # field width (liquid thickness of colony)  um
    Gamma=80.,  # colony-void relaxation parameter. Unit: um/s
    gamma=100.,  # surface tension， default 3.5 pN
    zs=16,  # substrate height coordinate, Unit: um
    delta=16.,  # substrate width, suppose delta = epsilon
    Adh=10,  # substrate adhesion, < 1e4
    g=1e7,  # substrate penetration barrier, should greater than 4e4
    xis=0,  # substrate friction. Unit: Pa s / um
    # =============== Cell physiology ==========================
    lambda1=1.6,
    lambda2=.4,
    C_m=20.e-3,
    C2_m=5.,
    f1=(25 / 18) * 1e3 * (1 + .98),  # rho_DW / Da_carbon, Y is yield factor
    f2=(25 / 18) * 1e3 * 2 * (1 + .273),
    p2=(25 / 18) * 1e3 * 2 * 0.44,
    # ============= Nutrients field ============================
    c1=20.,  # mM
    c2=0.,
    c1_init_ratio=.1,
    # ============= Toggle Switch ============================
    green0=43.6,
    red0=5.4,
    alphaG_pars=(1.1, 40.609, 276.15846024, 1.0, 0.6568289999999999, 8.649725),
    alphaR_pars=(1.1, 26.836, 320.215, 1.0, 0.47195400000000015, 5.950950000000001),
    hillG=(0.13, 15.5, 4.0),
    hillR=(0.015, 10.0, 2.0),
    # ============= init conditions ==============================
    zg=5,
    phi0_r=300,
    phi0_z=30,
    r0=10,  # initial radius,  Deprecated.
    m=2 ** 7 + 2 ** 6,  # box number in r dir
    n=2 ** 6,  # box number in z dir
    Lr=350,
    Lz=100,
    Lz_up=100,

    save_dir=r'F:\example_data\colony_phase_field_sim_rets',  # Windows
    # save_dir=r'/media/fulab/fulab-nas/chupan/Data_Raid/Colony_Project/colony_phase_field_sim_rets/GColony_CroFed_Test',  # Linux platform
    sim_name='GColony_CroFed_Test')
# # evolve the Toggle
# t = Toggle(43.6, 5.4, gr=0.2)
#
# ret = t.ivp_odeint(max_time=20, dt=.1e-3)
#
# step = 100
# time_r = ret[1][::step]
# green_red_r = ret[0][::step, ...]
#
# fig, ax = plt.subplots(1, 1)
# ax.plot(ret[1], ret[0][:, 0], '-g', label='G')
# ax.plot(ret[1], ret[0][:, 1], '-r', label='R')
# ax.legend(loc='best')
# fig.show()

# %%
if __name__ == '__main__':

    if os.path.exists(os.path.join(pars_model['save_dir'], pars_model['sim_name'])):
        print('Clean the dir.')
        shutil.rmtree(os.path.join(pars_model['save_dir'], pars_model['sim_name']))
    colony = ColonyPaseField(pars_model)

    # # ================ if start the model de novo
    colony.runSim(velocity_verbose=False)
    # # ================ if start model from pickled data.
    # colony.save_parameters()
    # colony.generateField()
    # colony.load_field(
    #     r'''.\sim_rets\Results_GColony_Gamma=10_kg=1.6_Dc=100_Lz=50_delta=1_Conc2_test_1_index-0400_time-4.00.pkl''')
    # # revise the fields
    # v_box = colony.phi0 > 0.95
    # colony.green = 100. * v_box
    # colony.red = 30. * v_box
    # # zs_index = np.argmin(np.abs(colony.z[:, 0] - colony.zs))
    # # colony.c_0 = colony.c_0 / 5
    # # colony.c_0[zs_index:, :] = 0.
    # # colony.c_1 = colony.c_1 / 5
    # # colony.c_all = colony.c_all / 5
    # # colony.c2_0 = np.ones(colony.phi0.shape) * colony.c2
    # # colony.c2_0[zs_index:, :] = 0.
    # # colony.c2_1 = np.ones(colony.phi0.shape) * colony.c2
    # # colony.c2_all = np.ones(colony.phi0.shape) * colony.c2
    # colony.iterate_model(False)
