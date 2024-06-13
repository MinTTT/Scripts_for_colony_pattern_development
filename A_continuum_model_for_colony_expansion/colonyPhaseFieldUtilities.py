# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
from scipy.optimize import fsolve
from scipy.integrate import RK45, odeint
from scipy.interpolate import LinearNDInterpolator, interpn, RegularGridInterpolator

import numpy as np  # Or any other
from typing import Tuple, Union, Optional
# […]
# from numba import njit
import skfmm
import matplotlib.pyplot as plt
from numba import njit, jit, prange

from scipy.interpolate import LinearNDInterpolator, interpn, RegularGridInterpolator
# Own modules
from joblib import Parallel, delayed, dump, load

ndarray = np.ndarray

import subprocess as sp
import os
from threading import Thread, Timer
import sched, time


def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values


@njit(cache=True)
def hillFunc(leakage, k, n, p):
    hill = leakage + (1. - leakage) / (1. + (p / k) ** n)
    return hill


@njit(cache=True)
def alphaG(gr: Union[float, ndarray],
           pars=(2.1, 33.8, 627.0, 1.6, 0.5, 6.2)):
    # alpha = 1 * gr * (16.609 + 627.747 / (1.0 + (gr / 0.865) ** 4.635))
    # alpha = 1.1 * (gr) * (34.0 + 604.4 / (1.4 + (gr / 0.6000000000000001) ** 7.3))
    # pars = (1.1, 34.0, 604.4, 1.4, 0.6000000000000001, 7.3)  # togge2
    # pars = (2.1, 34.0, 605.2, 1.4, 0.5, 8.2)  # togge3
    # pars = (2.1, 33.8, 627.0, 1.6, 0.5, 6.2)  # toggle4
    alpha = pars[0] * gr * (pars[1] + pars[2] / (pars[3] + (gr / pars[4]) ** pars[5]))
    return alpha


@njit(cache=True)
def alphaR(gr: Union[float, ndarray],
           pars=(2.3, 27.8, 320.99999999999994, 1.0, 0.4, 7.8)):
    # alpha = 1 * gr * (26.836 + 320.215 / (1.0 + (gr / 0.661) ** 4.09))
    # alpha = 1.1 * (gr) * (26.836 + 326.9 / (1.0 + (gr / 0.5) ** 5.9))
    # # (1.1, 26.836, 326.9, 1.0, 0.5, 5.9)
    # alpha = 2.3 * (gr) * (26.4 + 327.49999999999994 / (1.0 + (gr / 0.4) ** 6.800000000000001))
    # pars = (2.3, 26.4, 327.49999999999994, 1.0, 0.4, 6.800000000000001)  # Toggle 3
    # pars = (2.3, 27.8, 320.99999999999994, 1.0, 0.4, 7.8)  # Toggle 4
    alpha = pars[0] * gr * (pars[1] + pars[2] / (pars[3] + (gr / pars[4]) ** pars[5]))
    return alpha


# Own modules
@jit(nopython=True, cache=True)
def change_value_2d(target, value, index):
    """
    change the values in the target matrix 2D.

    Parameters
    ----------
    target : ndarray
        2d matrix
    value : ndarray
        1d matrix
    index : tuple[ndarray, ndarray]
        tuple of two axis (0, 1)

    Returns
    -------
    target
        ndarray
    """
    # 2.49 µs ± 50.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    index_len = index[0].shape[0]
    for i in prange(index_len):
        target[index[0][i], index[1][i]] = value[i]
    return target


@jit(nopython=True, cache=True)
def mask_change_value(target, value: float, mask):
    index = np.nonzero(mask)
    index_len = index[0].shape[0]
    for i in prange(index_len):
        target[index[0][i], index[1][i]] = value
    return target


def get_low_res(mat: np.ndarray, space: int) -> np.ndarray:
    new_mat = mat[::space, ::space]
    return new_mat


def interpolate_mat(r_axis_low, z_axis_low, mat_low, r_high, z_high):
    """Interpolate a matrix using a linear interpolation."""
    interp = RegularGridInterpolator((z_axis_low, r_axis_low), mat_low,
                                     bounds_error=False, fill_value=None)
    mat_hight = interp((z_high, r_high))
    return mat_hight


@jit(nopython=True, cache=True)
def get_value_2d(target, index):
    target_shape = target.shape
    target_1d = np.ravel(target)
    index_1d = index[0] * target_shape[1] + index[1]
    value = target_1d[index_1d]
    return value


def find_boundary(box):
    grad_b_r, grad_map_z = grad_2d_cart_iso(box, 1, 1, 0)
    boundary = np.abs(grad_b_r) + np.abs(grad_map_z)
    boundary = boundary.astype(bool)
    return boundary


def get_colony_radius_height(phi: ndarray, r: ndarray, z: ndarray,
                             lambda_: float = 0.5) -> tuple:
    """
    Compute the radius and length of colony.

    Parameters
    ----------
    phi : ndarray
        phase field
    r : ndarray
        radius coordinate
    z : ndarray
        z coordiante
    lambda_ : float


    Returns
    -------


    """
    phi_ext_mask = phi >= lambda_
    # phi[~phi_ext_mask] = 0.

    r_mask = np.any(phi_ext_mask, axis=0)
    # r_bound = np.argmax(r_mask)
    radius = r[0, r_mask].max()

    z_mask = np.any(phi_ext_mask, axis=1)
    # z_bound = np.argmax(z_mask)
    height = z[z_mask, 0].max()

    return radius, height


# def crate_kappa(len_r, len_z, dr, dz):
#     kappa_r = 2 * np.pi * fftfreq(len_r, dr).astype(np.float64)  # FFT wave number
#     kappa_z = 2 * np.pi * fftfreq(len_z, dz).astype(np.float64)
#     kp_sub = np.array(np.meshgrid(kappa_z, kappa_r, indexing='ij'))
#     kp2_sub = np.sum(kp_sub * kp_sub, axis=0, dtype=np.float64)
#     if gpu:
#         kp2_sub = torch.from_numpy(kp2_sub).to(device)
#     yield kp2_sub


def calculate_colony_volume(phi, r, z, lambda_=0.5):
    # delta_r = red[0, 1] - red[0, 0]
    delta_z = z[1, 0] - z[0, 0]

    phi_ext_mask = phi >= lambda_
    z_mask = np.any(phi_ext_mask, axis=1)
    ColonyRegion = phi_ext_mask[z_mask, ...]
    delta_v = 0
    for r_section in ColonyRegion:
        radius_z = r[0, r_section].max()
        delta_v += np.pi * radius_z ** 2 * delta_z
    return delta_v


def data_type_convert(args, target=np.float32):
    """
    Convert data in a tuple
    Parameters
    ----------
    args : Tuple
        tuple with data
    target : object
        to the data type for conversion

    Returns
    -------

    """
    args = (arg.astype(target) if isinstance(arg, np.ndarray) else arg for arg in args)
    return args


@jit(nopython=True, fastmath=True, cache=True)
def shift_up(f):
    """ shift the upper element down
    the matrix 0 -> 1 along 0 axis. put [i-1, j] to [i, j]"""
    fu = np.empty(f.shape, dtype=f.dtype)
    fu[0, :] = f[-1, :]
    fu[1:, :] = f[:-1, :]
    return fu


@jit(nopython=True, fastmath=True, cache=True)
def shift_down(f):
    """ shift the lower element up
     the matrix 1 -> 0 along 0 axis. put [i+1, j] to [i, j]
    """

    fd = np.empty(f.shape, dtype=f.dtype)
    fd[-1, :] = f[0, :]
    fd[:-1, :] = f[1:, :]
    return fd


@jit(nopython=True, fastmath=True, cache=True)
def shift_left(f: ndarray):
    """
    shift the left element to right dirt, 0 -> 1. put [i, j-1] to [i, j]
    Parameters
    ----------
    f : np.ndarray

    Returns
    -------

    """
    fr = np.empty(f.shape, dtype=f.dtype)
    fr[:, 0] = f[:, -1]
    fr[:, 1:] = f[:, :-1]
    return fr


@njit(fastmath=True, cache=True)
def Sp_r(f: ndarray):
    """
    S+(f) = (f_{i+1) + f_{i}) / 2
    """
    S_p_r = (shift_right(f) + f) / 2.
    return S_p_r


@njit(fastmath=True, cache=True)
def Sp_z(f: ndarray):
    """
    S+(f) = (f_{j+1) + f_{j}) / 2
    """
    S_p_z = (shift_down(f) + f) / 2.
    return S_p_z


@njit(fastmath=True, cache=True)
def Sm_r(f: ndarray, case: int = 0):
    """
    S-(f) = (f_{i-1) + f_{i}) / 2
    """
    Sm_r = (shift_left(f) + f) / 2.
    if case % 2 == 0:
        Sm_r[:, 0] = (f[:, 0] + f[:, 1]) / 2.
    if case % 2 == 1:
        Sm_r[:, 0] = (f[:, 0] - f[:, 1]) / 2.

    return Sm_r


@njit(fastmath=True, cache=True)
def Dp_r(f: ndarray, dr):
    fp1 = shift_right(f)
    Dm_r = (fp1 - f) / dr
    return Dm_r


@njit(fastmath=True, cache=True)
def Dm_r(f: ndarray, dr, case: int = 0):
    fm1 = shift_left(f)
    if case % 2 == 0:
        fm1[:, 0] = f[:, 1]
    else:
        fm1[:, 0] = 2 * f[:, 0] - f[:, 1]
    Dm_r = (f - fm1) / dr
    return Dm_r


@njit(fastmath=True, cache=True)
def Dp_z(f: ndarray, dz):
    fp1 = shift_down(f)
    Dp_z = (fp1 - f) / dz
    return Dp_z


@njit(fastmath=True, cache=True)
def Dm_z(f: ndarray, dz):
    fm1 = shift_up(f)
    Dm_z = (f - fm1) / dz
    return Dm_z


@njit(fastmath=True, cache=True)
def Sm_z(f: ndarray):
    """
    S-(f) = (f_{i-1) + f_{i}) / 2
    """
    Sm_z = (shift_up(f) + f) / 2.
    return Sm_z


@njit(fastmath=True, cache=True)
def shift_right(f):
    """
    shift the right element to left dirt, 1 -> 0. put [i, j+1] to [i, j]

    Parameters
    ----------
    f :

    Returns
    -------

    """
    fl = np.empty(f.shape, dtype=f.dtype)
    fl[:, -1] = f[:, 0]
    fl[:, :-1] = f[:, 1:]
    return fl


@jit(nopython=True, fastmath=True, cache=True)
def func_ext(f, case):
    """
    extend the function, horizontally.
    extend along the 1 axis. (red)

    v1.0:
            if case % 2 == 1:
            f_ext = np.hstack((-np.fliplr(f[:, 1:-1]), f))
        elif case % 2 == 0:
            f_ext = np.hstack((np.fliplr(f[:, 1:-1]), f))
    Parameters
    ----------
    f : np.ndarray
        array [m, n]
    case : int
        if n == 0, even extend.
            [f(-x), f(x)]
        if n == 1, odd extend.
            [-f(-x), f(x)]

    Returns
    -------
    f_ext: np.ndarray
        extended function [m , 2n-1]
    """

    f_len_0 = f.shape[0]
    f_len_1 = f.shape[1]
    f_ext_len_1 = 2 * f_len_1 - 2
    f_ext = np.empty((f_len_0, f_ext_len_1), dtype=f.dtype)
    f_ext[:, f_len_1 - 2:] = f
    if case % 2 == 1:
        f_ext[:, :f_len_1 - 2] = -np.fliplr(f[:, 1:-1])
    elif case % 2 == 0:
        f_ext[:, :f_len_1 - 2] = np.fliplr(f[:, 1:-1])

    return f_ext


@njit(cache=True)
def func_red(f_ext):
    """
    Reduce the extended function.

    Parameters
    ----------
    f_ext : np.ndarray

    Returns
    -------
    f_red: np.ndarray
        function reduced
    """
    f_ext_len_1 = f_ext.shape[1]
    reduced_index = int(f_ext_len_1 / 2 - 1)
    f_red = f_ext[:, reduced_index:]
    return f_red


@njit(fastmath=True, cache=True)
def dfdr_2d_cart_iso(f, dr, case: int = 0):
    """

    df/dr with isotropic difference.
    # check
    different to origin code, let both boundary have same extension properties, origin one only
    apply the property at left boundary.

    V1.0:
        f_r_conv = dfdr(f, dr, case)

    # f_r_conv_d = np.roll(f_r_conv, shift=(-1, 0), axis=(0, 1))
    f_r_conv_d = shift_down(f_r_conv)
    # f_r_conv_u = np.roll(f_r_conv, shift=(1, 0), axis=(0, 1))
    f_r_conv_u = shift_up(f_r_conv)
    f_r_iso = 2. / 3. * f_r_conv + 1. / 6. * f_r_conv_d + 1. / 6. * f_r_conv_u
    Parameters
    ----------
    f : np.ndarray
    dr : float
    case : int

    Returns
    -------
    f_r_iso: np.ndarray
        df/dr
    """
    fr = shift_right(f)
    fl = shift_left(f)
    # fu = shift_up(f)
    # fd = shift_down(f)
    fru = shift_up(fr)
    frd = shift_down(fr)
    flu = shift_up(fl)
    fld = shift_down(fl)

    if case % 2 == 0:  # even
        fl[:, 0] = f[:, 1]
        fr[:, -1] = f[:, -2]
        # flu[:, 0] = fu[:, 1]
        flu[1:, 0] = f[:-1, 1]
        flu[0, 0] = f[-1, 1]
        # fru[:, -1] = fu[:, -2]
        fru[1:, -1] = f[:-1, -2]
        fru[0, -1] = f[-1, -2]
        # fld[:, 0] = fd[:, 1]
        fld[:-1, 0] = f[1:, 1]
        fld[-1, 0] = f[0, 1]
        # frd[:, -1] = fd[:, -2]
        frd[:-1, -1] = f[1:, -2]
        frd[-1, -1] = f[0, -2]
    else:  # odd case
        fl[:, 0] = -f[:, 1]
        fr[:, -1] = -f[:, -2]
        # flu[:, 0] = -fu[:, 1]
        flu[1:, 0] = -f[:-1, 1]
        flu[0, 0] = -f[-1, 1]
        # fru[:, -1] = -fu[:, -2]
        fru[1:, -1] = -f[:-1, -2]
        fru[0, -1] = -f[-1, -2]
        # fld[:, 0] = -fd[:, 1]
        fld[:-1, 0] = -f[1:, 1]
        fld[-1, 0] = -f[0, 1]
        # frd[:, -1] = -fd[:, -2]
        frd[:-1, -1] = -f[1:, -2]
        frd[-1, -1] = -f[0, -2]

    dfdr_iso = ((fr - fl) / 3. + (fru - flu) / 12. + (frd - fld) / 12.) / dr
    return dfdr_iso


@jit(nopython=True, fastmath=True, cache=True)
def dfdr_2d_cart(f, dr, case: int = 0):
    """

    df/dr with isotropic difference.
    # check
    different to origin code, let both boundary have same extension properties, origin one only
    apply the property at left boundary.

    V1.0:
        f_r_conv = dfdr(f, dr, case)

    # f_r_conv_d = np.roll(f_r_conv, shift=(-1, 0), axis=(0, 1))
    f_r_conv_d = shift_down(f_r_conv)
    # f_r_conv_u = np.roll(f_r_conv, shift=(1, 0), axis=(0, 1))
    f_r_conv_u = shift_up(f_r_conv)
    f_r_iso = 2. / 3. * f_r_conv + 1. / 6. * f_r_conv_d + 1. / 6. * f_r_conv_u
    Parameters
    ----------
    f : np.ndarray
    dr : float
    case : int

    Returns
    -------
    f_r_iso: np.ndarray
        df/dr
    """
    fr = shift_right(f)
    fl = shift_left(f)

    if case % 2 == 0:  # even
        fl[:, 0] = f[:, 1]
        fr[:, -1] = f[:, -2]

    else:  # odd case
        fl[:, 0] = -f[:, 1]
        fr[:, -1] = -f[:, -2]

    dfdr = (fr - fl) / dr

    return dfdr


@njit(fastmath=True, cache=True)
def dfdz_2d_cart_iso(f, dz) -> np.ndarray:
    """
    df/dz
    # check

    Parameters
    ----------
    f : np.ndarray
    dz : float

    Returns
    -------

    """

    # v2.0
    f_d = shift_down(f)
    f_u = shift_up(f)
    f_rd = shift_right(f_d)
    f_ld = shift_left(f_d)
    f_ru = shift_right(f_u)
    f_lu = shift_left(f_u)
    # boundary conditions, we suppose an even function at r=0.
    f_ld[:, 0] = f_rd[:, 0]
    f_lu[:, 0] = f_ru[:, 0]
    f_rd[:, -1] = f_ld[:, -1]
    f_ru[:, -1] = f_lu[:, -1]

    f_z_iso = ((f_rd - f_ru) / 12 + (f_ld - f_lu) / 12 + (f_d - f_u) / 3) / dz
    return f_z_iso
    # f_d = shift_down(f)
    # f_u = shift_up(f)
    # f_z = (f_d - f_u) / dz
    #
    # return f_z


@jit(nopython=True, fastmath=True, cache=True)
def dfdz_2d_cart(f, dz) -> np.ndarray:
    """
    df/dz
    # check

    Parameters
    ----------
    f : np.ndarray
    dz : float

    Returns
    -------

    """

    # v2.0
    f_d = shift_down(f)
    f_u = shift_up(f)
    f_z = (f_d - f_u) / dz

    return f_z


@jit(nopython=True, fastmath=True, cache=True)
def ddFdrz_2d_cart_iso(f, dr, dz, case):
    """
    cross derivative of f.

    Parameters
    ----------
    f: np.ndarray
        function for finite differences.
    dr
    dz
    case

    Returns
    -------
    frz: np.ndarray
    """

    # if dr != dz:
    #     raise TypeError('the isotropic finite differences only apply for isotropic grid.')
    f = func_ext(f, case)
    fr = shift_right(f)
    fl = shift_left(f)
    fru = shift_up(fr)  # i-1, j+1
    flu = shift_up(fl)  # i-1, j-1
    frd = shift_down(fr)  # i+1, j+1
    fld = shift_down(fl)  # i+1, j-1
    frz = (frd + flu - fru - fld) / (4. * dr * dz)
    ddfdrdz_iso = func_red(frz)
    return ddfdrdz_iso


@jit(nopython=True, fastmath=True, cache=True)
def ddF_2d_cart_iso(f_half: np.ndarray, dr: float, dz: float, case: int) -> tuple:
    """
    2nd derivative of f with isotropic finite differences.
    ref:
    Kumar, A. Isotropic finite-differences. J. Comput. Phys. 201, 109–118 (2004).



    Parameters
    ----------
    f_half: np.ndarray
        function for second derivative
    dr: float
        delta r
    dz: float
        delta z
    case:
        the case type for extend the function.

    Returns
    -------
    drddf, dzddf: tuple
        2nd derivative in r direction, 2nd derivative in z direction.

    """

    if dr != dz:
        raise TypeError('the isotropic finite differences only apply for isotropic grid.')

    f = func_ext(f_half, case)
    fd = shift_down(f)  # i+1, j
    fu = shift_up(f)  # i-1, j
    fr = shift_right(f)
    fl = shift_left(f)
    # fru = shift_right(fu)
    # flu = shift_left(fu)
    # frd = shift_right(fd)
    # fld = shift_left(fd)

    d2 = dr * dr

    # frr = ((frd - 2. * fd + fld) / 12.
    #        + (fru - 2. * fu + flu) / 12.
    #        + (fr - 2. * f + fl) * 5. / 6.) / d2
    #
    # fzz = ((fd - 2. * f + fu) * 5. / 6.
    #        + (frd - 2. * fr + fru) / 12.
    #        + (fld - 2. * fd + flu) / 12.) / d2

    frr = (fr - 2. * f + fl) / d2

    fzz = (fd - 2. * f + fu) / d2

    drddf = func_red(frr)
    dzddf = func_red(fzz)

    return drddf, dzddf


@njit(cache=True)
def grad_2d_cart_iso(f, dr, dz, case=0):
    """ Calculate grad(f), f is a scalar filed.

    Parameters
    ----------
    f : np.ndarray
    dr : float
    dz : float
    case: int (default 0)
        odd or even extend.

    Returns
    -------
    tuple
        par_f_r, par_f_z
    """
    divf_r = dfdr_2d_cart_iso(f, dr, case)
    divf_z = dfdz_2d_cart_iso(f, dz)

    return (divf_r, divf_z)


@jit(nopython=True, fastmath=True, cache=True)
def grad_f_g_cart_iso(f, g, dr, dz):
    """

    Parameters
    ----------
    f :
    g :
    dr :
    dz :

    Returns
    -------

    """
    dfdr, dfdz = grad_2d_cart_iso(f, dr, dz, 0)
    dgdr, dgdz = grad_2d_cart_iso(g, dr, dz, 0)

    dgf_dr = dfdr * g + dgdr * f
    dgf_dz = dfdz * g + dgdz * f
    return dgf_dr, dgf_dz


@njit(cache=True)
def grad_f_g_consv(f, g, dr, dz):
    """
    conservation law of /nabla{f g}
    Parameters
    ----------
    f :
    g :
    dr :
    dz :

    Returns
    -------

    """
    Dr_gf = (Sp_r(f) * Sp_r(g) - Sm_r(f) * Sm_r(g)) / dr
    Dz_gf = (Sp_z(f) * Sp_z(g) - Sm_z(f) * Sm_z(g)) / dz
    return Dr_gf, Dz_gf


@njit(cache=True)
def div_f_div_g_consv(f, g, dr, dz):
    Drr = (Sp_r(f) * Dp_r(g, dr) - Sm_r(f) * Dm_r(g, dr)) / dr
    Dzz = (Sp_z(f) * Dp_z(g, dz) - Sm_z(f) * Dm_z(g, dz)) / dz
    return Drr + Dzz


@njit(fastmath=True, cache=True)
def lap_2d_cart_iso(f, dr, dz, case=0):
    """  The scalar Laplacian
    Calculate the Laplacian of scaler field f.
    Lap(f) = Dr(DrF) + Dz(DzF)
    Parameters
    ----------
    dz: float
        z step
    f: np.ndarray
        scaler field
    dr: float
        r step
    Returns
    -------
    ddf: np.ndarray

    """

    frr, fzz = ddF_2d_cart_iso(f, dr, dz, case)
    ddf = fzz + frr
    return ddf


@njit(fastmath=True, cache=True)
def div_fu_2d_cart_iso(f, ur, uz, dr, dz):
    """
    Calculte div(f * U)

    div(f * U) = grad f * U + f div U

    Parameters
    ----------
    f : np.ndarray
        scalar field
    ur : np.ndarray
        scalar field of vector field U, U = [ur, uz]
    uz : np.ndarray
        scalar field of vector field U
    dr : float
        grid size of r
    dz : float
        grid size of z

    Returns
    -------

    """
    dfdr = dfdr_2d_cart_iso(f, dr, 0)
    dfdz = dfdz_2d_cart_iso(f, dz)

    dUr_dr = dfdr_2d_cart_iso(ur, dr, 1)
    dUz_z = dfdz_2d_cart_iso(uz, dz)

    div_fu = f * (dUr_dr + dUz_z) + dfdr * ur + dfdz * uz
    return div_fu


@njit(fastmath=True, cache=True)
def div_U_2d_cart_iso(ur, uz, dr, dz):
    """
    Calculte div(U)



    Parameters
    ----------
    ur : np.ndarray
        scalar field of vector field U, U = [ur, uz]
    uz : np.ndarray
        scalar field of vector field U
    dr : float
        grid size of r
    dz : float
        grid size of z

    Returns
    -------

    """

    dUr_dr = dfdr_2d_cart_iso(ur, dr, 1)
    dUz_dz = dfdz_2d_cart_iso(uz, dz)

    div_fu = dUr_dr + dUz_dz
    return div_fu


@njit(fastmath=True, cache=True)
def div_f_grad_g_2d_cart_iso(f, g, dr, dz):
    """Calculate div (f * grad g)


    div (f * grad g) = dFdr * dGdr + dFdz * dGdz + f * (ddGdr+ ddGdz)

    Parameters
    ----------
    f : np.ndarray
        scalar field
    g : np.ndarray
        scalar field
    dr : float
        step r
    dz : float
        step z

    Returns
    -------

    """

    ddGddr, ddGddz = ddF_2d_cart_iso(g, dr, dz, 0)
    dFdz = dfdz_2d_cart_iso(f, dz)
    dGdz = dfdz_2d_cart_iso(g, dz)
    dFdr = dfdr_2d_cart_iso(f, dr, 0)
    dGdr = dfdr_2d_cart_iso(g, dr, 0)
    divFdG = dFdr * dGdr + dFdz * dGdz + f * (ddGddr + ddGddz)
    return divFdG


@njit(fastmath=True, cache=True)
def div_f_grad_u_2d_cart_iso(f, ur, uz, dr, dz, case=0):
    """
    compute div ( phi * (grad U + grad U^T - 2/3 grad U cdot I) )
    Parameters
    ----------
    f :  np.ndarray
        scalar field (phi)
    ur :
    uz :
    dr :
    dz :

    Returns
    -------

    """
    # 1st derivative
    dfdr, dfdz = grad_2d_cart_iso(f, dr, dz, 0)
    dUrdr, dUrdz = grad_2d_cart_iso(ur, dr, dz, 1)
    dUzdr, dUzdz = grad_2d_cart_iso(uz, dr, dz, 0)
    # 2nd derivative
    ddUrdr, ddUrdz = ddF_2d_cart_iso(ur, dr, dz, 1)
    ddUzdr, ddUzdz = ddF_2d_cart_iso(uz, dr, dz, 0)
    # cross derivateive
    ddUrdrdz = ddFdrz_2d_cart_iso(ur, dr, dz, 1)
    ddUzdrdz = ddFdrz_2d_cart_iso(uz, dr, dz, 0)

    # 1. partial f
    par_f_term_r = 2. * dfdr * dUrdr + dfdz * (dUrdz + dUzdr)
    par_f_term_z = 2. * dfdz * dUzdz + dfdr * (dUzdr + dUrdz)
    # 2. partial stress tensor
    par_s_term_r = f * (2. * ddUrdr + ddUrdz + ddUzdrdz)
    par_s_term_z = f * (2. * ddUzdz + ddUzdr + ddUrdrdz)

    div_f_grad_U_r = par_f_term_r + par_s_term_r
    div_f_grad_U_z = par_f_term_z + par_s_term_z

    if case == 1:
        dT_r = f * (ddUrdr + ddUzdrdz) + dfdr * (dUrdr + dUzdz)
        dT_z = f * (ddUzdz + ddUrdrdz) + dfdz * (dUrdr + dUzdz)
        div_f_grad_U_r = div_f_grad_U_r - 2 / 3 * dT_r
        div_f_grad_U_z = div_f_grad_U_z - 2 / 3 * dT_z
    return div_f_grad_U_r, div_f_grad_U_z


def div_stress_imcomp_cart_iso(f, ur, uz, kg, dr, dz):
    """
    simplified version of div stress tensor.
    Parameters
    ----------
    f :
    ur :
    uz :
    dr :
    dz :

    Returns
    -------

    """
    lap_ur = lap_2d_cart_iso(ur, dr, dz, 1)
    lap_uz = lap_2d_cart_iso(uz, dr, dz, 0)
    dkg_dr, dkg_dz = grad_2d_cart_iso(kg, dr, dz, 0)

    div_stress_r = f * (lap_ur + dkg_dr)
    div_stress_z = f * (lap_uz + dkg_dz)
    return div_stress_r, div_stress_z


@jit(nopython=True, fastmath=True, cache=True)
def div4_2d_cart_iso(f, g, ur, uz, dr, dz):
    """ calculate div (f * g * U)
        = dr(f * g* ur) + dz(f * g * uz)

    Parameters
    ----------
    f: np.ndarray
        scalar fields
    g: np.ndarray
        scalar fields
    ur: np.ndarray
        Ur of U (vector fields)
    uz: np.ndarray
        Uz of U (vector fields)
    dr: float
        step r
    dz: float
        step z

    Returns
    -------

    """

    Ur_r = dfdr_2d_cart_iso(ur, dr, 1)
    Uz_z = dfdz_2d_cart_iso(uz, dz)

    dfdr = dfdr_2d_cart_iso(f, dr, 0)
    dgdr = dfdr_2d_cart_iso(g, dr, 0)
    dfdz = dfdz_2d_cart_iso(f, dz)
    dgdz = dfdz_2d_cart_iso(g, dz)

    advec = (dfdr * g + dgdr * f) * ur + (dfdz * g + dgdz * f) * uz + f * g * (Ur_r + Uz_z)

    return advec


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


def create_circle_hyper(center: Tuple[float], radius: float, epsilon: float,
                        r: np.ndarray, z: np.ndarray) -> np.ndarray:
    phi0_r, phi0_z = center
    dis = np.sqrt((r - phi0_r) ** 2 +
                  (z - phi0_z) ** 2)
    phi0 = 0.5 + 0.5 * np.tanh(3. * (radius - dis) / epsilon)
    return phi0


def create_dome_hyper(height: float, radius: float,
                      r_cord: np.ndarray, z_cord: np.ndarray) -> np.ndarray:
    a = radius
    z, r = -1., 1
    m, n = 1, 0
    h = height
    dx = .01
    width = 16
    rec_top_x = np.arange(0.0, a, dx)
    rec_top_y = np.ones_like(rec_top_x) * (z + 2. * r)

    theta = np.arange(np.pi / 2, -np.pi / 2, -dx)
    round_y = r * np.sin(theta) + z + r
    round_x = r * np.cos(theta) + a

    rec_btm_x = np.arange(a, 0., -dx)
    rec_btm_y = np.ones_like(rec_btm_x) * z

    total_x = np.hstack((rec_top_x, round_x, rec_btm_x))
    total_y = np.hstack((rec_top_y, round_y, rec_btm_y))

    y_scale = - m * (total_y - (2 * r + z)) * (total_y - (z)) ** n + 1.
    total_y = y_scale * total_y

    y_scale2 = h * (total_x[total_y >= (z + r)] - (a + r)) * (total_x[total_y >= (z + r)] - (-a - r)) / (
            (a + r) * (-a - r)) + 1
    total_y[total_y >= (z + r)] = y_scale2 * total_y[total_y >= (z + r)]

    # map them to girds
    grids = np.ones((len(z_cord), len(r_cord)))
    x_mapped_index = np.array([np.argmin(np.abs(r_cord - x)) for x in total_x])
    z_mapped_index = np.array([np.argmin(np.abs(z_cord - y)) for y in total_y])

    for j, i in zip(z_mapped_index, x_mapped_index):
        grids[j, i] = -1
    index_along_z = np.argmax(grids == -1, axis=1)
    np.argmax(grids == -1, axis=1)
    for j in range(len(z_cord)):
        if index_along_z[j] != 0:
            grids[j, :index_along_z[j]] = -1

    dx = (z_cord[1] - z_cord[0], r_cord[1] - r_cord[0])
    dist_grids = skfmm.distance(phi=grids, dx=dx)
    field_grids = 0.5 + 0.5 * np.tanh(3. * -dist_grids / width)  # generate hyperbolic field

    return field_grids


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
        c_max = np.quantile(phi, 0.98)
        mp1 = ax.imshow(phi, origin='lower', extent=[0., Lr, -Lz, Lz_up], cmap='coolwarm', vmax=c_max)
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


def save_pickle(save_dict, save_dir, filename=None, separate=False):
    if filename is None:
        filename = 'result.pickled'

    for name, data in save_dict.items():
        if hasattr(data, 'device'):
            if data.device.type == 'cuda':  # covert tensor to ndarray
                data = data.cpu()
                save_dict[name] = data

    if separate is True:
        for name, data in save_dict.items():
            dump(data, os.path.join(save_dir, f'{name}.pickled'))
    else:
        dump(save_dict, os.path.join(save_dir, filename))


#%%
class Toggle:
    def __init__(self, green0, red0, gr):
        self.gr = gr
        self.green = green0
        self.red = red0

    def field_flow(self, GR_array):
        # G, R = GR_array
        f_green = self.f_G(GR_array[0], GR_array[1])
        f_red = self.f_R(GR_array[0], GR_array[1])
        field_flow = np.empty(2)
        field_flow[0] = f_green
        field_flow[1] = f_red
        return field_flow

    def evolve(self):

        return self.field_flow((self.green, self.red))

    def _field_flow_rk45(self, t, GR_array):
        return self.field_flow(GR_array)

    def ivp_rk45(self, dt=1e-6, max_time=10., t0=0.):
        y0 = np.array((self.green, self.red))
        ivp = RK45(fun=self._field_flow_rk45, t0=t0, y0=y0, first_step=dt, t_bound=max_time)
        return ivp

    def ivp_odeint(self, dt=1e-6, max_time=10., t0=0.):
        y0 = np.array([self.green, self.red])
        t = np.arange(t0, max_time, dt)
        y_t = odeint(func=self._field_flow_rk45, y0=y0, t=t, tfirst=True)
        return y_t, t

    def prod_G(self, R):
        return self.alphaG(self.gr) * hillFunc(0.015, 10., 2., R)

    def prod_R(self, G):
        return self.alphaR(self.gr) * hillFunc(0.13, 15.5, 4., G)

    # @njit(cache=True)
    def alphaG(self, gr):
        alpha = 1 * gr * (16.609 + 627.747 / (1.0 + (gr / 0.865) ** 4.635))
        # alpha = 1.1 * gr * (40.609 + 627.747 / (1.0 + (gr / 0.765) ** 5.635))

        return alpha

    # @njit(cache=True)
    def alphaR(self, gr):
        # alpha = 1 * gr * (26.836 + 320.215 / (1.0 + (gr / 0.661) ** 4.09))
        alpha = 1.1 * gr * (26.836 + 320.215 / (1.0 + (gr / 0.661) ** 4.09))

        return alpha

    def null_cline_R(self, G):
        return self.prod_R(G) / self.gr

    def null_cline_G(self, R):
        return self.prod_G(R) / self.gr

    def f_G(self, G, R):
        return self.prod_G(R) - self.gr * G

    def f_R(self, G, R):
        return self.prod_R(G) - self.gr * R

    def solve_sst(self, G_conc_list: np.ndarray = np.arange(0, 1000, 0.01), optimize: bool = False):
        """
        Class method for solving the fix points of the toggle.

        """
        sst_R = self.null_cline_R(G_conc_list)

        sign_dev_laci = np.sign(self.f_G(G_conc_list, sst_R))
        root = np.diff(sign_dev_laci)
        sst_index = np.nonzero(root)[0]
        self.sst_R_conc = self.null_cline_R(G_conc_list[sst_index])
        self.sst_G_conc = self.null_cline_G(self.sst_R_conc)
        if optimize is True:
            sst_tetr = []
            sst_laci = []
            for i in range(len(self.sst_R_conc)):
                sst_laci_tetr = fsolve(self.field_flow, np.array([self.sst_G_conc[i], self.sst_R_conc[i]]))
                sst_laci.append(sst_laci_tetr[0])
                sst_tetr.append(sst_laci_tetr[1])
            self.sst_G_conc = np.array(sst_laci)
            self.sst_R_conc = np.array(sst_tetr)
        self.sst_state = root[sst_index]
        if len(sst_index) == 3:
            self.bistable = True
        else:
            self.bistable = False


if __name__ == '__main__':
    # dr, dz = 0.1, 0.1
    # r_cord = np.arange(0, 500, step=dr)
    # z_cord = np.arange(-50, 100, step=dz)
    #
    # phi0 = create_dome_hyper(80, 400, r_cord, z_cord)
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(phi0, cmap='coolwarm', origin='lower',
    #           extent=(r_cord[0], r_cord[-1], z_cord[0], z_cord[-1]))
    # fig.show()

    # ================= test toggle ====================
    growth_array = np.linspace(0.2, 1.6, 100)


    def get_toggle_sst(gr):
        toggle = Toggle(1, 1, gr)
        toggle.solve_sst(optimize=True)
        return toggle.sst_G_conc, toggle.sst_R_conc


    sst_list = [get_toggle_sst(i) for i in growth_array]
    sst_GoverR = [gr[0] / gr[1] for gr in sst_list]
    sst_GoverR_max = np.array([i.max() for i in sst_GoverR])
    sst_bistable = np.array([True if len(i) >= 2 else False for i in sst_GoverR])

    fig, ax = plt.subplots(1, 1)
    ax.scatter(growth_array[sst_bistable], sst_GoverR_max[sst_bistable], color='r')
    ax.scatter(growth_array[~sst_bistable], sst_GoverR_max[~sst_bistable], color='g')
    fig.show()
