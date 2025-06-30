from neuralop.layers.fourier_continuation import FCLegendre
import torch
import warnings
pi = torch.pi


def fourier_derivative_1d(u, order=1, L=2*torch.pi, use_FC=False, FC_n=4, FC_d=20, FC_one_sided=False, low_pass_filter_ratio=None):
    """
    Compute the 1D Fourier derivative of a given tensor.
    Use Fourier continuation to extend the signal if it is non-periodic. 
    Use with care, as Fourier continuation and Fourier derivatives are not always stable.
    Parameters
    ----------
    u : torch.Tensor
        Input tensor. The derivative will be computed along the last dimension.
    order : int, optional
        Order of the derivative. Defaults to 1.
    L : float, optional
        Length of the domain considered. Defaults to 2*pi.
    use_FC : bool, optional
        Whether to use Fourier continuation. Use for non-periodic functions. Defaults to False.
    FC_n : int, optional
        Degree of the Fourier continuation. Defaults to 4.
    FC_d : int, optional
        Number of points to add using the Fourier continuation layer. Defaults to 40.
    FC_one_sided : bool, optional
        Whether to only add points on one side, or add an equal number of points on both sides. Defaults to False.
    low_pass_filter_ratio : float, optional
        If not None, apply a low-pass filter to the Fourier coefficients. Can help reduce artificial oscillations. 
        1.0 means no filtering, 0.5 means keep half of the coefficients, etc.
        Defaults to None.
    Returns
    -------
    torch.Tensor
        The derivative of the input tensor.
    """

    # Extend signal using Fourier continuation if specified
    if use_FC:
        FC = FCLegendre(n=FC_n, d=FC_d).to(u.device)
        u = FC(u, dim=1, one_sided=FC_one_sided)
        L = L *  (u.shape[-1] + FC_d) / u.shape[-1]     # Define extended length
    else:
        warnings.warn("Consider using Fourier continuation if the input is not periodic (use_FC=True).", category=UserWarning)

    nx = u.size(-1)    
    u_h = torch.fft.rfft(u, dim=-1) 
    k_x = torch.fft.rfftfreq(nx, d=1/nx, device=u_h.device).view(*([1] * (u_h.dim() - 1)), u_h.size(-1))

    if low_pass_filter_ratio is not None:
        # Apply a low-pass filter to the Fourier coefficients
        cutoff = int(u_h.shape[-1] * low_pass_filter_ratio)
        u_h[..., cutoff:] = 0

    # Fourier differentiation
    derivative_u_h = (1j * k_x * 2*torch.pi/L)**order * u_h 

    # Inverse Fourier transform to get the derivative in physical space
    derivative_u = torch.fft.irfft(derivative_u_h, dim=-1, n=nx) 

    # If Fourier continuation is used, crop the result to retrieve the derivative on the original interval
    if use_FC:
        if FC_one_sided:
            derivative_u = derivative_u[..., :-FC_d]
        else:
            derivative_u = derivative_u[..., FC_d//2: -FC_d//2]

    return derivative_u


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from neuralop.layers.fourier_continuation import FCLegendre128

def fourier_derivative_1d(u, order=1, L=2*torch.pi, use_FC=False, FC_n=4, FC_d=40, FC_one_sided=False, low_pass_filter_ratio=None):
    """
    Compute the 1D Fourier derivative of a given tensor.
    Use Fourier continuation to extend the signal if it is non-periodic. 
    Use with care, as Fourier continuation and Fourier derivatives are not always stable.

    Parameters
    ----------
    u : torch.Tensor
        Input tensor. The derivative will be computed along the last dimension.
    order : int, optional
        Order of the derivative. Defaults to 1.
    L : float, optional
        Length of the domain considered. Defaults to 2*pi.
    use_FC : bool, optional
        Whether to use Fourier continuation. Use for non-periodic functions. Defaults to False.
    FC_n : int, optional
        Degree of the Fourier continuation. Defaults to 4.
    FC_d : int, optional
        Number of points to add using the Fourier continuation layer. Defaults to 40.
    FC_one_sided : bool, optional
        Whether to only add points on one side, or add an equal number of points on both sides. Defaults to False.
    low_pass_filter_ratio : float, optional
        If not None, apply a low-pass filter to the Fourier coefficients. Can help reduce artificial oscillations. 
        1.0 means no filtering, 0.5 means keep half of the coefficients, etc.
        Defaults to None.

    Returns
    -------
    torch.Tensor
        The derivative of the input tensor.
    """
    
    # Extend signal using Fourier continuation if specified
    if use_FC:
        FC = FCLegendre128(n=FC_n, d=FC_d).to(u.device)
        u = FC(u, dim=1, one_sided=FC_one_sided)
        L = L *  (u.shape[-1] + FC_d) / u.shape[-1]     # Define extended length
    else:
        warnings.warn("Consider using Fourier continuation if the input is not periodic (use_FC=True).", category=UserWarning)

    nx = u.size(-1)    
    u_h = torch.fft.rfft(u, dim=-1) 
    k_x = torch.fft.rfftfreq(nx, d=1/nx, device=u_h.device).view(*([1] * (u_h.dim() - 1)), u_h.size(-1))
    
    if low_pass_filter_ratio is not None:
        # Apply a low-pass filter to the Fourier coefficients
        cutoff = int(u_h.shape[-1] * low_pass_filter_ratio)
        u_h[..., cutoff:] = 0
    
    # Fourier differentiation
    derivative_u_h = (1j * k_x * 2*torch.pi/L)**order * u_h 
    
    # Inverse Fourier transform to get the derivative in physical space
    derivative_u = torch.fft.irfft(derivative_u_h, dim=-1, n=nx) 

    # If Fourier continuation is used, crop the result to retrieve the derivative on the original interval
    if use_FC:
        if FC_one_sided:
            derivative_u = derivative_u[..., :-FC_d]
        else:
            derivative_u = derivative_u[..., FC_d//2: -FC_d//2]

    return derivative_u




def gradient_3D(w, order = 1, Lx=2*torch.pi, Ly=2*torch.pi, Lz=2*torch.pi, use_FC=False, FC_n=4, FC_d=40, spatial  = False,FC_one_sided=False, low_pass_filter_ratio=None):
    """
    Compute the gradient of a given tensor. Expect shape (b, c, x, y, z) or (b, c, x, y, t)
    """
    if use_FC:
        FC = FCLegendre128(n=FC_n, d=FC_d).to(w.device)
        w = FC(w, dim=1, one_sided=FC_one_sided)
        Lx = Lx *  (w.shape[-1] + FC_d) / w.shape[-1]
        Ly = Ly *  (w.shape[-2] + FC_d) / w.shape[-2]
        Lz = Lz *  (w.shape[-3] + FC_d) / w.shape[-3]

    
    dy = fourier_derivative_1d(w.permute(0, 1, 2, 4, 3), order = 1, L = Ly, use_FC=use_FC, FC_n=FC_n, FC_d=FC_d, FC_one_sided=FC_one_sided, low_pass_filter_ratio=low_pass_filter_ratio)
    dx = fourier_derivative_1d(w.permute(0, 1, 3, 4, 2), order = 1, L = Lx, use_FC=use_FC, FC_n=FC_n, FC_d=FC_d, FC_one_sided=FC_one_sided, low_pass_filter_ratio=low_pass_filter_ratio)
    dy = dy.permute(0, 1, 2, 4, 3)
    dx = dx.permute(0, 1, 2, 4, 3)

    if spatial is not False:
        dz = fourier_derivative_1d(w, order = 1, L = Lz, use_FC=use_FC, FC_n=FC_n, FC_d=FC_d, FC_one_sided=FC_one_sided, low_pass_filter_ratio=low_pass_filter_ratio)
        gradient = (dx, dy, dz)
    else:
        gradient = (dx, dy)

    return gradient


def laplacian_3D(w, Lx=2*torch.pi, Ly=2*torch.pi, Lz=2*torch.pi, use_FC=False, FC_n=4, FC_d=40, FC_one_sided=False, spatial = False, low_pass_filter_ratio=None):
    """
    Compute the Laplacian of a given tensor. Expect shape (b, c, x, y, z) or (b, c, x, y, t)
    """
    if use_FC:
        FC = FCLegendre128(n=FC_n, d=FC_d).to(w.device)
        w = FC(w, dim=1, one_sided=FC_one_sided)
        Lx = Lx * (w.shape[-1] + FC_d) / w.shape[-1]
        Ly = Ly * (w.shape[-2] + FC_d) / w.shape[-2]
        Lz = Lz * (w.shape[-3] + FC_d) / w.shape[-3]

    # Calculate second derivatives directly
    dxx = fourier_derivative_1d(w.permute(0,1,4,3,2), order=2, L=Lx, use_FC=use_FC, FC_n=FC_n, FC_d=FC_d, FC_one_sided=FC_one_sided, low_pass_filter_ratio=low_pass_filter_ratio)
    dxx = dxx.permute(0,1,4,3,2)
    
    # For y direction, permute to make y the last dimension, take derivative, then permute back
    w_y = w.permute(0, 1, 2, 4, 3)
    dyy = fourier_derivative_1d(w_y, order=2, L=Ly, use_FC=use_FC, FC_n=FC_n, FC_d=FC_d, FC_one_sided=FC_one_sided, low_pass_filter_ratio=low_pass_filter_ratio)
    dyy = dyy.permute(0, 1, 2, 4, 3)
    
    # For z direction, permute to make z the last dimension, take derivative, then permute back
    if spatial is not False:
        dzz = fourier_derivative_1d(w, order=2, L=Lz, use_FC=use_FC, FC_n=FC_n, FC_d=FC_d, FC_one_sided=FC_one_sided, low_pass_filter_ratio=low_pass_filter_ratio)
        laplacian_arr = (dxx, dyy, dzz)
    else:
        laplacian_arr = (dxx, dyy)

    return laplacian_arr


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from neuralop.layers.fourier_continuation import FCLegendre128
import torch
import warnings

def fourier_derivative_1d_128(u, order=1, L=2*torch.pi, use_FC=False, FC_n=4, FC_d=40, FC_one_sided=False, low_pass_filter_ratio=None):
    """
    Compute the 1D Fourier derivative of a given tensor.
    Use Fourier continuation to extend the signal if it is non-periodic. 
    Use with care, as Fourier continuation and Fourier derivatives are not always stable.

    Parameters
    ----------
    u : torch.Tensor
        Input tensor. The derivative will be computed along the last dimension.
    order : int, optional
        Order of the derivative. Defaults to 1.
    L : float, optional
        Length of the domain considered. Defaults to 2*pi.
    use_FC : bool, optional
        Whether to use Fourier continuation. Use for non-periodic functions. Defaults to False.
    FC_n : int, optional
        Degree of the Fourier continuation. Defaults to 4.
    FC_d : int, optional
        Number of points to add using the Fourier continuation layer. Defaults to 40.
    FC_one_sided : bool, optional
        Whether to only add points on one side, or add an equal number of points on both sides. Defaults to False.
    low_pass_filter_ratio : float, optional
        If not None, apply a low-pass filter to the Fourier coefficients. Can help reduce artificial oscillations. 
        1.0 means no filtering, 0.5 means keep half of the coefficients, etc.
        Defaults to None.

    Returns
    -------
    torch.Tensor
        The derivative of the input tensor.
    """
    
    # Extend signal using Fourier continuation if specified
    if use_FC:
        FC = FCLegendre128(n=FC_n, d=FC_d).to(u.device)
        u = FC(u, dim=1, one_sided=FC_one_sided)
        L = L *  (u.shape[-1] + FC_d) / u.shape[-1]     # Define extended length
    else:
        warnings.warn("Consider using Fourier continuation if the input is not periodic (use_FC=True).", category=UserWarning)

    nx = u.size(-1)    
    u_h = torch.fft.rfft(u, dim=-1) 
    k_x = torch.fft.rfftfreq(nx, d=1/nx, device=u_h.device).view(*([1] * (u_h.dim() - 1)), u_h.size(-1))
    
    if low_pass_filter_ratio is not None:
        # Apply a low-pass filter to the Fourier coefficients
        cutoff = int(u_h.shape[-1] * low_pass_filter_ratio)
        u_h[..., cutoff:] = 0
    
    # Fourier differentiation
    derivative_u_h = (1j * k_x * 2*torch.pi/L)**order * u_h 
    
    # Inverse Fourier transform to get the derivative in physical space
    derivative_u = torch.fft.irfft(derivative_u_h, dim=-1, n=nx) 

    # If Fourier continuation is used, crop the result to retrieve the derivative on the original interval
    if use_FC:
        if FC_one_sided:
            derivative_u = derivative_u[..., :-FC_d]
        else:
            derivative_u = derivative_u[..., FC_d//2: -FC_d//2]

    return derivative_u

