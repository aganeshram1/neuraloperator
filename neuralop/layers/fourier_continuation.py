import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial.legendre import Legendre


class FCLegendre(nn.Module):
    def __init__(self, n, d, dtype=torch.float32):
        super().__init__()

        self.dtype = dtype

        self.compute_extension_matrix(n, d)
    
    def compute_extension_matrix(self, n, d):
        self.n = n
        self.d = d 

        a = 0.0
        h = 0.1

        #Generate grid for the extension
        total_points = 2*n + d
        full_grid = a + h*np.arange(total_points, dtype=np.float64)
        fit_grid = np.concatenate((full_grid[0:self.n], full_grid[-self.n:]), 0)
        extension_grid = full_grid[self.n:-self.n]

        #Initialize orthogonal polynomial system
        I = np.eye(2*self.n, dtype=np.float64)
        polynomials = []
        for j in range(2*self.n):
            polynomials.append(Legendre(I[j,:], domain=[full_grid[0], full_grid[-1]]))

        #Compute data and evaluation matrices
        X = np.zeros((2*self.n,2*self.n), dtype=np.float64)
        Q = np.zeros((self.d, 2*self.n), dtype=np.float64)
        for j in range(2*self.n):
            Q[:,j] = polynomials[j](extension_grid)
            X[:,j] = polynomials[j](fit_grid)

        #Compute extension matrix
        ext_mat = np.matmul(Q, np.linalg.pinv(X, rcond=1e-31))
        self.register_buffer('ext_mat', torch.from_numpy(ext_mat).to(dtype=self.dtype))
        self.register_buffer('ext_mat_T', self.ext_mat.T.clone())

        return self.ext_mat

    def extend_left_right(self, x, one_sided):
        right_bnd = x[...,-self.n:]
        left_bnd = x[...,0:self.n]
        y = torch.cat((right_bnd, left_bnd), dim=-1)
        
        if x.is_complex():
            ext = torch.matmul(y, self.ext_mat_T + 0j)
        else:
            ext = torch.matmul(y, self.ext_mat_T)
        
        if one_sided:
            return torch.cat((x, ext), dim=-1)
        else:
            return torch.cat((ext[...,self.d//2:], x, ext[...,:self.d//2]), dim=-1)
    
    
    def extend_top_bottom(self, x, one_sided):
        bottom_bnd = x[...,-self.n:,:]
        top_bnd = x[...,0:self.n,:]
        y = torch.cat((bottom_bnd, top_bnd), dim=-2)
        
        if x.is_complex():
            ext = torch.matmul(self.ext_mat, y + 0j)
        else:
            ext = torch.matmul(self.ext_mat, y)
        
        if one_sided:
            return torch.cat((x, ext), dim=-2)
        else:
            return torch.cat((ext[...,self.d//2:,:], x, ext[...,:self.d//2,:]), dim=-2)

    
    def extend1d(self, x, one_sided):
        return self.extend_left_right(x, one_sided)
    
    def extend2d(self, x, one_sided):
        x = self.extend_left_right(x, one_sided)
        x = self.extend_top_bottom(x, one_sided)

        return x
    
    def forward(self, x, dim=2, one_sided=True):
        if dim == 1:
            return self.extend1d(x, one_sided)
        if dim == 2:
            return self.extend2d(x, one_sided)


class FCLegendre128(nn.Module):
    def __init__(self, n, d, dtype=torch.float64):
        super().__init__()

        self.dtype = dtype

        self.compute_extension_matrix(n, d)
    
    def compute_extension_matrix(self, n, d):
        self.n = n
        self.d = d 

        a = 0.0
        h = 0.1

        # Generate grid for the extension
        total_points = 2*n + d
        full_grid = a + h*np.arange(total_points, dtype=np.float128)
        fit_grid = np.concatenate((full_grid[0:self.n], full_grid[-self.n:]), 0)
        extension_grid = full_grid[self.n:-self.n]

        # Initialize orthogonal polynomial system
        I = np.eye(2*self.n, dtype=np.float128)
        polynomials = []
        for j in range(2*self.n):
            polynomials.append(Legendre(I[j,:], domain=[full_grid[0], full_grid[-1]]))

        # Compute data and evaluation matrices
        X = np.zeros((2*self.n,2*self.n), dtype=np.float128)
        Q = np.zeros((self.d, 2*self.n), dtype=np.float128)
        for j in range(2*self.n):
            Q[:,j] = polynomials[j](extension_grid)
            X[:,j] = polynomials[j](fit_grid)

        # Compute extension matrix using pseudo-inverse with QR decomposition
        # X_pinv = (X^T X)^(-1) X^T
        XT = X.T
        XTX = np.matmul(XT, X)
        
        # Add small regularization
        reg = 1e-16 * np.eye(XTX.shape[0], dtype=np.float128)
        XTX_reg = XTX + reg
        
        # Solve (X^T X) * X_pinv = X^T using Gaussian elimination
        X_pinv = self._solve_linear_system(XTX_reg, XT)
        
        ext_mat = np.matmul(Q, X_pinv)
        self.register_buffer('ext_mat', torch.from_numpy(ext_mat.astype(np.float64)).to(dtype=self.dtype))
        self.register_buffer('ext_mat_T', self.ext_mat.T.clone())

        return self.ext_mat
    
    def _solve_linear_system(self, A, b):
        """
        Solve A * x = b using Gaussian elimination with partial pivoting
        Works with float128 arrays (np.linalg.solve does not support float128)
        """
        n = A.shape[0]
        
        # Reshape b if needed
        b_reshaped = b.reshape(-1, 1) if len(b.shape) == 1 else b
        
        # Gaussian elimination with partial pivoting
        for i in range(n):
            # Find pivot and swap rows if necessary
            pivot_row = i
            for k in range(i + 1, n):
                if abs(A[k, i]) > abs(A[pivot_row, i]):
                    pivot_row = k
            if pivot_row != i:
                A[i, :], A[pivot_row, :] = A[pivot_row, :].copy(), A[i, :].copy()
                b_reshaped[i, :], b_reshaped[pivot_row, :] = b_reshaped[pivot_row, :].copy(), b_reshaped[i, :].copy()
            
            # Eliminate column i
            for k in range(i + 1, n):
                factor = A[k, i] / A[i, i]
                A[k, i:] -= factor * A[i, i:]
                b_reshaped[k, :] -= factor * b_reshaped[i, :]
        
        # Back substitution
        x = np.zeros_like(b_reshaped, dtype=np.float128)
        for i in range(n - 1, -1, -1):
            x[i, :] = (b_reshaped[i, :] - np.matmul(A[i, i+1:], x[i+1:, :])) / A[i, i]
        
        return x.reshape(b.shape) if len(b.shape) == 1 else x

    def extend_left_right(self, x, one_sided):
        right_bnd = x[...,-self.n:]
        left_bnd = x[...,0:self.n]
        y = torch.cat((right_bnd, left_bnd), dim=-1)
        ext_mat_T = self.ext_mat_T.to(dtype=x.dtype)
        
        if x.is_complex():
            ext = torch.matmul(y, ext_mat_T + 0j)
        else:
            ext = torch.matmul(y, ext_mat_T)
        
        if one_sided:
            return torch.cat((x, ext), dim=-1)
        else:
            return torch.cat((ext[...,self.d//2:], x, ext[...,:self.d//2]), dim=-1)
    
    
    def extend_top_bottom(self, x, one_sided):
        bottom_bnd = x[...,-self.n:,:]
        top_bnd = x[...,0:self.n,:]
        y = torch.cat((bottom_bnd, top_bnd), dim=-2)
        ext_mat = self.ext_mat.to(dtype=x.dtype)
        
        if x.is_complex():
            ext = torch.matmul(ext_mat, y + 0j)
        else:
            ext = torch.matmul(ext_mat, y)
        
        if one_sided:
            return torch.cat((x, ext), dim=-2)
        else:
            return torch.cat((ext[...,self.d//2:,:], x, ext[...,:self.d//2,:]), dim=-2)

    
    def extend1d(self, x, one_sided):
        return self.extend_left_right(x, one_sided)
    
    def extend2d(self, x, one_sided):
        x = self.extend_left_right(x, one_sided)
        x = self.extend_top_bottom(x, one_sided)
        return x
    
    def forward(self, x, dim=2, one_sided=True):
        if dim == 1:
            return self.extend1d(x, one_sided)
        if dim == 2:
            return self.extend2d(x, one_sided)
