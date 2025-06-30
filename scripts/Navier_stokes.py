import os, sys
HERE = os.path.dirname(__file__)                           # .../Benchmarking/FC-PINO-1D
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))   # .../Benchmarking
sys.path.insert(0, PROJECT_ROOT)


from Blowup_utils import *
from hyperparams import *
from wandb_utils import *
from Burger_utils import *

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
inf = torch.inf
from wandb_utils import *
from fixed_lbfgs import Fixed_LBFGS
from Burger_utils import *
import matplotlib.pyplot as plt
import os
from FNO1d import FNO1d
import torch.nn.functional as F
import torch.optim as optim
from neuraloperator.neuralop.losses.data_losses import LpLoss
from FCPINO1D.isolated_utils import FC1d_2d

FC_func = lambda x: FC1d_3d(FC1d_3d(FC1d_3d(x, 2, order=3), 3, order=3), 4, order=3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torch.autograd import grad
from neuraloperator.neuralop.layers.fourier_continuation import FCLegendre, FCLegendre128
from neuraloperator.neuralop.losses.fourier_diff import *
from neuraloperator.neuralop.models.fno import FNO, FC_FNO1d, FC_FNO2d, FC_FNO3d
from neuraloperator.neuralop.losses.equation_losses import BurgersEqnLoss
#from physicsnemo.sym.loss.aggregator import Relobralo
from hyperparams import * 
from FCPINO1D.isolated_utils import FC1d_3d
import wandb


def plt_show(label: str):
    if WANDB:
        plt.savefig("temp_image.png")
        wandb.log({label: wandb.Image("temp_image.png")})
    if PLOT_POPUPS:
        plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pi = torch.pi



momentum_weight = 100
ic_weight = 100
BC_weight = 100

torch.set_default_dtype(torch.float64)
WANDB = True
Re = 500
sub_s = 4
sub_t = 20
s = 256 // sub_s
T_in = 1000 # 1000*0.005 = 5s
T = 50 # 1000 + 50*20*0.005 = 10s

padding = 14


modes = 24
width = 64

wandb_name = "end-fc-NS"
#wandb.init(project = WANDB_PROJECT, name = wandb_name, entity = 'caltech-anima-group')


LX=1
LT = 0.005*sub_t*T
LZ=1

data_path = "/central/groups/tensorlab/aganeshram/FC/data_ns/cavity.mat"

batch_size = 1

reader = MatReader(data_path)

data_u = reader.read_field('u')[T_in:T_in+T*sub_t:sub_t, ::sub_s, ::sub_s].permute(1,2,0)
data_v = reader.read_field('v')[T_in:T_in+T*sub_t:sub_t, ::sub_s, ::sub_s].permute(1,2,0)

x = torch.stack([data_u, data_v],dim=-1)
print(x.shape)

data_output = torch.stack([data_u, data_v],dim=-1).reshape(batch_size,s,s,T,2)
data_input = data_output[:,:,:,:1,:].repeat(1,1,1,T,1).reshape(batch_size,s,s,T,2)



data_output = data_output.permute(0,4, 1,2,3)
data_input = data_input.permute(0,4, 1,2,3)

print('data_output.shape', data_output.shape)
print('data_input.shape', data_input.shape)


for i in ['beginning1', 'beginning', 'end', 'outside']:

    FC_type = 'beginning' if i == 'beginning1' else i

    directory = f'{i}_3d_NS'

    wandb_name = f"{i}_3d_NS"
    WANDB_PROJECT = "FC-PINO-3D"
    wandb.init(project = WANDB_PROJECT, name = wandb_name, entity = 'caltech-anima-group', reinit = True)

    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type(torch.DoubleTensor)

    os.makedirs(directory, exist_ok=True)

    model = FC_FNO3d(
        n_modes_height = 32 if i == 'beginning1' else 24,
        n_modes_width = 32 if i == 'beginning1' else 24,
        n_modes_depth = 25 if i == 'beginning1' else 20,
        hidden_channels = 60,
        in_channels = 2, ## check
        out_channels = 3,
        Lx = LX,
        Ly = LZ,
        Lz = LT,
        x_deriv = True,
        y_deriv = True,
        z_deriv = True,
        laplacian = True,
        x_res = 64,
        y_res = 64,
        z_res = 50,
        FC_func = FC_func,
        FC_type = FC_type,
        cont_points = 25,
        one_sided = True,
        non_linearity = F.tanh,
        fourier_diff = True,
    ).to('cuda')

    print("model dtype:", next(model.parameters()).dtype)

    os.makedirs(f'{wandb_name}_plots', exist_ok=True)

    SCHEDULER_KWARGS = process_hyperparams()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = SCHEDULER_TYPE(optimizer, **SCHEDULER_KWARGS)


    """def momentum_loss_NS(U, Dx_arr, laplacian):
        #U, Dx_arr, gradient, laplacian = model(input)
        gradient = (Dx_arr[0], Dx_arr[1]) 
        dt = Dx_arr[2]
        gradient_x = (gradient[0][..., 0] + gradient[1][..., 0])
        gradient_y = (gradient[0][..., 1] + gradient[1][..., 1])
        dx = gradient[0]
        dy = gradient[1]


        E1 = dt[..., 0] + torch.matmul(U[..., 0].T, gradient_x) + dx[..., 2] - 1/Re  * (laplacian[..., 0]) 
        E2 = dt[..., 1] + torch.matmul(U[..., 1].T, gradient_y) + dy[..., 2] - 1/Re  * (laplacian[..., 1]) 
        E3 = dx[..., 0] + dy[..., 1]

        res_target = torch.zeroes(E1.shape, device = U.device)

        E1_loss = F.mse_loss(E1, res_target)
        E2_loss = F.mse_loss(E2, res_target)
        E3_loss = F.mse_loss(E3, res_target)

        total_momentum = (E1_loss + E2_loss + E3_loss)
        
        return E1_loss, E2_loss, E3_loss #total_momentum"""

    def momentum_loss_NS(U, Dx_arr, laplacian):
        dx, dy, dt = Dx_arr

        E1 = dt[:, 0, :, :, :] + U[:, 0, :, :, :] * dx[:, 0, :, :, :] + U[:, 1, :, :, :] * dy[:, 0, :, :, :] + dx[:, 2, :, :, :] - 1/Re * (laplacian[0][:, 0, :, :, :] + laplacian[1][:, 0, :, :, :])
        E2 = dt[:, 1, :, :, :] + U[:, 0, :, :, :] * dx[:, 1, :, :, :] + U[:, 1, :, :, :] * dy[:, 1, :, :, :] + dy[:, 2, :, :, :] - 1/Re * (laplacian[0][:, 1, :, :, :] + laplacian[1][:, 1, :, :, :])
        E3 = dx[:, 0, :, :, :] + dy[:, 1, :, :, :]

        
        res_target = torch.zeros(E1.shape, device = U.device)
        
        E1_loss = F.mse_loss(E1, res_target)
        E2_loss = F.mse_loss(E2, res_target)
        E3_loss = F.mse_loss(E3, res_target)
        total_momentum = (E1_loss + E2_loss + E3_loss)
        return E1_loss, E2_loss, E3_loss


    """print("U.shape", U.shape)
        #print('dt.shape', dt.shape)
        #print('dx.shape', dx.shape)
        #print('dy.shape', dy.shape)
        
        # Get the first timestep
        dt_0 = dt[..., 0]  # [1, 32, 32]
        dx_0 = dx[..., 0]  # [1, 32, 32]
        dy_0 = dy[..., 0]  # [1, 32, 32]
        
        # Get U components at t=0
        u = U[:, 0, :, :, 0]  # [1, 32, 32]
        v = U[:, 1, :, :, 0]  # [1, 32, 32]
        p = U[:, 2, :, :, 0]  # [1, 32, 32]
        
        # Calculate convective terms using broadcasting
        convective_x = u * dx_0 + v * dy_0
        
        # First momentum equation (x-direction)
        E1 = dt_0 + convective_x - 1/Re * laplacian[0, :, :, 0]
        
        # Second momentum equation (y-direction)
        E2 = dt_0 + convective_x - 1/Re * laplacian[1, :, :, 0]
        
        # Continuity equation
        E3 = dx_0 + dy_0

        res_target = torch.zeros(E1.shape, device = U.device)

        E1_loss = F.mse_loss(E1, res_target)
        E2_loss = F.mse_loss(E2, res_target)
        E3_loss = F.mse_loss(E3, res_target)
        
        return E1_loss, E2_loss, E3_loss"""

    def initial_boundary_loss_NS(U, Dx_arr, laplacian, data_output):
        #print("U.shape", U.shape)
        #print("data_output.shape", data_output.shape)
        # U shape: [batch, 3, x, y, t]
        #print("U.shape", U.shape)
        ic_loss = F.mse_loss(U[:, :2, :, :, 0], data_output[:, :, :, :, 0])

        BC1 = F.mse_loss(U[:, :2, 0, :, :], data_output[:, :, 0, :, :])  # x = 0
        BC2 = F.mse_loss(U[:, :2, -1, :, :], data_output[:, :, -1, :, :])  # x = 1
        BC3 = F.mse_loss(U[:, :2, :, -1, :], data_output[:, :, :, -1, :])  # y = 1
        BC4 = F.mse_loss(U[:, :2, :, 0, :], data_output[:, :, :, 0, :])  # y = 0

        BC_loss = (BC1 + BC2 + BC3 + BC4) / 4

        return ic_loss, BC_loss



    data_input = data_input.cuda().double()
    data_output = data_output.cuda().double()


    for ep in range(1):
        optimizer.zero_grad()

        U, Dx_arr, laplacian = model(data_input)

        E1_loss, E2_loss, E3_loss = momentum_loss_NS(U, Dx_arr, laplacian)
        ic_loss, BC_loss = initial_boundary_loss_NS(U, Dx_arr, laplacian, data_output)

        total_loss = momentum_weight * (E1_loss + E2_loss + E3_loss) + ic_weight * ic_loss + BC_weight * BC_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        print(ep, ic_loss.item(), "\t", (E1_loss + E2_loss + E3_loss).item(), "\t", BC_loss.item(), "\t", total_loss.item())

        if WANDB:
            wandb.log({
                "epoch": ep,
                "log10 interior loss": np.log10((E1_loss + E2_loss + E3_loss).item()),
                "log10 initial condition loss": np.log10(ic_loss.item()),
                "log10 boundary condition loss": np.log10(BC_loss.item()),
                "log10 total loss": np.log10(total_loss.item()),
                "log10 optimizer learning rate": np.log10(optimizer.param_groups[0]['lr'])
            })

        if ep % 500 == 0:
            ##### WANDB PLOTTING #####

            y_plot = data_output[0, :2].cpu().numpy()
            out_plot = U[0, :2].detach().cpu().numpy()

            max_0 = np.max(y_plot[0, :, :, -1])
            max_1 = np.max(y_plot[1, :, :, -1])
            min_0 = np.min(y_plot[0, :, :, -1])
            min_1 = np.min(y_plot[1, :, :, -1])

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(y_plot[0, :, :, -1], vmax=max_0, vmin=min_0)
            ax[1].imshow(y_plot[1, :, :, -1], vmax=max_1, vmin=min_1)
            plt_show("Exact solution")

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(out_plot[0, :, :, -1], vmax=max_0, vmin=min_0)
            ax[1].imshow(out_plot[1, :, :, -1], vmax=max_1, vmin=min_1)
            plt_show("Model channels")

            ##### END WANDB PLOTTING #####

            ##### MATPLOTLIB PLOTTING #####
            
            # Convert tensors to numpy arrays for plotting
            y_plot = data_output[0, :2].cpu().numpy()  # Only u and v components
            out_plot = U[0, :2].detach().cpu().numpy()  # Only u and v components

            # Get min/max values for consistent color scaling
            max_u = max(np.max(y_plot[0]), np.max(out_plot[0]))
            max_v = max(np.max(y_plot[1]), np.max(out_plot[1]))
            min_u = min(np.min(y_plot[0]), np.min(out_plot[0]))
            min_v = min(np.min(y_plot[1]), np.min(out_plot[1]))

            # Create figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Epoch {ep}')

            # Plot exact solution
            im1 = axes[0,0].imshow(y_plot[0, :, :, -1], vmax=max_u, vmin=min_u)
            axes[0,0].set_title('Exact u')
            plt.colorbar(im1, ax=axes[0,0])

            im2 = axes[0,1].imshow(y_plot[1, :, :, -1], vmax=max_v, vmin=min_v)
            axes[0,1].set_title('Exact v')
            plt.colorbar(im2, ax=axes[0,1])

            # Plot model prediction
            im3 = axes[1,0].imshow(out_plot[0, :, :, -1], vmax=max_u, vmin=min_u)
            axes[1,0].set_title('Predicted u')
            plt.colorbar(im3, ax=axes[1,0])

            im4 = axes[1,1].imshow(out_plot[1, :, :, -1], vmax=max_v, vmin=min_v)
            axes[1,1].set_title('Predicted v')
            plt.colorbar(im4, ax=axes[1,1])


            # Save the plot
            plt.tight_layout()
            plt.savefig(f'{directory}/navier_stokes_epoch_{ep}.png', dpi=300, bbox_inches='tight')
            plt.close()
