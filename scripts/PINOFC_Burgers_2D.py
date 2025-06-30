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
#from fixed_lbfgs import Fixed_LBFGS
#from Burger_utils import *
import matplotlib.pyplot as plt
import os
from FNO1d import FNO1d
import torch.nn.functional as F
import torch.optim as optim
from neuraloperator.neuralop.losses.data_losses import LpLoss
from FCPINO1D.isolated_utils import FC1d_2d



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plt_show(label: str):
    if WANDB:
        plt.savefig("temp_image.png")
        wandb.log({label: wandb.Image("temp_image.png")})
    if PLOT_POPUPS:
        plt.show()

from torch.autograd import grad
from neuraloperator.neuralop.layers.fourier_continuation import FCLegendre, FCLegendre128
#from neuraloperator. neuralop.layers.fourier_derivatives import *
from neuraloperator.neuralop.models.fno import FNO, FC_FNO1d, FC_FNO2d
from neuraloperator.neuralop.losses.equation_losses import BurgersEqnLoss
#from physicsnemo.sym.loss.aggregator import Relobralo
from hyperparams import * 

WANDB = True

torch.manual_seed(23)
np.random.seed(23)
pi = np.pi
VISCOSITY=0.01
IC_weight = 5
BC_weight = 1
F_weight = 100


device = 'cuda' if torch.cuda.is_available() else 'cpu'

for i in ['beginning', 'end', 'outside']:

    FC_type = i

    directory = f'{FC_type}_2d_128'

    wandb_name = f"{FC_type}_2d_128"
    WANDB_PROJECT = "FC-PINO-2D"
    wandb.init(project = WANDB_PROJECT, name = wandb_name, entity = 'caltech-anima-group', reinit = True)

    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type(torch.DoubleTensor)

    os.makedirs(directory, exist_ok=True)

    cont_points = 25

    extension = FCLegendre128(n=3, d=cont_points).to('cuda')

    legendre_2d = lambda x : extension(x, dim=2, one_sided=False)

    cont_2d = lambda x : FC1d_2d(FC1d_2d(x, 2 , 3), 3, 3) # x1 = self.use_FC(x1, 2, 3) ---> x1 = self.use_FC(x1, 3, 3)


    torch.manual_seed(0)
    np.random.seed(0)
    pi = np.pi
    x_res = 256
    y_res = 256

    model = FC_FNO2d(
        in_channels= 2,
        Lx=2*pi,
        Ly=2*pi,
        out_channels = 1,
        n_modes_height=64,
        n_modes_width=64,
        hidden_channels = 64,
        n_layers = 4,
        FC_func=cont_2d,
        second_deriv=True,
        complex_data = False,
        one_sided=True,
        x_second_deriv = True,
        y_second_deriv = False,
        FC_type=FC_type,
        cont_points = cont_points,
        fourier_diff = True,
        x_res = x_res,
        y_res = y_res,
        non_linearity=F.tanh,
        ).to('cuda')

    print("model dtype:", next(model.parameters()).dtype)

    if C0_INITIAL is False:
        params = model.parameters()
    else:
        params = [*model.parameters(), model.c]
    # define an optimizer
    from torch.optim import Adam
    SCHEDULER_KWARGS = process_hyperparams()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = SCHEDULER_TYPE(optimizer, **SCHEDULER_KWARGS)

    x = torch.linspace(0, 2*pi, x_res+1, device=device, dtype=torch.float64, requires_grad=True)[:-1]
    y = torch.linspace(0, 2*pi, y_res+1, device=device, dtype=torch.float64, requires_grad=True)[:-1]
    grid = torch.cartesian_prod(x, y)
    grid = grid.reshape(1, 2, x_res, y_res)
    #print(grid.shape)

    def ground_truth(inputs, k = 1):
        return torch.sin(k*inputs)

    def losses(model, grid):
        u, grad = model(grid) #(b, c, x, t)

        num_t = u.size(-1)
        num_x = u.size(-2)

        u0 = grid.reshape(1, num_t, num_x, 2)[:, 0, :, 1]  # Take x-coordinate at t=0 ytgrid.reshape(1, num_t, num_x, 2)[:, 0, :, 1]
        u0 = ground_truth(u0)
        
        # Get derivatives from grad tuple
        ux, ut = grad[0].squeeze(), grad[1].squeeze()  # First derivatives
        uxx = grad[2].squeeze() if len(grad) > 2 else None  # Second derivative if available
        
        # PDE residual
        u = u.T
        interior_expr = ut + u*ux - VISCOSITY*uxx
        f = torch.zeros(interior_expr.shape, device=u.device)
        loss_f = F.mse_loss(interior_expr, f)
        
        # Initial condition loss 
        u = u.T
        initial_u = u[0, :].reshape(-1,)  # Changed from channel 1 to 0
        loss_ic = F.mse_loss(initial_u, u0.reshape(-1,))

        # Reshape for boundary conditions
        # Spatial boundary conditions - x=0 and x=L
        loss_bc = F.mse_loss(u[:, 0], torch.zeros(u[:, 0].shape, device=u.device)) \
                + F.mse_loss(u[:, -1], torch.zeros(u[:, -1].shape, device=u.device))

        return loss_ic, loss_f, loss_bc, u, ux, ut, uxx



    for ep in range (1):
        optimizer.zero_grad()


        loss_ic, loss_f, loss_bc, U, ux, ut, uxx = losses(model, grid)
        total_loss = IC_weight*loss_ic + F_weight*loss_f + BC_weight*loss_bc
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        print(ep, loss_ic.item(), "\t", loss_f.item(), "\t", total_loss.item())


        if WANDB:
            wandb.log({
                "epoch": ep,
                "log10 interior loss": np.log10(loss_f.item()),
                "log10 boundary loss": np.log10(loss_ic.item()),
                "log10 additional boundary loss": 0 if loss_bc.item() == 0 else np.log10(loss_bc.item()),
                "log10 total loss": np.log10(total_loss.item()),
                "log10 optimizer learning rate": np.log10(optimizer.param_groups[0]['lr']),
                #"log10 ground truth loss": np.log10(ground_truth_loss.item()),
                #"log10 implicit equation loss": np.log10(implicit_eqn_loss.item()),
            })


    # plot
        if ep % 1000 == 0:    
            #plot(U[0,:,:,0].cpu().detach().numpy(), "Model plot")

            #ux, ut, uxx = Du[0], Du[1], Du[2]
            #plot(ux[0,:,:].cpu().detach().numpy(), "ux plot")
            #plot(ut[0,:,:].cpu().detach().numpy(), "ut plot")
            #plot(uxx[0,:,:].cpu().detach().numpy(), "uxx plot")



            U_numpy = U.cpu().detach().numpy().squeeze()  # Remove the singleton dimension
            plt.figure(figsize=(8, 6))
            plt.imshow(U_numpy, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title("Model plot")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.savefig(f'{directory}/model_plot_{ep}.png')
            plt.close()
            """
            # Plot the derivatives
            ux_numpy = ux.cpu().detach().numpy().squeeze()
            ut_numpy = ut.cpu().detach().numpy().squeeze()
            uxx_numpy = uxx.cpu().detach().numpy().squeeze()

            plt.figure(figsize=(8, 6))
            plt.imshow(ux_numpy, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title("ux plot")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.savefig(f'plots_2d/ux_plot_{ep}.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.imshow(ut_numpy, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title("ut plot")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.savefig(f'plots_2d/ut_plot_{ep}.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.imshow(uxx_numpy, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title("uxx plot")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.savefig(f'plots_2d/uxx_plot_{ep}.png')
            plt.close()"""


    ########################################################################################



