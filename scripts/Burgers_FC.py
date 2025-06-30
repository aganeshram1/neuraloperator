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
from math import inf
from wandb_utils import *
from fixed_lbfgs import Fixed_LBFGS
from Burger_utils import *
import matplotlib.pyplot as plt
import os
from FNO1d import FNO1d
import torch.nn.functional as F
import torch.optim as optim
from neuraloperator.neuralop.losses.data_losses import LpLoss


device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torch.autograd import grad
from neuraloperator.neuralop.layers.fourier_continuation import FCLegendre, FCLegendre128
#from neuraloperator. neuralop.layers.fourier_derivatives import *
from neuraloperator.neuralop.models.fno import FNO, FC_FNO1d
from neuraloperator.neuralop.losses.equation_losses import BurgersEqnLoss
#from physicsnemo.sym.loss.aggregator import Relobralo
from hyperparams import *

torch.manual_seed(23)
np.random.seed(23)
pi = np.pi

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import numpy as np

from Blowup_utils import process_hyperparams, model_epoch
from hyperparams import *
from wandb_utils import *
from Burger_utils import *
from FNO1d import FNO1d
import wandb


wandb.init(project = WANDB_PROJECT, entity = 'caltech-anima-group', name = "Legendre_64_search")

cont_points = wandb.config.cont_points
n = wandb.config.n

# Update the run name

torch.set_default_dtype(torch.float64)


torch.manual_seed(0)
np.random.seed(0)
pi = np.pi

#LEARNING_RATE = wandb.config.LEARNING_RATE
#NUM_MODES = wandb.config.NUM_MODES
#WIDTH = wandb.config.WIDTH
#LAYERS = wandb.config.LAYERS
#PATIENCE = wandb.config.PATIENCE

#True for Gram, False for legendre

#gram_cont = lambda x : FC1d(x, 3) # Precomputed matrices

#extension = FCLegendre128(n=n, d=cont_points).to('cuda')
extension = FCLegendre(n=n, d=cont_points, dtype=torch.float64).to('cuda')

legendre_cont = lambda x : extension(x, dim=1, one_sided=one_sided)
one_sided = wandb.config.one_sided 


model = FC_FNO1d(
    in_channels=1,
    L=4,
    n_modes_height=32,
    hidden_channels = WIDTH,
    n_layers = DEPTH,
    FC_func=legendre_cont,
    FC_type='end',
    cont_points = cont_points, 
    grid_points = 400,
    one_sided = one_sided, 
    fourier_diff = True,
    non_linearity=F.tanh,
).to('cuda')



def get_losses(optimizer, model, ep, interior_term=INTERIOR_TERM):
    y = torch.linspace(INTERVAL[0], INTERVAL[1], m + 1, device='cuda', dtype=torch.float64, requires_grad=True)[:-1]
    optimizer.zero_grad()

    if TRACK_SMOOTHNESS_LOSS:
        U, Uy, Uyy, *_ = model(y.reshape(1,1, m))
    else:
        U, Uy, *_ = model(y.reshape(1,m,1), 1)

    U = U.squeeze()
    Uy= Uy.squeeze()
    Uyy = Uyy.squeeze()

        
    loss_b = BOUNDARY_FUNC(U)

    Burgers_expression = interior_term(U, Uy, y)
    loss_i = torch.norm( Burgers_expression )**2 / m

    if TRACK_SMOOTHNESS_LOSS:
        Burgers_gradient = INTERIOR_TERM_2(U, Uy, Uyy, y)
        smoothness_loss = torch.norm( Burgers_gradient )**2 / m
    else:
        smoothness_loss = 0
    
    #ground_truth_loss = torch.norm(U - torch.from_numpy(U_arr[0]).cuda())**2 / m

    #implicit_eqn_loss = torch.norm(y + U + torch.sign(U)*torch.abs(U)**(1+1/l))**2 / m
    

    loss = INTERIOR_WEIGHT*loss_i + BOUNDARY_WEIGHT*loss_b \
        + SMOOTHNESS_WEIGHT*smoothness_loss
    
    l2loss = LpLoss(d = 1, p = 2)

    l2 = l2loss(U, U_arr[0].to(U.device))

    return {
        "interior": loss_i,
        "boundary": loss_b, 
        "smoothness": smoothness_loss, 
       #"ground truth": ground_truth_loss, 
        #"implicit equation": implicit_eqn_loss, 
        "total": loss, 
        "y": y,
        #'l2': l2
    }


def model_epoch(optimizer, scheduler, model, ep, list_wrapper=None):
    if LBFGS_AFTER_EPOCH is False or ep <= LBFGS_AFTER_EPOCH:
        losses = get_losses(optimizer, model, ep)
        loss = losses["total"]
        #l2loss = losses['l2']

        update_model(loss, optimizer, scheduler, ep, model)
        if LBFGS_AFTER_EPOCH is not False and ep == LBFGS_AFTER_EPOCH:
            optimizer = torch.optim.LBFGS(
                    model.parameters(),
                    lr=0.1,               
                    max_iter=50,           
                    history_size=50,       
                    line_search_fn=None,
                    tolerance_grad=0,
                    tolerance_change=0,
                )
            scheduler = SCHEDULER_TYPE(optimizer, **process_hyperparams())


            if list_wrapper is None:
                raise ValueError("LBFGS requires passing a list_wrapper argument to model_epoch.")
            #total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            #print(f"Total trainable (grad) parameters: {total_params}")

            """optimizer = torch.optim.LBFGS(
                model.parameters(), lr=10000000, #(LBFGS_LR if LBFGS_LR else optimizer.param_groups[0]['lr']), 
                tolerance_grad=0, tolerance_change=0, history_size=LBFGS_HISTORY_SIZE, max_iter=50,
                max_eval=torch.inf, line_search_fn="strong_wolfe")
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=0.5,                # <— start closer to the right scale
                max_iter=40,           # <— give each *outer* call more room
                history_size=10,       # <— smaller, stabler Hessian approx.
                line_search_fn=None,
                tolerance_grad=1e-10,
                tolerance_change=1e-12,
            )
            scheduler = SCHEDULER_TYPE(optimizer, **process_hyperparams())"""
            #print(optimizer)

        print(f"epoch = {ep},\tloss = {loss.item()}") #, \tl2loss = {l2loss.item()}")
        wandb_log({
            "epoch": ep,
            "log10 interior loss": np.log10(losses["interior"].item()),
            "log10 boundary loss": np.log10(losses["boundary"].item()),
            "log10 total loss": np.log10(loss.item()),
            #"log10 l2 loss": np.log10(l2loss.item()),
            #"log10 ground truth loss": np.log10(losses["ground truth"].item()),
            #"log10 implicit equation loss": np.log10(losses["implicit equation"].item()),
            "log10 optimizer learning rate": np.log10(optimizer.param_groups[0]['lr'])
        } | ({
            "log10 smoothness loss": np.log10(losses["smoothness"].item()),
            } 
            if TRACK_SMOOTHNESS_LOSS else {}), ep)
        
        if list_wrapper is None:
            return losses["y"]
        list_wrapper.extend((losses, optimizer, scheduler))
    else:
        if list_wrapper is None:
            raise ValueError("Must pass list_wrapper list object for LBFGS.")
        n_lbfgs_steps = 5    
        for _ in range(n_lbfgs_steps):
            closure = lambda : closure_func(optimizer, scheduler, model, ep, list_wrapper)
            optimizer.step(closure)
        losses = list_wrapper[0]
        loss = losses["total"]

        print(f"epoch = {ep},\tloss = {loss.item()}")
        wandb_log({
            "epoch": ep,
            "log10 interior loss": np.log10(losses["interior"].item()),
            "log10 boundary loss": np.log10(losses["boundary"].item()),
            "log10 total loss": np.log10(loss.item()),
            #"log10 l2 loss": np.log10(l2loss.item()),
           # "log10 ground truth loss": np.log10(losses["ground truth"].item()),
           # "log10 implicit equation loss": np.log10(losses["implicit equation"].item()),
            "log10 optimizer learning rate": np.log10(optimizer.param_groups[0]['lr'])
        } | ({
            "log10 smoothness loss": np.log10(losses["smoothness"].item()),
            } 
            if TRACK_SMOOTHNESS_LOSS else {}), ep)

        if SCHEDULER_TYPE in (torch.optim.lr_scheduler.ReduceLROnPlateau, FixedReduceLROnPlateau):
            loss = get_losses(optimizer, model, ep)["total"]
            metrics = (loss,)
        else:
            metrics = ()
        scheduler.step(*metrics)

def update_model(loss, optimizer, scheduler, ep=-1, model=None):
    loss.backward()

    optimizer.step()

    metrics = (loss,) if SCHEDULER_TYPE in \
        (torch.optim.lr_scheduler.ReduceLROnPlateau, FixedReduceLROnPlateau) \
        else ()
    scheduler.step(*metrics)

def closure_func(optimizer, scheduler, model, ep, list_wrapper):
    optimizer.zero_grad()

    losses = get_losses(optimizer, model, ep) #, interior_term=INTERIOR_TERM)#get_losses(optimizer, model, ep) , U_arr, model, ep, interior_term=INTERIOR_TERM)
    loss = losses["total"]
    loss.backward()

    with torch.no_grad():
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        #print(f"[closure] loss={loss.item():.3e}  |grad|={gnorm:.2e}")

    del list_wrapper[:3]
    list_wrapper.extend((losses, optimizer, scheduler))

    return loss


################################################################
# Setup
################################################################
if (not TRACK_SMOOTHNESS_LOSS) and SMOOTHNESS_WEIGHT != 0:
    raise ValueError("Must track smoothness loss for nonzero smoothness weight")


#model = FNO1d(D=D).cuda()




print("model dtype:", next(model.parameters()).dtype)

if C0_INITIAL is False:
    params = model.parameters()
else:
    params = [*model.parameters(), model.c]
# define an optimizer
from torch.optim import Adam
SCHEDULER_KWARGS = process_hyperparams()
optimizer = Adam(params, lr=LEARNING_RATE, weight_decay=0) # use adamw
scheduler = SCHEDULER_TYPE(optimizer, **SCHEDULER_KWARGS)
print(SCHEDULER_TYPE)
print(scheduler)
print(l)

#wandb.config.update({
#    "NUM_MODES": wandb.config.NUM_MODES,
#    "LEARNING_RATE": wandb.config.LEARNING_RATE,
#    "WIDTH": wandb.config.WIDTH,
#    "DEPTH": DEPTH,
#    "SMOOTHNESS_WEIGHT": wandb.config.get("SMOOTHNESS_WEIGHT", 0)
#}, allow_val_change=True)

y = torch.linspace(INTERVAL[0], INTERVAL[1], m + 1, device='cuda', dtype=torch.float64, requires_grad=True)[:-1]

lr = LEARNING_RATE
# 2) run the SciPy solver (which now works on a NumPy array)
# before training loop
y_np = y.detach().cpu().numpy()
U_np   = get_U_solns(l, y_np)        # NumPy array
U_soln = torch.from_numpy(U_np)      # stays on CPU
U_arr   = [U_soln]
for _ in range(1,4):
    U_soln = take_derivative(U_soln) # still a CPU Tensor
    U_arr.append(U_soln)

os.makedirs('plots', exist_ok=True)






################################################################
# Main training loop
################################################################
for ep in range(NUM_EPOCHS):
    list_wrapper = []
    model_epoch(optimizer, scheduler, model, ep, list_wrapper)
    losses, optimizer, scheduler = list_wrapper
    y = losses["y"]
    if ep % 1000 == 0:
        U, Uy, Uyy, *_ = model(y.reshape(1,1,m))
        y_np       = y.cpu().detach().numpy()
        U_pred_np  = U.squeeze().detach().cpu().numpy

        U = U.detach().cpu().numpy()
        y = losses["y"]
        y = y.detach().cpu().numpy().flatten()

        wandb_save(y, "y")

        for i in range(4):
            plt_plot(y, U)
            wandb_plot(y, U, "y", "U" + "y"*i, ep)

            if BURGER_OVERLAY_PLOTS:
                plt_Burger_overlay(y, U, i, U_arr, True, ep)

            U = take_derivative(U)
            y = y[1:-1]

        #print(f"Epoch {ep}, LR = {optimizer.param_groups[0]['lr']}")
        # --------------------------------------------------------------------
        # Prepare grid and run a forward pass (no mode change / no_no_grad)
        # --------------------------------------------------------------------
        """y_plot  = torch.linspace(INTERVAL[0], INTERVAL[1], m + 1,
                                device=device, dtype=torch.float32)[:-1]   # (m,)
        y_batch = y_plot.view(1, 1, -1)                                     # (1,1,m)
        
        U, Uy, Uyy, *_ = model(y.reshape(1,1,m))                         # (1,1,m) each
        
        # Move to CPU → NumPy
        y_np       = y.cpu().detach().numpy()
        U_pred_np  = U.squeeze().detach().cpu().numpy()
        dU_pred_np = Uy#.squeeze().detach().cpu().numpy()
        dUU_pred_np= Uyy#.squeeze().detach().cpu().numpy()
    
        U_true_np = U_arr[0]

    
        
     # --------------------------------------------------------------------
        # 1)  U overlay
        # --------------------------------------------------------------------
        plt.figure(figsize=(6, 4))
        plt.plot(y_np, U_true_np,  label="U true")
        plt.plot(y_np, U_pred_np, "--", label="U predicted")
        plt.xlabel("y"); plt.ylabel("U"); plt.legend(); plt.tight_layout()
        plt.savefig(f"plots/U_overlay_ep{ep}.png", dpi=300); plt.close()

        plt.figure(figsize=(6, 4))
       # plt.plot(y_np, dU_true_np,  label="dU true")
        plt.plot(y_np, dU_pred_np.detach().cpu().numpy(), label="dU predicted")
        plt.xlabel("y"); plt.ylabel("dU/dy"); plt.legend(); plt.tight_layout()
        plt.savefig(f"plots/dU_overlay_ep{ep}.png", dpi=300); plt.close()

        plt.figure(figsize=(6, 4))
       # plt.plot(y_np, dUU_true_np,  label="d²U true")
        plt.plot(y_np, dUU_pred_np.detach().cpu().numpy(), label="d²U predicted")
        plt.xlabel("y"); plt.ylabel("d²U/dy²"); plt.legend(); plt.tight_layout()
        plt.savefig(f"plots/dUU_overlay_ep{ep}.png", dpi=300); plt.close()  """
     
        




#torch.save(model, "fc_pino_model.pt")
#print("Model saved to fc_pino_model.pt")

"""
higher_res_m = 900
def get_losses_higher_res(optimizer, U_arr, model, ep, interior_term=INTERIOR_TERM):
    y = torch.linspace(INTERVAL[0], INTERVAL[1], higher_res_m + 1, device='cuda', dtype=torch.float64, requires_grad=True)[:-1]
    optimizer.zero_grad()

    if TRACK_SMOOTHNESS_LOSS:
        U, Uy, Uyy, *_ = model(y.reshape(1,1, higher_res_m))
    else:
        U, Uy, *_ = model(y.reshape(1,higher_res_m,1), 1)

    U = U.squeeze()
    Uy= Uy.squeeze()
    Uyy = Uyy.squeeze()

        
    loss_b = BOUNDARY_FUNC_HIGHER_RES(U)

    Burgers_expression = interior_term(U, Uy, y)
    loss_i = torch.norm( Burgers_expression )**2 / higher_res_m

    if TRACK_SMOOTHNESS_LOSS:
        Burgers_gradient = INTERIOR_TERM_2(U, Uy, Uyy, y)
        smoothness_loss = torch.norm( Burgers_gradient )**2 / higher_res_m
    else:
        smoothness_loss = 0
    
    #ground_truth_loss = torch.norm(U - torch.from_numpy(U_arr[0]).cuda())**2 / m

    #implicit_eqn_loss = torch.norm(y + U + torch.sign(U)*torch.abs(U)**(1+1/l))**2 / m
    

    loss = loss_i + BOUNDARY_WEIGHT*loss_b \
        + SMOOTHNESS_WEIGHT*smoothness_loss
    
    #l2loss = LpLoss(d = 1, p = 2)

    #l2 = l2loss(U, U_arr[0].to(U.device))

    return {
        "interior": loss_i,
        "boundary": loss_b, 
        "smoothness": smoothness_loss, 
       #"ground truth": ground_truth_loss, 
        #"implicit equation": implicit_eqn_loss, 
        "total": loss, 
        "y": y,
        #'l2': l2
    }


def model_epoch_higher_res(optimizer, U_arr, scheduler, model, ep, list_wrapper=None):
    if LBFGS_AFTER_EPOCH is False or ep <= LBFGS_AFTER_EPOCH:
        losses = get_losses_higher_res(optimizer, U_arr, model, ep)
        loss = losses["total"]
        #l2loss = losses['l2']

        update_model_higher_res(loss, optimizer, scheduler, ep, model)

        if LBFGS_AFTER_EPOCH is not False and ep == LBFGS_AFTER_EPOCH:
            if list_wrapper is None:
                raise ValueError("LBFGS requires passing a list_wrapper argument to model_epoch.")

            optimizer = Fixed_LBFGS(
                model.parameters(), lr=(LBFGS_LR if LBFGS_LR else optimizer.param_groups[0]['lr']), 
                tolerance_grad=0, tolerance_change=0, history_size=LBFGS_HISTORY_SIZE, max_iter=1,
                max_eval=inf, line_search_fn="strong_wolfe")
            scheduler = SCHEDULER_TYPE(optimizer, **process_hyperparams())

        print(f"epoch = {ep},\tloss = {loss.item()}") #, \tl2loss = {l2loss.item()}")
        wandb_log({
            "epoch": ep,
            "log10 interior loss": np.log10(losses["interior"].item()),
            "log10 boundary loss": np.log10(losses["boundary"].item()),
            "log10 total loss": np.log10(loss.item()),
            #"log10 l2 loss": np.log10(l2loss.item()),
            #"log10 ground truth loss": np.log10(losses["ground truth"].item()),
            #"log10 implicit equation loss": np.log10(losses["implicit equation"].item()),
            "log10 optimizer learning rate": np.log10(optimizer.param_groups[0]['lr'])
        } | ({
            "log10 smoothness loss": np.log10(losses["smoothness"].item()),
            } 
            if TRACK_SMOOTHNESS_LOSS else {}), ep)
        
        if list_wrapper is None:
            return losses["y"]
        list_wrapper.extend((losses, optimizer, scheduler))
    else:
        if list_wrapper is None:
            raise ValueError("Must pass list_wrapper list object for LBFGS.")
        closure = lambda : closure_func_higher_res(optimizer, scheduler, model, ep, list_wrapper)
        optimizer.step(closure)
        losses = list_wrapper[0]
        loss = losses["total"]

        print(f"epoch = {ep},\tloss = {loss.item()}")
        wandb_log({
            "epoch": ep,
            "log10 interior loss": np.log10(losses["interior"].item()),
            "log10 boundary loss": np.log10(losses["boundary"].item()),
            "log10 total loss": np.log10(loss.item()),
            #"log10 l2 loss": np.log10(l2loss.item()),
           # "log10 ground truth loss": np.log10(losses["ground truth"].item()),
           # "log10 implicit equation loss": np.log10(losses["implicit equation"].item()),
            "log10 optimizer learning rate": np.log10(optimizer.param_groups[0]['lr'])
        } | ({
            "log10 smoothness loss": np.log10(losses["smoothness"].item()),
            } 
            if TRACK_SMOOTHNESS_LOSS else {}), ep)

        if SCHEDULER_TYPE in (torch.optim.lr_scheduler.ReduceLROnPlateau, FixedReduceLROnPlateau):
            loss = get_losses_higher_res(optimizer, model, ep, U_arr)["total"]
            metrics = (loss,)
        else:
            metrics = ()
        scheduler.step(*metrics)

def update_model_higher_res(loss, optimizer, scheduler, U_arr, ep=-1, model=None):
    loss.backward()

    optimizer.step()

    metrics = (loss,) if SCHEDULER_TYPE in \
        (torch.optim.lr_scheduler.ReduceLROnPlateau, FixedReduceLROnPlateau) \
        else ()
    scheduler.step(*metrics)

def closure_func_higher_res(optimizer, scheduler, model, ep, list_wrapper):
    optimizer.zero_grad()

    losses = get_losses_higher_res(optimizer, model, ep)
    loss = losses["total"]
    loss.backward()

    del list_wrapper[:3]
    list_wrapper.extend((losses, optimizer, scheduler))

    return loss


################################################################
# Setup
################################################################
if (not TRACK_SMOOTHNESS_LOSS) and SMOOTHNESS_WEIGHT != 0:
    raise ValueError("Must track smoothness loss for nonzero smoothness weight")


#model = FNO1d(D=D).cuda()




print("model dtype:", next(model.parameters()).dtype)

if C0_INITIAL is False:
    params = model.parameters()
else:
    params = [*model.parameters(), model.c]
# define an optimizer
from torch.optim import Adam
SCHEDULER_KWARGS = process_hyperparams()
optimizer = Adam(params, lr=LEARNING_RATE, weight_decay=0) # use adamw
scheduler = SCHEDULER_TYPE(optimizer, **SCHEDULER_KWARGS)
print(SCHEDULER_TYPE)
print(scheduler)
print(l)

#wandb.config.update({
#    "NUM_MODES": wandb.config.NUM_MODES,
#    "LEARNING_RATE": wandb.config.LEARNING_RATE,
#    "WIDTH": wandb.config.WIDTH,
#    "DEPTH": DEPTH,
#    "SMOOTHNESS_WEIGHT": wandb.config.get("SMOOTHNESS_WEIGHT", 0)
#}, allow_val_change=True)

y = torch.linspace(INTERVAL[0], INTERVAL[1], higher_res_m + 1, device='cuda', dtype=torch.float64, requires_grad=True)[:-1]

lr = LEARNING_RATE
# 2) run the SciPy solver (which now works on a NumPy array)
# before training loop
y_np = y.detach().cpu().numpy()
U_np   = get_U_solns(l, y_np)        # NumPy array
U_soln = torch.from_numpy(U_np)      # stays on CPU
U_arr   = [U_soln]
for _ in range(1,4):
    U_soln = take_derivative(U_soln) # still a CPU Tensor
    U_arr.append(U_soln)

os.makedirs('plots_higher_res', exist_ok=True)


for ep in range(60001):
    list_wrapper = []
    model_epoch_higher_res(optimizer, U_arr, scheduler, model, ep, list_wrapper)
    losses, optimizer, scheduler = list_wrapper
    y = losses["y"]
    if ep % 1000 == 0:
        U, Uy, Uyy, *_ = model(y.reshape(1,1,higher_res_m))
        y_np       = y.cpu().detach().numpy()
        U_pred_np  = U.squeeze().detach().cpu().numpy

        y = y.detach().cpu().numpy()
        U = U.detach().cpu().numpy()

        #wandb_save(y, "y")

        #for i in range(4):
        #    plt_plot(y, U)
        #    wandb_plot(y, U, "y", "U" + "y"*i, ep)

        #    if BURGER_OVERLAY_PLOTS:
        #        plt_Burger_overlay(y, U, i, U_arr, True, ep)

        #    U = take_derivative(U)
        #    y = y[1:-1]
        #print(f"Epoch {ep}, LR = {optimizer.param_groups[0]['lr']}")
        # --------------------------------------------------------------------
        # Prepare grid and run a forward pass (no mode change / no_no_grad)
        # --------------------------------------------------------------------
        y_plot  = torch.linspace(INTERVAL[0], INTERVAL[1], higher_res_m + 1,
                                device=device, dtype=torch.float64)[:-1]   # (m,)
        y_batch = y_plot.view(1, 1, -1)                                     # (1,1,m)
        
        #U, Uy, Uyy, *_ = model(y.reshape(1,1,m))                         # (1,1,m) each
        
        # Move to CPU → NumPy
        y_np       = y#.cpu().detach().numpy()
        U_pred_np  = U.squeeze()#.detach().cpu().numpy()
        dU_pred_np = Uy.squeeze()#.detach().cpu().numpy()
        dUU_pred_np= Uyy.squeeze()#.detach().cpu().numpy()
    
        U_true_np = U_arr[0]

    
        
     # --------------------------------------------------------------------
        # 1)  U overlay
        # --------------------------------------------------------------------
        plt.figure(figsize=(6, 4))
        plt.plot(y_np, U_true_np,  label="U true")
        plt.plot(y_np, U_pred_np, "--", label="U predicted")
        plt.xlabel("y"); plt.ylabel("U"); plt.legend(); plt.tight_layout()
        plt.savefig(f"plots_higher_res/U_overlay_ep{ep}.png", dpi=300); plt.close()

        plt.figure(figsize=(6, 4))
       # plt.plot(y_np, dU_true_np,  label="dU true")
        plt.plot(y_np, dU_pred_np.detach().cpu().numpy(), label="dU predicted")
        plt.xlabel("y"); plt.ylabel("dU/dy"); plt.legend(); plt.tight_layout()
        plt.savefig(f"plots_higher_res/dU_overlay_ep{ep}.png", dpi=300); plt.close()

        plt.figure(figsize=(6, 4))
       # plt.plot(y_np, dUU_true_np,  label="d²U true")
        plt.plot(y_np, dUU_pred_np.detach().cpu().numpy(), label="d²U predicted")
        plt.xlabel("y"); plt.ylabel("d²U/dy²"); plt.legend(); plt.tight_layout()
        plt.savefig(f"plots_higher_res/dUU_overlay_ep{ep}.png", dpi=300); plt.close()  
     """