from functools import partialmethod
from typing import Tuple, List, Union, Literal

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from ..layers.spectral_convolution import SpectralConv
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.channel_mlp import ChannelMLP, LinearChannelMLP
from ..layers.complex import ComplexValued
from .base_model import BaseModel
from ..losses.fourier_diff import fourier_derivative_1d, laplacian_3D, fourier_derivative_1d_128

torch.set_default_dtype(torch.float64)

#from FCPINO1D.hyperparams import *
#from FCPINO1D.isolated_utils import FC1d_2d



#fc1 = nn.Linear(100, 256)
#fc2 = nn.Linear(256, 1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, use_FC, return_full=False):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.use_FC = use_FC
        self.return_full = return_full

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2, dtype=torch.double))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # 

        if self.use_FC: 
            x = CONTINUATION_FUNC(x)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cdouble)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], torch.view_as_complex(self.weights1))

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        if self.return_full:
            return x[..., :-CONTINUATION_GRIDPOINTS], x
        elif self.use_FC:
            return x[..., :-CONTINUATION_GRIDPOINTS]
        else:
            return x



class FNO(BaseModel, name='FNO'):
    """N-Dimensional Fourier Neural Operator. The FNO learns a mapping between
    spaces of functions discretized over regular grids using Fourier convolutions, 
    as described in [1]_.
    
    The key component of an FNO is its SpectralConv layer (see 
    ``neuralop.layers.spectral_convolution``), which is similar to a standard CNN 
    conv layer but operates in the frequency domain.

    For a deeper dive into the FNO architecture, refer to :ref:`fno_intro`.

    Parameters
    ----------
    n_modes : Tuple[int]
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the FNO is inferred from ``len(n_modes)``
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    n_layers : int, optional
        Number of Fourier Layers, by default 4

    Documentation for more advanced parameters is below.

    Other parameters
    ------------------
    lifting_channel_ratio : int, optional
        ratio of lifting channels to hidden_channels, by default 2
        The number of liting channels in the lifting block of the FNO is
        lifting_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels)
    projection_channel_ratio : int, optional
        ratio of projection channels to hidden_channels, by default 2
        The number of projection channels in the projection block of the FNO is
        projection_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels)
    positional_embedding : Union[str, nn.Module], optional
        Positional embedding to apply to last channels of raw input
        before being passed through the FNO. Defaults to "grid"

        * If "grid", appends a grid positional embedding with default settings to 
        the last channels of raw input. Assumes the inputs are discretized
        over a grid with entry [0,0,...] at the origin and side lengths of 1.

        * If an initialized GridEmbedding module, uses this module directly
        See :mod:`neuralop.embeddings.GridEmbeddingND` for details.

        * If None, does nothing

    non_linearity : nn.Module, optional
        Non-Linear activation function module to use, by default F.gelu
    norm : Literal ["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use, by default None
    complex_data : bool, optional
        Whether data is complex-valued (default False)
        if True, initializes complex-valued modules.
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default True
    channel_mlp_dropout : float, optional
        dropout parameter for ChannelMLP in FNO Block, by default 0
    channel_mlp_expansion : float, optional
        expansion parameter for ChannelMLP in FNO Block, by default 0.5
    channel_mlp_skip : Literal['linear', 'identity', 'soft-gating'], optional
        Type of skip connection to use in channel-mixing mlp, by default 'soft-gating'
    fno_skip : Literal['linear', 'identity', 'soft-gating'], optional
        Type of skip connection to use in FNO layers, by default 'linear'
    resolution_scaling_factor : Union[Number, List[Number]], optional
        layer-wise factor by which to scale the domain resolution of function, by default None
        
        * If a single number n, scales resolution by n at each layer

        * if a list of numbers [n_0, n_1,...] scales layer i's resolution by n_i.
    domain_padding : Union[Number, List[Number]], optional
        If not None, percentage of padding to use, by default None
        To vary the percentage of padding used along each input dimension,
        pass in a list of percentages e.g. [p1, p2, ..., pN] such that
        p1 corresponds to the percentage of padding along dim 1, etc.
    domain_padding_mode : Literal ['symmetric', 'one-sided'], optional
        How to perform domain padding, by default 'symmetric'
    fno_block_precision : str {'full', 'half', 'mixed'}, optional
        precision mode in which to perform spectral convolution, by default "full"
    stabilizer : str {'tanh'} | None, optional
        whether to use a tanh stabilizer in FNO block, by default None

        Note: stabilizer greatly improves performance in the case
        `fno_block_precision='mixed'`. 

    max_n_modes : Tuple[int] | None, optional

        * If not None, this allows to incrementally increase the number of
        modes in Fourier domain during training. Has to verify n <= N
        for (n, m) in zip(max_n_modes, n_modes).

        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    factorization : str, optional
        Tensor factorization of the FNO layer weights to use, by default None.

        * If None, a dense tensor parametrizes the Spectral convolutions

        * Otherwise, the specified tensor factorization is used.
    rank : float, optional
        tensor rank to use in above factorization, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : str {'factorized', 'reconstructed'}, optional

        * If 'factorized', implements tensor contraction with the individual factors of the decomposition 
        
        * If 'reconstructed', implements with the reconstructed full tensorized weight.
    decomposition_kwargs : dict, optional
        extra kwargs for tensor decomposition (see `tltorch.FactorizedTensor`), by default dict()
    separable : bool, optional (**DEACTIVATED**)
        if True, use a depthwise separable spectral convolution, by default False   
    preactivation : bool, optional (**DEACTIVATED**)
        whether to compute FNO forward pass with resnet-style preactivation, by default False
    conv_module : nn.Module, optional
        module to use for FNOBlock's convolutions, by default SpectralConv
    
    Examples
    ---------
    
    >>> from neuralop.models import FNO
    >>> model = FNO(n_modes=(12,12), in_channels=1, out_channels=1, hidden_channels=64)
    >>> model
    FNO(
    (positional_embedding): GridEmbeddingND()
    (fno_blocks): FNOBlocks(
        (convs): SpectralConv(
        (weight): ModuleList(
            (0-3): 4 x DenseTensor(shape=torch.Size([64, 64, 12, 7]), rank=None)
        )
        )
            ... torch.nn.Module printout truncated ...

    References
    -----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential 
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    """

    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int=4,
        lifting_channel_ratio: Number=2,
        projection_channel_ratio: Number=2,
        positional_embedding: Union[str, nn.Module]="grid",
        non_linearity: nn.Module=F.gelu,
        norm: Literal ["ada_in", "group_norm", "instance_norm"]=None,
        complex_data: bool=False,
        use_channel_mlp: bool=True,
        channel_mlp_dropout: float=0,
        channel_mlp_expansion: float=0.5,
        channel_mlp_skip: Literal['linear', 'identity', 'soft-gating']="soft-gating",
        fno_skip: Literal['linear', 'identity', 'soft-gating']="linear",
        resolution_scaling_factor: Union[Number, List[Number]]=None,
        domain_padding: Union[Number, List[Number]]=None,
        domain_padding_mode: Literal['symmetric', 'one-sided']="symmetric",
        fno_block_precision: str="full",
        stabilizer: str=None,
        max_n_modes: Tuple[int]=None,
        factorization: str=None,
        rank: float=1.0,
        fixed_rank_modes: bool=False,
        implementation: str="factorized",
        decomposition_kwargs: dict=dict(),
        separable: bool=False,
        preactivation: bool=False,
        conv_module: nn.Module=SpectralConv,
        **kwargs
    ):
        
        super().__init__()
        self.n_dim = len(n_modes)
        
        # n_modes is a special property - see the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * self.hidden_channels)

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * self.hidden_channels)

        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision
        
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0., 1.]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(in_channels=self.in_channels,
                                                        dim=self.n_dim, 
                                                        grid_boundaries=spatial_grid_boundaries)
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(f'Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}')
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding == None:
            self.positional_embedding = None
        else:
            raise ValueError(f"Error: tried to instantiate FNO positional embedding with {positional_embedding},\
                              expected one of \'grid\', GridEmbeddingND")
        
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode
        self.complex_data = self.complex_data

        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            **kwargs
        )
        
        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # if lifting_channels is passed, make lifting a Channel-Mixing MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=non_linearity
            )
        # Convert lifting to a complex ChannelMLP if self.complex_data==True
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def forward(self, x, output_shape=None, **kwargs):
        """FNO's forward pass
        
        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent space

        3. Applies optional domain padding to high-dimensional intermediate function representation

        4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity) 

        5. If domain padding was applied, domain padding is removed

        6. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor
        
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            
            * If None, don't specify an output shape

            * If tuple, specifies the output-shape of the **last** FNO Block

            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes

class FC_FNO1d(FNO):
    """1D FC-FNO.

    You can apply FC either at the beginning or the end of the model or outisde the model.

    Beginning FC is applied before the inital lifiting layer. 

    End FC is right before the last projection layer using the multivariable chain rule. This is just a means to compute derivatives.

    Outside FC is applied after the last projection layer. This is again just a means to compute derivatives.

    ** THERE ARE UN NEEDED PARAMETERS IN THIS CLASS. WILL BE FIXED IN THE FUTURE. **
    
    The where_restriction and where_projection parameters are used to specify the location of the FC function. 

    This was being tested, and the best were obtained when we applied the FC function before the lifting layer and 
    after the projection layer. 

    This will be defualt in the future. 



    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    modes_height : int
        number of Fourier modes to keep along the height
        FC_type: str
            'beginning': FC is applied before the inital lifiting layer. 
            'end': FC is applied right before the last projection layer using the multivariable chain rule. This is just a means to compute derivatives.
            'outside': FC is applied after the last projection layer. This is again just a means to compute derivatives.
        FC_func: function
            FC function to apply.
        cont_points: int
            number of points to use for the continuous part of the domain.
    """

    def __init__(
        self,
        n_modes_height,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        FC_location = str,
        where_restriction = str,
        where_projection = str,
        L = int,
        max_n_modes=None,
        n_layers=4,
        resolution_scaling_factor=None,
        non_linearity=F.gelu,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="symmetric",
        conv_module: nn.Module=SpectralConv,
        grid_points=None,
        FC_func=None,
        FC_type=None,
        cont_points = None,
        one_sided = None,
        fourier_diff = None,
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height,),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            max_n_modes=max_n_modes,
            norm=norm,
            skip=None,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            L = L,
            grid_points = grid_points,
            FC_func=FC_func,
            FC_type=FC_type,
            cont_points = cont_points,
            fourier_diff = fourier_diff,
            one_sided = one_sided,
            FC_location = FC_location,
            where_restriction = where_restriction,
            where_projection = where_projection)
        

        """self.fno_blocks_fc = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            FC_func=FC_func,
            FC_type=FC_type,
            cont_points = cont_points,
            one_sided = one_sided,
            **kwargs
        )"""

        self.n_modes_height = n_modes_height
        self.FC_location = FC_location



        self.L = L
        self.grid_points = grid_points
        self.FC_func=FC_func
        self.FC_type=FC_type
        self.cont_points = cont_points
        self.one_sided = one_sided
        self.fourier_diff = fourier_diff
        self.where_restriction = where_restriction
        self.where_projection = where_projection
    
        
        self.projection = LinearChannelMLP(
            layers=[hidden_channels, projection_channels, out_channels],
            non_linearity=non_linearity,
        )
    
    def dQ(self, X1, DX_arr, num_derivs, Q1, Q2):

        """
        
        Chain rule of model, call it Q, to compute dQ/d(inputs). Uses einsum for efficiency. 

        ***** Assumes that the final nonlinearity is tanh. ******
        
        Gradient chain rule: D(f o g) = D(f(g)) o Dg
        
        Hessian chain rule: D2(f o g) = D2(f(g)) o Dg^2 + D(f(g)) o D2g

        As implemented, Q1 is the first projection layer of the model, and Q2 is the last projection layer.         
        
        Also as implemented, this works up to the second derivative in the 1-D case.

        You can generalize this to higher dimensions by using the chain rule for higher derivatives. 
      
        In the 3rd derivative case, the  chain rule is:
        D3(f o g) = D3(f(g)) o Dg^3 + 3 * D2(f(g)) o Dg * D2g + D(f(g)) o D3g
        
        In the general case, the chain rule is:
        D^n(f o g) = summation(k=0 to n-1) (n-1 choose k) * D^(k+1)g(x)D^(n-k)(f'(g(x)))

        see https://math.stackexchange.com/questions/4046174/nth-derivative-with-chain-rule for more details. 
        
        """
        
        DX = DX_arr[0]           # (b, i, x)
        D2X = DX_arr[1] if num_derivs == 2 else None



        X1 = X1.permute(0, 2, 1)  # (b, m, x)
        b, m, x = X1.shape
        i = Q1.weight.shape[1]    # input dim of Q1
        o = Q2.weight.shape[1]    # output dim 

        DW1 = Q1.weight           # (m, i)
        DW1 = DW1.squeeze(-1)
        DW2 = Q2.weight.reshape(m, 1)  # (m, 1)

        Dtanh = 1 / torch.cosh(X1)**2     # (b, m, x)

        W = (DW1 * DW2).T   # (i, m)


        DQ = torch.matmul(W.unsqueeze(0), Dtanh)  # (1, i, m) @ (b, m, x) => (b, i, x)

        DX_out = (DQ * DX).sum(dim=1)  # (b, x)


        
        #DQ = torch.einsum("mi,bmx->bix", DW1 * DW2, Dtanh)
        #DX_out = torch.einsum("bix,bix->bx", DQ, DX)  # (b, x)

        if num_derivs == 1:
            return DX_out

        # Second derivative
        #Htanh = -2 * D_act * torch.tanh(X1)  # (b, m, x)
        #H_deriv = fourier_derivative_1d(act, order = 2, L = self.L, FC_func=True)
       
        Htanh = -2*Dtanh*torch.tanh(X1)

        H2 = DW2.reshape(1, m, 1) * Htanh    # (b, m, x)

        # Compute D2Q1 = d/dx^2 (Q2(tanh(Q1(x))))
        D2X_1 = (torch.matmul(DW1, DX) * H2 * torch.matmul(DW1, DX)).sum(dim=1)
        D2X_2 = (DQ * D2X).sum(dim=1)
        #D2X_1 = torch.einsum("bix,mi,bmx,mj,bjx->bx", DX, DW1, H2, DW1, DX)
        #D2X_2 = torch.einsum("bix,bix->bx", DQ, D2X)

        D2X_out = D2X_1 + D2X_2
        return DX_out, D2X_out




    def forward(self, x, output_shape=None, **kwargs):
            """FNO's forward pass
            
            1. Applies optional positional encoding

            2. Sends inputs through a lifting layer to a high-dimensional latent space

            3. Applies optional domain padding to high-dimensional intermediate function representation

            4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity) 

            5. If domain padding was applied, domain padding is removed

            6. Projection of intermediate function representation to the output channels

            Parameters
            ----------
            x : tensor
                input tensor
            
            output_shape : {tuple, tuple list, None}, default is None
                Gives the option of specifying the exact output shape for odd shaped inputs.
                
                * If None, don't specify an output shape

                * If tuple, specifies the output-shape of the **last** FNO Block

                * If tuple list, specifies the exact output-shape of each FNO Block
            """

            if output_shape is None:
                output_shape = [None]*self.n_layers
            elif isinstance(output_shape, tuple):
                output_shape = [None]*(self.n_layers - 1) + [output_shape]

            # append spatial pos embedding if set
            if self.positional_embedding is not None:
                x = self.positional_embedding(x) 

            if self.FC_type == 'beginning' and self.where_projection == 'before':
                x = self.FC_func(x)
            
            x = self.lifting(x)

            if self.FC_type == 'beginning' and self.where_projection == 'after':
                x = self.FC_func(x)
            
            if self.domain_padding is not None:
                x = self.domain_padding.pad(x)

            for layer_idx in range(self.n_layers):
                x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

            if self.domain_padding is not None:
                x = self.domain_padding.unpad(x)

            
            if self.FC_type == 'end':
                x1 = self.FC_func(x)
            elif self.FC_type == 'beginning':
                x1 = x

 
            if self.fourier_diff and (self.FC_type == 'end' or self.FC_type == 'beginning'):
                dx = fourier_derivative_1d(x1, order = 1, L = self.L * (self.grid_points+self.cont_points)/self.grid_points)
                dxx = fourier_derivative_1d(x1, order = 2, L = self.L * (self.grid_points+self.cont_points)/self.grid_points)
                if self.one_sided:
                    dx = dx[..., :-self.cont_points]
                    dxx = dxx[..., :-self.cont_points]
                else:
                    dx = dx[..., self.cont_points//2:-self.cont_points//2]
                    dxx = dxx[..., self.cont_points//2:-self.cont_points//2]


                Dx_arr = (dx, dxx)

                Q1 = self.projection.fcs[0]  # first Linear layer
                Q2 = self.projection.fcs[-1]




                if self.FC_type == 'beginning': 
                    if self.one_sided:
                        x1 = x[..., :-self.cont_points]
                else:
                    x1 = x 

                if self.where_restriction == 'before':
                    x = x[..., :-self.cont_points]
                #print(x1.shape)
                #print(x1.transpose(1,2).shape)
                X1 = Q1(x1.transpose(1,2))

                Dx_arr = self.dQ(X1, Dx_arr, 2, Q1, Q2)
                
                x = self.projection(x.transpose(1,2))
            
                x = x.transpose(1,2)

                if self.where_restriction == 'after':
                    x = x[..., :-self.cont_points]
            
            if self.FC_type == 'outside':
                x = self.projection(x.transpose(1,2))
                x1 = self.FC_func(x)
                dx = fourier_derivative_1d(x1, order = 1, L = self.L * (self.grid_points+self.cont_points)/self.grid_points)
                dxx = fourier_derivative_1d(x1, order = 2, L = self.L * (self.grid_points+self.cont_points)/self.grid_points)
                if self.one_sided:
                    dx = dx[..., :-self.cont_points]
                    dxx = dxx[..., :-self.cont_points]
                else:
                    dx = dx[..., self.cont_points//2:-self.cont_points//2]
                    dxx = dxx[..., self.cont_points//2:-self.cont_points//2]

                Dx_arr = (dx, dxx)

            
            return x.squeeze(), Dx_arr[0].squeeze(), *[deriv.squeeze() for deriv in Dx_arr[1:]]


class FC_FNO2d(FNO):
    """2D Fourier Neural Operator. As implemented, the FC_FNO2d works for the 2D Burgers equation. 

     ** THERE ARE UN NEEDED PARAMETERS IN THIS CLASS. WILL BE FIXED IN THE FUTURE. **
    
    Will be generalized to other PDEs in the future.


    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    n_modes_height : int
        number of Fourier modes to keep along the height
    n_modes_width : int
        number of Fourier modes to keep along the width
    FC_type: str
        'beginning': FC is applied before the inital lifiting layer. 
        'end': FC is applied right before the last projection layer using the multivariable chain rule. This is just a means to compute derivatives.
        'outside': FC is applied after the last projection layer. This is again just a means to compute derivatives.
    FC_func: function
        FC function to apply. Assumes it extends the function in 2D
    cont_points: int
        number of points to use for the continuous part of the domain.
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        FC_location = str,
        Lx = int,
        Ly = int,
        max_n_modes=None,
        n_layers=4,
        resolution_scaling_factor=None,
        non_linearity=F.gelu,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="symmetric",
        conv_module: nn.Module=SpectralConv,
        x_second_deriv = None,
        y_second_deriv = None,
        x_res = None,
        y_res = None,
        FC_func=None,   
        FC_type=None,
        cont_points = None,
        one_sided = None,
        fourier_diff = None,
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            max_n_modes=max_n_modes,
            norm=norm,
            skip=None,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            Lx = Lx,
            Ly = Ly,
            FC_func=FC_func,
            FC_type=FC_type,
            cont_points = cont_points,
            fourier_diff = fourier_diff,
            one_sided = one_sided,
            x_second_deriv = x_second_deriv,
            x_res = x_res,
            y_res = y_res,
            y_second_deriv = y_second_deriv)
        

        """self.fno_blocks_fc = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            FC_func=FC_func,
            FC_type=FC_type,
            cont_points = cont_points,
            one_sided = one_sided,
            **kwargs
        )"""

        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.FC_location = FC_location



        self.Lx = Lx
        self.Ly = Ly
        self.FC_func=FC_func
        self.FC_type=FC_type
        self.cont_points = cont_points
        self.one_sided = one_sided
        self.fourier_diff = fourier_diff
        self.x_res = x_res
        self.y_res = y_res
        self.x_second_deriv = x_second_deriv
        self.y_second_deriv = y_second_deriv
        
        self.projection = LinearChannelMLP(
            layers=[hidden_channels, projection_channels, out_channels],
            non_linearity=non_linearity,
        )
    
    def dQ_2D(self, X1, DX_arr, x_second_deriv, y_second_deriv, Q1, Q2):
        """
        
        Chain rule of model, call it Q, to compute dQ/d(inputs). Uses einsum for efficiency. 

        ***** Assumes that the final nonlinearity is tanh. ******
        
        Gradient chain rule: D(f o g) = D(f(g)) o Dg
        
        Hessian chain rule: D2(f o g) = D2(f(g)) o Dg^2 + D(f(g)) o D2g

        As implemented, Q1 is the first projection layer of the model, and Q2 is the last projection layer.         
        
        Also as implemented, this works up for the derivatives in the 2-D Burgers equation. 
        In future, this will be generalized and more robust for any PDE

        You can generalize this to higher dimensions by using the chain rule for higher derivatives. 
      
        In the 3rd derivative case, the chain rule is:
        D3(f o g) = D3(f(g)) o Dg^3 + 3 * D2(f(g)) o Dg * D2g + D(f(g)) o D3g
        
        In the general case, the chain rule is:
        D^n(f o g) = summation(k=0 to n-1) (n-1 choose k) * D^(k+1)g(x)D^(n-k)(f'(g(x)))

        see https://math.stackexchange.com/questions/4046174/nth-derivative-with-chain-rule for more details. 
        
        """
        
        
            
        wx, wt = DX_arr[0], DX_arr[1] # shapes (B,C,T,X)

        # X1 (B, X, T, C)
        X1 = X1.permute(0, 3, 1, 2)  #
        # now X1 (B, C, T, X)

        b = X1.shape[0]
        t = X1.shape[3]
        x = X1.shape[2]
        i = self.hidden_channels
        m = 256 #self.x_res *4
        n = 128 #self.x_res * 2#self.hidden_channels*2
        #o = self.out_dim

        ### Gradient: D(f o g) = D(f(g)) o Dg
        DW1 = Q1.weight #(m, i)
        Dtanh = 1/torch.cosh(X1)**2  # (b,m,x)
        DW2 = Q2.weight.reshape(2*n,)  # (m, o)
        DQ = torch.einsum("mi,bmtx,m->bitx", DW1, Dtanh, DW2)

        wxQ = torch.einsum("bitx, bitx->btx", DQ, wx)
        wtQ = torch.einsum("bitx, bitx->btx", DQ, wt)

        ### Hessian: D^2(f o g) = Dg o Hf o Dg + Df o Hg

        if self.x_second_deriv:
            wxx = DX_arr[2]
        if self.y_second_deriv:
            wyy = DX_arr[3]

        Htanh = -2*Dtanh*torch.tanh(X1)
        H2 = DW2.reshape(1,m,1,1)*Htanh # (b,m,x,y,t)

        if self.x_second_deriv:
            wxx1 = torch.einsum("bitx,mi,bmtx,mj,bjtx->btx", wx,DW1,H2,DW1,wx) # (b,t,x)
            wxx2 = torch.einsum("bitx,bitx->btx", DQ.reshape(b,i,t,x), wxx)
            wxxQ = wxx1 + wxx2
        
        if self.y_second_deriv:
            wyy1 = torch.einsum("bitx,mi,bmtx,mj,bjtx->btx", wy,DW1,H2,DW1,wy) # (b,t,y)
            wyy2 = torch.einsum("bitx,bitx->btx", DQ.reshape(b,i,t,y), wyy)
            wyyQ = wyy1 + wyy2

        if self.x_second_deriv and self.y_second_deriv:
            Dx_arr = (wxQ, wtQ, wxxQ, wyyQ)
        elif self.x_second_deriv and not self.y_second_deriv:
            Dx_arr = (wxQ, wtQ, wxxQ)
        elif not self.x_second_deriv and self.y_second_deriv:
            Dx_arr = (wxQ, wtQ, wyyQ)
        else:
            Dx_arr = (wxQ, wtQ)

            
        return Dx_arr

    def forward(self, x, output_shape=None, **kwargs):
            """FNO's forward pass
            
            1. Applies optional positional encoding

            2. Sends inputs through a lifting layer to a high-dimensional latent space

            3. Applies optional domain padding to high-dimensional intermediate function representation

            4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity) 

            5. If domain padding was applied, domain padding is removed

            6. Projection of intermediate function representation to the output channels

            Parameters
            ----------
            x : tensor
                input tensor
            
            output_shape : {tuple, tuple list, None}, default is None
                Gives the option of specifying the exact output shape for odd shaped inputs.
                
                * If None, don't specify an output shape

                * If tuple, specifies the output-shape of the **last** FNO Block

                * If tuple list, specifies the exact output-shape of each FNO Block
            """

            if output_shape is None:
                output_shape = [None]*self.n_layers
            elif isinstance(output_shape, tuple):
                output_shape = [None]*(self.n_layers - 1) + [output_shape]

            # append spatial pos embedding if set
            if self.positional_embedding is not None:
                x = self.positional_embedding(x)
            
            x = self.lifting(x)

            if self.FC_type == 'beginning':
                x = self.FC_func(x)
            
            if self.domain_padding is not None:
                x = self.domain_padding.pad(x)

            for layer_idx in range(self.n_layers):
                x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

            if self.domain_padding is not None:
                x = self.domain_padding.unpad(x)

            
            if self.FC_type == 'end':
                x1 = self.FC_func(x)
            elif self.FC_type == 'beginning':  
                x1 = x 

 
            if self.fourier_diff and (self.FC_type == 'end' or self.FC_type == 'beginning'):
                dy = fourier_derivative_1d(x1, order = 1, L = self.Ly * (self.y_res+self.cont_points)/self.y_res)
                dx = fourier_derivative_1d(x1.permute(0, 1, 3, 2), order = 1, L = self.Lx * (self.x_res+self.cont_points)/self.x_res)
                dx = dx.permute(0, 1, 3, 2)
                dyy = fourier_derivative_1d(x1, order = 2, L = self.Ly * (self.y_res+self.cont_points)/self.y_res)
                dxx = fourier_derivative_1d(x1.permute(0, 1, 3, 2), order = 2, L = self.Lx * (self.x_res+self.cont_points)/self.x_res)
                dxx = dxx.permute(0, 1, 3, 2)

                #dx, dy, dxx, dyy = fourier_derivative_2d(x1, order = 1, Lx = self.Lx * (self.x_res+self.cont_points)/self.x_res, Ly = self.Ly * (self.y_res+self.cont_points)/self.y_res, x_second_deriv = self.x_second_deriv, y_second_deriv = self.y_second_deriv)
                if self.one_sided:
                    dx = dx[..., :-self.cont_points, :-self.cont_points]
                    dy = dy[..., :-self.cont_points, :-self.cont_points]
                    if self.x_second_deriv:
                        dxx = dxx[..., :-self.cont_points, :-self.cont_points]
                    if self.y_second_deriv:
                        dyy = dyy[..., :-self.cont_points, :-self.cont_points]
                else:
                    dx = dx[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    dy = dy[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.x_second_deriv:
                        dxx = dxx[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.y_second_deriv:
                        dyy = dyy[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
    
                if self.x_second_deriv:
                    Dx_arr = (dx, dy, dxx)
                    #print(len(Dx_arr))
                elif self.y_second_deriv:
                    Dx_arr = (dx, dy, dyy)
                elif self.x_second_deriv and self.y_second_deriv:
                    Dx_arr = (dx, dy, dxx, dyy)
                else:
                    Dx_arr = (dx, dy)

                Q1 = self.projection.fcs[0]  # first Linear layer
                Q2 = self.projection.fcs[-1]

        

                if self.FC_type == 'beginning': 
                    if self.one_sided:
                        x1 = x[..., :-self.cont_points, :-self.cont_points,]
                    else:
                        x1 = x[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                if self.FC_type == 'end':
                    x1 = x

                # Reshape x1 to be compatible with Q1 while preserving orientation
                b, c, h, w = x1.shape
                x1_flat = x1.permute(0, 2, 3, 1)
                X1 = Q1(x1_flat) 
                Dx_arr = self.dQ_2D(X1, Dx_arr, self.x_second_deriv, self.y_second_deriv, Q1, Q2)

                x = x.permute(0, 2, 3, 1) 
                x = self.projection(x)  
                x = x.permute(0, 3, 1, 2)
                #print(x.shape)
                

                if self.FC_type == 'beginning':
                    if self.one_sided:
                        x = x[..., :-self.cont_points, :-self.cont_points]
                    else:
                        x = x[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                x = x.squeeze()
                x = x.T
            
            if self.FC_type == 'outside':
                x = x.permute(0, 2, 3, 1) 
                x = self.projection(x)  
                x = x.permute(0, 3, 1, 2)
                #x = x.transpose(1, 2)
                x1 = self.FC_func(x)
                dy = fourier_derivative_1d(x1, order = 1, L = self.Ly * (self.y_res+self.cont_points)/self.y_res)
                dx = fourier_derivative_1d(x1.permute(0, 1, 3, 2), order = 1, L = self.Lx * (self.x_res+self.cont_points)/self.x_res)
                dx = dx.permute(0, 1, 3, 2)
                dxx = fourier_derivative_1d(x1.permute(0, 1, 3, 2), order = 2, L = self.Lx * (self.x_res+self.cont_points)/self.x_res)
                dxx = dxx.permute(0, 1, 3, 2)
                dyy = fourier_derivative_1d(x1, order = 2, L = self.Ly * (self.y_res+self.cont_points)/self.y_res)
                dyy = dyy

                #dx, dy, dxx, dyy = fourier_derivative_2d(x1, order = 1, Lx = self.Lx * (self.x_res+self.cont_points)/self.x_res, Ly = self.Ly * (self.y_res+self.cont_points)/self.y_res, x_second_deriv = self.x_second_deriv, y_second_deriv = self.y_second_deriv)
                if self.one_sided:
                    dx = dx[..., :-self.cont_points, :-self.cont_points]
                    dy = dy[..., :-self.cont_points, :-self.cont_points]
                    if self.x_second_deriv:
                        dxx = dxx[..., :-self.cont_points, :-self.cont_points]
                    if self.y_second_deriv:
                        dyy = dyy[..., :-self.cont_points, :-self.cont_points]
                else:
                    dx = dx[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    dy = dy[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.x_second_deriv:
                        dxx = dxx[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.y_second_deriv:
                        dyy = dyy[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
    
                if self.x_second_deriv:
                    Dx_arr = (dx, dy, dxx)
                    #print(len(Dx_arr))
                elif self.y_second_deriv:
                    Dx_arr = (dx, dy, dyy)
                elif self.x_second_deriv and self.y_second_deriv:
                    Dx_arr = (dx, dy, dxx, dyy)
                else:
                    Dx_arr = (dx, dy)
                
                Dx_arr = (Dx_arr[0].squeeze(), *[deriv.squeeze() for deriv in Dx_arr[1:]])
                x = x.squeeze()
                x = x.T
                

            
            #print(x.shape)
            return x.squeeze(), Dx_arr #[0].squeeze(), *[deriv.squeeze() for deriv in Dx_arr[1:]]

class FC_FNO3d(FNO):
    """3D FC-FNO. As implemented, the FC_FNO3d works for the 2D + time Navier-Stokes equation. 


    ** THERE ARE UN NEEDED PARAMETERS IN THIS CLASS. WILL BE FIXED IN THE FUTURE. **
    
    Will be generalized to other PDEs in the future.

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    modes_width : int
        number of modes to keep in Fourier Layer, along the width
    modes_height : int
        number of Fourier modes to keep along the height
    modes_depth : int
        number of Fourier modes to keep along the depth
    FC_type: str
        'beginning': FC is applied before the inital lifiting layer. 
        'end': FC is applied right before the last projection layer using the multivariable chain rule. This is just a means to compute derivatives.
        'outside': FC is applied after the last projection layer. This is again just a means to compute derivatives.
    FC_func: function
        FC function to apply. Assumes it extends the function in 3D
    cont_points: int
        number of points to use for the continuous part of the domain.
    laplacian: bool
        whether to compute the Laplacian.
    x_second_deriv: bool
        whether to compute the second derivative along the x-direction.
    y_second_deriv: bool
        whether to compute the second derivative along the y-direction.
    z_second_deriv: bool
        whether to compute the second derivative along the z-direction.
    gradient: bool
        whether to compute the gradient.
    x_res: int
        number of grid points along the x-direction.
    y_res: int
        number of grid points along the y-direction.
    z_res: int  
        number of grid points along the z-direction.
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        n_modes_depth,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        resolution_scaling_factor=None,
        max_n_modes=None,
        non_linearity=F.gelu,
        Lx = int,
        Ly = int,
        Lz = int,
        x_deriv = None,
        y_deriv = None,
        z_deriv = None,
        x_second_deriv = None,
        y_second_deriv = None,
        z_second_deriv = None,
        gradient = None, 
        laplacian = None,   
        x_res = None,
        y_res = None,
        z_res = None,
        FC_func=None,
        FC_type = None,
        cont_points = None,
        one_sided = None,
        fourier_diff = None,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="symmetric",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width, n_modes_depth),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            max_n_modes=max_n_modes,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            FC_func=FC_func,
            FC_type=FC_type,
            cont_points=cont_points,
            one_sided=one_sided,
            fourier_diff=fourier_diff,
            x_deriv=x_deriv,
            y_deriv=y_deriv,
            z_deriv=z_deriv,
            x_second_deriv=x_second_deriv,
            y_second_deriv=y_second_deriv,
            z_second_deriv=z_second_deriv,
            gradient=gradient,
            x_res=x_res,
            y_res=y_res,
            z_res=z_res,
            laplacian=laplacian,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_modes_depth = n_modes_depth
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.FC_func = FC_func
        self.FC_type = FC_type
        self.cont_points = cont_points
        self.one_sided = one_sided
        self.fourier_diff = fourier_diff
        self.x_deriv = x_deriv
        self.y_deriv = y_deriv
        self.z_deriv = z_deriv
        self.x_second_deriv = x_second_deriv
        self.y_second_deriv = y_second_deriv
        self.z_second_deriv = z_second_deriv
        self.gradient = gradient
        self.x_res = x_res
        self.y_res = y_res
        self.z_res = z_res
        self.laplacian = laplacian
        self.projection = LinearChannelMLP(
            layers=[hidden_channels, projection_channels, out_channels],
            non_linearity=non_linearity,)

    def dQ_3D(self, X1, DX_arr, laplacian, x_second_deriv, y_second_deriv, Q1, Q2):

        """
        Chain rule of model, call it Q, to compute dQ/d(inputs). Uses einsum for efficiency. 

        ***** Assumes that the final nonlinearity is tanh. ******
        
        Gradient chain rule: D(f o g) = D(f(g)) o Dg
        
        Hessian chain rule: D2(f o g) = D2(f(g)) o Dg^2 + D(f(g)) o D2g

        As implemented, Q1 is the first projection layer of the model, and Q2 is the last projection layer.         
        
        Also as implemented, this works for the Navier-Stokes 2D + time case.
        In future, this will be generalized and more robust for any PDE

        You can generalize this to higher dimensions by using the chain rule for higher derivatives. 
      
        In the 3rd derivative case, the  chain rule is:
        D3(f o g) = D3(f(g)) o Dg^3 + 3 * D2(f(g)) o Dg * D2g + D(f(g)) o D3g
        
        In the general case, the chain rule is:
        D^n(f o g) = summation(k=0 to n-1) (n-1 choose k) * D^(k+1)g(x)D^(n-k)(f'(g(x)))

        see https://math.stackexchange.com/questions/4046174/nth-derivative-with-chain-rule for more details. 
        
        """

        # DX_arr = (dx, dy, dt)
        # second_derivs = (dxx, dyy)
        dx, dy, dt = DX_arr
        dxx, dyy   = laplacian      # (b,i,t,x,z)

        # Put channels first: (B,C,T,X,Z)
        X1 = X1.permute(0,4,1,2,3)
        B, C, T, X, Z = X1.shape
        I = self.hidden_channels
        O = self.out_channels


        DW1 = Q1.weight           
        DW2 = Q2.weight.t()            # (C_in, O)

        Dtanh = 1/torch.cosh(X1)**2           # (B,C,T,X,Z)

        # First-order term J_f(g(x))*J_g(x)
        
        DQ   = torch.einsum(
            "ci, bctxz, co -> boitxz",
            DW1, Dtanh, DW2
        )      # (B,O,I,T,X,Z)

        wxQ = torch.einsum("boitxz,bitxz->botxz", DQ, dx)
        wzQ = torch.einsum("boitxz,bitxz->botxz", DQ, dy)
        wtQ = torch.einsum("boitxz,bitxz->botxz", DQ, dt)

        # Hessian term J_g^T * H_f * J_g  +  J_f * H_g
        Htanh = -2*Dtanh*torch.tanh(X1)                         # (B,C,T,X,Z)
        H2    = torch.einsum("co,bctxz->bcotxz", DW2, Htanh)    # (B,C,O,T,X,Z)

        wxx1  = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz",
                            dx, DW1, H2, DW1, dx)
        wxx2  = torch.einsum("boitxz,bitxz->botxz", DQ, dxx)
        wxxQ  = wxx1 + wxx2

        wzz1  = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz",
                            dy, DW1, H2, DW1, dy)
        wzz2  = torch.einsum("boitxz,bitxz->botxz", DQ, dyy)
        wzzQ  = wzz1 + wzz2

        DX_out       = (wxQ, wzQ, wtQ)
        laplacian_out = (wxxQ, wzzQ)
        return DX_out, laplacian_out


    def forward(self, x, output_shape=None, **kwargs):
            """FNO's forward pass
            
            1. Applies optional positional encoding

            2. Sends inputs through a lifting layer to a high-dimensional latent space

            3. Applies optional domain padding to high-dimensional intermediate function representation

            4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity) 

            5. If domain padding was applied, domain padding is removed

            6. Projection of intermediate function representation to the output channels

            Parameters
            ----------
            x : tensor
                input tensor
            
            output_shape : {tuple, tuple list, None}, default is None
                Gives the option of specifying the exact output shape for odd shaped inputs.
                
                * If None, don't specify an output shape

                * If tuple, specifies the output-shape of the **last** FNO Block

                * If tuple list, specifies the exact output-shape of each FNO Block
            """

            if output_shape is None:
                output_shape = [None]*self.n_layers
            elif isinstance(output_shape, tuple):
                output_shape = [None]*(self.n_layers - 1) + [output_shape]

            # append spatial pos embedding if set
            if self.positional_embedding is not None:
                x = self.positional_embedding(x)
            
            x = self.lifting(x)

            if self.FC_type == 'beginning':
                x = self.FC_func(x)
            
            if self.domain_padding is not None:
                x = self.domain_padding.pad(x)

            for layer_idx in range(self.n_layers):
                x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

            if self.domain_padding is not None:
                x = self.domain_padding.unpad(x)

            
            if self.FC_type == 'end':
                x1 = self.FC_func(x)
                #x1 = self.FC_func(x, 2, 3)
                #x1 = self.FC_func(x1, 3, 3)
            elif self.FC_type == 'beginning':
                x1 = x #(b, c, x, y, z) 

            
            if self.fourier_diff and (self.FC_type == 'end' or self.FC_type == 'beginning'):
                dz = fourier_derivative_1d(x1, order = 1, L = self.Lz * (self.z_res+self.cont_points)/self.z_res)
                dy = fourier_derivative_1d(x1.permute(0, 1, 2, 4, 3), order = 1, L = self.Ly * (self.y_res+self.cont_points)/self.y_res)
                dy = dy.permute(0, 1, 2, 4, 3)
                dx = fourier_derivative_1d(x1.permute(0, 1, 3, 4, 2), order = 1, L = self.Lx * (self.x_res+self.cont_points)/self.x_res)
                dx = dx.permute(0, 1, 4, 2, 3)
                dxx = fourier_derivative_1d(x1.permute(0, 1, 3, 4, 2), order = 2, L = self.Lx * (self.x_res+self.cont_points)/self.x_res)
                dxx = dxx.permute(0, 1, 4, 2, 3)
                dyy = fourier_derivative_1d(x1.permute(0, 1, 2, 4, 3), order = 2, L = self.Ly * (self.y_res+self.cont_points)/self.y_res)
                dyy = dyy.permute(0, 1, 2, 4, 3)
                if self.gradient:
                    gradient = gradient_3D(x1, order = 1, Lx = self.Lx * (self.x_res+self.cont_points)/self.x_res, Ly = self.Ly * (self.y_res+self.cont_points)/self.y_res, Lz = self.Lz * (self.z_res+self.cont_points)/self.z_res)
                if self.laplacian:
                    laplacian_arr = (dxx, dyy)

                #dx, dy, dxx, dyy = fourier_derivative_2d(x1, order = 1, Lx = self.Lx * (self.x_res+self.cont_points)/self.x_res, Ly = self.Ly * (self.y_res+self.cont_points)/self.y_res, x_second_deriv = self.x_second_deriv, y_second_deriv = self.y_second_deriv)
                if self.one_sided:
                    dx = dx[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    dy = dy[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    dz = dz[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    #print("dx.shape", dx.shape)
                    #print('dy.shape', dy.shape)
                    #print('dz.shape', dz.shape)
                    if self.x_second_deriv:
                        dxx = dxx[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    if self.y_second_deriv:
                        dyy = dyy[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    if self.z_second_deriv:
                        dzz = dzz[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    if self.gradient:
                        gradient = gradient[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    if self.laplacian:
                        lap_list = []
                        for i in laplacian_arr:
                            lap_list.append(i[..., :-self.cont_points, :-self.cont_points, :-self.cont_points])
                        laplacian = lap_list
                else:
                    dx = dx[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    dy = dy[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    dz = dz[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.x_second_deriv:
                        dxx = dxx[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.y_second_deriv:
                        dyy = dyy[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.z_second_deriv:
                        dzz = dzz[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.gradient:
                        gradient = gradient[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.laplacian:
                        dxx = dxx[..., -self.cont_points, -self.cont_points, -self.cont_points]
                        dyy = dyy[..., -self.cont_points, -self.cont_points, -self.cont_points]
                        laplacian = (dxx, dyy)
                        #dzz = dzz[..., -self.cont_points, -self.cont_points, -self.cont_points]
                        #laplacian = laplacian_3D(x1, Lx = self.Lx * (self.x_res+self.cont_points)/self.x_res, Ly = self.Ly * (self.y_res+self.cont_points)/self.y_res, Lz = self.Lz * (self.z_res+self.cont_points)/self.z_res, spatial = True)
                Dx_arr = (dx, dy, dz)
                if self.x_second_deriv:
                    Dx_arr = Dx_arr.append(dxx)
                if self.y_second_deriv:
                    Dx_arr.append(dyy)
                if self.z_second_deriv:
                    Dx_arr.append(dzz)


                Q1 = self.projection.fcs[0]  # first Linear layer
                Q2 = self.projection.fcs[-1]

                if self.FC_type == 'beginning': 
                    if self.one_sided:
                        x1 = x[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    else:
                        x1 = x[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                if self.FC_type == 'end':
                    x1 = x

                #if self.fourier_diff:
                #    x1 = x
                #else:
                #    x1 = x 

                b, c, h, w, d = x1.shape
                x1_flat = x1.permute(0, 2, 3, 4, 1)
                X1 = Q1(x1_flat) 
                #for i in Dx_arr:
                #    print("i.shape", i.shape)
                Dx_arr, laplacian = self.dQ_3D(X1, Dx_arr, laplacian, self.x_second_deriv, self.y_second_deriv, Q1, Q2)
                #laplacian = laplacian[0] + laplacian[1]
                #gradient = self.dQ_3D(X1, gradient, self.x_second_deriv=False, self.y_second_deriv=False, self.z_second_deriv=False, Q1, Q2)

                x = x.permute(0, 2, 3, 4, 1) 
                x = self.projection(x)  
                x = x.permute(0, 4, 1, 2, 3)
            

                if self.FC_type == 'beginning':
                    if self.one_sided:
                        x = x[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    else:
                        x = x[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
            
            if self.FC_type == 'outside':
                # Reshape x for projection: [b, c, h, w] -> [b, h*w, c]
                x = x.permute(0, 2, 3, 4, 1) # .reshape(b, h*w, c)
                x = self.projection(x)  # Apply projection
                x = x.permute(0, 4, 1, 2, 3)
                x1 = self.FC_func(x)
                dz = fourier_derivative_1d(x1, order = 1, L = self.Lz * (self.z_res+self.cont_points)/self.z_res)
                dy = fourier_derivative_1d(x1.permute(0, 1, 2, 4, 3), order = 1, L = self.Ly * (self.y_res+self.cont_points)/self.y_res)
                dy = dy.permute(0, 1, 2, 4, 3)
                dx = fourier_derivative_1d(x1.permute(0, 1, 3, 4, 2), order = 1, L = self.Lx * (self.x_res+self.cont_points)/self.x_res)
                dx = dx.permute(0, 1, 4, 2, 3)
                dxx = fourier_derivative_1d(x1.permute(0, 1, 3, 4, 2), order = 2, L = self.Lx * (self.x_res+self.cont_points)/self.x_res)
                dxx = dxx.permute(0, 1, 4, 2, 3)
                dyy = fourier_derivative_1d(x1.permute(0, 1, 2, 4, 3), order = 2, L = self.Ly * (self.y_res+self.cont_points)/self.y_res)
                dyy = dyy.permute(0, 1, 2, 4, 3)
                if self.gradient:
                    gradient = gradient_3D(x1, order = 1, Lx = self.Lx * (self.x_res+self.cont_points)/self.x_res, Ly = self.Ly * (self.y_res+self.cont_points)/self.y_res, Lz = self.Lz * (self.z_res+self.cont_points)/self.z_res)
                if self.laplacian:
                    laplacian = laplacian_3D(x1, Lx = self.Lx * (self.x_res+self.cont_points)/self.x_res, Ly = self.Ly * (self.y_res+self.cont_points)/self.y_res, Lz = self.Lz * (self.z_res+self.cont_points)/self.z_res)

                #dx, dy, dxx, dyy = fourier_derivative_2d(x1, order = 1, Lx = self.Lx * (self.x_res+self.cont_points)/self.x_res, Ly = self.Ly * (self.y_res+self.cont_points)/self.y_res, x_second_deriv = self.x_second_deriv, y_second_deriv = self.y_second_deriv)
                if self.one_sided:
                    dx = dx[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    dy = dy[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    dz = dz[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    if self.x_second_deriv:
                        dxx = dxx[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    if self.y_second_deriv:
                        dyy = dyy[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    if self.z_second_deriv:
                        dzz = dzz[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    if self.gradient:
                        gradient = gradient[..., :-self.cont_points, :-self.cont_points, :-self.cont_points]
                    if self.laplacian:
                        lap_list = []
                        for i in laplacian:
                            lap_list.append(i[..., :-self.cont_points, :-self.cont_points, :-self.cont_points])
                        laplacian = lap_list
                        #laplacian = lap_list[0] + lap_list[1]
                else:
                    dx = dx[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    dy = dy[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    dz = dz[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.x_second_deriv:
                        dxx = dxx[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.y_second_deriv:
                        dyy = dyy[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.z_second_deriv:
                        dzz = dzz[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.gradient:
                        gradient = gradient[..., self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2, self.cont_points//2:-self.cont_points//2]
                    if self.laplacian:
                        laplacian_arr = laplacian_3D(x1, Lx = self.Lx * (self.x_res+self.cont_points)/self.x_res, Ly = self.Ly * (self.y_res+self.cont_points)/self.y_res, Lz = self.Lz * (self.z_res+self.cont_points)/self.z_res, spatial = True)
                        #laplacian = laplacian_arr[0] + laplacian_arr[1]
                Dx_arr = (dx, dy, dz)
          

            if self.gradient and self.laplacian:
                return x, Dx_arr, gradient, laplacian
            elif self.gradient:
                return x, Dx_arr, gradient
            elif self.laplacian:
                #print("laplacian.shape", laplacian.shape)
                #print("x.shape", x.shape)
                return x, Dx_arr, laplacian
            else:
                return x, Dx_arr

class FNO1d(FNO):
    """1D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    modes_height : int
        number of Fourier modes to keep along the height
    """

    def __init__(
        self,
        n_modes_height,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        max_n_modes=None,
        n_layers=4,
        resolution_scaling_factor=None,
        non_linearity=F.gelu,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="symmetric",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height,),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            max_n_modes=max_n_modes,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
        )
        self.n_modes_height = n_modes_height


class FNO2d(FNO):
    """2D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    n_modes_width : int
        number of modes to keep in Fourier Layer, along the width
    n_modes_height : int
        number of Fourier modes to keep along the height
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        resolution_scaling_factor=None,
        max_n_modes=None,
        non_linearity=F.gelu,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="symmetric",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            max_n_modes=max_n_modes,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width


class FNO3d(FNO):
    """3D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    modes_width : int
        number of modes to keep in Fourier Layer, along the width
    modes_height : int
        number of Fourier modes to keep along the height
    modes_depth : int
        number of Fourier modes to keep along the depth
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        n_modes_depth,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        resolution_scaling_factor=None,
        max_n_modes=None,
        non_linearity=F.gelu,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="symmetric",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width, n_modes_depth),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            max_n_modes=max_n_modes,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_modes_depth = n_modes_depth


def partialclass(new_name, cls, *args, **kwargs):
    """Create a new class with different default values

    Notes
    -----
    An obvious alternative would be to use functools.partial
    >>> new_class = partial(cls, **kwargs)

    The issue is twofold:
    1. the class doesn't have a name, so one would have to set it explicitly:
    >>> new_class.__name__ = new_name

    2. the new class will be a functools object and one cannot inherit from it.

    Instead, here, we define dynamically a new class, inheriting from the existing one.
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    new_class = type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "forward": cls.forward,
        },
    )
    return new_class


TFNO = partialclass("TFNO", FNO, factorization="Tucker")
TFNO1d = partialclass("TFNO1d", FNO1d, factorization="Tucker")
TFNO2d = partialclass("TFNO2d", FNO2d, factorization="Tucker")
TFNO3d = partialclass("TFNO3d", FNO3d, factorization="Tucker")


