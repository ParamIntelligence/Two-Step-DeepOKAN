import jax
import numpy as np
import scipy.io as io
from scipy import signal
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
import optax
import pickle
import matplotlib.pyplot as plt
import sys
from flax import nnx
from jax import grad, jit, vmap, value_and_grad
from jax import random
key = random.PRNGKey(1234)
from jax.example_libraries import optimizers
import jaxkan

import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec

from jaxKAN.jaxkan.layers.SplineLayer import SplineLayer
from jaxKAN.jaxkan.layers.RBFLayer import RBFLayer

import orbax.checkpoint as ocp
import torch

plt.rcParams.update({
'font.family': 'serif',
'font.size': 22
})

class Trunk(nnx.Module):
    def __init__(self,
                 layer_dims, required_parameters, Am
                ):

        if required_parameters is None:
            raise ValueError("required_parameters must be provided as a dictionary for the selected layer_type.")

        self.rngs = nnx.Rngs(42)
        self.layer_dims = layer_dims
        self.layers = [
                SplineLayer(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    **required_parameters,
                    rngs=self.rngs
                )
                for i in range(len(layer_dims) - 1)
            ]
        self.biases = [
            nnx.Param(jnp.zeros((dim,))) for dim in layer_dims[1:]
            ]
        self.Am = nnx.Param(Am)

    def __call__(self, x, both=False, pred_only=False):

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x += self.biases[i].value

        if pred_only:
            return x
        if not both:
            output = jnp.einsum('im,jm->ij', self.Am, x) 
            return output
        if both:
            return x, self.Am

class Branch(nnx.Module):
    def __init__(self,
                 layer_dims, required_parameters
                ):

        if required_parameters is None:
            raise ValueError("required_parameters must be provided as a dictionary for the selected layer_type.")

        self.rngs = nnx.Rngs(42)
        self.layer_dims = layer_dims
        self.layers = [
                RBFLayer(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    **required_parameters,
                    seed = 444
                )
                for i in range(len(layer_dims) - 1)
            ]

        self.biases = [
            nnx.Param(jnp.zeros((dim,))) for dim in layer_dims[1:]
            ]
    
    def __call__(self, v):

        for i, layer in enumerate(self.layers): 
            v = layer(v)
            v += self.biases[i].value
        return v


#Fourier Transform
def fft_(y_true, Par):
    y_true = torch.from_numpy(y_true)

    signal_length = y_true.shape[-1]
    duration = Par['t_max'] #0.0001
    sampling_rate = duration / signal_length  

    freqs = torch.fft.rfftfreq(signal_length, d=float(sampling_rate))
    fft_true = torch.fft.rfft(y_true,dim=-1)  
    fft_mag_true = torch.abs(fft_true)

    return freqs, fft_mag_true


def loaddata(data):
    X_fun = jnp.array(data["P"])
    X_loc = jnp.array(data["t"]).squeeze()
    y = jnp.array(data["R"])
    namelist = data['names']
    return X_fun,X_loc,y,namelist


def nondim2D(X_func, R0, X_loc, y, Par):
    rho = 1e+03
    tau = Par['t_max']
    R0max = Par['R0_max']
    R0_bar = R0 / R0max
    scale = 1/R0_bar
    Par['scale'] = scale
    if Par['case'] == 'singleR': 
        P_star = (1000*R0max**2*rho)/(tau**2) #for KM and RP, single bubble
    elif Par['case'] == 'multiR':
        P_star = (250*scale**2*R0**2*rho)/(tau**2)

    X_func_bar = X_func/P_star[0]
    X_loc_bar = X_loc / tau
    # y_bar = y/R0.reshape(1, len(R0), 1) 
    y_bar = y/R0max
    # R0 = R0/R0max

    return (
        X_func_bar,
        X_loc_bar,
        y_bar,
        R0_bar,
        Par
    )


def dimensional2D(X_loc_bar, X_func_bar, y_pred_bar, y_true_bar, R0, Par):
    rho = 1e+03
    tau = Par['t_max']
    R0max = Par['R0_max']
    P_star = (250*(R0max)**2*rho)/(tau**2)
    
    R0 = R0*R0max
    X_func = X_func_bar*P_star
    X_loc = X_loc_bar*tau
    
    y_pred = y_pred_bar*R0max#.reshape(1,-1,1)
    y_true = y_true_bar*R0max#.reshape(1,-1,1)

    return y_pred, y_true,X_loc, R0

#Fourier Transform
def fft(y_pred, y_true, Par):

    y_pred = y_pred.__array__()
    y_pred = torch.from_numpy(y_pred)

    y_true = y_true.__array__()
    y_true = torch.from_numpy(y_true)

    signal_length = y_pred.shape[-1]
    duration = Par['t_max'] #0.0001
    sampling_rate = duration / signal_length  

    freqs = torch.fft.rfftfreq(signal_length, d=float(sampling_rate))
    fft_pred = torch.fft.rfft(y_pred,dim=-1)
    fft_true = torch.fft.rfft(y_true,dim=-1)
    fft_mag_pred = torch.abs(fft_pred)  
    fft_mag_true = torch.abs(fft_true)

    return freqs, fft_mag_pred, fft_mag_true

def plotfourier(freq, y_pred, y_true):
    np.random.seed(23)
    
    domain = 151#len(freq) 221; 151 for extrapolation; 121 for interpolation 
    plt.figure(figsize=(10, 5))
    plt.plot(freq[1:domain],np.log(y_pred[1:domain]),'r--') #[:domain]
    plt.plot(freq[1:domain],np.log(y_true[1:domain]),'b-')
    plt.xlabel("Frequency (Hz)",fontsize=18)
    plt.ylabel("Magnitude",fontsize=18)
    plt.legend(['Prediction', 'True'], fontsize=16)
    plt.grid(True)

def load_model_b(path, layer_dims, req_params):
    # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
    abstract_model = Branch(layer_dims=layer_dims, required_parameters=req_params)

    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.PyTreeCheckpointer()
    state_restored = checkpointer.restore(os.path.join(path, f'best_model'), item=abstract_state)

    # The model is now good to use!
    model = nnx.merge(graphdef, state_restored)
    return model


def load_model(path, layer_dims, req_params, A):
    # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
    abstract_model = Trunk(layer_dims=layer_dims, required_parameters=req_params, Am=A)

    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.PyTreeCheckpointer()
    state_restored = checkpointer.restore(os.path.join(path, 'best_model'), item=abstract_state)

    # The model is now good to use!
    model = nnx.merge(graphdef, state_restored)
    return model


def infer_solution(inputs):

    foldert = inputs['Model_folder']['Trunk']
    folderb = inputs['Model_folder']['Branch']
    layers_x = inputs['Architecture']['Trunk']
    layers_f = inputs['Architecture']['Branch']
    Result_folder = inputs['Results_folder']
    radius = inputs['Radius']

    command = 'rm -r '+Result_folder
    os.system(command)
    command = 'mkdir '+Result_folder
    os.system(command)

    G_dim = layers_x[-1]


    ########################################################################################################
    test_set= np.load(f"../data/bubble/validation_5R.npz")
    X_func_test, X_loc_test, y_test, testlist = loaddata(test_set)
    R0_test = y_test[0,:,0]

    # normalization parameters
    Par = {}
    Par['case'] = 'multiR' 
    Par['t_max'] = np.max(X_loc_test)
    Par['R0_max'] = np.max(R0_test)

    # Normalize data
    X_func_test, X_loc_test, y_test, R0_test, Par = nondim2D(X_func_test, R0_test, X_loc_test, y_test, Par)
    ########################################################################################################

    ############################################ Extract a specific range of frequencies ###################
    id_freq = []
    amp_press = []
    for i in range(X_func_test.shape[0]):
        actual = np.array(X_func_test[i,:].ravel())
        freqs, fft_mag_true = fft_(actual, Par)
        domain = 151
        freq = freqs[27:domain][fft_mag_true[27:].argmax()]
        id_freq.append(float(freq)/1000)
    
    args = np.argsort(id_freq)
    id_highfreq = args[-58:]
    batch = np.sort(id_highfreq)
    y_test = y_test[batch]
    X_func_test = X_func_test[batch]
    ###########################################################################################################
    for idx,i in enumerate(R0_test):
        x = jnp.ones(X_loc_test.shape)*i
        if idx == 0:
            tR0 = jnp.hstack([X_loc_test.reshape([-1,1]), x.reshape([-1,1])])
        else:
            tR0 = jnp.vstack([tR0,jnp.hstack([X_loc_test.reshape([-1,1]), x.reshape([-1,1])])])

    path = os.getcwd()
    path = os.path.join(path, folderb)
    path = os.path.join(path, 'best_model_details')
    with open(path, 'rb') as file:
        data = pickle.load(file)

    initializer = jax.nn.initializers.glorot_normal()

    def matrix_init(N,K,key):
        in_dim = N
        out_dim = K
        std = np.sqrt(2.0/(in_dim+out_dim))
        W = initializer(key, (in_dim, out_dim), jnp.float32)*std
        return W

    keym = random.PRNGKey(4234)
    keysm = random.split(keym, num=1)

    ############################### Load trunk ###################################################
    G_dimt = layers_x[-1]
    Am = []
    for i in range(1):
        A = matrix_init(G_dimt, 200, keysm[i])
        Am.append(A)
    path = os.getcwd()
    path = os.path.join(path, foldert)
    print(f'Trunk path: {path}')
    print(f'Shape of Am: {Am[0].shape}')
    model_T = load_model(path, layers_x, {'k':2, 'G':40}, Am[0]) 

    ############################### Load branch ###################################################
    path = os.getcwd()
    path = os.path.join(path, folderb)
    print(f'Branch path: {path}')
    model_B = load_model_b(path, layers_f, {'D':6}) 

    # Open the file in binary read mode for loading the R-matrix
    with open(os.path.join(folderb,'Rmatrix'), 'rb') as file:
        R = pickle.load(file)[0]

    tr = model_T(tR0, pred_only=True)
    output = jnp.einsum('ik,kj->ij', tr, jnp.linalg.inv(R))
    br = model_B(X_func_test)
    output = jnp.einsum('im,jm->ij', br, output)
    output = output.reshape([58,5,2000])

    # Dimensionalize the non-dimensional variables
    y_pred, y_test, X_loc, R0 = dimensional2D(X_loc_test, X_func_test, output, y_test, R0_test, Par)
    y_pred = y_pred*1e6
    y_test = y_test*1e6
    X_loc = X_loc*1e6
    print(f'Shape of actual: {y_test.shape}')
        
    
    # Pressure in kPa
    t_set= np.load(f"../data/bubble/validation_5R.npz")
    press, _, _, _ = loaddata(test_set)
    # Radius in microns
    rad = np.array([50, 60, 70, 80, 90])
    j = np.argwhere(rad == radius)[0,0]
    
    # Set the result directory
    path = os.getcwd()
    res_dir = os.path.join(path, Result_folder)
    print(res_dir)

    for i in range(58):
        acc = jnp.linalg.norm(y_pred[i,j,:].ravel() - y_test[i,j,:].ravel())/jnp.linalg.norm(y_test[i,j,:].ravel())
        print(f'Pressure: {round(press[i].max()/1e5)*1e5} Pa')
        print(f'Frequency: {id_freq[i]} kHz')
        print(f'Radius: {rad[j]} mu')
        print('relative L2 error: {}'.format(acc))
        print()
        
        # Create a figure with custom size
        fig = plt.figure(figsize=(8, 12))

        # Create a GridSpec with 3 rows and 1 column, and custom height ratios
        gs = GridSpec(3, 1, height_ratios=[1, 0.5, 1], figure=fig)

        # First subplot: sin(x)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(X_loc_test.ravel(), y_test[i,j,:].ravel(), color='blue', label='Ground truth')
        ax1.plot(X_loc_test.ravel(), y_pred[i,j,:].ravel(), color='red', label='Prediction', linestyle='--')
        ax1.set_ylabel(r'R ($\mu$m)')
        ax1.set_xticks([])
        ax1.set_title(f'Freq: {round(id_freq[i])} kHz, P: {round(press[i].max()/1e5)} x $10^5$ Pa')
        ax1.xaxis.set_ticks_position('bottom')


        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(X_loc.ravel(), np.abs(y_pred[i,j,:].ravel() - y_test[i,j,:].ravel()), color='k')
        ax2.set_ylabel(r'Abs. Err. ($\mu$m)')
        ax2.set_xlabel(r't ($\mu$s)')

        freqs, fft_mag_pred, fft_mag_true = fft(y_pred[i,j,:].ravel(), y_test[i,j,:].ravel(), Par)

        domain = 151
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(freqs[1:domain],np.log(fft_mag_true[1:domain]),'b-', label='Ground truth')
        ax3.plot(freqs[1:domain],np.log(fft_mag_pred[1:domain]),'r--', linestyle='--', label='Prediction')
        ax3.set_ylabel(r'Magnitude')
        ax3.set_xlabel(r'Frequency (kHz)')
        ax3.legend(loc='best')

        # Adjust layout and save the figure
        fig.tight_layout()

        plt.subplots_adjust(hspace=0.3)
        plt.savefig(os.path.join(res_dir, f'radius_{rad[j]}um_pressure_{round(press[i].max()/1e5)*1e5}Pa_freq_{round(id_freq[i])}kHz_error__{i}_{j}.png'), dpi=500)