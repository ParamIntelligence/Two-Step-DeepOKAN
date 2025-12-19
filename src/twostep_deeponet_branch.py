import jax
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
import optax
import pickle
import jaxopt
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.example_libraries import optimizers
import jaxkan
from flax import nnx

from jaxKAN.jaxkan.layers.SplineLayer import SplineLayer
from jaxKAN.jaxkan.layers.RBFLayer import RBFLayer

from typing import List
import optax
import orbax.checkpoint as ocp
import torch

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

def fft_(y_true, Par):
    """
    Extract the frequencies present in bubble dynamics through FFT
    
    Parameters:
        y_true (array): bubble dynamics represented in time domain
        Par (dict): parameters relavant for FFT
    Returns:
        freqs (array [torch]): frequencies
        fft_mag_true (array [torch]): magnitude
    """
    y_true = torch.from_numpy(y_true)
    signal_length = y_true.shape[-1]
    duration = Par['t_max'] #0.0001
    sampling_rate = duration / signal_length  
    freqs = torch.fft.rfftfreq(signal_length, d=float(sampling_rate))
    fft_true = torch.fft.rfft(y_true,dim=-1)  
    fft_mag_true = torch.abs(fft_true)
    return freqs, fft_mag_true

def loaddata(data):
    """
    Read and extract relavant variables present in the dataset
    
    Parameters:
        data (.npz format) : bubble dynamics represented in time domain
    Returns:
        X_fun (array): Pressure
        X_loc (array): Time
        y (array): Radius
        name_list (list): variable names
    """
    X_fun = jnp.array(data["P"])
    X_loc = jnp.array(data["t"]).squeeze()
    y = jnp.array(data["R"])
    namelist = data['names']
    return X_fun,X_loc,y,namelist


def nondim2D(X_func, R0, X_loc, y, Par):
    """
    Non-dimensionalize the variables relavant for bubble dynamics
    
    Parameters:
        X_fun (array): Pressure
        R0 (array): Initial Radius
        X_loc (array): Time
        y (array): Radius
        Par (dict): Parameters
    Returns:
        X_fun_bar (array): Non-dimensional Pressure
        X_loc_bar (array): Non-dimensional Time
        y_bar (array): Non-dimensional Radius
        R0_bar (array): Non-dimensional Initial Radius
        Par (dict): Parameters
    """
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
    y_bar = y/R0max

    return (
        X_func_bar,
        X_loc_bar,
        y_bar,
        R0_bar,
        Par
    )

def train_branchnet(inputs):
    """
    Train the trunk network and save the model at intermediate states during training
    
    Parameters:
        inputs : User inputs for training branch network
    Returns:
        None
    """
    num_epochs = inputs['Epochs']['Branch'] + 1
    lr = inputs['learning_rate']
    foldert = inputs['Model_folder']['Trunk']
    folder = inputs['Model_folder']['Branch']
    FSM = inputs['FSM']['Branch']
    FRL = inputs['FRL']['Branch']
    layers_f = inputs['Architecture']['Branch']
    layers_t = inputs['Architecture']['Trunk']
    decomposition = inputs['Decomposition']
    params_f = inputs['Basis']['Branch']
    params_t = inputs['Basis']['Trunk']

    command = 'rm -r '+folder
    os.system(command)
    command = 'mkdir '+folder
    os.system(command)

    ########################################################################################################
    train_set= np.load(f"../data/bubble/train_5R.npz")
    X_func_train, X_loc, y_train, trainlist = loaddata(train_set)
    R0 = y_train[0,:,0]

    # normalization parameters
    Par = {}
    Par['case'] = 'multiR' 
    Par['t_max'] = np.max(X_loc)
    Par['R0_max'] = np.max(R0)
    
    # Normalize data
    X_func_train, X_loc, y_train, R0, Par = nondim2D(X_func_train, R0, X_loc, y_train, Par)
    ########################################################################################################

    ############################################ Extract a specific range of frequencies ###################
    id_freq = []
    for i in range(200):
        actual = np.array(X_func_train[i,:].ravel())
        freqs, fft_mag_true = fft_(actual, Par)
        domain = 151
        freq = freqs[27:domain][fft_mag_true[27:].argmax()]
        id_freq.append(float(freq)/1000)
    

    args = np.argsort(id_freq)
    print(np.array(id_freq)[args])
    print(np.array(id_freq)[args][-100:])
    print(np.array(id_freq)[args][-75:])
    print(np.array(id_freq)[args][-50:])

    print(args)
    id_highfreq = args[-200:-150]
    id_highfreq2 = args[-150:-100]
    id_highfreq3 = args[-100:]
    #id_highfreq4 = args[-50:]

    batch = np.sort(id_highfreq)
    batch2 = np.sort(id_highfreq2)
    batch3 = np.sort(id_highfreq3)
    #batch4 = np.sort(id_highfreq4)

    y_train_high = y_train[batch]
    X_func_train_high = X_func_train[batch]

    y_train_high2 = y_train[batch2]
    X_func_train_high2 = X_func_train[batch2]

    y_train_high3 = y_train[batch3]
    X_func_train_high3 = X_func_train[batch3]

    #y_train_high4 = y_train[batch4]
    #X_func_train_high4 = X_func_train[batch4]

    print(batch)
    print(batch2)
    print(batch3)
    #print(batch4)
    
    #raise AssertionError
    ###########################################################################################################

    print(X_func_train.shape)
    print(y_train.shape)

    print(X_func_train_high.shape)
    print(y_train_high.shape)

    print(X_func_train_high2.shape)
    print(y_train_high2.shape)

    print(X_func_train_high3.shape)
    print(y_train_high3.shape)

    #raise AssertionError

    for idx,i in enumerate(R0):
        x = jnp.ones(X_loc.shape)*i
        if idx == 0:
            tR0 = jnp.hstack([X_loc.reshape([-1,1]), x.reshape([-1,1])])
        else:
            tR0 = jnp.vstack([tR0,jnp.hstack([X_loc.reshape([-1,1]), x.reshape([-1,1])])])

    initializer = jax.nn.initializers.glorot_normal()

    G_dim = int(np.ceil(layers_f[-1]/3))
    print("branch layers:\t",layers_f)

    key = random.PRNGKey(1234)
    keys = random.split(key, num=1)
    print("keysss=\t",keys)

    #output dimension for Branch and Trunk Net
    G_dimt = layers_t[-1]

    def matrix_init(N,K,key):
        in_dim = N
        out_dim = K
        std = np.sqrt(2.0/(in_dim+out_dim))
        W = initializer(key, (in_dim, out_dim), jnp.float32)*std
        return W

    keym = random.PRNGKey(4234)
    keysm = random.split(keym, num=1)

    W_trunk, b_trunk = [], []
    Am = []
    for i in range(1):
        A = matrix_init(200, G_dimt, keysm[i])
        Am.append(A)

    def load_model(path, layer_dims, req_params, A):
        # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
        abstract_model = Trunk(layer_dims=layer_dims, required_parameters=req_params, Am=A)

        graphdef, abstract_state = nnx.split(abstract_model)

        checkpointer = ocp.PyTreeCheckpointer()
        state_restored = checkpointer.restore(os.path.join(path, 'best_model'), item=abstract_state)

        # The model is now good to use!
        model = nnx.merge(graphdef, state_restored)
        return model

    def load_model_b(path, layer_dims, req_params):
        # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
        abstract_model = Branch(layer_dims=layer_dims, required_parameters=req_params)

        graphdef, abstract_state = nnx.split(abstract_model)

        checkpointer = ocp.PyTreeCheckpointer()
        state_restored = checkpointer.restore(os.path.join(path, f'best_model'), item=abstract_state)

        # The model is now good to use!
        model = nnx.merge(graphdef, state_restored)
        return model


    def save_model(model,epoch):
        if epoch == None:
            path = os.getcwd()
            path = os.path.join(path, folder)
            _, state = nnx.split(model)
            checkpointer = ocp.StandardCheckpointer()

            command = 'rm -r '+os.path.join(path, f'best_model')
            os.system(command)

            checkpointer.save(os.path.join(path, f'best_model'), state)
        else:
            path = os.getcwd()
            path = os.path.join(path, folder)
            _, state = nnx.split(model)
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(os.path.join(path, f'state_{epoch}'), state)

    path = os.getcwd()
    path = os.path.join(path, foldert)
    model = load_model(path, layers_t, params_t, Am[0])

    batches = [batch, batch2, batch3]
    X_func_trains = [X_func_train_high, X_func_train_high2, X_func_train_high3]
    u_train1 = []
    Rm = []
    for i, batch_ in enumerate(batches):
        phi , Am = model(tR0, both=True)
        if decomposition == "SVD":
            Q,Sd,Vd = jnp.linalg.svd(phi, full_matrices=False)
            R = jnp.matmul(jnp.diag(Sd),Vd)
        else:
            Q,R = jnp.linalg.qr(phi, mode='reduced')
        
        Rm.append(R)
        print()
        temp = jnp.einsum('im,jm->ji', R, Am[batch_])
        print("phi=\t",phi.shape,i)
        print("Am=\t",Am[batch_].shape,i)
        print("Q=\t",Q.shape,i)
        print("R=\t",R.shape,i)
        print("u_train1=\t",temp.shape,i)
        
        u_train1.append(temp)

    model_B = Branch(layers_f, params_f)

    def loss(model, data1, data2, u1, u2):
        u_preds1 = model(data1)
        u_preds2 = model(data2)

        loss_data = 1.*jnp.mean((u_preds1.flatten() - u1.flatten())**2) + 1.*jnp.mean((u_preds2.flatten() - u2.flatten())**2)
    
        mse = loss_data
        return mse
    

    # optimizer
    optim = optax.adamw(lr, weight_decay=0.1)
    optimizer = nnx.Optimizer(model_B, optim)
    
    train_loss, test_loss = [], []
    epo = []
    trn_err1 =[]
    trn_err2 =[]
    trn_err3 =[]
    trn_err4 =[]

    path = os.getcwd()
    path = os.path.join(path, folder)
    filename = os.path.join(path, 'Rmatrix')
    with open(filename, 'wb') as file:
        pickle.dump(Rm, file)
    time.sleep(2)
    
    best_loss = 10000000000.0
    n = 0
    i = 0
    ls = np.array([0,1])

    start_time = time.time()
    start_time1= time.time()
    for epoch in range(num_epochs):
        value, grads = nnx.value_and_grad(loss)(model_B, X_func_trains[ls[0]], X_func_trains[ls[1]],\
                                                u_train1[ls[0]], u_train1[ls[1]])
        optimizer.update(grads)

        if epoch % FRL ==0:
            epoch_time = time.time() - start_time
            u_train_pred1 = model_B(X_func_trains[ls[0]])
            u_train_pred2 = model_B(X_func_trains[ls[1]])

            # training error
            err_train1 = jnp.mean(jnp.linalg.norm(u_train1[ls[0]].ravel() - u_train_pred1.ravel(), 2, axis=0)/\
                np.linalg.norm(u_train1[ls[0]].ravel(), 2, axis=0))
            
            err_train2 = jnp.mean(jnp.linalg.norm(u_train1[ls[1]].ravel() - u_train_pred2.ravel(), 2, axis=0)/\
                np.linalg.norm(u_train1[ls[1]].ravel(), 2, axis=0))

            l1 = loss(model_B, X_func_trains[ls[0]], X_func_trains[ls[1]],\
                                u_train1[ls[0]], u_train1[ls[1]])
            train_loss.append(l1)

            print("Epoch: {} | T: {:0.6f} | Trunk {} | Train MSE: {:0.3e} | Batch-{} L2: {:0.6f} | Batch-{} L2: {:0.6f}".format(epoch, epoch_time, 0,\
                                                                l1, ls[0], err_train1, ls[1], err_train2))

            if ls.all() == np.array([0,1]).all():
                trn_err1.append(err_train1)
                trn_err2.append(err_train2)
                ls = np.array([1,2])
            elif ls.all() == np.array([1,2]).all():
                trn_err2.append(err_train1)
                trn_err3.append(err_train2)
                ls = np.array([2,0])
            elif ls.all() == np.array([2,0]).all():
                trn_err3.append(err_train1)
                trn_err1.append(err_train2)
                ls = np.array([0,1])
                
            if l1 < best_loss:
                save_model(model_B,None)

                path = os.getcwd()
                path = os.path.join(path, folder)
                filename = os.path.join(path, 'best_model_details')
                
                command = 'rm -r '+filename
                os.system(command)

                with open(filename, 'wb') as file:
                    pickle.dump([i, epoch, l1, ls[0], ls[1], err_train1, err_train2], file)
                time.sleep(1)
                best_loss = l1
            epo.append(epoch)


        if epoch % FSM ==0:
            save_model(model_B,epoch)

            path = os.getcwd()
            path = os.path.join(path, folder)
            filename = os.path.join(path, 'loss')
            with open(filename, 'wb') as file:
                pickle.dump((epo,train_loss), file)

            filename = os.path.join(path, 'trn_error1')
            with open(filename, 'wb') as file:
                pickle.dump(trn_err1, file)

            filename = os.path.join(path, 'trn_error2')
            with open(filename, 'wb') as file:
                pickle.dump(trn_err2, file)

            filename = os.path.join(path, 'trn_error3')
            with open(filename, 'wb') as file:
                pickle.dump(trn_err3, file)

        start_time = time.time()

    total_time = time.time() - start_time1
    print("training time for the branch net=\t",total_time)
    
    # save trained model
    for i in range(1):
        save_model(model_B,num_epochs)
    time.sleep(120)

    path = os.getcwd()
    path = os.path.join(path, folder)
    filename = os.path.join(path, 'Rmatrix')
    with open(filename, 'wb') as file:
        pickle.dump(Rm, file)
    time.sleep(120)

    path = os.getcwd()
    path = os.path.join(path, folder)
    filename = os.path.join(path, 'trn_error1')
    with open(filename, 'wb') as file:
        pickle.dump(trn_err1, file)
    time.sleep(120)

    path = os.getcwd()
    path = os.path.join(path, folder)
    filename = os.path.join(path, 'trn_error2')
    with open(filename, 'wb') as file:
        pickle.dump(trn_err2, file)
    time.sleep(120)

    path = os.getcwd()
    path = os.path.join(path, folder)
    filename = os.path.join(path, 'trn_error3')
    with open(filename, 'wb') as file:
        pickle.dump(trn_err3, file)
    time.sleep(120)