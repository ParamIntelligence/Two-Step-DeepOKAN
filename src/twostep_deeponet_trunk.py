import jax
import os
import numpy as np
import time
import scipy.io as io
import jax.numpy as jnp
import pickle
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.example_libraries import optimizers
import jaxkan
from flax import nnx

from jaxKAN.jaxkan.layers.SplineLayer import SplineLayer

from typing import List
import optax
import orbax.checkpoint as ocp
from copy import deepcopy
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

    def update_grid(self, x, G):
        for i in range(len(self.layer_dims) - 1):
            key = jax.random.PRNGKey(42)
            x_batch = jax.random.uniform(key, shape=(100, self.layer_dims[i]), minval=-4.0, maxval=4.0)
            self.layers[i].update_grid(x=x_batch, G_new=G)

    def __call__(self, x, batch):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x += self.biases[i].value

        output = jnp.einsum('im,jm->ij', self.Am[batch], x) 
        return output

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
    duration = Par['t_max']
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
    return X_fun, X_loc, y, namelist


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


def train_trunknet(inputs):
    """
    Train the trunk network and save the model at intermediate states during training
    
    Parameters:
        inputs : User inputs for training trunk network
    Returns:
        None
    """

    num_epochs = inputs['Epochs']['Trunk'] + 1
    lr = inputs['learning_rate']
    folder = inputs['Model_folder']['Trunk']
    FSM = inputs['FSM']['Trunk']
    FRL = inputs['FRL']['Trunk']
    layers_t = inputs['Architecture']['Trunk']
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
    for i in range(200):
        actual = np.array(y_train[i,0,:].ravel())
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
    id_highfreq = args[-50:]
    id_highfreq2 = args[-100:-50]
    id_highfreq3 = args[-200:-100]

    batch = np.sort(id_highfreq)
    batch2 = np.sort(id_highfreq2)
    batch3 = np.sort(id_highfreq3)

    y_train_high = y_train[batch]
    X_func_train_high = X_func_train[batch]

    y_train_high2 = y_train[batch2]
    X_func_train_high2 = X_func_train[batch2]

    y_train_high3 = y_train[batch3]
    X_func_train_high3 = X_func_train[batch3]


    print(batch)
    print(batch2)
    print(batch3)
    #raise AssertionError
    ###########################################################################################################

    for idx,i in enumerate(R0):
        x = jnp.ones(X_loc.shape)*i
        if idx == 0:
            tR0 = jnp.hstack([X_loc.reshape([-1,1]), x.reshape([-1,1])])
        else:
            tR0 = jnp.vstack([tR0,jnp.hstack([X_loc.reshape([-1,1]), x.reshape([-1,1])])])
    
    initializer = jax.nn.initializers.glorot_normal()

    def load_model(path, layer_dims, req_params, A):
        # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
        abstract_model = Trunk(layer_dims=layer_dims, required_parameters=req_params, Am=A)

        graphdef, abstract_state = nnx.split(abstract_model)

        checkpointer = ocp.PyTreeCheckpointer()
        state_restored = checkpointer.restore(os.path.join(path, 'best_model'), item=abstract_state)

        # The model is now good to use!
        model = nnx.merge(graphdef, state_restored)
        return model

    def matrix_init(N,K,key):
        in_dim = N
        out_dim = K
        std = np.sqrt(2.0/(in_dim+out_dim))
        W = initializer(key, (in_dim, out_dim), jnp.float32)*std
        return W

    key = random.PRNGKey(1234)
    keys = random.split(key, num=1)
    keym = random.PRNGKey(4234)
    keysm = random.split(keym, num=1)
    print("keysss=\t",keys)

    #output dimension for Branch and Trunk Net
    G_dim = layers_t[-1]

    Am = []
    for i in range(1):
        A = matrix_init(200, G_dim, keysm[i])
        Am.append(A)

    print(f'Shape of Am: {Am[0].shape}')
    print(f'Params: {params_t}')
    
    model = Trunk(layers_t, params_t, Am[0])
    '''
    ############################################## load model (continual learning) ##################################################
    path = os.getcwd()
    path = os.path.join(path, '0_model_T_noactive_3ls_150_150_k2_G40_lr1em4_dumm_slrm_p1_main_highfreq_100batch_cv_')
    print(f'Trunk path: {path}')
    model = load_model(path, layers_t, params_t, Am[0]) 
    print()
    print(f'Shape of loaded Am: {model.Am.shape}')
    print()
    print('Model loaded!')
    #################################################################################################################################
    '''

    def loss(model, bat1, bat2, bat3, data, u1, u2, u3):
        u_preds1 = model(data, bat1)
        u_preds1 = u_preds1.reshape([bat1.shape[0],5,2000])

        u_preds2 = model(data, bat2)
        u_preds2 = u_preds2.reshape([bat2.shape[0],5,2000])

        u_preds3 = model(data, bat3)
        u_preds3 = u_preds3.reshape([bat3.shape[0],5,2000])

        loss_data = jnp.mean((u_preds1.flatten() - u1.flatten())**2) + jnp.mean((u_preds2.flatten() - u2.flatten())**2) +\
                    jnp.mean((u_preds3.flatten() - u3.flatten())**2)
        mse = loss_data
        return mse
    
    # optimizer
    optim = optax.adamw(lr, weight_decay=0.1)
    print(lr)
    optimizer = nnx.Optimizer(model, optim)
    
    train_loss, test_loss = [], []
    epo = []
    iters = []
    trn_err1 = []
    trn_err2 = []
    trn_err3 = []

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
    
    batches = [batch, batch2, batch3]
    print(batches)

    #raise AssertionError
    best_loss = 10000000000.0
    n = 0
    i = 0
    start_time1 = time.time()

    start_time = time.time()
    for epoch in range(num_epochs):
        trn_bat1 = batches[0]
        trn_bat2 = batches[1]
        trn_bat3 = batches[2]

        value, grads = nnx.value_and_grad(loss)(model, trn_bat1, trn_bat2, trn_bat3, tR0,\
                                                y_train[trn_bat1], y_train[trn_bat2], y_train[trn_bat3])
        optimizer.update(grads)
    
        if epoch % FRL ==0:
            epoch_time = time.time() - start_time
            y_out_pred_trn1 = model(tR0, trn_bat1)
            y_out_pred_trn2 = model(tR0, trn_bat2)
            y_out_pred_trn3 = model(tR0, trn_bat3)
            
            y_out_pred_trn1 = y_out_pred_trn1.reshape([trn_bat1.shape[0],5,2000])
            y_out_pred_trn2 = y_out_pred_trn2.reshape([trn_bat2.shape[0],5,2000])
            y_out_pred_trn3 = y_out_pred_trn3.reshape([trn_bat3.shape[0],5,2000])
            
            # training error
            err_train1 = jnp.mean(jnp.linalg.norm(y_train[trn_bat1].ravel() - y_out_pred_trn1.ravel(), 2, axis=0)/\
                np.linalg.norm(y_train[trn_bat1].ravel(), 2, axis=0))

            err_train2 = jnp.mean(jnp.linalg.norm(y_train[trn_bat2].ravel() - y_out_pred_trn2.ravel(), 2, axis=0)/\
                np.linalg.norm(y_train[trn_bat2].ravel(), 2, axis=0))

            err_train3 = jnp.mean(jnp.linalg.norm(y_train[trn_bat3].ravel() - y_out_pred_trn3.ravel(), 2, axis=0)/\
                np.linalg.norm(y_train[trn_bat3].ravel(), 2, axis=0))
            
            l1 = loss(model, trn_bat1, trn_bat2, trn_bat3, tR0, y_train[trn_bat1,:], y_train[trn_bat2,:], y_train[trn_bat3,:])
            
            train_loss.append(l1)
            
            trn_err1.append(err_train1)
            trn_err2.append(err_train2)
            trn_err3.append(err_train3)

            print("Epoch: {} | T: {:0.6f} | Trunk {} | Train MSE: {:0.3e} | High L2: {:0.6f} | Inter L2: {:0.6f} | Low L2: {:0.6f}".format(epoch, epoch_time, i,\
                                                                l1, err_train1, err_train2, err_train3))

            if l1 < best_loss:
                print(best_loss)
                print(l1)
                save_model(model,None)

                path = os.getcwd()
                path = os.path.join(path, folder)
                filename = os.path.join(path, 'best_model_details')
                
                command = 'rm -r '+filename
                os.system(command)

                with open(filename, 'wb') as file:
                    pickle.dump([i, epoch, l1, err_train1, err_train2, err_train3], file)
                time.sleep(1)
                best_loss = l1
            #raise AssertionError
            epo.append(epoch)

        
        if epoch % FSM ==0:
            save_model(model,epoch)
            path = os.getcwd()
            path = os.path.join(path, folder)
            filename = os.path.join(path, 'loss')
            with open(filename, 'wb') as file:
                pickle.dump((epo,train_loss), file)

            filename = os.path.join(path, 'trn_error1')
            with open(filename, 'wb') as file:
                pickle.dump((epo,trn_err1), file)

            filename = os.path.join(path, 'trn_error2')
            with open(filename, 'wb') as file:
                pickle.dump((epo,trn_err2), file)

            filename = os.path.join(path, 'trn_error3')
            with open(filename, 'wb') as file:
                pickle.dump((epo,trn_err3), file)
        
        start_time = time.time()

    total_time = time.time() - start_time1
    print("training time for the trunk net=\t",total_time)
    
    # save trained model
    for i in range(1):
        save_model(model,num_epochs)
    time.sleep(120)

    path = os.getcwd()
    path = os.path.join(path, folder)
    filename = os.path.join(path, 'loss')
    with open(filename, 'wb') as file:
        pickle.dump((epo,train_loss), file)
    time.sleep(120)

    path = os.getcwd()
    path = os.path.join(path, folder)
    filename = os.path.join(path, 'trn_error1')
    with open(filename, 'wb') as file:
        pickle.dump((epo,trn_err1), file)
    time.sleep(120)

    path = os.getcwd()
    path = os.path.join(path, folder)
    filename = os.path.join(path, 'trn_error2')
    with open(filename, 'wb') as file:
        pickle.dump((epo,trn_err2), file)
    time.sleep(120)

    path = os.getcwd()
    path = os.path.join(path, folder)
    filename = os.path.join(path, 'trn_error3')
    with open(filename, 'wb') as file:
        pickle.dump((epo,trn_err3), file)
    time.sleep(120)