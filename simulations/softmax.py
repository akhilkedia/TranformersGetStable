import os
import torch
from tqdm import tqdm
import itertools
import numpy as np
from collections import defaultdict
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.use_deterministic_algorithms(True)

RUNS = [1]

assert (torch.cuda.device_count()==1)
torch.set_default_tensor_type("torch.cuda.FloatTensor")


BATCH_SIZE = [10000]
SIGMA_X_INPUT = np.logspace(-2, 0, num=5, endpoint=True)
CORR_X_INPUT = np.linspace(0.0, 1, num=5, endpoint=False)
SIGMA_G_INPUT = np.logspace(-0.5, 0.5, num=5, endpoint=True)
CORR_G_INPUT = [0.0]
DIM1 = np.logspace(np.log10(300), 4, num=5, endpoint=True, dtype=np.int32)
ERROR_MAP = ['x mean', 'x var full', 'x var approx', 'g var full', 'g var approx']

all_args = [SIGMA_X_INPUT, CORR_X_INPUT, SIGMA_G_INPUT, CORR_G_INPUT, DIM1, BATCH_SIZE, [1], RUNS]

all_args = list(itertools.product(*all_args))
random.shuffle(all_args)

def print_errs(errs, label):
    errs = np.sort(errs)*100.0
    total = len(errs)
    percentiles = [50, 90, 99]
    percentile_errors = [errs[int(total*percentile/100)] for percentile in percentiles]
    percentile_errors = np.array(percentile_errors)
    np.set_printoptions(precision=1)
    print(f'Error {label}: ', np.array2string(percentile_errors, separator=', '))
    return

def simulate_all():
    all_errs = defaultdict(list)
    for args in tqdm(all_args):
        result = simulate_one(args)
        for i in range(len(result)):
            all_errs[i].append(result[i])

    for error_idx in range(len(ERROR_MAP)):
        all_errs[error_idx] = np.array(all_errs[error_idx])
        print_errs(all_errs[error_idx], label=ERROR_MAP[error_idx])

    return all_errs


def calc_err(actual, predicted, pred_var = 0):
    if predicted == 0:
        return min(actual, actual/pred_var)
    return np.abs((actual - predicted) / actual)


def simulate_one(args):
    # SIGMA_X_INPUT, CORR_X, SIGMA_G_INPUT, DIM1, BATCH_SIZE
    sigma_x_inp, corr_x_inp, sigma_g_inp, corr_g_inp, dim1, batch_size, err_idx, runs = args
    dim2 = dim1
    # print(args)
    x_out_sum = 0
    x_out_sq_sum = 0
    cov_sum = 0

    g_out_sum = 0
    g_out_sq_sum = 0

    x_count = 0
    g_count = 0
    cov_count = 0

    loc = torch.zeros(dim1)
    covariance_matrix = torch.ones((dim1, dim1)) * corr_x_inp * sigma_x_inp * sigma_x_inp
    covariance_matrix.fill_diagonal_(sigma_x_inp * sigma_x_inp)

    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix)
    batch_size = batch_size * runs

    # input tensor
    input_tensor = dist.sample((batch_size,))
    input_tensor.requires_grad = True

    # The layer to test
    layer = torch.nn.Softmax(dim=-1)

    layer_output = layer(input_tensor)

    layer_output.retain_grad()

    # # simulating a random gradient coming from above
    grads_in = torch.Tensor(batch_size, dim2).normal_(mean=0, std=sigma_g_inp)

    # # This is to make the loss what we need
    loss = torch.sum(layer_output * grads_in)
    loss.backward()

    x_out_sum += torch.sum(layer_output).item()
    x_out_sq_sum += torch.sum(layer_output * layer_output).item()
    x_count += torch.numel(layer_output)

    g_out_sum += torch.sum(input_tensor.grad).item()
    g_out_sq_sum += torch.sum(input_tensor.grad * input_tensor.grad).item()
    g_count += torch.numel(input_tensor.grad)
    del input_tensor, layer, layer_output, grads_in, loss

    # calculate mean and var observed
    x_out_mean = x_out_sum / x_count
    x_out_var = (x_out_sq_sum / x_count) - (x_out_mean * x_out_mean)

    g_out_mean = g_out_sum / g_count
    g_out_var = (g_out_sq_sum / g_count) - (g_out_mean * g_out_mean)

    # calculate mean and var predicted
    pred_x_out_mean = 1/dim1

    # V1
    pred_x_out_var1 = (np.exp((1 - corr_x_inp) * sigma_x_inp * sigma_x_inp) - 1) / (dim1**2)

    # V2
    exp_val = np.exp((1 - corr_x_inp) * sigma_x_inp * sigma_x_inp * dim1 / (dim1 - 1))
    pred_x_out_numer = (exp_val - 1) * exp_val * exp_val
    pred_x_out_demon = ((dim1 - 1) * np.exp((1 - corr_x_inp) * sigma_x_inp * sigma_x_inp) + 1)**2
    pred_x_out_var2 = pred_x_out_numer / pred_x_out_demon

    # V1
    pred_g_out_var1 = np.exp(sigma_x_inp * sigma_x_inp * (1 - corr_x_inp)) * sigma_g_inp * sigma_g_inp / (dim1 * dim1)
    # V2
    pred_g_out_var2 = sigma_g_inp * sigma_g_inp * (pred_x_out_numer / pred_x_out_demon + 1/(dim1*dim1))

    # calcualte differennce in errors
    errs = np.array([
        calc_err(x_out_mean, pred_x_out_mean, pred_x_out_var2),
        calc_err(x_out_var, pred_x_out_var2),
        calc_err(x_out_var, pred_x_out_var1),
        calc_err(g_out_var, pred_g_out_var2),
        calc_err(g_out_var, pred_g_out_var1)])

    return errs


simulate_all()



