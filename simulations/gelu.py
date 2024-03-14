import numpy as np
import itertools
from tqdm import tqdm
import torch
from collections import defaultdict
import random
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.use_deterministic_algorithms(True)

MEAN_INP = [0]
SIGMA_X_INPUT = np.logspace(-0.5, 0.5, num=3, endpoint=True)
CORR_X_INPUT = np.linspace(0.0, 1, num=5, endpoint=False)
SIGMA_G_INPUT = np.logspace(-0.5, 0.5, num=3, endpoint=True)
CORR_G_INPUT = np.linspace(0.0, 1, num=5, endpoint=False)
DIM1 = [1]
SEQ_LEN = np.logspace(2, 3, num=2, endpoint=True, dtype=np.int32)
ERROR_MAP = ['x mean', 'x var', 'g var', 'x cov', 'g cov']
# ERROR_MAP = ['x mean']

RUNS = [100000]

func_vals = []
# assert (torch.cuda.device_count()<1)
torch.set_default_tensor_type("torch.cuda.FloatTensor")

varying_vals = [MEAN_INP, SIGMA_X_INPUT, CORR_X_INPUT, SIGMA_G_INPUT, CORR_G_INPUT, DIM1, SEQ_LEN, [1], RUNS]
all_args = list(itertools.product(*varying_vals))
random.shuffle(all_args)


def print_errs(errs, label):
    errs = np.sort(errs) * 100.0
    total = len(errs)
    percentiles = [50, 90, 99]
    percentile_errors = [errs[int(total * percentile / 100)] for percentile in percentiles]
    percentile_errors = np.array(percentile_errors)
    np.set_printoptions(precision=1)
    print(f'Error {label}: ', np.array2string(percentile_errors, separator=', '))
    return


def simulate_all():
    all_errs = defaultdict(list)

    for args in tqdm(all_args, smoothing=0):
        result = simulate_one(args)
        for i in range(len(result)):
            all_errs[i].append(result[i])

    for error_idx in range(len(ERROR_MAP)):
        all_errs[error_idx] = np.array(all_errs[error_idx])
        print_errs(all_errs[error_idx], label=ERROR_MAP[error_idx])

    return all_errs


def calc_err(actual, predicted, pred_var=0):
    if pred_var == 0:
        return np.abs((actual - predicted) / actual)
    if predicted == 0:
        return min(abs(actual), abs(actual / pred_var))
    return min(np.abs((actual - predicted) / actual), np.abs((actual - predicted) / pred_var))


def simulate_one(args):
    mean_inp, sigma_x_inp, corr_x_inp, sigma_g_inp, corr_g_inp, dim1, seq_len, err_idx, runs = args
    # print(args)
    x_out_sum = 0
    x_out_sq_sum = 0

    g_out_sum = 0
    g_out_sq_sum = 0

    x_count = 0
    g_count = 0

    cov_sum_x = 0
    cov_sum_g = 0

    loc_x = torch.zeros(seq_len) + mean_inp
    covariance_matrix_x = torch.ones((seq_len, seq_len)) * corr_x_inp * sigma_x_inp * sigma_x_inp
    covariance_matrix_x.fill_diagonal_(sigma_x_inp * sigma_x_inp)

    dist_x = torch.distributions.multivariate_normal.MultivariateNormal(loc_x, covariance_matrix_x)

    # for i in range(runs):
    # input tensor
    input_tensor = dist_x.sample((runs, dim1,))
    input_tensor = input_tensor.transpose(1, 2)
    input_tensor.requires_grad = True

    # The layer to test

    layer_output = input_tensor/2 * (1 + torch.special.erf(input_tensor/np.sqrt(2)))
    # layer_output = torch.nn.functional.gelu(input_tensor)

    # The layer to test
    # layer = torch.nn.Linear(dim1, dim2, bias=False)
    # torch.nn.init.normal_(layer.weight, std=sigma_weight)
    # layer_output = layer(input_tensor)

    layer_output.retain_grad()

    # simulating a random gradient coming from above

    loc_g = torch.zeros(seq_len)
    covariance_matrix_g = torch.ones((seq_len, seq_len)) * corr_g_inp * sigma_g_inp * sigma_g_inp
    covariance_matrix_g.fill_diagonal_(sigma_g_inp * sigma_g_inp)
    dist_g = torch.distributions.multivariate_normal.MultivariateNormal(loc_g, covariance_matrix_g)

    grads_in = dist_g.sample((runs, dim1,))
    grads_in = grads_in.transpose(1, 2)

    # grads_in = torch.Tensor(runs, seq_len, dim2).normal_(mean=0, std=sigma_g_inp)

    # This is to make the loss what we need
    loss = torch.sum(layer_output * grads_in)
    loss.backward()

    # calculate sum of variable and squared variable
    # x_out_sum += layer_output[0][0].item()
    # x_out_sq_sum += (layer_output[0][0] * layer_output[0][0]).item()
    # x_count += 1

    # g_out_sum += torch.sum(input_tensor.grad).item()
    # g_out_sq_sum += torch.sum(input_tensor.grad * input_tensor.grad).item()
    # g_count += input_tensor.grad.size()[-1]

    # cov_mat = torch.cov(layer_output)
    cov_mat_x = torch.cov(layer_output.transpose(0, 1).reshape(seq_len, -1))
    cov_mat_x.fill_diagonal_(0)
    cov_sum_x += torch.mean(cov_mat_x).item() * seq_len / (seq_len - 1)

    x_out_sum += torch.sum(layer_output).item()
    x_out_sq_sum += torch.sum(layer_output * layer_output).item()
    x_count += torch.numel(layer_output)

    g_out_sum += torch.sum(input_tensor.grad).item()
    g_out_sq_sum += torch.sum(input_tensor.grad * input_tensor.grad).item()
    g_count += torch.numel(input_tensor.grad)

    cov_mat_g = torch.cov(input_tensor.grad.transpose(0, 1).reshape(seq_len, -1))
    cov_mat_g.fill_diagonal_(0)
    cov_sum_g += torch.mean(cov_mat_g).item() * seq_len / (seq_len - 1)

    # del input_tensor, layer_output, grads_in, loss

    # calculate mean and var observed
    x_out_mean = x_out_sum / x_count
    # x_out_var = x_out_var_sum / x_count
    x_out_var = (x_out_sq_sum / x_count) - (x_out_mean * x_out_mean)
    # x_out_var = (x_out_sq_sum / x_count) - (x_out_mean*x_out_mean)

    g_out_mean = g_out_sum / g_count
    g_out_var = (g_out_sq_sum / g_count) - (g_out_mean * g_out_mean)

    # cov_out = cov_sum / runs
    x_out_cov = cov_sum_x
    g_out_cov = cov_sum_g

    # calculate mean and var predicted
    sq = sigma_x_inp * sigma_x_inp
    a = sq / (1 + sq)
    pred_x_out_mean = (sigma_x_inp / np.sqrt(2 * np.pi)) * np.sqrt(a)
    pred_x_out_var = (sq / (2 * np.pi)) * (np.pi - a - np.arccos(a) + 2 * a * np.sqrt((1 - a) / (1 + a)))
    # pred_x_out_var = sq / 4 + sq / (2 * np.pi) * (2 * sq / ((1 + sq) * np.sqrt(1 + 2 * sq)) + np.arcsin(sq / (1 + sq)))
    # pred_x_out_cov = sq * (np.pi * corr_x_inp + np.sqrt(1 - corr_x_inp**2) - corr_x_inp*np.arccos(corr_x_inp) - 1) / (2 * np.pi)
    pred_x_in_cov = sq *  corr_x_inp
    # pred_x_out_cov = pred_x_in_cov * (1 / 4 + np.arcsin(a) / (2 * np.pi)) + sq / (2 * np.pi) * (
    #     sq * (1 - corr_x_inp * corr_x_inp) + (1 + corr_x_inp * corr_x_inp)) / ((sq + 1) * np.sqrt((sq + 1) ** 2 - (corr_x_inp * sq)**2)) - pred_x_out_mean * pred_x_out_mean
    pred_x_out_cov = sq / 4 * (corr_x_inp + corr_x_inp * np.arcsin(corr_x_inp * a) * (2 / np.pi) + sq * (2 / np.pi) * (
        sq * (1 - corr_x_inp * corr_x_inp) + (1 + corr_x_inp * corr_x_inp)) / ((sq + 1) * np.sqrt((sq + 1) ** 2 - (corr_x_inp * sq)**2))) - pred_x_out_mean * pred_x_out_mean
    # pred_g_out_mean = 0
    pred_g_out_var = sigma_g_inp * sigma_g_inp * (0.5 + (1 / (2 * np.pi)) * (-np.arccos(a) + a * (5 * sq + 3) / (1 + 2 * sq)**1.5))
    pred_g_out_cov = (0.5 + 1 / (2 * np.pi) * (-np.arccos(corr_x_inp * a) + corr_x_inp * a * ((sq + 1)*(2 * sq + 3) - 2*(corr_x_inp * sq)**2) / (((sq + 1) ** 2 - (corr_x_inp * sq)**2) **1.5 ))) * sigma_g_inp * sigma_g_inp * corr_g_inp


    # calcualte differennce in errors
    errs = np.array([
        calc_err(x_out_mean, pred_x_out_mean, pred_x_out_var),
        calc_err(x_out_var, pred_x_out_var),
        calc_err(g_out_var, pred_g_out_var),
        calc_err(x_out_cov, pred_x_out_cov, pred_x_out_var),
        calc_err(g_out_cov, pred_g_out_cov, pred_g_out_var),
                     ])
    # errs = np.array([errs[4]])

    # if errs[err_idx] > 0.03:
    #     # hack to get away with fewer runs if all is good :P
    #     if runs <= RUNS[0] * 100:
    #         args_new = tuple(list(args)[:-1] + [runs * 10])
    #         return simulate_one(args_new)
    #     else:
    #         if errs[err_idx] > 0.1:
    #             print('aaaaa', errs[err_idx], args)
    # print('done')
    return errs


simulate_all()
