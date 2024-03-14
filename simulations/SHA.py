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

SIGMA_X_INPUT = [1]
MEAN_INP = [0]
CORR_X_INPUT = np.linspace(0.0, 1, num=3, endpoint=False)

SIGMA_W_V = np.logspace(-0.5, 0.5, num=3, endpoint=True)
SIGMA_W_Q = np.logspace(-1, np.log10(0.6), num=3, endpoint=True)
DROPOUT = np.linspace(0, 1.0, num=3, endpoint=False)

SIGMA_G_INPUT = np.logspace(-0.5, 0.5, num=3, endpoint=True)
CORR_G_INPUT = np.linspace(0.0, 1, num=3, endpoint=False)

DIM_INPUT = np.logspace(2, 3, num=3, endpoint=True, dtype=np.int32)
DIM_OUTPUT = [32, 64, 128, 256]
SEQ_LEN = np.logspace(np.log10(300), 3, num=3, endpoint=True, dtype=np.int32)

ERROR_MAP = ['x mean', 'x var', 'g var', 'x cov', 'g cov']

RUNS = [64]

func_vals = []
torch.set_default_tensor_type("torch.cuda.FloatTensor")

varying_vals = [MEAN_INP, SIGMA_X_INPUT, CORR_X_INPUT, SIGMA_W_V, SIGMA_W_Q, DROPOUT, SIGMA_G_INPUT, CORR_G_INPUT, DIM_INPUT, DIM_OUTPUT, SEQ_LEN, [1], RUNS]
all_args = list(itertools.product(*varying_vals))
random.shuffle(all_args)

np.set_printoptions(precision=1)


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
    # MEAN_INP, SIGMA_X_INPUT, CORR_X_INPUT, SIGMA_W_V, SIGMA_W_Q, SIGMA_G_INPUT, CORR_G_INPUT, DIM_INPUT, DIM_OUTPUT, SEQ_LEN, [1], RUNS
    mean_inp, sigma_x_inp, corr_x_inp, sigma_w_v, sigma_w_q, dropout, sigma_g_inp, corr_g_inp, dim_input, dim_output, seq_len, err_idx, runs = args

    sigma_w_v = sigma_w_v / np.sqrt(dim_input)
    sigma_w_q = sigma_w_q / np.sqrt(dim_input)
    sigma_w_k = sigma_w_q

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

    batch_size = runs

    # input tensor
    input_tensor = dist_x.sample((batch_size, dim_input,))
    input_tensor = input_tensor.transpose(1, 2)
    input_tensor.requires_grad = True

    # The layer to test

    q_layer = torch.nn.Linear(dim_input, dim_output, bias=False)
    k_layer = torch.nn.Linear(dim_input, dim_output, bias=False)
    v_layer = torch.nn.Linear(dim_input, dim_output, bias=False)
    softmax_layer = torch.nn.Softmax(dim=-1)
    dropout_layer = torch.nn.Dropout(p=dropout)

    # Initialize the layer to test
    torch.nn.init.normal_(q_layer.weight, std=sigma_w_q)
    torch.nn.init.normal_(k_layer.weight, std=sigma_w_k)
    torch.nn.init.normal_(v_layer.weight, std=sigma_w_v)

    Q = q_layer(input_tensor)
    K = k_layer(input_tensor)
    K = torch.transpose(K, -1, -2)
    V = v_layer(input_tensor)
    scores = torch.matmul(Q, K) / torch.sqrt(torch.tensor(dim_output))
    attn_scores = softmax_layer(scores)
    attn_scores = dropout_layer(attn_scores)
    layer_output = torch.matmul(attn_scores, V)

    layer_output.retain_grad()

    # simulating a random gradient coming from above

    loc_g = torch.zeros(seq_len)
    covariance_matrix_g = torch.ones((seq_len, seq_len)) * corr_g_inp * sigma_g_inp * sigma_g_inp
    covariance_matrix_g.fill_diagonal_(sigma_g_inp * sigma_g_inp)
    dist_g = torch.distributions.multivariate_normal.MultivariateNormal(loc_g, covariance_matrix_g)

    grads_in = dist_g.sample((batch_size, dim_output,))
    grads_in = grads_in.transpose(1, 2)

    # grads_in = torch.Tensor(1, seq_len, dim_output).normal_(mean=0, std=sigma_g_inp)

    # This is to make the loss what we need
    loss = torch.sum(layer_output * grads_in)
    loss.backward()

    # cov_mat = torch.cov(layer_output)
    cov_mat_x = torch.cov(layer_output.transpose(0, 1).reshape(seq_len, -1))
    cov_mat_x.fill_diagonal_(0)
    cov_sum_x += torch.mean(cov_mat_x).item() * seq_len / (seq_len - 1)

    x_out_sum += torch.sum(layer_output[:, :, :]).item()
    x_out_sq_sum += torch.sum(layer_output[:, :, :] * layer_output[:, :, :]).item()
    x_count += torch.numel(layer_output[:, :, :])

    g_out_sum += torch.sum(input_tensor.grad).item()
    g_out_sq_sum += torch.sum(input_tensor.grad * input_tensor.grad).item()
    g_count += torch.numel(input_tensor.grad)

    cov_mat_g = torch.cov(input_tensor.grad.transpose(0, 1).reshape(seq_len, -1))
    cov_mat_g.fill_diagonal_(0)
    cov_sum_g += torch.mean(cov_mat_g).item() * seq_len / (seq_len - 1)

    del input_tensor, q_layer, k_layer, v_layer, dropout_layer, softmax_layer, layer_output, grads_in, loss

    # calculate mean and var observed
    x_out_mean = x_out_sum / x_count
    # x_out_var = x_out_var_sum / x_count
    x_out_var = (x_out_sq_sum / x_count) - (x_out_mean * x_out_mean)
    # x_out_var = (x_out_sq_sum / x_count) - (x_out_mean*x_out_mean)

    g_out_mean = g_out_sum / g_count
    g_out_var = (g_out_sq_sum / g_count) - (g_out_mean * g_out_mean)

    # x_out_cov = cov_sum_x / runs
    # g_out_cov = cov_sum_g / runs
    x_out_cov = cov_sum_x
    g_out_cov = cov_sum_g

    # calculate mean and var predicted
    pred_x_out_mean = 0

    sigma_qk = sigma_w_q * sigma_w_k
    d_mult_v = dim_input * sigma_w_v**2

    sigma_w_sq = dim_input* sigma_x_inp**2 * sigma_qk**2
    pred_x_out_var_term1 = (1-corr_x_inp)**2 * (seq_len - 1) * sigma_w_sq * sigma_x_inp**4
    pred_x_out_var_term2 = np.exp((1-corr_x_inp) * dim_input * sigma_w_sq * sigma_x_inp**2 ) / (1-dropout)
    pred_x_out_var_term3 = (1-corr_x_inp)**2 * 4* sigma_w_sq * sigma_x_inp**4 + (1-corr_x_inp) * sigma_x_inp**2
    pred_x_out_var_term4 = corr_x_inp* sigma_x_inp**2
    pred_x_out_var = ((pred_x_out_var_term1 + (pred_x_out_var_term2 * pred_x_out_var_term3))/seq_len + pred_x_out_var_term4)*d_mult_v


    pred_x_out_cov_term1 = 2 * (1-corr_x_inp)**2 * dim_input * sigma_qk**2 * sigma_x_inp**6
    pred_x_out_cov_term2 = 1 * (1-corr_x_inp) * sigma_x_inp**2
    pred_x_out_cov = (corr_x_inp * sigma_x_inp**2 + (pred_x_out_cov_term1 + pred_x_out_cov_term2) / seq_len)*d_mult_v

    pred_g_out_var = sigma_g_inp * sigma_g_inp * dim_output * sigma_w_v**2 * (1 + (seq_len - 1) * corr_g_inp * (1 - dropout)) / (seq_len * (1 - dropout))
    pred_g_out_cov = sigma_g_inp * sigma_g_inp * dim_output * sigma_w_v**2 * (1 + (seq_len - 1) * corr_g_inp) / (seq_len)

    errs = np.array([calc_err(x_out_mean, pred_x_out_mean, pred_x_out_var),
                     calc_err(x_out_var, pred_x_out_var),
                     calc_err(g_out_var, pred_g_out_var),
                     calc_err(x_out_cov, pred_x_out_cov, pred_x_out_var),
                     calc_err(g_out_cov, pred_g_out_cov),
                     ])

    return errs


simulate_all()
