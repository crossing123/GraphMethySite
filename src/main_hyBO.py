import sys
import time
import argparse
import os
import torch

import sys
from HyBO_model import *

sys.path.append('./dep/HyBO')
from GPmodel.kernels.mixeddiffusionkernel import MixedDiffusionKernel
from GPmodel.models.gp_regression import GPRegression
from GPmodel.sampler.sample_mixed_posterior import posterior_sampling

from GPmodel.sampler.tool_partition import group_input
from GPmodel.inference.inference import Inference

from acquisition.acquisition_optimization import next_evaluation
from acquisition.acquisition_functions import expected_improvement,upper_confidence_bounds,probability_of_improvement
from acquisition.acquisition_marginalization import inference_sampling
from experiments.random_seed_config import generate_random_seed_coco
from experiments.test_functions.mixed_integer import MixedIntegerCOCO
from experiments.test_functions.weld_design import Weld_Design
from experiments.test_functions.speed_reducer import SpeedReducer 
from experiments.test_functions.pressure_vessel_design import Pressure_Vessel_Design
# from experiments.test_functions.push_robot_14d import Push_robot_14d 
from experiments.test_functions.nn_ml_datasets import NN_ML_Datasets 
from experiments.test_functions.em_func import EM_func 

# from config import experiment_directory
from utils import model_data_filenames, load_model_data, displaying_and_logging
import time


def HyBO(objective=None, n_eval=200, path=None,ac_kind=0, parallel=False, store_data=True, problem_id=None,gpu_id="0",sitekind="K",iter_num=1,**kwargs):
    """
    :param objective:
    :param n_eval:
    :param path:s
    :param parallel:
    :param kwargs:
    :return:
    """
    acquisition_func_list = [expected_improvement,upper_confidence_bounds,probability_of_improvement]
    acquisition_func = acquisition_func_list[ac_kind]
    print("ac_kind:",ac_kind)

    n_vertices = adj_mat_list = None
    eval_inputs = eval_outputs = log_beta = sorted_partition = lengthscales = None
    time_list = elapse_list = pred_mean_list = pred_std_list = pred_var_list = None

    if objective is not None:
        exp_dir = f"/home/Users/gly/gly_github_code/graphmethysite/exp/HyBO_Result"
        objective_id_list = [objective.__class__.__name__] #['MixedIntegerCOCO']
        if hasattr(objective, 'random_seed_info'): #test走这条
            objective_id_list.append(objective.random_seed_info)
        if hasattr(objective, 'data_type'):
            objective_id_list.append(objective.data_type)
        objective_id_list.append('HyBO')
        if problem_id is None:
            objective_id_list.append(sitekind)
        objective_id_list.append(f"{acquisition_func}")
        objective_name = '_'.join(objective_id_list)
        model_filename, data_cfg_filaname, logfile_dir = model_data_filenames(exp_dir=exp_dir,
                                                                              objective_name=objective_name)


        n_vertices = objective.n_vertices #长度为8，每个离散变量的节点数节点数
        adj_mat_list = objective.adjacency_mat #长度为8，离散的图的邻接矩阵？
        grouped_log_beta = torch.ones(len(objective.fourier_freq)) #长度为8，离散变量的核 的超参beta
        log_order_variances = torch.zeros((objective.num_discrete + objective.num_continuous)) #长度为离散8+连续2
        fourier_freq_list = objective.fourier_freq #傅里叶频率，长度为8
        fourier_basis_list = objective.fourier_basis #傅里叶基础，长度为8
        suggested_init = objective.suggested_init  # suggested_init should be 2d tensor 20x10
        n_init = suggested_init.size(0) #随机初始化的个数，20
        num_discrete = objective.num_discrete #离散8
        num_continuous = objective.num_continuous #连续2
        lengthscales = torch.zeros((num_continuous))
        print(f"------------------------------- initializing kernel -----------------")
        kernel = MixedDiffusionKernel(log_order_variances=log_order_variances, grouped_log_beta=grouped_log_beta, fourier_freq_list=fourier_freq_list, 
                            fourier_basis_list=fourier_basis_list, lengthscales=lengthscales,
                            num_discrete=num_discrete, num_continuous=num_continuous)# 内核
        surrogate_model = GPRegression(kernel=kernel) #高斯过程
        eval_inputs = suggested_init #选定的超参
        # print("eval_inputs:",eval_inputs)
        eval_outputs = torch.zeros(eval_inputs.size(0), 1, device=eval_inputs.device) #超参的输出
        hop_list=[]
        distance_list=[]
        for i in range(eval_inputs.size(0)):
            # print(kwag_)
            eval_outputs[i],hop_return,distance_return = objective.evaluate_hybo(x_unorder=eval_inputs[i],sitetype=sitekind,cuda=gpu_id,iter_num=iter_num,ac_kind=acquisition_func_list[ac_kind])
            hop_list.append(int(eval_inputs[i][0])+1) #优化的超参数 跳数
            distance_list.append(eval_inputs[i][-1])
            # print(f"真正的hop:{hop}  distance:{distance}")
        assert not torch.isnan(eval_outputs).any()
        log_beta = eval_outputs.new_zeros(num_discrete)
        log_order_variance = torch.zeros((num_discrete + num_continuous)) #离散和连续
        sorted_partition = [[m] for m in range(num_discrete)]
        lengthscale = torch.zeros((num_continuous))


        time_list = [time.time()] * n_init
        elapse_list = [0] * n_init
        pred_mean_list = [0] * n_init
        pred_std_list = [0] * n_init
        pred_var_list = [0] * n_init

        surrogate_model.init_param(eval_outputs)
        print('(%s) Burn-in' % time.strftime('%H:%M:%S', time.localtime()))
        sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list, log_order_variance,
                                             log_beta, lengthscale, sorted_partition, n_sample=1, n_burn=1, n_thin=1)
        # print(sample_posterior)
        log_order_variance = sample_posterior[1][0]
        log_beta = sample_posterior[2][0]
        lengthscale = sample_posterior[3][0]
        sorted_partition = sample_posterior[4][0]
        print('')
    else:
        surrogate_model, cfg_data, logfile_dir = load_model_data(path, exp_dir=exp_dir)

    for _ in range(n_eval):
        print(f"--------------{_}--------------")
        start_time = time.time()
        reference = torch.min(eval_outputs, dim=0)[0].item() # output的最小值
        print('(%s) Sampling' % time.strftime('%H:%M:%S', time.localtime()))
        sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list, log_order_variance,
                                              log_beta, lengthscale,  sorted_partition, n_sample=10, n_burn=0, n_thin=1)
        hyper_samples, log_order_variance_samples, log_beta_samples, lengthscale_samples, partition_samples, freq_samples, basis_samples, edge_mat_samples = sample_posterior
        log_order_variance = log_order_variance_samples[-1]
        log_beta = log_beta_samples[-1]
        lengthscale = lengthscale_samples[-1]
        sorted_partition = partition_samples[-1]
        print('\n')
        # print(hyper_samples[0])
        # print(log_order_variance)
        # print(log_beta)
        # print(lengthscale)
        # print(sorted_partition)
        # print('')

        x_opt = eval_inputs[torch.argmin(eval_outputs)]
        inference_samples = inference_sampling(eval_inputs, eval_outputs, n_vertices,
                                               hyper_samples, log_order_variance_samples, log_beta_samples, lengthscale_samples, partition_samples,
                                               freq_samples, basis_samples, num_discrete, num_continuous)
        print(f"objective:{objective}, x_opt:{x_opt}, eval_inputs:{eval_inputs}, inference_samples:{inference_samples}, partition_samples:{partition_samples}, edge_mat_samples:{edge_mat_samples},n_vertices:{n_vertices}, acquisition_func:{acquisition_func}, reference:{reference}, parallel:{parallel}")
        # print(acquisition_func)
        suggestion = next_evaluation(objective, x_opt, eval_inputs, inference_samples, partition_samples, edge_mat_samples,
                                     n_vertices, acquisition_func, reference, parallel)
        next_eval, pred_mean, pred_std, pred_var = suggestion

        processing_time = time.time() - start_time
        print("next_eval", next_eval)

        eval_inputs = torch.cat([eval_inputs, next_eval.view(1, -1)], 0)
        print(eval_inputs)
        tensor_evaluate,hop,distance=objective.evaluate_hybo(eval_inputs[-1],sitetype=sitekind,cuda=gpu_id,iter_num=iter_num,ac_kind=acquisition_func_list[ac_kind])
        eval_outputs = torch.cat([eval_outputs, tensor_evaluate.view(1, 1)])
        assert not torch.isnan(eval_outputs).any()
        time_list.append(time.time())
        elapse_list.append(processing_time)
        pred_mean_list.append(pred_mean.item())
        pred_std_list.append(pred_std.item())
        pred_var_list.append(pred_var.item())

        displaying_and_logging(logfile_dir, eval_inputs, eval_outputs, pred_mean_list, pred_std_list, pred_var_list,
                               time_list, elapse_list, hyper_samples, log_beta_samples, lengthscale_samples, log_order_variance_samples, hop,distance,store_data)
        print('Optimizing %s with regularization %.2E up to %4d visualization random seed : %s'
              % (objective.__class__.__name__, objective.lamda if hasattr(objective, 'lamda') else 0, n_eval,
                 objective.random_seed_info if hasattr(objective, 'random_seed_info') else 'none'))

 
if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='Hybrid Bayesian optimization using additive diffusion kernels')
    parser_.add_argument('--n_eval', dest='n_eval', type=int, default=220)
    parser_.add_argument('--objective', dest='objective')
    parser_.add_argument('--problem_id', type=int, default=None, help="The id of the problem")
    parser_.add_argument('--ac_kind', type=int, default=0, help="The id of the problem")
    parser_.add_argument('--path', type=str, default="", help="The path")
    parser_.add_argument('--store_data', type=bool, default=True, help="")
    parser_.add_argument('--parallel', type=bool, default=False, help="")
    parser_.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs to train.")
    parser_.add_argument('-lr', type=float, default=1e-4, help="Initial learning rate.")
    parser_.add_argument('--iter_num', type=int, default=5, help="Train the i fold in 5fold.")
    parser_.add_argument('--sitekind', type=str, default="K", help="The type of the histone methylation site.")
    parser_.add_argument('--gpu_id', type=str, default='0',help="id(s) for CUDA_VISIBLE_DEVICES")

    args_ = parser_.parse_args()
    kwag_ = vars(args_)
    objective_ = kwag_['objective']
    # print(kwag_)
    for i in range(1):
        if objective_ == 'coco':
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_['objective'] = MixedIntegerCOCO(random_seed_, problem_id=kwag_['problem_id'])
        elif objective_ == 'weld_design':
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_['objective'] = Weld_Design(random_seed_, problem_id=kwag_['problem_id'])
        elif objective_ == 'speed_reducer':
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_['objective'] = SpeedReducer(random_seed_, problem_id=kwag_['problem_id'])
        elif objective_ == 'pressure_vessel':
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_['objective'] = Pressure_Vessel_Design(random_seed_, problem_id=kwag_['problem_id'])
        #elif objective_ == 'push_robot':
        #   random_seed_ = sorted(generate_random_seed_coco())[i]
        #   kwag_['objective'] = Push_robot_14d(random_seed_, problem_id=kwag_['problem_id'])
        elif objective_ == 'em_func':
           random_seed_ = sorted(generate_random_seed_coco())[i]
           kwag_['objective'] = EM_func(random_seed_, problem_id=kwag_['problem_id'])
        elif objective_ == 'nn_ml_datasets':
            random_seed_ = sorted(generate_random_seed_coco())[i]
            kwag_['objective'] = NN_ML_Datasets(random_seed_, problem_id=kwag_['problem_id'])
        elif objective_ == 'HyBO':
            print(f"--------------------HyBO----------------------")
            kwag_['objective'] = HyBO_model(n_points=3)
        else:
            raise NotImplementedError
        HyBO(**kwag_)
