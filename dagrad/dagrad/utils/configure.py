from typing import Dict, Any
import typing
from ..score.score import loss_fn, reg_fn, nl_loss_fn
from ..hfunction.h_functions import h_fn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
# Definitions of methods, algorithms, functions, and regularizers

METHODS = ['notears', 'topo', 'dagma']
OPTIMIZERS = ['adam', 'lbfgs','sklearn']
LOSS_FUNCTIONS = ['l2', 'logistic', 'logll', 'logdetll_ev', 'logdetll_nv', 'user_loss']
H_FUNCTIONS = ['h_exp_sq', 'h_logdet_sq', 'h_poly_sq', 'h_exp_abs', 'h_logdet_abs', 'h_poly_abs',
               'h_exp_topo', 'h_logdet_topo', 'h_poly_topo', 'user_h']
REGULARIZERS = ['l1', 'l2', 'mcp', 'none', 'user_reg']


allowed_general_options = {
    # this options include hyperparameters for regularization, general purpose options,
    'lambda1': (float,int,np.ndarray,torch.Tensor), # weight for l1 penalty: scalar or (p,p) per-edge matrix (negative entries encourage an edge)
    'lambda2': (float,int), # weight for l2 penalty
    'gamma': (float,int), # hyperparamter for quasi-MCP
    'var_floor': (float,int), # minimum per-node residual variance for logdet losses (0 = off); floors RSS so a (near-)deterministic node can't send log(RSS) -> -inf
    'w_threshold': (float,int),
    'initialization': (type(None), np.ndarray, torch.Tensor, nn.Module),
    'tuning_method': (str, type(None)),
    'K': int, # the number of folds for cross-validation
    'reg_paras': list,
    'user_params': dict,
    'mask': (np.ndarray, torch.Tensor, list),  # (d,d) allow-mask: 0 forbids edge i->j
    'given_hinge_mask': (np.ndarray, torch.Tensor),  # bool (p,p) marking given edges to encourage via the magnitude-floor hinge; torch only
    'given_hinge_margin': (float, int),  # m: magnitude floor each given edge is pushed to clear (0 = off)
    'given_hinge_weight': (float, int),  # rho: strength of the given-edge hinge (0 = off)
    'given_hinge_al': bool,  # torch only: use the augmented-Lagrangian given-edge floor (per-edge dual, reaches the margin exactly) instead of the fixed-weight hinge. Default False keeps the original soft hinge.
    'reg_on_adj': bool,  # nonlinear only: apply L1 (and per-edge lambda1) to the column-space edge magnitudes instead of the scalar fc1 weight sum; required for a (p,p) lambda1 / given-edge encouragement. Default False keeps the original scalar fc1 L1.
}

allowed_method_options = {
    'linear':{
    # this options include hyperparameters for linear methods
    'notears':{
    'rho_max': int,
    'main_iter': int,
    'rho': float,
    'h_tol': float,
    'dtype':type,
    'verbose': bool},

    'topo':{
    'no_large_search': int,
    'size_small': int,
    'size_large': int,
    'topo': (list, type(None)),
    'dtype':type,
    'verbose': bool
    },


    'dagma':{
    'T': int,
    'mu_init': float,
    'mu_factor': float,
    's': typing.Union[typing.List[float], float], 
    'warm_iter': int,
    'main_iter': int,
    'dtype':type,
    'exclude_edges': typing.Optional[typing.List[typing.Tuple[int, int]]] or None,
    'include_edges': typing.Optional[typing.List[typing.Tuple[int, int]]] or None,
    'verbose': bool
    }

    },
    
    'nonlinear':{
    
    'notears':{
    'bias': bool,
    'activation': str,
    'rho_max': int,
    'main_iter': int,
    'h_tol': float,
    'dims': typing.List,
    'dtype':type,
    'verbose': bool},

    'topo':{
    'no_large_search': int,
    'size_small': int,
    'size_large': int,
    'topo': (list, type(None)),
    'dims': typing.List,
    'bias': bool,
    'activation': str,
    "dtype": type,
    'verbose': bool,
    },


    'dagma':{
    'T': int,
    'mu_init': float,
    'mu_factor': float,
    's': typing.Union[typing.List[float], float], # hyperparameter for h log det function
    'warm_iter': int,
    'main_iter': int,
    'dtype':type,
    'dims': typing.List,
    'verbose': bool,
    'bias': bool,
    }



    }

    

}


allowed_optimizer_options: Dict[str, Dict] = {
    'numpy':{
        'adam':{
            'opt_config':{
            'lr':float,
            'betas':typing.Tuple[float,float],
            'eps':float,
            },
            'opt_settings':{
                'check_iterate':int,
                'tol':float,
                'num_steps':int,
            }
        },
        'lbfgs':{
            'opt_config':{
            'disp':(int, type(None)),
            'maxcor':int,
            'ftol':float,
            'gtol':float,
            'eps':float,
            'maxfun':int,
            'maxiter':int,
            'iprint':int,
            'maxls':int,}
            ,
            'opt_settings':{
            }
        },
        'sklearn':{
            'opt_config':{
            },
            'opt_settings':{
            }
        }
    },
    'torch':{                  
        'adam':{
            'opt_config':{
            'lr':float,
            'betas':typing.Tuple[float,float],
            'eps':float,
            },
            'opt_settings':{
            'tol':float, # self-added
            'num_steps':int, #self-added
            'check_iterate':int, # self-added
            'lr_decay':bool # self-added
            }
        },
        'lbfgs':{
            'opt_config':{
            'lr':float,
            'max_iter':int,
            'max_eval ':int or None,
            'tolerance_grad ':float,
            'tolerance_change':float,
            'history_size':int,
            'line_search_fn':str or None,
            },
            'opt_settings':{
                'num_steps':int, #self-added
                'tol':float, #self-added
                'check_iterate':int, # self-added
                'lr_decay':bool # self-added
            }

            
        }

    
}
}




optimizer_functions = {
    'adam': optim.Adam,
    'lbfgs': optim.LBFGS,
}

loss_functions = {
    'l2': loss_fn.l2_loss,
    'logistic': loss_fn.logistic_loss,
    'logll': loss_fn.logll_loss,
    'logdetll_ev': loss_fn.logdetll_ev_loss,
    'logdetll_nv': loss_fn.logdetll_nv_loss,
    'user_loss': loss_fn.user_loss,
    # Add more mappings as needed
}
nl_loss_functions = {
    'l2': nl_loss_fn.nl_l2_loss,
    'logistic': nl_loss_fn.nl_logistic_loss,
    'logll': nl_loss_fn.nl_logll_loss,
    'user_loss': nl_loss_fn.user_nl_loss,
    # Add more mappings as needed
}

h_functions = {
    'h_exp_abs': h_fn.h_exp_abs,
    'h_poly_abs': h_fn.h_poly_abs,
    'h_logdet_abs': h_fn.h_logdet_abs,
    'h_exp_sq': h_fn.h_exp_sq,
    'h_poly_sq': h_fn.h_poly_sq,
    'h_logdet_sq': h_fn.h_logdet_sq,
    'h_exp_topo': h_fn.h_exp_topo,
    'h_poly_topo': h_fn.h_poly_topo,
    'h_logdet_topo': h_fn.h_logdet_topo,
    'user_h': h_fn.user_h
    # Add more mappings as needed
}

reg_functions = {
    'l1': reg_fn.l1_reg,
    'l2': reg_fn.l2_reg,
    'mcp': reg_fn.mcp_reg,
    'none': reg_fn.no_reg,
    'user_reg': reg_fn.user_reg
    # Add more mappings as needed
}