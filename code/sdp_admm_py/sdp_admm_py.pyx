# distutils: language = c++
from sdp_admm cimport List, MatrixXd, SDPResult
from sdp_admm cimport set_value, sdp1_admm, sdp_admm_sbm_2, sdp_admm_sbm_si_2, get_mat, set_list_value
import numpy as np

def sdp1_admm_py(pyMat, K, rho=0.1, T=10000, tol=1e-5, report_interval=100):
    return _admm_py(pyMat, K=K, rho=rho, T=T, tol=tol, report_interval=report_interval, method='sdp1_admm')

def sdp_admm_sbm_2_py(pyMat, rho=0.1, T=10000, tol=1e-5, report_interval=100):
    return _admm_py(pyMat, rho=rho, T=T, tol=tol, report_interval=report_interval, method='sdp_admm_sbm_2')

def sdp_admm_sbm_si_2_py(pyMat, rho=0.1, T=10000, tol=1e-5, report_interval=100):
    return _admm_py(pyMat, rho=rho, T=T, tol=tol, report_interval=report_interval, method='sdp_admm_sbm_si_2')

def _admm_py(pyMat, K=2, rho=0.1, T=10000, tol=1e-5, report_interval=100, method='sdp1_admm'):
    cdef List* param_list
    cdef SDPResult result_list

    
    param_list = new List()
    set_list_value(param_list[0], rho, T, tol, report_interval)
    
    n_rows, n_cols = pyMat.shape
    _m_r = new MatrixXd(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            set_value(_m_r[0], i, j, pyMat[i, j])
    if method == 'sdp1_admm':
        result_list = sdp1_admm(_m_r[0], K, param_list[0])
    elif method == 'sdp_admm_sbm_2':
        result_list = sdp_admm_sbm_2(_m_r[0], param_list[0])
    elif method == 'sdp_admm_sbm_si_2':
        result_list = sdp_admm_sbm_si_2(_m_r[0], param_list[0])
    else:
        raise ValueError("unknown method " + method)
    get_mat(_m_r[0], result_list)
    py_result_mat = np.zeros([n_rows, n_cols])
    for i in range(n_rows):
        for j in range(n_cols):
            py_result_mat[i, j] = _m_r[0](i, j)
    del param_list
    del _m_r

    return py_result_mat

