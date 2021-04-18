from libcpp.map cimport map
from libcpp.string cimport string

cdef extern from "sdp_admm.h":
    cdef struct SDPResult:
        pass

cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass MatrixXd:
        MatrixXd()
        MatrixXd(int, int) except +
        double operator()(int, int)
cdef extern from "helper.h":
    ctypedef map[string, float] List
    cdef void set_value(MatrixXd, int, int, double)
    cdef SDPResult sdp1_admm(MatrixXd, int, List)
    cdef SDPResult sdp_admm_sbm_2(MatrixXd, List)
    cdef SDPResult sdp_admm_sbm_si_2(MatrixXd, List)
    cdef void get_mat(MatrixXd, SDPResult)
    cdef void set_list_value(List, double, int, double, int)