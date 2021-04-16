cdef map[str, float] List
cdef extern from "sdp_admm.h":
    struct SDPResult:
        pass

cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass MatrixXd:
        MatrixXd()
        MatrixXd(int, int) except +
        double operator()(int, int)
cdef extern from "helper.h":
    cdef void set_value(MatrixXd, int, int, double)
    cdef List sdp1_admm(MatrixXd, int, List)
    cdef List sdp1_admm_si(MatrixXd, List)
    cdef void get_mat(MatrixXd, SDPResult)
    cdef void set_list_value(List, double, int, double, int)