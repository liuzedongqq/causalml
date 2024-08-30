import multiprocessing as mp  # 导入多进程库  
from collections import defaultdict  # 导入默认字典工具  

import logging  # 导入日志库  
import cython  # 导入Cython库  
import numpy as np  # 导入NumPy库  
cimport numpy as np  # Cython 导入 NumPy 的 C 版本  
import pandas as pd  # 导入Pandas库  
import scipy.stats as stats  # 导入SciPy统计库  
import sklearn  # 导入Scikit-learn库  
from joblib import Parallel, delayed  # 导入并行和延迟执行工具  
from packaging import version  # 导入版本检测工具  
from sklearn.model_selection import train_test_split  # 导入数据集拆分工具  
from sklearn.utils import check_X_y, check_array, check_random_state  # 导入数据检查工具  

# 检测 Scikit-learn 的版本。如果版本 >= 0.22.0，使用新的 ignore_warnings。  
if version.parse(sklearn.__version__) >= version.parse('0.22.0'):  
    from sklearn.utils._testing import ignore_warnings  # 导入新的忽略警告方法  
else:  
    from sklearn.utils.testing import ignore_warnings  # 导入旧的忽略警告方法  

# 定义数据类型  
N_TYPE = np.int32  # 定义样本数量数据类型为32位整数  
TR_TYPE = np.int8  # 定义处理结果数据类型为8位整数  
Y_TYPE = np.int8  # 定义标签数据类型为8位整数  
P_TYPE = np.float64  # 定义概率数据类型为64位浮点数  

# Cython 类型定义  
ctypedef np.int32_t N_TYPE_t  # Cython 类型定义：样本数量  
ctypedef np.int8_t TR_TYPE_t  # Cython 类型定义：处理结果  
ctypedef np.int8_t Y_TYPE_t  # Cython 类型定义：标签  
ctypedef np.float64_t P_TYPE_t  # Cython 类型定义：概率  

# 定义最大整数值  
MAX_INT = np.iinfo(np.int32).max  # 获取32位整数的最大值  

# 创建日志记录器  
logger = logging.getLogger("causalml")  # 用于记录程序的信息和错误  

# Cython 外部函数声明，调用 C 的数学函数  
cdef extern from "math.h":  
    double log(double x) nogil  # 声明无GIL的log函数  
    double fabs(double x) nogil  # 声明无GIL的绝对值函数  
    double sqrt(double x) nogil  # 声明无GIL的平方根函数  

@cython.cfunc  
def kl_divergence(pk: cython.float, qk: cython.float) -> cython.float:  
    '''  
    计算二分类的 KL 散度。  

    Args  
    ----  
    pk : float  
        一个分布中 1 的概率。  
    qk : float  
        另一个分布中 1 的概率。  

    Returns  
    -------  
    S : float  
        KL 散度的值。  
    '''  
    eps: cython.float = 1e-6  # 设置一个很小的常数，用于避免计算中的除零错误  
    S: cython.float  

    # 如果 qk 为零，直接返回零  
    if qk == 0.:  
        return 0.  

    # 将 qk 限制在 eps 和 1-eps 之间  
    qk = min(max(qk, eps), 1 - eps)  

    # 根据 pk 和 qk 的值分别计算 KL 散度  
    if pk == 0.:  
        S = -log(1 - qk)  # 当 pk 为零时，KL 散度公式的一种特例  
    elif pk == 1.:  
        S = -log(qk)  # 当 pk 为一时，KL 散度公式的一种特例  
    else:  
        S = pk * log(pk / qk) + (1 - pk) * log((1 - pk) / (1 - qk))  # 常规 KL 散度计算  

    return S  # 返回计算的 KL 散度  

@cython.cfunc  
def entropyH(p: cython.float, q: cython.float=-1.) -> cython.float:  
    '''  
    计算熵。  

    Args  
    ----  
    p : float  
        用于熵计算的概率。  

    q : float, optional, (default = -1.)  
        用于熵计算的第二个概率。  

    Returns  
    -------  
    entropy : float  
        计算的熵值。  
    '''  

    # 根据 q 的不同情况计算熵  
    if q == -1. and p > 0.:  
        return -p * log(p)  # 只有 p 时的熵计算  
    elif q > 0.:  
        return -p * log(q)  # 有 p 和 q 时的熵计算  
    else:  
        return 0.  # 如果 p 和 q 都不满足条件，返回零




class DecisionTree:  
    """ Tree Node Class  

    Tree node class to contain all the statistics of the tree node.  

    Parameters  
    ----------  
    classes_ : list of str  
        A list of the control and treatment group names.  

    col : int, optional (default = -1)  
        The column index for splitting the tree node to children nodes.  

    value : float, optional (default = None)  
        The value of the feature column to split the tree node to children nodes.  

    trueBranch : object of DecisionTree  
        The true branch tree node (feature > value).  

    falseBranch : object of DecisionTree  
        The false branch tree node (feature <= value).  

    results : list of float  
        The classification probability P(Y=1|T) for each of the control and treatment groups  
        in the tree node.  

    summary : list of list  
        Summary statistics of the tree nodes, including impurity, sample size, uplift score, etc.  

    maxDiffTreatment : int  
        The treatment index generating the maximum difference between the treatment and control groups.  

    maxDiffSign : float  
        The sign of the maximum difference (1. or -1.).  

    nodeSummary : list of list  
        Summary statistics of the tree nodes [P(Y=1|T), N(T)], where y_mean stands for the target metric mean  
        and n is the sample size.  

    backupResults : list of float  
        The positive probabilities in each of the control and treatment groups in the parent node. The parent node  
        information is served as a backup for the children node, in case no valid statistics can be calculated from the  
        children node, the parent node information will be used in certain cases.  

    bestTreatment : int  
        The treatment index providing the best uplift (treatment effect).  

    upliftScore : list  
        The uplift score of this node: [max_Diff, p_value], where max_Diff stands for the maximum treatment effect, and  
        p_value stands for the p_value of the treatment effect.  

    matchScore : float  
        The uplift score by filling a trained tree with validation dataset or testing dataset.  
    """  

    def __init__(self, classes_, col=-1, value=None, trueBranch=None, falseBranch=None, results=None, summary=None,  
                  maxDiffTreatment=None, maxDiffSign=1., nodeSummary=None, backupResults=None, bestTreatment=None,  
                  upliftScore=None, matchScore=None):  
        self.classes_ = classes_  # 控制组和处理组名称列表  
        self.col = col  # 用于分割节点的列索引  
        self.value = value  # 特征列的值，用于分割树节点  
        self.trueBranch = trueBranch  # 为trueBranch的DecisionTree对象, 条件为 feature > value  
        self.falseBranch = falseBranch  # 为falseBranch的DecisionTree对象, 条件为 feature <= value  
        self.results = results  # 每个组的分类概率，叶节点计算值不为None  
        self.summary = summary  # 节点的汇总统计信息  
        self.maxDiffTreatment = maxDiffTreatment  # 最大差异生成的处理组索引  
        self.maxDiffSign = maxDiffSign  # 最大差异的符号  
        self.nodeSummary = nodeSummary  # 节点的汇总统计，[P(Y=1|T), N(T)]  
        self.backupResults = backupResults  # 父节点的正概率  
        self.bestTreatment = bestTreatment  # 提供最佳提升的处理组索引  
        self.upliftScore = upliftScore  # 节点的提升分数  
        self.matchScore = matchScore  # 填充训练好的树与验证集或测试集的提升分数  


def group_uniqueCounts_to_arr(treatment_idx, y, out_arr):  
    '''  
    Count sample size by experiment group.  

    Args  
    ----  
    treatment_idx : array-like, shape = [num_samples]  
        An array containing the treatment group index for each unit.  
        Should be of type numpy.int8  
    y : array-like, shape = [num_samples]  
        An array containing the outcome of interest for each unit.  
        Should be of type numpy.int8  
    out_arr : array-like, shape = [2 * n_class]  
        An array to store the output counts, should have type numpy.int32  

    Returns  
    -------  
    No return value, but modified the out_arr to hold the negative and positive  
    outcome sample sizes for each of the control and treatment groups.  
        out_arr[2*i] is N(Y = 0, T = i) for i = 0, ..., n_class  
        out_arr[2*i+1] is N(Y = 1, T = i) for i = 0, ..., n_class  
    '''  
    cdef int out_arr_len = out_arr.shape[0]  # 获取输出数组的长度  
    cdef int n_class = out_arr_len / 2  # 计算实验组的数量  
    cdef int num_samples = treatment_idx.shape[0]  # 样本数量  
    cdef int yv = 0  # 当前样本的输出  
    cdef int tv = 0  # 当前样本的处理组索引  
    cdef int i = 0  # 循环变量  

    # 首先清空输出数组  
    for i in range(out_arr_len):  
        out_arr[i] = 0  
    
    # 循环处理每个样本，根据处理组累加计数  
    for i in range(num_samples):  
        tv = treatment_idx[i]  # 获取当前样本的处理组索引  
        out_arr[2 * tv] += 1  # 统计当前处理组的样本数量  
        out_arr[2 * tv + 1] += y[i]  # 统计当前处理组中y的值（0或1）的总和  
    
    # 将负样本数调整为 N(Y = 0, T = i) = N(T = i) - N(Y = 1, T = i)  
    for i in range(n_class):  
        out_arr[2 * i] -= out_arr[2 * i + 1]  
    # 执行完成，修改了 out_arr，因此不需要返回  


def group_counts_by_divide(  
        col_vals, threshold_val, is_split_by_gt,  
        treatment_idx, y, out_arr):  
    '''  
    Count sample size by experiment group for the left branch,  
    after splitting col_vals by threshold_val.  
    If is_split_by_gt, the left branch is (col_vals >= threshold_val),  
    otherwise the left branch is (col_vals == threshold_val).  

    Args  
    ----  
    col_vals : array-like, shape = [num_samples]  
        An array containing one column of x values.  
    threshold_val : compatible value with col_vals  
        A value for splitting col_vals.  
        If is_split_by_gt, the left branch is (col_vals >= threshold_val),  
        otherwise the left branch is (col_vals == threshold_val).  
    is_split_by_gt : bool  
        Whether to split by (col_vals >= threshold_val).  
        If False, will split by (col_vals == threshold_val).  
    treatment_idx : array-like, shape = [num_samples]  
        An array containing the treatment group index for each unit.  
        Should be of type numpy.int8  
    y : array-like, shape = [num_samples]  
        An array containing the outcome of interest for each unit.  
        Should be of type numpy.int8  
    out_arr : array-like, shape = [2 * n_class]  
        An array to store the output counts, should have type numpy.int32  

    Returns  
    -------  
    len_X_l: the number of samples in the left branch.  
    Also modify the out_arr to hold the negative and positive  
    outcome sample sizes for each of the control and treatment groups.  
        out_arr[2*i] is N(Y = 0, T = i) for i = 0, ..., n_class  
        out_arr[2*i+1] is N(Y = 1, T = i) for i = 0, ..., n_class  
    '''  
    cdef int out_arr_len = out_arr.shape[0]  # 输出数组长度  
    cdef int n_class = out_arr_len / 2  # 计算实验组数量  
    cdef int num_samples = treatment_idx.shape[0]  # 样本数量  
    cdef int yv = 0  
    cdef int tv = 0  
    cdef int i = 0  
    cdef N_TYPE_t len_X_l = 0  # 左分支样本数量  
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] filt  # 过滤条件数组  

    # 首先清空输出数组  
    for i in range(out_arr_len):  
        out_arr[i] = 0  

    # 根据是否大于阈值分割  
    if is_split_by_gt:  
        filt = col_vals >= threshold_val  
    else:  
        filt = col_vals == threshold_val  

    # 循环处理样本，根据滤波条件统计左分支的数量  
    for i in range(num_samples):  
        if filt[i]:  # 如果满足过滤条件  
            len_X_l += 1  # 左分支样本数量加一  
            tv = treatment_idx[i]  # 获取处理组索引  
            out_arr[2 * tv] += 1  # 统计当前处理组的样本数量  
            out_arr[2 * tv + 1] += y[i]  # 统计当前处理组中y的值（0或1）的总和  
    
    # 将负样本数调整为 N(Y = 0, T = i) = N(T = i) - N(Y = 1, T = i)  
    for i in range(n_class):  
        out_arr[2 * i] -= out_arr[2 * i + 1]  
    
    return len_X_l  # 返回左分支的样本数量



















