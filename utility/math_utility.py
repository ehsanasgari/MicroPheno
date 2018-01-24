__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"
__project__= "LLP - MicroPheno"

from scipy import stats
from sklearn.preprocessing import normalize

def get_kl_rows(A):
    '''
    :param A: matrix A
    :return: Efficient implementation to calculate kl-divergence between rows in A
    '''
    norm_A=normalize(A+1e-100, norm='l1')
    return stats.entropy(norm_A.T[:,:,None], norm_A.T[:,None,:])
