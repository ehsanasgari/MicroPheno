__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu or ehsaneddin.asgari@helmholtz-hzi.de"
__project__ = "LLP - MicroPheno"
__website__ = "https://llp.berkeley.edu/micropheno/"

import re
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt


class HierarchicalClutering(object):
    '''
    classdocs
    '''


    def __init__(self, distance_matrix, labels_out):
        '''
        Constructor
        '''
        z = hac.complete(distance_matrix)
        hac.dendrogram(z,labels=labels_out)
        tree = hac.to_tree(z,False)
        self.nwk=self.getNewick(tree, "", tree.dist, labels_out)
        
    
    def getNewick(self, node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
        else:
            if len(newick) > 0:
                newick = "):%.2f%s" % (parentdist - node.dist, newick)
            else:
                newick = ");"
            newick = self.getNewick(node.get_left(), newick, node.dist, leaf_names)
            newick = self.getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
            newick = "(%s" % (newick)
            return newick
