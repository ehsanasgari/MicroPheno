import sys
sys.path.append('../')
from sklearn.utils import resample
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import codecs
import random
import numpy as np
from math_utility import get_kl_rows
import matplotlib.pyplot as plt
from IPython import display
from Bio import SeqIO
from file_utility import FileUtility

class BootStrapping(object):
    def __init__(self, input_dir, output_dir, sampling_sizes=[10,20,50,100,200,500,1000,2000,5000,10000], n_resamples=10, seqtype='fasta', N_files=10):
        '''
        :param input_dir: input directory or a  list of files
        :param output_dir: a directory to generate the output files
        :param sampling_sizes: a list of sampling sizes
        :param n_resamples: number of resamples from each file
        :param seqtype: file suffixes
        :param N_files: number of files from the directory to make samplings from
        '''
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.N_files=N_files
        self.seqtype=seqtype
        self.sampling_sizes=sampling_sizes
        self.n_resamples=n_resamples
        # dictionary for different k-values
        # x's are the sampling sizes
        self.x=dict()
        # y's are mean of distances among k-mer distribution resamples
        self.y=dict()
        # errors's are std of distances among k-mer distribution resamples
        self.error=dict()
        # y_tot's are mean of distances between k-mer distribution resamples and the whole sample k-mer distribution
        self.y_tot=dict()
        # error_tot's are std of distances between k-mer distribution resamples and the whole sample k-mer distribution
        self.error_tot=dict()


    def add_kmer_sampling(self, k_mer):
        '''
        :param k_mer: k_mer bootstrapping to add in the analysis
        :return:
        '''
        self.x[k_mer], self.y[k_mer], self.error[k_mer], self.y_tot[k_mer], self.error_tot[k_mer] = self.get_stats_samples(k_mer)

    def get_stats_samples(self, k_mer):
        x=[]
        y=[]
        y_tot=[]
        error=[]
        error_tot=[]
        # To find the files
        if isinstance(self.input_dir, str):
            sample_files=FileUtility.recursive_glob(self.input_dir,"*"+self.seqtype)
        else:
            sample_files=self.input_dir
        sample_files=random.sample(sample_files, self.N_files)

        # To iterate over the sampling sizes
        for sample_size in self.sampling_sizes:
            distance_i=[]
            tot_dist_i=[]
            print(' sampling size ' ,sample_size , ' is started ...')
            # To iterate over random files
            for sample_file in sample_files:
                comp_dist=self._get_kmer_distribution(sample_file, k_mer, -1, 1)
                resamples_kmers=self._get_kmer_distribution(sample_file, k_mer, sample_size, self.n_resamples)
                distance_i.append(np.mean(get_kl_rows(np.array(resamples_kmers))))
                tot_dist_i=tot_dist_i+list(get_kl_rows(np.vstack((np.array(resamples_kmers),comp_dist[0])))[0:10,10])
            print(' sampling size ' ,sample_size ,  ' is completed.')
            mean_distance=np.mean(distance_i)
            std_distance=np.std(distance_i)
            mean_total_distance=np.mean(tot_dist_i)
            std_total_distance=np.std(tot_dist_i)
            x.append(sample_size)
            y.append(mean_distance)
            error.append(std_distance)
            y_tot.append(mean_total_distance)
            error_tot.append(std_total_distance)
        return x, y, error, y_tot, error_tot

    def _get_kmer_distribution(self,sample_file, k, sample_size, number_of_samples):
        '''
        :param sample_file:
        :param k:
        :param sample_size:
        :param number_of_samples:
        :return:
        '''
        vocab = [''.join(xs) for xs in itertools.product('atcg', repeat=k)]
        vectorizer = TfidfVectorizer(use_idf=False, vocabulary=vocab, analyzer='char', ngram_range=(k, k),
                                              norm=None, stop_words=[], lowercase=True, binary=False)
        corpus=[]
        for cur_record in SeqIO.parse(sample_file, 'fasta' if self.seqtype=='fsa' else self.seqtype):
            corpus.append(str(cur_record.seq).lower())
        if sample_size==-1:
            sample_size=len(corpus)
            resamples=[corpus]
        else:
            resamples=[]
            for i in range(number_of_samples):
                resamples.append(resample(corpus, replace=True, n_samples=sample_size))
        vect=[]
        for rs in resamples:
            vect.append(normalize(np.sum(vectorizer.fit_transform(rs).toarray(), axis=0).reshape(1, -1), axis=1, norm='l1')[0])
        return vect

    def _plotting(self, file_name):
        fig = plt.figure()
        plt.subplot(211)
        k_mers=list(self.x.keys())
        legend_vals=[]
        for k in k_mers:
            plt.plot(self.x[k], np.array(self.y[k]))
            plt.fill_between(self.x[k], np.array(self.y[k])-np.array(self.error[k]),np.array(self.y[k])+np.array(self.error[k]),alpha=0.2, linewidth=4, linestyle='dashdot', antialiased=True)
            legend_vals.append('k = '+str(k))
        plt.legend(legend_vals, loc='upper right')
        plt.xlabel('Sequence sampling size')
        plt.ylabel('KL-divergence')
        plt.xlim([0,10000])
        plt.ylim([0,10])
        plt.title('The averaged pair-wise KL-divergences between k-mer disributions \n in 10 resamples with respect  to the sampling size', fontsize=36)

        plt.subplot(212)

        k_mers=list(self.x.keys())
        legend_vals=[]
        for k in k_mers:
            plt.plot(self.x[k], np.array(self.y_tot[k]))
            plt.fill_between(self.x[k], np.array(self.y_tot[k])-np.array(self.error_tot[k]), np.array(self.y_tot[k])+np.array(self.error_tot[k]),alpha=0.2, linewidth=4, linestyle='dashdot', antialiased=True)
            legend_vals.append('k = '+str(k))
        plt.legend(legend_vals, loc='upper right')
        plt.xlim([0,10000])
        plt.ylim([0,0.1])
        plt.xlabel('Sequence sampling size')
        plt.ylabel('KL-divergence')
        plt.title('The averaged KL-divergences between k-mer disributions in 10 resamples \n and the whole data usage with respect to the sampling size', fontsize=36)

        params = {
           'legend.fontsize': 36,
           'xtick.labelsize': 36,
           'ytick.labelsize': 36,
           'text.usetex': True,
           }
        plt.rcParams.update(params)
        plt.rc('font', family='serif', serif='Times')
        fig.tight_layout()
        FileUtility.save_obj([self.x, self.y, self.error,self.y_tot,self.error_tot],'paper_figs/'+file_name+'.obj')
        plt.savefig('paper_figs/'+file_name+'.pdf')

if __name__=='__main__':
    files=FileUtility.recursive_glob('/mounts/data/proj/asgari/github_repos/microbiomephenotype/data_config/bodysites/','*.txt')
    list_of_files=[]
    for file in files:
        list_of_files+=FileUtility.load_list(file)
    list_of_files=[x+'.fsa' for x in list_of_files]
    fasta_files, mapping = FileUtility.read_fasta_directory('/mounts/data/proj/asgari/dissertation/datasets/deepbio/microbiome/hmb_data/','fsa',only_files=list_of_files)
    BS=BootStrapping(fasta_files,'body', seqtype='fsa', N_files=10)
    for k in [3,4,5,6,7,8]:
        print(k)
        BS.add_kmer_sampling(k)
