__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu or ehsaneddin.asgari@helmholtz-hzi.de"
__project__ = "LLP - MicroPheno"
__website__ = "https://llp.berkeley.edu/micropheno/"



import sys
sys.path.append('../')
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import numpy as np
from multiprocessing import Pool
import tqdm
import random
from scipy import sparse
from utility.file_utility import FileUtility
from Bio import SeqIO
import timeit


class Metagenomic16SRepresentation:
    '''
        Make k-mer from directory of fasta files
    '''

    def __init__(self, fasta_files, indexing, sampling_number=3000, num_p=20):
        '''
        :param fasta_files: list of fasta files
        :param indexing: the index
        :param sampling_number:
        :param num_p:
        '''
        self.fasta_files=fasta_files
        self.num_p=num_p
        self.sampling_number=sampling_number
        self.indexing=indexing


    def get_corpus(self, file_name_sample):
        '''
        :param file_name_sample:
        :return:
        '''
        file_name=file_name_sample[0]
        sample_size=file_name_sample[1]
        corpus=[]
        if file_name[-1]=='q':
            for cur_record in SeqIO.parse(file_name, "fastq"):
                corpus.append(str(cur_record.seq).lower())
        else:
            for cur_record in SeqIO.parse(file_name, "fasta"):
                corpus.append(str(cur_record.seq).lower())
        return file_name, random.sample(corpus, min(sample_size,len(corpus))) 
    

    def generate_kmers_all(self, k, save=False):
        '''
        :param k:
        :param save:
        :return:
        '''
        self.k=k
        self.vocab = [''.join(xs) for xs in itertools.product('atcg', repeat=k)]
        self.vocab.sort()
        self.vectorizer = TfidfVectorizer(use_idf=False, vocabulary=self.vocab, analyzer='char', ngram_range=(k, k),
                                          norm=None, stop_words=[], lowercase=True, binary=False)

        data = np.zeros((len(self.fasta_files), len(self.vocab))).astype(np.float64)

        # multi processing extraction of k-mer distributions
        t_steps=[]
        s_steps=[]
        pool = Pool(processes=self.num_p)
        for ky, (v,t,s) in tqdm.tqdm(pool.imap_unordered(self.get_kmer_distribution, self.fasta_files, chunksize=1),
                               total=len(self.fasta_files)):
            data[self.indexing[ky], :] = v
            t_steps.append(t)
            s_steps.append(s)

        # normalize the frequencies
        data = normalize(data, axis=1, norm='l1')
        data = sparse.csr_matrix(data)

        if save:
            FileUtility.save_sparse_csr(save, data)
            FileUtility.save_list(save+'_meta',self.fasta_files)
            FileUtility.save_list(save+'_log',[': '.join(['mean_time', str(np.mean(t_steps))]), ': '.join(['std_time', str(np.std(t_steps))]), ': '.join(['mean_size', str(np.mean(s_steps))]), ': '.join(['std_size', str(np.std(s_steps))])])

        return data

    def get_kmer_distribution(self, file_name):
        '''

        :param file_name:
        :return:
        '''
        start = timeit.timeit()
        corpus=[]
        if file_name[-1]=='q':
            for cur_record in SeqIO.parse(file_name, "fastq"):
                corpus.append(str(cur_record.seq).lower())
        else:
            for cur_record in SeqIO.parse(file_name, "fasta"):
                corpus.append(str(cur_record.seq).lower())
        tot_size=len(corpus)
        if self.sampling_number==-1:
            random.shuffle(corpus)
        else:
            corpus = random.sample(corpus, min(self.sampling_number,len(corpus)))
        end = timeit.timeit()
        return file_name, (np.sum(self.vectorizer.fit_transform(corpus).toarray(), axis=0),end - start,tot_size)


class FastaRepresentations(object):
    '''
        Make k-mer from single fasta file
        where the headers contain info about the label
    '''
    def __init__(self, fasta_address, label_modifying_func=str):
        '''
        :param fasta_address:
        :param label_modifying_func: extract label from the header
        '''
        self.labels=[]
        self.corpus=[]
        for cur_record in SeqIO.parse(fasta_address, 'fasta'):
            self.corpus.append(str(cur_record.seq).lower())
            self.labels.append(str(cur_record.id).lower())
        self.labels=[label_modifying_func(l) for l in self.labels]

    def get_samples(self, envs, N):
        '''
        :param envs: list of labels
        :param N: sample size
        :return: extract stratified with size N corpus and label list
        '''
        labels=[]
        corpus=[]
        for env in envs:
            selected=[idx for idx,v in enumerate(self.labels) if env==v]
            if N==-1:
                N=len(selected)
            idxs=random.sample(selected, N)
            corpus=corpus+[self.corpus[idx] for idx in idxs]
            labels=labels+[self.labels[idx] for idx in idxs]
        return corpus, labels

    def get_vector_rep(self, corpus, k, restricted=True):
        '''
        :param corpus:
        :param k: k-mer size
        :param restricted: restricted to known values
        :return:
        '''
        if restricted:
            vocab = [''.join(xs) for xs in itertools.product('atcg', repeat=k)]
            tf_vec = TfidfVectorizer(use_idf=True, vocabulary=vocab, analyzer='char', ngram_range=(k, k),
                                                  norm='l1', stop_words=[], lowercase=True, binary=False)
        else:
            tf_vec = TfidfVectorizer(use_idf=True, analyzer='char', ngram_range=(k, k),
                                                  norm='l1', stop_words=[], lowercase=True, binary=False)
        return tf_vec.fit_transform(corpus)

if __name__=='__main__':
    FR=FastaRepresentations('sample.fasta')
    MR=Metagenomic16SRepresentation('16ssamples/')

