__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"
__project__= "LLP - MicroPheno"

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


def body_sites():
    files=FileUtility.recursive_glob('/mounts/data/proj/asgari/github_repos/microbiomephenotype/data_config/bodysites/','*.txt')
    list_of_files=[]
    for file in files:
        print (file)
        list_of_files+=FileUtility.load_list(file)
    list_of_files=[x+'.fsa' for x in list_of_files]
    fasta_files, mapping = FileUtility.read_fasta_directory('/mounts/data/proj/asgari/dissertation/datasets/deepbio/microbiome/hmb_data/','fsa',only_files=list_of_files)
    
    sampling_dict={3:[20,3000],4:[100,1000],5:[500,5000],6:[2000,10000],7:[5000,10000],8:[8000,16000]}
    for k in range(3,7):
        for s in sampling_dict[k]:
            print(k)
            RS=Metagenomic16SRepresentation(fasta_files, mapping, s, 20)
            RS.generate_kmers_all(k, save='datasets/bodysites/'+str(k)+'-mers_rate_'+str(s))

    
def crohn_disease():
    fasta_files, mapping = FileUtility.read_fasta_directory('/mounts/data/proj/asgari/dissertation/datasets/deepbio/microbiome/crohn/','fastq')
    RS=Metagenomic16SRepresentation(fasta_files, mapping, 3000, 20)
    for k in range(3,9):
        RS.generate_kmers_all(k, save='datasets/crohn/'+str(k)+'-mers_rate_3000')

if __name__=='__main__':
    crohn_disease()

