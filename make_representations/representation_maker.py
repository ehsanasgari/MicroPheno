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
from make_representations.cpe_apply import BPE
from Bio import SeqIO
from make_representations.cpe_efficient import train_cpe
import timeit


class Metagenomic16SRepresentation:

    def __init__(self, fasta_files, indexing, sampling_number=3000, num_p=10):
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

    def generate_cpe_all(self, size, save, directory='data_config/',post_fix=''):
        f=open(directory+'cpe_16s_'+str(size)+post_fix,'r')
        self.CPE_Applier=BPE(f,separator='')
        self.cpe_vocab=[x.lower() for x in FileUtility.load_list('data_config/cpe_16s_'+str(size)+'_vocab')]
        self.cpe_vocab.sort()
        self.cpe_vectorizer = TfidfVectorizer(use_idf=False, vocabulary=self.cpe_vocab, analyzer='word',
                                          norm=None, stop_words=[], lowercase=True, binary=False, tokenizer=str.split)

        data = np.zeros((len(self.fasta_files), len(self.cpe_vocab))).astype(np.float64)

        # multi processing extraction of k-mer distributions
        pool = Pool(processes=self.num_p)
        for ky, v in tqdm.tqdm(pool.imap_unordered(self.get_cpe_distribution, self.fasta_files, chunksize=1),
                               total=len(self.fasta_files)):
            data[self.indexing[ky], :] = v

        # normalize the frequencies
        data = normalize(data, axis=1, norm='l1')
        data = sparse.csr_matrix(data)

        if save:
            FileUtility.save_sparse_csr(save, data)
            FileUtility.save_list(save+'_meta',self.fasta_files)
        return data
    
    def train_cpe_transformation(self, size, sample_size, directory):
        
        pool = Pool(processes=self.num_p)
        fasta_sample_files=[[x,sample_size] for x in self.fasta_files]
        corpus=[]
        for ky, v in tqdm.tqdm(pool.imap_unordered(self.get_corpus, fasta_sample_files, chunksize=5),
                               total=len(self.fasta_files)):
            corpus = corpus+ v
        print('Corpus size for traing CPE is ',len(corpus))
        train_cpe(corpus, directory+'cpe_16s_'+str(size), size, directory+'cpe_16s_'+str(size)+'_freq')
        

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
    
    def get_cpe_distribution(self, file_name):
        corpus=[]
        if file_name[-1]=='q':
            for cur_record in SeqIO.parse(file_name, "fastq"):
                corpus.append(str(cur_record.seq).lower())
        else:
            for cur_record in SeqIO.parse(file_name, "fasta"):
                corpus.append(str(cur_record.seq).lower())
        if self.sampling_number==-1:
            random.shuffle(corpus)
        else:
            corpus = random.sample(corpus, min(self.sampling_number,len(corpus)))
        corpus=[self.CPE_Applier.segment(x) for x in corpus]
        return file_name, np.sum(self.cpe_vectorizer.fit_transform(corpus).toarray(), axis=0)

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

def bowel():
    fasta_files, mapping = FileUtility.read_fasta_directory('/mounts/data/proj/asgari/dissertation/datasets/deepbio/microbiome/BOWEL/fasta/','fasta')
    RS=Metagenomic16SRepresentation(fasta_files, mapping, 3000, 40)
    for k in range(3,9):
        RS.generate_kmers_all(k, save='datasets/bowel/'+str(k)+'-mers_rate_3000')
    RS.generate_cpe_all(1000, save='datasets/bowel/cpe_1000_rate_3000')
    RS.generate_cpe_all(2000, save='datasets/bowel/cpe_2000_rate_3000')
    RS.generate_cpe_all(5000, save='datasets/bowel/cpe_5000_rate_3000')
    RS.generate_cpe_all(10000, save='datasets/bowel/cpe_10000_rate_3000')

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
            #RS=Representation_16S_sample(fasta_files, mapping, s, 20)
            #RS.generate_kmers_all(k, save='datasets/bodysites/'+str(k)+'-mers_rate_'+str(s))
    RS=Metagenomic16SRepresentation(fasta_files, mapping, 5000, 20)
    RS.generate_cpe_all(10000, save='datasets/bodysites/cpeself_10000_rate_5000', directory='datasets/bodysites/data_config/')
    RS=Metagenomic16SRepresentation(fasta_files, mapping, 10000, 20)
    RS.generate_cpe_all(10000, save='datasets/bodysites/cpeself_10000_rate_10000', directory='datasets/bodysites/data_config/')
    RS=Metagenomic16SRepresentation(fasta_files, mapping, 2000, 20)
    RS.generate_cpe_all(10000, save='datasets/bodysites/cpesilva_10000_rate_2000', post_fix='_lower')
    RS=Metagenomic16SRepresentation(fasta_files, mapping, 5000, 20)
    RS.generate_cpe_all(10000, save='datasets/bodysites/cpesilva_10000_rate_5000', post_fix='_lower')

def dental():
    fasta_files, mapping = FileUtility.read_fasta_directory('/mounts/data/proj/asgari/dissertation/datasets/deepbio/microbiome/dental/','fastq')
    
    sampling_dict={3:[20,1000],4:[50,1000],5:[500,5000],6:[2000,10000],7:[5000,10000],8:[8000,16000]}
    for k in range(7,9):
        for s in sampling_dict[k]:
            print(k)
            RS=Metagenomic16SRepresentation(fasta_files, mapping, s, 20)
            RS.generate_kmers_all(k, save='datasets/dental/'+str(k)+'-mers_rate_'+str(s))
    #RS=Representation_16S_sample(fasta_files, mapping, 5000, 20)
    #RS.generate_cpe_all(10000, save='datasets/dental/cpeself_10000_rate_5000', directory='datasets/dental/data_config/')
    #RS=Representation_16S_sample(fasta_files, mapping, 10000, 20)
    #RS.generate_cpe_all(10000, save='datasets/dental/cpeself_10000_rate_10000', directory='datasets/dental/data_config/')
    #RS=Representation_16S_sample(fasta_files, mapping, 2000, 20)
    #RS.generate_cpe_all(10000, save='datasets/dental/cpesilva_10000_rate_2000', post_fix='_lower')
    #RS=Representation_16S_sample(fasta_files, mapping, 5000, 20)
    #RS.generate_cpe_all(10000, save='datasets/dental/cpesilva_10000_rate_5000', post_fix='_lower')
    
def crohn_disease():
    fasta_files, mapping = FileUtility.read_fasta_directory('/mounts/data/proj/asgari/dissertation/datasets/deepbio/microbiome/crohn/','fastq')
    RS=Metagenomic16SRepresentation(fasta_files, mapping, 3000, 20)
    for k in range(3,9):
        RS.generate_kmers_all(k, save='datasets/crohn/'+str(k)+'-mers_rate_3000')
    RS.generate_cpe_all(1000, save='datasets/crohn/cpe_1000_rate_3000')
    RS.generate_cpe_all(2000, save='datasets/crohn/cpe_2000_rate_3000')
    RS.generate_cpe_all(5000, save='datasets/crohn/cpe_5000_rate_3000')
    RS.generate_cpe_all(10000, save='datasets/crohn/cpe_10000_rate_3000')

def cpe_dental():
    fasta_files, mapping = FileUtility.read_fasta_directory('/mounts/data/proj/asgari/dissertation/datasets/deepbio/microbiome/dental/','fastq')
    RS=Metagenomic16SRepresentation(fasta_files, mapping, 3000, 10)
    RS.train_cpe_transformation(10000, 100,'datasets/dental/')

def cpe_bodysite():
    files=FileUtility.recursive_glob('/mounts/data/proj/asgari/github_repos/microbiomephenotype/data_config/bodysites/','*.txt')
    list_of_files=[]
    for file in files:
        print (file)
        list_of_files+=FileUtility.load_list(file)
    list_of_files=[x+'.fsa' for x in list_of_files]
    fasta_files, mapping = FileUtility.read_fasta_directory('/mounts/data/proj/asgari/dissertation/datasets/deepbio/microbiome/hmb_data/','fsa',only_files=list_of_files)
    RS=Metagenomic16SRepresentation(fasta_files, mapping, 3000, 10)
    RS.train_cpe_transformation(10000, 20,'datasets/bodysites/')  

if __name__=='__main__':
    dental()

