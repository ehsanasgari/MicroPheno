#! /usr/bin/python

# -*- coding: utf-8 -*-
__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu or ehsaneddin.asgari@helmholtz-hzi.de"
__project__ = "LLP - MicroPheno"
__website__ = "https://llp.berkeley.edu/micropheno/"

import re
import sys
import os
import argparse
import os.path
import codecs
import random
from Bio import SeqIO
import numpy as np

class FastaCutter:
    def __init__(self, fasta_path, ngram_range):
        '''
        :param corpus: list of sentences
        '''
        f=dict()
        for i, ngram in enumerate(ngram_range):
            f[i]=codecs.open('/mounts/data/proj/asgari/other/swiss_notshuff_'+str(ngram)+'.txt','w','utf-8')

        #f_rand=codecs.open(fasta_path.split('.')[0]+'_rand.txt','w','utf-8')
        for cur_record in SeqIO.parse(fasta_path, "fasta") :
            seq=str(cur_record.seq).upper()
            for i, ngram in enumerate(ngram_range):
                f[i].write('\n'.join([' '.join(chops) for chops in self.generate_sent_ngrams_overlapping(seq, ngram)]))
                f[i].write('\n')
            #f_rand.write('\n'.join([' '.join(chops) for chops in self.generate_random_cutter(seq)]))
            #f_rand.write('\n')
        for i, ngram in enumerate(ngram_range):
            f[i].close()
        #f_rand.close()

    def generate_sent_ngrams_overlapping(self, sentence, n, whitespace_mark='@', padding=False):
        '''
        :param sentence: sentence t
        :param n: n of n-gram
        :param padding: to pad whitespaces before and after sentence
        :param cookie_cut: generate all ways or only overlapping
        :return:
        '''
        if whitespace_mark:
            sentence = re.sub(r"\s+", whitespace_mark, sentence)

        if padding:
            sentence = whitespace_mark * (n - 1) + sentence + whitespace_mark * (n - 1)

        # generate all ways of n-gram sequences
        all_ngrams = [(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]
        return [all_ngrams[i::n] for i in range(n)]

    def generate_random_cutter(self,sentence):
        result=[]
        for i in range(0,len(sentence)):
            seq_cut=[]
            cuts=np.random.randint(5,size=(1,len(sentence)))+2
            l=0
            count=0
            while (l<len(sentence)):
                seq_cut.append(sentence[l:min(l+cuts[count])])
                l=l+min(l+cuts[count])
                count+=1
            result.append(seq_cut)
        return result



def checkArgs(args):
    '''
        This function checks the input arguments and returns the errors (if exist) otherwise reads the parameters
    '''
    # keep all errors
    err = "";
    # Using the argument parser in case of -h or wrong usage the correct argument usage
    # will be prompted
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument('--in', action='store', dest='input_file', default='english_brown_corpus.txt', type=str,
                        help='Input corpus file')
    parser.add_argument('--n', action='store', dest='ngram', type=int, default=4, help='N in n-grams')
    parser.add_argument('--out', action='store', dest='output_file', type=str, default='output_ngram_corpus.txt',
                        help='Output corpus file')
    parser.add_argument('--shuffle', action='store', dest='shuffle_out', type=int, default=1,
                        help='Whether to shuffle the output or not')
    parsedArgs = parser.parse_args()

    if parsedArgs.input_file != None:
        if (not os.access(parsedArgs.input_file, os.F_OK)):
            err = err + "\nError: Permission denied or could not find the file!"
    else:
        err = err + "\nError: You need to specify an input corpus file!"
    return [err, parsedArgs.input_file, parsedArgs.ngram, parsedArgs.output_file, parsedArgs.shuffle_out, parsedArgs.isFasta]


if __name__ == '__main__':
    CC = FastaCutter('/mounts/Users/student/asgari/PycharmProjects/bioinformatics/embeddings/datasets/proteins/swissprot/swiss_prot.fasta',range(2,3))
