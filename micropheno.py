#! /usr/bin/python

# -*- coding: utf-8 -*-
__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu or ehsaneddin.asgari@helmholtz-hzi.de"
__project__ = "LLP - MicroPheno"
__website__ = "https://llp.berkeley.edu/micropheno/"

import argparse
import os
import os.path
import re
import sys

from bootstrapping.bootstrapping import BootStrapping
from utility.file_utility import FileUtility


class MicroPheno:
    def __init__(self):
        '''
            MicroPheno commandline use
            For interactive interface please see the ipython notebooks
            in the notebook directory
        '''
        print('MicroPheno 1.0.0 == HTTP://LLP.BERKELEY.EDU/MICROPHENO')

    @staticmethod
    def bootstrapping(inp_dir, out_dir, dataset_name, filetype='fastq', k_values=[3, 4, 5, 6, 7, 8],
                      sampling_sizes=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]):
        '''
        :param inp_dir:
        :param out_dir:
        :param filetype:
        :param k_values:
        :param sampling_sizes:
        :return:
        '''
        fasta_files, mapping = FileUtility.read_fasta_directory(inp_dir, filetype)
        BS = BootStrapping(fasta_files, out_dir, seqtype=filetype, sampling_sizes=sampling_sizes,
                           n_resamples=10, M=10)

        for k in k_values:
            print(k, '-mer bootstrapping started')
            BS.add_kmer_sampling(k)
            print(k, '-mer bootstrapping completed')

        BS.plotting('results_bootstrapping' + '_' + dataset_name, dataset_name)

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
    parser.add_argument('--bootstrap', action='store', dest='input_dir_bootstrapping', default=False, type=str,
                        help='directory of 16S rRNA samples')

    parser.add_argument('--filetype', action='store', dest='filetype', type=str, default='fastq',
                        help='fasta fsa fastq etc')
    parser.add_argument('--kvals', action='store', dest='kvals', type=str, default='3,4,5,6,7,8',
                        help='Comma separated k-mer values 2,3,4,5,6')
    parser.add_argument('--nvals', action='store', dest='nvals', type=str,
                        default='10,20,50,100,200,500,1000,2000,5000,10000', help='Comma separated sample sizes')

    parser.add_argument('--genrep', action='store', dest='input_addr', default=False, type=str,
                        help='Generate representations for input fasta file or directory of 16S rRNA samples')

    parser.add_argument('--KN', action='store', dest='K_N', default=None, type=str,
                        help='pair of comma separated Kmer:sub-sample-size ==> 2:100,6:-1 (N=-1 means using all sequences)')

    parser.add_argument('--out', action='store', dest='output_addr', type=str, default='out', help='Out put directory')

    parser.add_argument('--in', action='store', dest='input_addr', type=str, default=None,
                        help='Input fasta file or directory of samples')
    parser.add_argument('--name', action='store', dest='data_name', type=str, default=None, help='name of the dataset')

    parsedArgs = parser.parse_args()

    if parsedArgs.input_dir_bootstrapping:
        '''
            bootstrapping functionality
        '''
        print('Bootstrapping requested..\n')
        if (not os.access(parsedArgs.input_dir_bootstrapping, os.F_OK)):
            err = err + "\nError: Permission denied or could not find the directory!"
            return err
        else:
            try:
                os.stat(parsedArgs.output_addr)
            except:

                os.mkdir(parsedArgs.output_addr)

            if len(FileUtility.recursive_glob(parsedArgs.input_dir_bootstrapping, '*'+parsedArgs.filetype))==0:
                err = err + "\nThe filetype "+parsedArgs.filetype+" could not find the directory!"
                return err

            if not parsedArgs.output_addr:
                parsedArgs.data_name=parsedArgs.input_dir_bootstrapping.split('/')[-1]

            try:
                k_values=[int(x) for x in parsedArgs.kvals.split(',')]
                n_values=[int(x) for x in parsedArgs.nvals.split(',')]
            except:
                err = err + "\n k-mers or sampling sizes are not fed correctly; see the help with -h!"
                return err
            MicroPheno.bootstrapping( parsedArgs.input_dir_bootstrapping, parsedArgs.output_addr, parsedArgs.output_addr, filetype = parsedArgs.filetype, k_values = k_values, sampling_sizes = n_values)

    else:
        err = err + "\nError: You need to specify an input corpus file!"
        print('others')

    return False

if __name__ == '__main__':
    err = checkArgs(sys.argv)
    if  err :
        print(err)
        exit()
