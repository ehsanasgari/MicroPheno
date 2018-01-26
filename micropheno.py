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
import sys

from bootstrapping.bootstrapping import BootStrapping
from make_representations.representation_maker import Metagenomic16SRepresentation
from utility.file_utility import FileUtility
from classifier.classical_classifiers import RFClassifier,SVM
from classifier.DNN import DNNMutliclass16S

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

    @staticmethod
    def representation_creation_dir(inp_dir, out_dir, dataset_name, num_p, filetype='fastq',
                                    sampling_dict={3: [20], 4: [100], 5: [500], 6: [100, 1000, 2000, 5000, 10000, -1],
                                                   7: [5000], 8: [8000]}):

        fasta_files, mapping = FileUtility.read_fasta_directory(inp_dir, filetype)

        for k in sampling_dict.keys():
            for N in sampling_dict[k]:
                print(k, '-mers with sampling size ', N)
                RS = Metagenomic16SRepresentation(fasta_files, mapping, N, num_p)
                # path to save the generated files
                RS.generate_kmers_all(k, save=out_dir + '_'.join([dataset_name, str(k) + '-mers', str(N)]))

    @staticmethod
    def classical_classifier(X_file, Y_file, model, out_dir, dataset_name, cores):
        #
        X=FileUtility.load_sparse_csr(X_file)
        # labels
        Y=FileUtility.load_list(Y_file)

        if model=='RF':
            #### Random Forest classifier
            MRF = RFClassifier(X, Y)
            # results containing the best parameter, confusion metrix, best estimator, results on fold will be stored in this address
            MRF.tune_and_eval(out_dir+'/classification_results_'+dataset_name, n_jobs=cores)
        else:
            #### Support Vector Machine classifier
            MSVM = SVM(X, Y)
            # results containing the best parameter, confusion metrix, best estimator, results on fold will be stored in this address
            MSVM.tune_and_eval(out_dir+'/classification_results_'+dataset_name, n_jobs=cores)

    @staticmethod
    def DNN_classifier(X_file, Y_file, arch, out_dir, dataset_name, gpu_id, epochs, batch_size):
        # k-mer data
        X=FileUtility.load_sparse_csr(X_file).toarray()
        # labels
        Y=FileUtility.load_list(Y_file)
        DNN=DNNMutliclass16S(X,Y,model_arch=arch)
        DNN.cross_validation(out_dir+'nn_classification_results_'+dataset_name, gpu_dev=gpu_id, n_fold=10, epochs=epochs, batch_size=batch_size, model_strct='mlp')


def checkArgs(args):
    '''
        This function checks the input arguments and returns the errors (if exist) otherwise reads the parameters
    '''
    # keep all errors
    err = "";
    # Using the argument parser in case of -h or wrong usage the correct argument usage
    # will be prompted
    parser = argparse.ArgumentParser()

    # top level ######################################################################################################
    parser.add_argument('--bootstrapping', action='store_true', help='To enable classification and parameter tuning')

    parser.add_argument('--genkmer', action='store_true',
                        help='To enable generation of representations for input fasta file or directory of 16S rRNA samples')

    parser.add_argument('--train_predictor', action='store_true', help='To enable classification and parameter tuning')

    # boot strapping #################################################################################################
    parser.add_argument('--indir', action='store', dest='input_dir_bootstrapping', default=False, type=str,
                        help='bootstrapping: directory of 16S rRNA samples', required='--bootstrapping' in sys.argv)

    # generate k-mers ################################################################################################
    parser.add_argument('--inaddr', action='store', dest='genrep_input_addr', default=False, type=str,
                        help='genkmer: Generate representations for input fasta file or directory of 16S rRNA samples',
                        required='--genkmer' in sys.argv)

    # classification ################################################################################################

    parser.add_argument('--x', action='store', dest='X', type=str, default=False,
                        help='train_predictor: The data in the npy format rows are instances and columns are features')

    parser.add_argument('--y', action='store', dest='Y', type=str, default=False,
                        help='train_predictor: The labels associated with the rows of classifyX, each line is a associated with a row')

    parser.add_argument('--model', action='store', dest='model', type=str, default=False,
                        choices=[False, 'RF', 'SVM', 'DNN'],
                        help='train_predictor: choice of classifier from RF, SVM, DNN')

    parser.add_argument('--batchsize', action='store', dest='batch_size', type=int, default=10,
                        help='train_predictor-model/DNN: batch size for deep learning')

    parser.add_argument('--gpu_id', action='store', dest='gpu_id', type=str, default='0',
                        help='train_predictor-model/DNN: GPU id for deep learning')

    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=100,
                        help='train_predictor-model/DNN: number of epochs for deep learning')


    parser.add_argument('--arch', action='store', dest='dnn_arch', type=str, default='1024,0.2,512',
                        help='train_predictor-model/DNN: The comma separated definition of neural network layers connected to eahc other, you do not need to specify the input and output layers, values between 0 and 1 will be considered as dropouts')

    # general to bootstrap  and rep ##################################################################################
    parser.add_argument('--filetype', action='store', dest='filetype', type=str, default='fastq',
                        help='fasta fsa fastq etc')

    # bootstrap ################################################################################
    parser.add_argument('--kvals', action='store', dest='kvals', type=str, default='3,4,5,6,7,8',
                        help='Comma separated k-mer values 2,3,4,5,6')

    parser.add_argument('--nvals', action='store', dest='nvals', type=str,
                        default='10,20,50,100,200,500,1000,2000,5000,10000', help='Comma separated sample sizes')

    # rep / classifier ################################################################################
    parser.add_argument('--cores', action='store', dest='cores', default=4, type=int,
                        help='Number of cores to be used')

    # rep ##################################################################################
    parser.add_argument('--KN', action='store', dest='K_N', default=None, type=str,
                        help='pair of comma separated Kmer:sub-sample-size ==> 2:100,6:-1 (N=-1 means using all sequences)')

    parser.add_argument('--out', action='store', dest='output_addr', type=str, default='out', help='Out put directory')

    parser.add_argument('--in', action='store', dest='input_addr', type=str, default=None,
                        help='Input fasta file or directory of samples')

    parser.add_argument('--name', action='store', dest='data_name', type=str, default=None, help='name of the dataset')

    parsedArgs = parser.parse_args()

    if parsedArgs.bootstrapping:
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

            if len(FileUtility.recursive_glob(parsedArgs.input_dir_bootstrapping, '*' + parsedArgs.filetype)) == 0:
                err = err + "\nThe filetype " + parsedArgs.filetype + " could not find the directory!"
                return err

            if not parsedArgs.data_name:
                parsedArgs.data_name = parsedArgs.input_dir_bootstrapping.split('/')[-1]

            try:
                k_values = [int(x) for x in parsedArgs.kvals.split(',')]
                n_values = [int(x) for x in parsedArgs.nvals.split(',')]
            except:
                err = err + "\n k-mers or sampling sizes are not fed correctly; see the help with -h!"
                return err
            MicroPheno.bootstrapping(parsedArgs.input_dir_bootstrapping, parsedArgs.output_addr, parsedArgs.data_name,
                                     filetype=parsedArgs.filetype, k_values=k_values, sampling_sizes=n_values)
        return False

    if parsedArgs.genkmer:
        '''
            Representation creation functionality
        '''
        if (not os.access(parsedArgs.genrep_input_addr, os.F_OK)):
            err = err + "\nError: Permission denied or could not find the directory!"
            return err
        elif os.path.isdir(parsedArgs.genrep_input_addr):
            print('Representation creation requested for directory ' + parsedArgs.genrep_input_addr + '\n')
            try:
                os.stat(parsedArgs.output_addr)
            except:
                os.mkdir(parsedArgs.output_addr)

            if len(FileUtility.recursive_glob(parsedArgs.genrep_input_addr, '*' + parsedArgs.filetype)) == 0:
                err = err + "\nThe filetype " + parsedArgs.filetype + " could not find the directory!"
                return err

            if not parsedArgs.data_name:
                parsedArgs.data_name = parsedArgs.input_dir_bootstrapping.split('/')[-1]

            try:
                sampling_dict = dict()
                for x in parsedArgs.K_N.split(','):
                    k, n = x.split(':')
                    k = int(k)
                    n = int(n)
                    if k in sampling_dict:
                        sampling_dict[k].append(n)
                    else:
                        sampling_dict[k] = [n]
            except:
                err = err + "\nWrong format for KN (k-mer sample sizes)!"
                return err

            MicroPheno.representation_creation_dir(parsedArgs.genrep_input_addr, parsedArgs.output_addr,
                                                   parsedArgs.data_name, parsedArgs.cores, filetype=parsedArgs.filetype,
                                                   sampling_dict=sampling_dict)
        else:
            print('Representation creation requested for file ' + parsedArgs.genrep_input_addr + '\n')

    if parsedArgs.train_predictor:
        print('Classification and parameter tuning requested..\n')
        if not parsedArgs.model:
            err = err + "\nNo classification model is specified"
        if (not os.access(parsedArgs.X, os.F_OK)):
            err = err + "\nError: Permission denied or could not find the X!"
            return err
        if (not os.access(parsedArgs.Y, os.F_OK)):
            err = err + "\nError: Permission denied or could not find the Y!"
            return err
        else:
            try:
                os.stat(parsedArgs.output_addr)
            except:
                os.mkdir(parsedArgs.output_addr)
                print (parsedArgs.output_addr ,' directory created')

        if not parsedArgs.data_name:
            parsedArgs.data_name = parsedArgs.X.split('/')[-1].split('.')[0]


        if parsedArgs.model=='DNN':
            '''
                Deep learning
            '''
            arch=[int(layer) if float(layer)>1 else float(layer) for layer in parsedArgs.dnn_arch.split(',')]
            MicroPheno.DNN_classifier(parsedArgs.X, parsedArgs.Y, arch, parsedArgs.output_addr, parsedArgs.data_name, parsedArgs.gpu_id,parsedArgs.epochs, parsedArgs.batch_size)
        else:
            '''
                SVM and Random Forest
            '''
            if parsedArgs.model in ['SVM','RF']:
                MicroPheno.classical_classifier( parsedArgs.X,  parsedArgs.Y, parsedArgs.model, parsedArgs.output_addr, parsedArgs.data_name, parsedArgs.cores)
            else:
                return  "\nNot able to recognize the model!"

    else:
        err = err + "\nError: You need to specify an input corpus file!"
        print('others')

    return False


if __name__ == '__main__':
    err = checkArgs(sys.argv)
    if err:
        print(err)
        exit()
