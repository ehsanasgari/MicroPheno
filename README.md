# MicroPheno
<table style="height: 48px; width: 812px;">
<table style="width: 802px;">
<tbody>
<tr>
<td style="width: 450px;" colspan="2"><span style="font-size: 14pt; font-family: helvetica,arial,sans-serif;"><span style="color: #0000ff;"><strong>Predicting environments and host phenotypes from 16S rRNA gene sequencing using a k-mer based representation of shallow sub-samples</strong></span></span></td>
</tr>
<tr>
<td style="width: 450px;"><img class="alignnone size-medium wp-image-82" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/Micropheno_v2-226x300.png" alt="" width="450" height="300" /></td>
<td style="width: 500px;"><span style="font-family: helvetica,arial,sans-serif;"><span style="color: #800000; font-size: 14pt;"><strong>MicroPheno </strong></span>is a reference- and alignment-free approach for predicting the environment or host phenotype from microbial community samples based on k-mer distributions in shallow sub-samples of 16S rRNA data.</span>
	<small>".. And a microphone for microbes to speak out about their host phenotypes and environments"</small>
	</td>
</tr>
</tbody>
</table>

<span style="color: #800000;"><strong>MicroPheno's paper is still under review. Please cite the <a style="color: #800000;" href="https://www.biorxiv.org/content/early/2018/01/28/255018">bioarXiv</a> version  <a href="https://www.biorxiv.org/highwire/citation/78275/bibtext"><img class="alignnone wp-image-142" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/bibtex-icon.png" alt="" width="44" height="44" /></a> <a href="https://www.biorxiv.org/highwire/citation/78275/mendeley"><img class="alignnone wp-image-143" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/Apps-Mendeley-icon-150x150.png" alt="" width="47" height="41" /></a>:
 
Asgari E, Garakani K, McHardy AC and Mofrad MRK, MicroPheno: Predicting environments and host phenotypes from 16S rRNA gene sequencing using a k-mer based representation of shallow sub-samples. bioRxiv, 2018. Available at: https://www.biorxiv.org/content/early/2018/01/28/255018.


 
 The datasets </strong> are also available for download <img class="alignnone wp-image-36" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/zip.png" alt="" width="33" height="33" />.

<strong>Contact</strong>: Ehsaneddin Asgari (<span style="color: #0000ff;">asgari [at] berkeley [dot] edu</span>)
<br/>
<strong>Project page:</strong>: <a href="http://llp.berkeley.edu/micropheno">http://llp.berkeley.edu/micropheno</a>
<hr />

<span style="font-family: helvetica,arial,sans-serif; font-size: 24pt;"><strong>Summary</strong></span>

&nbsp;

<img class="alignnone wp-image-112 size-large" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/Screen-Shot-2018-01-24-at-11.13.26-PM-1024x256.png" alt="" width="960" height="240" />

<span style="font-family: helvetica,arial,sans-serif;"><strong><span style="font-size: 24pt;">M</span>otivation:</strong> Microbial communities play important roles in the function and maintenance of various biosystems, ranging from the human body to the environment. A major challenge in microbiome research is the classification of microbial communities of different environments or host phenotypes. The most common and cost-effective approach for such studies to date is 16S rRNA gene sequencing. Recent falls in sequencing costs have increased the demand for simple, efficient, and accurate methods for rapid detection or diagnosis with proved applications in medicine, agriculture, and forensic science. We describe a reference- and alignment-free approach for predicting environments and host phenotypes from 16S rRNA gene sequencing based on k-mer representations that benefits from a bootstrapping framework for investigating the sufficiency of shallow sub-samples. Deep learning methods as well as classical approaches were explored for predicting environments and host phenotypes. </span>

<span style="font-family: helvetica,arial,sans-serif;"><strong><span style="font-size: 24pt;">R</span>esults:</strong> k-mer distribution of shallow sub-samples outperformed the computationally costly Operational Taxonomic Unit (OTU) features in the tasks of body-site identification and Crohn's disease prediction. Aside from being more accurate, using k-mer features in shallow sub-samples allows (i) skipping computationally costly sequence alignments required in OTU-picking, and (ii) provided a proof of concept for the sufficiency of shallow and short-length 16S rRNA sequencing for phenotype prediction. In addition, k-mer features predicted representative 16S rRNA gene sequences of 18 ecological environments, and 5 organismal environments with high macro-F1 scores of 0.88 and 0.87. For large datasets, deep learning outperformed classical methods such as Random Forest and SVM.</span>

&nbsp;</td>
</tr>
</tbody>

</table>


<h1>Installation</h1>

MicroPheno is implemented in Python3.x and uses ScikitLearn and Keras frameworks for machine learning. To install the dependencies use the following command:
```
pip install -r requirements.txt
```

Please cite the <a style="color: #800000;" href="https://www.biorxiv.org/content/early/2018/01/28/255018">bioarXiv</a> version  <a href="https://www.biorxiv.org/highwire/citation/78275/bibtext"><img class="alignnone wp-image-142" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/bibtex-icon.png" alt="" width="44" height="44" /></a> <a href="https://www.biorxiv.org/highwire/citation/78275/mendeley"><img class="alignnone wp-image-143" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/Apps-Mendeley-icon-150x150.png" alt="" width="47" height="41" /></a>

```
@article {Asgari255018,
	author = {Asgari, Ehsaneddin and Garakani, Kiavash and McHardy, Alice Carolyn and Mofrad, Mohammad R.K.},
	title = {MicroPheno: Predicting environments and host phenotypes from 16S rRNA gene sequencing using a k-mer based representation of shallow sub-samples},
	year = {2018},
	doi = {10.1101/255018},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2018/01/28/255018},
	eprint = {https://www.biorxiv.org/content/early/2018/01/28/255018.full.pdf},
	journal = {bioRxiv}
}

```

<h1> User Manual </h1>
MicroPheno can be used either via the templates provided in the <a href="https://github.com/ehsanasgari/MicroPheno/tree/master/notebooks">ipython notebooks</a> or the command-line interface.

<h2>Bootstrapping</h2>
An example of bootstrapping provided in the <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/1.Bootstrapping.ipynb">notebooks</a>.

<b>Command line use:</b> Argument to be used are the input/output directories, the sequence filetype, the k-mers and the sample size. Use argument '-h' to see the helpers.
```
python3 micropheno.py --bootstrap --indir /path/to/16srRNAsamples/ --out output_dir/ --filetype fastq --kvals 3,4,5,6 --nvals 10,100,200,500,1000 --name crohs
```
The output would be generating the following plot in the specified output directory. See the related <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/1.Bootstrapping.ipynb">notebook</a> for more details.
<img src="https://user-images.githubusercontent.com/8551117/35446008-af953ad6-02b3-11e8-9b33-06d1f4b429f3.png" alt="bootstrapping" />


<h2>Representation Creation</h2>
Two examples of representation creation are provided in the <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/2.%20k-mer%20Representation%20Creation%20with%20sub-sampling%20or%20without.ipynb">notebooks</a>, one with sampling from sequence files and the other for mapping the representative sequences.

<b>Command line use:</b> Argument to be used are the input/output directories, the sequence filetype, the k-mers and their sample size as well as number of cores to be used. Use argument '-h' to see the helpers.

```
python3 micropheno.py --genkmer --inaddr /path/to/16srRNAsamples/ --out output_dir/ --filetype fastq --cores 20 --KN 6:100,6:1000,2:100 --name test_crohn
```

<h2>Classification with Random Forest and SVM</h2>

The trained representation in the previous step in the input for classification.
See an example in the<a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/3.%20Classification_classical_classifiers.ipynb"> notebooks</a>.

<b>Command line use:</b> Argument to be used are the X and Y, the classification algorithm (SVM, or RF), output directory as well as number of cores to be used. Use argument '-h' to see the helpers.

The following command will do tuning the parameters as well as evaluation within a 10xFold corss-validation scheme. Details on how to parse the results (scores, confusion matrix, best estimator, etc) are provided <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/3.%20Classification_classical_classifiers.ipynb"> here</a>.

```
python3 micropheno.py --train_predictor --model RF (or SVM) --x k-mer.npz --y labels_phenotypes.txt --cores 20 --name test_crohn  --out output_dir/
```

<h2>Classification with Deep Neural Network</h2>
We use the Multi-Layer-Perceptrons (MLP) Neural Network architecture with several hidden layers using Rectified Linear Unit (ReLU) as the nonlinear activation function. We use softmax activation function at the last layer to produce the probability vector that can be regarded as representing posterior probabilities (Goodfellow-et-al-2016). To avoid overfitting we perform early stopping and also use dropout at hidden layers (Srivastava2014). A schematic visualization of our Neural Networks is depicted in the Figure.

<img src="https://user-images.githubusercontent.com/8551117/35446216-4ec1eb7c-02b4-11e8-9421-043ec1f9ed96.png" alt="dnn" />

Our objective is minimizing the loss, i.e. cross entropy between output and the one-hot vector representation of the target class. The error (the distance between the output and the target) is used to update the network parameters via a Back-propagation algorithm using Adaptive Moment Estimation (Adam) as optimizer (Kingma2015).

You can see an example in the notebooks <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/4.%20Classification%20Deep%20Learning.ipynb">here</a>, showing how to see the learning curves and also getting the activation function of the neural network from the trained model.

<b>Command line use:</b> Argument to be used are the X and Y, the DNN flag, the neural network architecture (hidden-sizes and dropouts), batch size, number of epochs, output directory as well as the GPU id to be used. Use argument '-h' to see the helpers.

```
python3 micropheno.py --train_predictor --model DNN --arch  --batchsize 10 --epochs  100 --x k-mer.npz --y labels_phenotypes.txt --name test_crohn  --out output_dir/
```


<h2>Visualization</h2>

An example of visualization using PCA, t-SNE, as well as t-SNE over the activation function of the last layer of the neural network is provided in <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/5.%20Visualization.ipynb">this notebook</a>.


![vis](https://user-images.githubusercontent.com/8551117/35447281-8f58b064-02b7-11e8-9a97-affe35573ba5.png)


