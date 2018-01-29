# MicroPheno
<h1 id="micropheno">MicroPheno</h1>
<table style="height: 48px; width: 812px;">
<tbody>
<tr>
<td style="width: 802px;">
<table style="width: 798px;">
<tbody>
<tr style="height: 57px;">
<td style="width: 945px; height: 57px;" colspan="2"><span style="font-size: 14pt; font-family: helvetica,arial,sans-serif;"><span style="color: #0000ff;"><strong>Predicting environments and host phenotypes from 16S rRNA gene sequencing using a k-mer based representation of shallow sub-samples</strong></span></span></td>
</tr>
<tr style="height: 230.867px;">
<td style="width: 331.533px; height: 230.867px;"><img class="wp-image-140 aligncenter" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/Micropheno_v2-226x300.png" alt="" width="154" height="205" />

<span style="font-family: helvetica,arial,sans-serif; font-size: 8pt;"><em>A microphone for microbes to speak out about their host phenotypes and environments</em></span></td>
<td style="width: 613.467px; height: 230.867px;"><span style="color: #800000; font-size: 14pt;"><strong>MicroPheno </strong></span>is a reference- and alignment-free approach for predicting the environment or host phenotype from microbial community samples based on k-mer distributions in shallow sub-samples of 16S rRNA data.

<span style="color: #ff0000;">MicroPheno's paper is still under review. Please cite the <a href="https://www.biorxiv.org/content/early/2018/01/28/255018">bioarXiv</a> version:</span>
<div class="gmail_default"><span style="font-size: 10pt; color: #000000;">Asgari E, Garakani K, McHardy AC and Mofrad MRK (2018) </span></div>
<div class="gmail_default"><span style="font-size: 10pt; color: #000000;"><strong>MicroPheno: Predicting environments and host phenotypes from 16S rRNA gene sequencing using a k-mer based representation of shallow sub-samples</strong>. <em>bioRxiv.</em></span></div>
<div class="gmail_default"><span style="font-size: 10pt; color: #000000;"> Available at: https://www.biorxiv.org/content/early/2018/01/28/255018.</span></div>
<div> <a href="https://www.biorxiv.org/highwire/citation/78275/bibtext"><img class="alignnone wp-image-142" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/bibtex-icon.png" alt="" width="44" height="44" /></a> <a href="https://www.biorxiv.org/highwire/citation/78275/mendeley"><img class="alignnone wp-image-143" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/Apps-Mendeley-icon-150x150.png" alt="" width="47" height="41" /></a></div></td>
</tr>
</tbody>
</table>
<strong>The implementation</strong> is available at <a href="https://github.com/ehsanasgari/MicroPheno"><img class="alignnone size-full wp-image-85" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/github-e1516824281561.png" alt="" width="50" height="50" /></a> with detailed <a href="https://github.com/ehsanasgari/MicroPheno/tree/master/notebooks">ipython notebooks</a> and a command-line interface.

<strong>The datasets </strong> are also available for download <img class="alignnone size-full wp-image-36" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/zip.png" alt="" width="50" height="50" />.

<strong><span style="color: #800000;">This repository and datasets will be publicized after acceptance of the journal paper.</span></strong>

<hr />

<span style="font-family: helvetica,arial,sans-serif; font-size: 24pt;"><strong>Summary</strong></span>

&nbsp;

<img class="alignnone wp-image-112 size-large" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/Screen-Shot-2018-01-24-at-11.13.26-PM-1024x256.png" alt="" width="960" height="240" />

<span style="font-family: helvetica,arial,sans-serif;"><strong><span style="font-size: 24pt;">M</span>otivation:</strong> Microbial communities play important roles in the function and maintenance of various biosystems, ranging from the human body to the environment. A major challenge in microbiome research is the classification of microbial communities of different environments or host phenotypes. The most common and cost-effective approach for such studies to date is 16S rRNA gene sequencing. Recent falls in sequencing costs have increased the demand for simple, efficient, and accurate methods for rapid detection or diagnosis with proved applications in medicine, agriculture, and forensic science. We describe a reference- and alignment-free approach for predicting environments and host phenotypes from 16S rRNA gene sequencing based on k-mer representations that benefits from a bootstrapping framework for investigating the sufficiency of shallow sub-samples. Deep learning methods as well as classical approaches were explored for predicting environments and host phenotypes. </span>

<span style="font-family: helvetica,arial,sans-serif;"><strong><span style="font-size: 24pt;">R</span>esults:</strong> k-mer distribution of shallow sub-samples outperformed the computationally costly Operational Taxonomic Unit (OTU) features in the tasks of body-site identification and Crohn's disease prediction. Aside from being more accurate, using k-mer features in shallow sub-samples allows (i) skipping computationally costly sequence alignments required in OTU-picking, and (ii) provided a proof of concept for the sufficiency of shallow and short-length 16S rRNA sequencing for phenotype prediction. In addition, k-mer features predicted representative 16S rRNA gene sequences of 18 ecological environments, and 5 organismal environments with high macro-F1 scores of 0.88 and 0.87. For large datasets, deep learning outperformed classical methods such as Random Forest and SVM.</span>

<hr />

<table style="height: 48px; width: 812px;">
<tbody>
<tr>
<td style="width: 802px;">
<table style="width: 802px;">
<tbody>
<tr>
<td style="width: 450px;" colspan="2"><span style="font-size: 14pt; font-family: helvetica,arial,sans-serif;"><span style="color: #0000ff;"><strong>Predicting environments and host phenotypes from 16S rRNA gene sequencing using a k-mer based representation of shallow sub-samples</strong></span></span></td>
</tr>
<tr>
<td style="width: 450px;"><img class="alignnone size-medium wp-image-82" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/Microphenp-256x300.png" alt="" width="450" height="300" /></td>
<td style="width: 500px;"><span style="font-family: helvetica,arial,sans-serif;"><span style="color: #800000; font-size: 14pt;"><strong>MicroPheno </strong></span>is a reference- and alignment-free approach for predicting the environment or host phenotype from microbial community samples based on k-mer distributions in shallow sub-samples of 16S rRNA data.</span></td>
</tr>
</tbody>
</table>
<strong>The implementation</strong> is available on <a href="https://github.com/ehsanasgari/MicroPheno"><img class="alignnone size-full wp-image-85" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/github-e1516824281561.png" alt="" width="50" height="50" /></a> with detailed <a href="https://github.com/ehsanasgari/MicroPheno/tree/master/notebooks">ipython notebooks</a> command-line interface.

<strong>The datasets </strong> are also available for download <img class="alignnone size-full wp-image-36" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/zip.png" alt="" width="50" height="50" />.

<hr />

<span style="font-family: helvetica,arial,sans-serif; font-size: 24pt;"><strong>Summary</strong></span>

&nbsp;

<img class="alignnone wp-image-112 size-large" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/Screen-Shot-2018-01-24-at-11.13.26-PM-1024x256.png" alt="" width="960" height="240" />

<span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;"><span style="font-size: 36pt;"><strong>M</strong></span>icrobial communities play important roles in the function and maintenance of various biosystems, ranging from the human body to the environment. A major challenge in microbiome research is the classification of microbial communities of different environments or host phenotypes. The most common and cost-effective approach for such studies to date is 16S rRNA gene sequencing. Recent falls in sequencing costs increased the demand for simple, efficient, and accurate methods for rapid detection or diagnosis with proved applications in medicine, agriculture, and forensic science. Here we propose MicroPheno to facilitate environments and host phenotype prediction from 16S rRNA gene sequences:</span>
<ul>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;"> We propose a bootstrapping framework to investigate the sufficiency of a shallow sub-sample for prediction. </span>
<ul>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;"> We showed that a shallow sub-sample of 16S rRNA samples alone can be sufficient for producing a proper k-mer representation of data. Aside from being more accurate, using k-mer features in shallow sub-samples provided the following benefits: </span>
<ul>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;">(i) skipping computationally costly sequence alignments required in OTU-picking, </span></li>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;">(ii) proof of concept for the sufficiency of a shallow and short-length 16S rRNA sequencing for environment/host phenotype prediction.</span></li>
</ul>
</li>
</ul>
</li>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;"> We study the use of deep learning methods as well as classic machine learning approaches for distinguishing among human body-sites, diagnosis of Crohn's disease, and predicting the environments (18 ecological and 5 organismal environments) from represetnative 16S sequences. </span>
<ul>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;">We demonstrated that k-mer representations outperform Operational Taxonomic Unit (OTU) features in distinguishing among 5 major body-sites as well as predicting Crohn's disease using 16S rRNA sequencing samples. </span></li>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;">In addition, k-mer features were able to accurately predict representative sequences of 18 ecological and 5 organismal environments with relatively high macro-F1 scores. </span></li>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;">Deep Neural Network outperformed Random Forest and Support Vector Machine in classification of large datasets.</span></li>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;">We explore the use of unsupervised dimensionality reduction methods as well as supervised deep representation learning for visualizing microbial data of different environments and host phenotypes. </span></li>
</ul>
</li>
</ul>

<hr />

&nbsp;</td>
</tr>
</tbody>

</table>


<h1>Installation</h1>

MicroPheno is implemented in Python3.x and uses ScikitLearn and Keras frameworks for machine learning. To install the dependencies use the following command:
```
pip install -r requirements.txt
```

Please cite the MicroPheno if you use this tool:



<h1> User Manual </h1>
You may use MicroPheno either using the templates provided in the <a href="https://github.com/ehsanasgari/MicroPheno/tree/master/notebooks">ipython notebooks</a> or the command-line interface.

<h2>Bootstrapping</h2>
An example of bootstrapping provided in the <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/1.Bootstrapping.ipynb">notebooks</a>.

<b>Command line use:</b> Argument to be used are the input/output directories, the sequence filetype, the k-mers and the sample size. Use argument '-h' to see the helpers.
```
python3 micropheno.py --bootstrap --indir /path/to/16srRNAsamples/ --out output_dir/ --filetype fastq --kvals 3,4,5,6 --nvals 10,100,200,500,1000 --name crohs
```
The output would be generating the following plot in the output directory. the <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/1.Bootstrapping.ipynb">notebooks</a> for more details.
![bootstrapping](https://user-images.githubusercontent.com/8551117/35446008-af953ad6-02b3-11e8-9b33-06d1f4b429f3.png)


<h2>Representation Creation</h2>
Two examples of representation creation are provided in the <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/2.%20k-mer%20Representation%20Creation%20with%20sub-sampling%20or%20without.ipynb">notebooks</a>, one with sampling from sequence files and one for representative sequences.

<b>Command line use:</b> Argument to be used are the input/output directories, the sequence filetype, the k-mers and their sample size as well as number of cores to be used. Use argument '-h' to see the helpers.

```
python3 micropheno.py --genkmer --inaddr /path/to/16srRNAsamples/ --out output_dir/ --filetype fastq --cores 20 --KN 6:100,6:1000,2:100 --name test_crohn
```

<h2>Classification with Random Forest and SVM</h2>

You can use the trained represenation in the previous step for classification.
See <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/3.%20Classification_classical_classifiers.ipynb" > this notebooks</a>.

<b>Command line use:</b> Argument to be used are the X and Y, the classification algorithm (SVM, or RF), output directory as well as number of cores to be used. Use argument '-h' to see the helpers.

The following command will do tuning the parameters as well as evaluation within a 10xFold corss-validation scheme. Details on how to parse the results (scores, confusion matrix, best estimator, etc) is provided <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/3.%20Classification_classical_classifiers.ipynb" > here</a>.

```
python3 micropheno.py --train_predictor --model RF (or SVM) --x k-mer.npz --y labels_phenotypes.txt --cores 20 --name test_crohn  --out output_dir/
```

<h2>Classification with Deep Neural Network</h2>

 We use the Multi-Layer-Perceptrons (MLP) Neural Network architecture with several hidden layers using Rectified Linear Unit (ReLU) as the nonlinear activation function. We use softmax activation function at the last layer to produce the probability vector that can be regarded as representing posterior probabilities (Goodfellow-et-al-2016). To avoid overfitting we perform early stopping and also use dropout at hidden layers (Srivastava2014). A schematic visualization of our Neural Networks is depicted in the Figure.

![dnn](https://user-images.githubusercontent.com/8551117/35446216-4ec1eb7c-02b4-11e8-9421-043ec1f9ed96.png)

Our objective is minimizing the loss, i.e. cross entropy between output and the one-hot vector representation of the target class. The error (the distance between the output and the target) is used to update the network parameters via a Back-propagation algorithm using Adaptive Moment Estimation (Adam) as optimizer (Kingma2015).

You can see an example in the notebooks <a href='https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/4.%20Classification%20Deep%20Learning.ipynb'>here</a>, showing how to see the learning curves and also getting the activation function of the neural network from the trained model.

<b>Command line use:</b> Argument to be used are the X and Y, the DNN flag, the neural network architecture (hidden-sizes and dropouts), batch size, number of epochs, output directory as well as the GPU id to be used. Use argument '-h' to see the helpers.

```
python3 micropheno.py --train_predictor --model DNN --arch  --batchsize 10 --epochs  100 --x k-mer.npz --y labels_phenotypes.txt --name test_crohn  --out output_dir/
```


<h2>Visualization</h2>

An example of visualization using PCA, t-SNE, as well as t-SNE over the activation function of the last layer of the neural network is provided in <a href="https://github.com/ehsanasgari/MicroPheno/blob/master/notebooks/5.%20Visualization.ipynb">this notebook</a>.



![vis](https://user-images.githubusercontent.com/8551117/35447281-8f58b064-02b7-11e8-9a97-affe35573ba5.png)


