# MicroPheno

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

<img class="alignnone wp-image-112 size-large" src="http://llp.berkeley.edu/wp-content/uploads/2018/01/Screen-Shot-2018-01-24-at-11.13.26-PM-1024x256.png" alt="" width="960" height="240" />

<span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;"><span style="font-size: 36pt;"><strong>M</strong></span>icrobial communities play important roles in the function and maintenance of various biosystems, ranging from the human body to the environment. A major challenge in microbiome research is the classification of microbial communities of different environments or host phenotypes. The most common and cost-effective approach for such studies to date is 16S rRNA gene sequencing. Recent falls in sequencing costs increased the demand for simple, efficient, and accurate methods for rapid detection or diagnosis with proved applications in medicine, agriculture, and forensic science. Here we propose MicroPheno to facilitate environments and host phenotype prediction from 16S rRNA gene sequences:</span>
<ul>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;"> We propose a bootstrapping framework to investigate the sufficiency of a shallow sub-sample for prediction. </span>
<ul>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;"> We also showed that a shallow sub-sample of 16S rRNA samples alone can be sufficient for producing a proper k-mer representation of data. Aside from being more accurate, using k-mer features in shallow sub-samples provided the following benefits: (i) skipping computationally costly sequence alignments required in OTU-picking, (ii) proof of concept for the sufficiency of a shallow and short-length 16S rRNA sequencing for environment/host phenotype prediction.</span></li>
</ul>
</li>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;"> We study the use of deep learning methods as well as classic machine learning approaches for distinguishing among human body-sites, diagnosis of Crohn's disease, and predicting the environments (18 ecological and 5 organismal environments) from represetnative 16S sequences. </span>
<ul>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;">We demonstrated that k-mer representations outperform Operational Taxonomic Unit (OTU) features in distinguishing among 5 major body-sites as well as predicting Crohn's disease using 16S rRNA sequencing samples. </span></li>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;">In addition, k-mer features were able to accurately predict representative sequences of 18 ecological and 5 organismal environments with relatively high macro-F1 scores. </span></li>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;">Deep Neural Network outperformed Random Forest and Support Vector Machine in classification of large datasets.</span></li>
</ul>
</li>
 	<li><span style="font-family: helvetica, arial, sans-serif; font-size: 10pt; color: #333333;">We explore the use of unsupervised dimensionality reduction methods as well as supervised deep representation learning for visualizing microbial data of different environments and host phenotypes. </span></li>
</ul>

<hr />

</td>
</tr>
</tbody>
</table>

<h1>Installation</h1>

MicroPheno is implemented in Python3.x and uses ScikitLearn and Keras frameworks for machine learning. To install the dependencies use the following command:
```
pip install -r requirements.txt
```

Please cite the MicroPheno if you use this tool:


<h1> User manual </h1>

The users may use the examples provided in the notebooks or the command-line interface.
For the

<h2>Bootstrapping</h2>

See notebook . Alternatively you can use:
```
python3 micropheno.py --bootstrap /path/to/16srRNAsamples/ --out output_dir/ --filetype fastq --kvals 3,4,5,6 --nvals 10,100,200,500,1000 --name crohs
```

<h2>Representation Creation</h2>

See notebook . Alternatively you can use:
```
python3 micropheno.py --genrep /path/to/16srRNAsamples/ --out output_dir/ --filetype fastq --cores 20 --KN 6:100,6:1000,2:100 --name test_crohn
```


<h2>Classification with Random Forest and SVM</h2>

You can use the trained represenation in the previous step for classification.
See notebook . Alternatively you can use:
```
python3 micropheno.py --genrep /path/to/16srRNAsamples/ --out output_dir/ --filetype fastq --KN 6:100,6:1000,2:100 --name test_crohn
```


<h2>Classification with Deep Neural Network</h2>

See notebook . Alternatively you can use:
```
python3 micropheno.py --genrep /path/to/16srRNAsamples/ --out output_dir/ --filetype fastq --KN 6:100,6:1000,2:100 --name test_crohn
```

<h2>Visualization</h2>

See notebook . Alternatively you can use:
```
python3 micropheno.py --genrep /path/to/16srRNAsamples/ --out output_dir/ --filetype fastq --KN 6:100,6:1000,2:100 --name test_crohn
```






