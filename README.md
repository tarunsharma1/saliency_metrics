# saliency_metrics

PLEASE CHECK THE CODE YOURSELF BEFORE USING !!! DO NOT TRUST THIS REPO FOR RESEARCH.

I don't want to waste people's time so full disclosure : 
I did not get too much time to make sure the results are right, this was implemented in a day. But nevertheless can be used as foundation code with changes made if necessary. I am not sure if all methods are correct. So please check with some examples and check the implementation yourself. I repeat, do NOT trust all methods I am pretty sure some of them have some bugs. 

Metrics for comparing two salience maps or two fixation maps. These metrics are implemented in matlab here https://github.com/cvzoya/saliency/tree/master/code_forMetrics . This is just a reimplementation of the metrics in python. 
The salience benchmark (in matlab) and this code produce the same measures for the same set of inputs. 


Metrics implemented are ->
1. AUC Judd
2. AUC Borji
3. AUC shuffled
4. NSS
5. Info Gain
6. SIM 
7. CC
8. KL divergence

The two maps to be compared are gt and s_map. Each method also discretizes (if needed) and normalizes the maps. Details about each 
of the methods can be found in the MIT Salience Benchmark paper (What do different evaluation metrics tell us about saliency models?)




Citations -
@misc{mit-saliency-benchmark,
  author       = {Zoya Bylinskii and Tilke Judd and Fr{\'e}do Durand and Aude Oliva and Antonio Torralba},
  title        = {MIT Saliency Benchmark},
  howpublished = {http://saliency.mit.edu/}
}

@article{salMetrics_Bylinskii,
    title    = {What do different evaluation metrics tell us about saliency models?},
    author   = {Zoya Bylinskii and Tilke Judd and Aude Oliva and Antonio Torralba and Fr{\'e}do Durand},
    journal  = {arXiv preprint arXiv:1604.03605},
    year     = {2016}
}
