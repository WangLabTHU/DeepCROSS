# DeepCROSS

*qxdu edited on Jan 5, 2025*

The code for computational implementation of "Systematic representation and optimization enable the inverse design of cross-species regulatory sequences in bacteria". This codebase has been updated to a PyTorch version to facilitate easier implementation, as compared to the previously provided, now-deprecated TensorFlow [version](https://github.com/WangLabTHU/DeepCROSS/tree/tensorflow_old). 



# Introduction

DeepCROSS is a generative inverse design tool for cross-species and species-preferred 5' regulatory sequences (RSs) in bacteria. By constructing the meta representation of 1.8 million 5’ RSs from thousands of bacterial genomes, DeepCROSS extracts the fundamental sequence features into a statistical embedding and forms the species-specific subspaces.


![Figure 1](./figs/github/figure1.png)


We validated the DeepCROSS-designed RSs by massively parallel reporter assay experiments. The DeepCROSS model refined by this strategy demonstrated a significant improvement in predicting RS activity performance. The experimental validation confirmed that the final DeepCROSS model achieved 90.0% and 93.3% accuracy in designing species-preferred and cross-species RSs respectively. These synthetic RSs exhibited high diversity and low sequence similarity to their natural counterparts within bacterial genomes.


# Preparation

## Environment Setup

For the convenience of researchers seeking seamless utilization, we have adhered to all environmental configurations as previously established in our released GPro toolkit. To set up the virtual environment for DeepCROSS, please follow the instructions provided in our [wiki](https://github.com/WangLabTHU/GPro/wiki/2.-Installation#installation). All dependencies will be automatically installed as part of this process, including appropriate version of PyTorch. The requirement of hardware and detailed [session](https://github.com/WangLabTHU/GPro/wiki/6.-Session-Information) information have also been previously provided.

Additionally, we have gone a step further to provide an installation guide specifically for servers without internet access. The necessary instructions for this offline installation can be found in the following [wiki](https://github.com/WangLabTHU/GPro/wiki/2.-Installation#alternative-choose-envs-for-offline-machine) section, which outlines the process for successful setup.

DeepCROSS is built upon the previously released toolkit [GPro](https://academic.oup.com/bioinformatics/article/40/3/btae123/7617687), facilitating convenient and rapid sequence reading, encoding, as well as the implementation of Early Stopping mechanisms. 


## Dataset Preparation

This study is primarily based on the aae_meta version of DeepCROSS. The relevant training data is stored in the dataset folders. The following table conclude the functionality of these data and how to obtain them.


| Folder Name | Description| Acquisition|
| ------ | ------ |------ |
| AAE_pretrain | Large-scale unsupervised training on 1.8 million 5’ RSs from 2621 representative bacteria genomes | 10.5281/zenodo.14598566
| AAE_finetune | Representative Enterobacterales and Pseudomonadales bacteria species | 10.5281/zenodo.14598566
| AAE_represent | Testing dataset for the representation of RSs from various bacteria genomes o assess DeepCROSS’s ability to derive meaningful features | This repository
| AAE_supervised | Supervised training dataset from previous [research](https://www.nature.com/articles/nmeth.4633), with tag `NM2018`; MPRA results from our experiments, with tag `0201` for Lib-1 and `0713` for Lib-2 | This repository
| AAE_univ | Testing data for evaluating the k-mer similarity performance of AAE model to check the quality of training process | This repository
| PredNet_round1 | Initial training dataset for training DenseLSTM predictor, dataset are from `NM2018`, processed already | This repository
| PredNet_final | The training dataset for the second round of training DenseLSTM predictor, dataset are from the combination of Lib-1 and Lib-2, processed already | This repository


# Design cross-species regulatory sequences

DeepCROSS is composed of a number of modules that can be imported as follows:

**Round 1 :**

- Generative model for representation (pretrain on `AAE_pretrain`, finetune on `AAE_finetune`, semi-supervised on `NM2018`)
- Predictive model for selection (directly trained on `NM2018`)

**Round 2 / Final :**

- Generative model for representation (pretrain on `AAE_pretrain`, finetune on `AAE_finetune`, semi-supervised on `Lib-1` and `Lib-2`)
- Predictive model for selection (follow the strategy in article, pretrain on top-90% filtered version of `NM2018`, and finetune on `Lib-1` and `Lib-2`)
- Optimization (Genetic Algorithms, based on trained Predictive model)

## Generation

The codes in `aae_meta.py` provide all necessary process for training and evaluating generative models, in all two rounds.

## Prediction

The codes in `prednet_r1.py` and `prednet_final.py` provide all necessary process for training and evaluating predictive models for two rounds, separately.

## Optimization

The codes in `optimization_final.py` provide the process needed for genetic algorithm, for getting the species-specific and cross-species data in the final round.


# Other Information

The necessary verification information for the figures in the main text and supplementaries has been partially provided in the `figs` folder. To ensure the cleanliness of the repository, we have not provided other redundant information, including the large-scale sampling required for Figure 5, DNAshape information, etc. The codes for obtaining these data have been fully provided. You can access further supports through issues or email.

Our original codes used absolute paths, which I have manually changed to relative paths. However, there may still be some path errors, and we welcome your feedback.

Some interval output files, including specific calibration strategies and column information in `final_overlap.csv` under the AAE_supervised folder, may cause confusion. For example, the `EC(2021)` column actually corresponds to the data of `Lib-1`. This is due to historical reasons; we initially designed these sequences in 2021 but updated the MPRA results on February 1, 2024 (`0201`), resulting in two tag information. We have tried our best to minimize these interval details that can easily cause confusion.

# License

For academic use, this project is licensed under the MIT License (see the LICENSE file for details). For commercial use, please contact the authors.

# Citations

~~~
[1] Haochen Wang, Qixiu Du, Ye Wang, Hanwen Xu, Zheng Wei, Xiaowo Wang, GPro: generative AI-empowered toolkit for promoter design, Bioinformatics, Volume 40, Issue 3, March 2024, btae123, https://doi.org/10.1093/bioinformatics/btae123

[2] cblaster: a remote search tool for rapid identification and visualization of homologous gene clusters, Bioinformatics Advances, Volume 1, Issue 1, 2021, vbab016, https://doi.org/10.1093/bioadv/vbab016
~~~