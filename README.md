# Meta learning for mutant HLA class I epitope immunogenicity prediction to accelerate cancer clinical immunotherapy
Accurate prediction of binding between human leukocyte antigen (HLA) class I molecules and antigenic peptide segments is a challenging task and a key bottleneck in personalized immunotherapy for cancer. Although existing prediction tools have demonstrated significant results using established datasets, most can only predict the binding affinity of antigenic peptides to HLA and do not enable the immunogenic interpretation of new antigenic epitopes. This limitation results from the training data for the computational models relying heavily on a large amount of peptide-HLA eluting ligand data, in which most of the candidate epitopes lack immunogenicity. Here, we propose an adaptive immunogenicity prediction model, named MHLAPre, which is trained on the large-scale MS-derived HLA I eluted ligandome (mostly presented by epitopes) that are immunogenic. Allele-specific and pan-allelic prediction models are also provided for endogenous peptide presentation. Using a meta-learning strategy, MHLAPre rapidly assessed HLA class I peptide affinities across the whole peptide-HLA (pHLA) pairs and accurately identified tumor-associated endogenous antigens. During the process of adaptive immune response of T cells, pHLA-specific binding in the antigen presentation is only a pre-task for CD8+ T cell recognition. The key factor in activating the immune response is the interaction between pHLA complexes and T cell receptors (TCRs). Therefore, we performed transfer learning on the pHLA model using the pHLA-TCR dataset. In pHLA binding task, MHLAPre demonstrated significant improvement in identifying neoepitope immunogenicity compared with five state-of-the-art models, proving its effectiveness and robustness. After transfer learning of the pHLA-TCR data, MHLAPre also exhibited relatively superior performance in revealing the mechanism of immunotherapy. MHLAPre is a powerful tool to identify neoepitopes that can interact with TCR and induce immune responses. We believe that the proposed method will greatly contribute to clinical immunotherapy, such as anti-tumor immunity, tumor-specific T-cell engineering and personalized tumor vaccine.


# The environment of MHLAPre
```
python==3.9.13
numpy==1.21.2
pandas==1.4.4
torch==1.12.1
scikit-learn>=1.0.2
pandas>=1.2.4
rdkit~=2021.03.2
```

# Installation Guide
Clone this Github repo and set up a new conda environment. It normally takes about 10 minutes to install on a normal desktop computer.
```
# create a new conda environment
$ conda create --name MHLAPre python=3.9.13
$ conda activate MHLAPre

# install requried python dependencies
$ conda install pytorch==1.12.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install -c conda-forge rdkit==2021.03.2
$ pip install -U scikit-learn

# clone the source code of MHLAPre
$ git clone https://github.com/
$ cd MHLAPre
```

# Dataset description
In this paper, epitope presentation and immunogenicity data sources are used, which are freely downloaded from IEDB (https://www.iedb.org)
By default, you can run our model using the immunogenicity dataset with:
```
python Pretreatment.py

python TransfomerEncoder.py

python TextCNN.py
```


# Acknowledgments
The authors sincerely hope to receive any suggestions from you!
Of note, the raw data and procedural data of immunogenicity are too large to upload. If you want to further study the immunogenicity of HLA-class I based on MHLAPre, please contact us. Email: 23B903048@stu.hit.edu.cn
## Architecture
<p align="center">
<img src="https://github.com/ChanganMakeYi/MHLAPre/blob/main/1.png" align="middle" height="80%" width="80%" />
</p >
