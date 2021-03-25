# wav2vec_xlsr_cv_exp
Experiments on out of training languages (from common voice https://commonvoice.mozilla.org/) using Wav2Vec

# Repo Description
**notebook_template.ipynb** is a notebook that you can use to experiment with wav2vec2 models. \
The folder **utils** is a python package where I put all the necessary functions ro run the notebook_template. \
The folder **exp** contains notebooks of our experiments (the notebooks are not the same as the template one,
they are not well organized but they follow the same procedure/approach as the notebook_template.) \
**trainer.py** is a module made to run a training.

# Introduction
Low resources languages have always been an issue for automatic speech recognition. Indeed, most speech recognition systems were trained using the available transcriptions, and when they were not sufficient, the model would be trained on a language with huge resources (typically english, french, etc...), and then finetuned on the low resource language. Although this approach was very intuitive and feasible, it still made use of annotated data. The problem is that it is very exhausting to annotate speech samples, especially for low resource languages. On the other hand, we gather every day countless of new unlabelled data, that can't be used for training with the precedent approach.
Fortunately, unsupervised pretraining on large quantity of audio has proven to be very successful in learning speech features. Pretrained models achieve very good results on several speech tasks on low resource languages (i.e languages with few amount of labelled data 10mn - 10h), and have outperformed fully supervised approaches. In this project, we will examine if the features learned in one language (Wav2Vec2-LV60) or in multiple languages (XLSR-53) are good enough on other out-of-training low resource languages.

# Wav2Vec 2.0 Model 
The model is composed of a multi-layer convolutional feature encoder
*f* : *X* → *Z* which takes as input raw audio *X* and outputs the
latent speech representations *z*<sub>1</sub>, ...., *z*<sub>*T*</sub>.
This latter is fed to a context network *g* : *Z* → *C* that follows the
Transformer architecture and uses convolutional layers to learn relative
positional embedding, which output representations
*c*<sub>1</sub>, ....., *c*<sub>*T*</sub> capturing information from the
entire sequence.  
The output of the feature encoder (and not of the context transformer)
is discretized to *q*<sub>*t*</sub> in parallel using a quantization
module *Z* → *Q* that relies on product quantization. These
*q*<sub>1</sub>, ....., *q*<sub>*T*</sub> represent the targets in the
pretraining task.  
The model architecture is summarized below (Source : [2]) : \
\
![plot](./img/wav2vec2.png)
 
 
The pretraining of this model consists of masking a certain proportion of time steps in the latent feature encoder space, and the objective is to identify the correct quantized latent audio representation in a set of distractors of each masked time step.

For the experimentation we consider two versions of the wav2vec 2.0 architecure [2].
The first one is the XLSR-53 [3], which was pretrained on 56k  hours of speech samples, which represent fifty three different languages from the multilingual librispeech, the common voice dataset and the babel dataset. \
The second is the wav2vec LARGE LV-60 which was pretrained on sixty thousand hours of only english speech samples from LibriVox. \


# Dataset Preparation
**Cross-Lingual** : For the cross-lingual experimentation, we consider
out-of-training language datasets from the Common Voice Dataset [4] . We
have identified languages that were not used for the pretraining of the
XLSR-53 model and for which we have sufficient annotated data to work
with (Ukrainian (30h) and Czech (36h)).

**Monolingual** : To examine the impact of multilingual training, we
took an english audio dataset that none of these models have been
pretrained on, thus we consider the TIMIT dataset.\

For both experimentation, we retrieve the non-aligned phoneme
transcription of each audio sample by running the open-source tool
phonemizer[1] on their corresponding text scripts. We follow partially
the defined splits in the dataset sources : the test set was kept
intact, we merge the training and validation set into one set, and we
split this latter into two set (80% training / 20% validation). For the
cross-lingual experimentation, we used different subsets of the training
set (10mn, 1h and the maximum number of hours ≈ 8h/9h).

# Results
## Monolingual : TIMIT English Dataset
<div class="center">

|    Model    | PER on dev. set | PER on test set |
|:-----------:|:---------------:|:---------------:|
|   XLSR-53   |      4.7%       |      7.3%       |
| Large LV-60 |      3.9%       |      6.1%       |

</div>

It seems like on an english task, the wav2vec 2.0 Large LV60 is more suited. This actually shows a very 
important result : cross-lingual pretraining does not improve the performances over a robust automatic 
speech recognition system trained on a very large dataset in english.

## Cross-Lingual : Ukrainian / Czech
<div class="tabular">

|      Language       | 10min |  1h  | 10h  |
|:-------------------:|:-----:|:----:|:----:|
|    Czech (XLSR)     | 19.0  | 15.5 | 11.4 |
| Czech (wav2vec 2.0) | 24.7  | 17.2 | 11.7 |
|   Czech (scratch)   |  \-   |  \-  | 35.8 |
|  Ukrainian (XLSR)   | 27.2  | 21.4 | 16.7 |
| Ukrainian (scratch) |  \-   |  \-  | 48.2 |

</div>

Using pretrained model (English Only LV60 or Multilingual XLSR) was more
efficient than training from scratch the model. We can see that using
only 10 minutes of training data, we achieve a 20% gain in PER compared
to the training from scratch with 10 hours of data. Furthermore, XLSR
performs better on the Czech language in comparison with the Wav2Vec 2.0
LV model, this shows that Cross-Lingual pretraining still preferred for
low resource languages.

# Models
Models can be found here : [Google Drive](https://drive.google.com/drive/folders/1D7wWF74esD93pe6C5UX52RLkQJzNPn2U?usp=sharing)

# References

[1] https://pypi.org/project/phonemizer (built from source espeak-ng to
access ukrainian support)

[2] A. Baevski, H. Zhou, A. Mohamed, and M. Auli.   wav2vec2.0:  A framework for self-supervised learning of speech rep-resentations, 2020.

[3] A.  Conneau,  A.  Baevski,  R.  Collobert,  A.  Mohamed,  andM. Auli.  Unsupervised cross-lingual representation learningfor speech recognition, 2020.

[4] R.  Ardila,  M.  Branson,  K.  Davis,  M.  Henretty,  M.  Kohler,J. Meyer, R. Morais, L. Saunders, F. M. Tyers, and G. Weber.Common voice:  A massively-multilingual speech corpus.  InProceedings of the 12th Conference on Language Resourcesand Evaluation (LREC 2020), pages 4211–4215, 2020