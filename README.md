# Mandarin Phonetics Modeling

## Motivation

>In many representation learning tasks, researchers aim to extract embeddings of objects that **reflect similarity** along a target feature, such as phonetic, semantic, or visual attributes, rather than merely optimizing a discriminative loss. To address this, I present a case study in the NLP subfield of Mandarin phonetics modeling, experimenting with several explicit strategies that enforce target‑based grouping with BiLSTM encoder. These strategies include **a shared input basis** (radical decomposition), **a continuous acoustic target space** (wav2vec2 embeddings), and **dimensionality reduction to reveal latent structure**. Furthermore, I design an evaluation framework that decouples the influence of different factors (visual, phonetic, acoustic, semantic) on hidden‑space similarity, using both regression analysis and decoder dynamics as diagnostics.

## Notebooks Structure

In the repository I provide the final versions of the notebooks that are loaded from Kaggle. These notebooks provide the results but are not supposed to be run outside of the Kaggle environment. For testing the notebooks with different versions, follow the links attached below:

### Tokenization
- [wubi-tokenizer.ipynb](https://www.kaggle.com/code/davidvista/wubi-tokenizer): Notebook for Wubi tokenization implementation and testing.

### Datasets

#### Pinyin Embeddings
- [mandarin-sounds-dataset.ipynb](https://www.kaggle.com/code/davidvista/mandarin-sounds-dataset): Main notebook for creating and processing the Mandarin sounds dataset (using the finetuned Wav2Vec model).
- [evaluation-of-mandarin-sounds-dataset-base.ipynb](https://www.kaggle.com/code/davidvista/evaluation-of-mandarin-sounds-dataset?scriptVersionId=310537595): Notebook for evaluating the Mandarin sounds dataset using embeddings from the base Wav2Vec model.
- [evaluation-of-mandarin-sounds-dataset-zh-ch.ipynb](https://www.kaggle.com/code/davidvista/evaluation-of-mandarin-sounds-dataset?scriptVersionId=310537423): Notebook for evaluating the Mandarin sounds dataset using embeddings from the finetuned Wav2Vec model.

#### Pinyin Labeling
- [char2pinyin-pinyin2char-framework.ipynb](https://www.kaggle.com/code/davidvista/char2pinyin-pinyin2char-framework?scriptVersionId=305074711): Notebook for training a character-to-pinyin model on a different corpus. The corpus is not primarily chosen for modeling due to unknown domain and too broad vocabulary. However, the general pinyin labeling is sufficient with this corpus.
- [pinyin-dataset-labelling.ipynb](https://www.kaggle.com/code/davidvista/pinyin-dataset-labelling): Notebook for labeling the [main pinyin dataset](https://huggingface.co/datasets/AIxBlock/Chinese-short-sentences) using a pre-trained character-to-pinyin model.
- [pinyin-eval-dataset-labelling.ipynb](https://www.kaggle.com/code/davidvista/pinyin-eval-dataset-labelling/notebook): Notebook for labeling the Pinyin evaluation dataset.

### Training

#### Regular Encoder
- [wubi-tokenizer-mandarin-encoder.ipynb](https://www.kaggle.com/code/davidvista/wubi-tokenizer-mandarin-encoder): Training of Mandarin encoder with Wubi tokenizer.
- [wubi-tokenizer-mandarin-frozen-pipeline.ipynb](https://www.kaggle.com/code/davidvista/wubi-tokenizer-mandarin-frozen-pipeline): Frozen encoder training pipeline (decoder is trained only).
- [wubi-tokenizer-mandarin-full-pipeline.ipynb](https://www.kaggle.com/code/davidvista/wubi-tokenizer-mandarin-full-pipeline): Complete training pipeline for Mandarin encoder with Wubi tokenizer.
- [wubi-tokenizer-scratch-pipeline.ipynb](https://www.kaggle.com/code/davidvista/wubi-tokenizer-scratch-pipeline): Training pipeline for Wubi tokenizer from scratch (both encoder and decoder).

#### Phonetic Aware Encoder
- [phonetic-aware-representaions-learning.ipynb](https://www.kaggle.com/code/davidvista/phonetic-aware-representaions-learning): Training of Mandarin encoder with Wubi tokenizer and phonetic alignment. Note that the link refers to the Kaggle version of the encoder with a stable training, for unstable version view "Version 2".
- [phonetic-aware-frozen-pipeline.ipynb](https://www.kaggle.com/code/davidvista/phonetic-aware-frozen-pipeline): Frozen encoder training pipeline (decoder is trained only).
- [phonetic-aware-frozen-pipeline-unstable-encoder.ipynb](https://www.kaggle.com/code/davidvista/phonetic-aware-frozen-pipeline-unstable-encoder): Frozen encoder training pipeline (decoder is trained only, training of encoder was not stable due to small alignment pressure). This notebook is used for contrasting the effect of proper alignment.
- [phonetic-aware-full-pipeline.ipynb](https://www.kaggle.com/code/davidvista/phonetic-aware-full-pipeline): Complete training pipeline for phonetic-aware encoder (decoder and stable trained encoder).

### Regression Analysis
- [context-averaged-representations.ipynb](https://www.kaggle.com/code/davidvista/context-averaged-representations): Extraction of context-averaged and context-aware representations for phonetic similarity analysis. The notebook provides the data about characters in a given sentence from the evaluation corpus: hidden state, pinyin embedding, semantic embedding (BERT), decomposed pinyin, and Wubi tokens. Note that the link refers to the final version of the notebook (representations for regular finetuned encoder), browse versions in Kaggle to see preparation other representations.
- [regular-encoder-representations.ipynb](https://www.kaggle.com/code/davidvista/mandarin-regression-analysis?scriptVersionId=312000883): Analysis of regular encoder representations.
- [phonetic-aware-encoder-representations.ipynb](https://www.kaggle.com/code/davidvista/mandarin-regression-analysis?scriptVersionId=311999196): Analysis of phonetic-aware encoder representations.
- [finetuned-regular-encoder-representations.ipynb](https://www.kaggle.com/code/davidvista/mandarin-regression-analysis?scriptVersionId=312005472): Analysis of finetuned regular encoder representations.
- [finetuned-phonetic-aware-encoder-representations.ipynb](https://www.kaggle.com/code/davidvista/mandarin-regression-analysis?scriptVersionId=312007699): Analysis of finetuned phonetic-aware encoder representations.

## Training Pipeline

![](images\pipeline.png)

### Loss Functions

$$
\mathcal{L}=\text{CE}_{\text{dec}} + \mathbb{\lambda}_{\text{enc}} \Sigma_{\text{CE}}
$$

$$
\mathcal{L}_{\text{aligned}}=\text{CE}_{\text{dec}} + \mathbb{\lambda}_{\text{enc}} (\Sigma_{\text{CE}} + \mathbb{\lambda}_{\text{MSE}} \text{MSE})
$$

## Evaluation Method

![](images\evaluation.png)

## Results

### Probing (Decoder Performance)

![](images\decoder.png)

### Regression Results

![](images\regression.png)


## Conclusion

In conclusion, I find that a BiLSTM encoder trained only to predict pinyin from Chinese characters does not naturally organize its hidden space by phonetic similarity; instead, the space is dominated by visual radical overlap. By adding an explicit MSE loss that aligns the hidden representations with pre‑computed acoustic embeddings (wav2vec2), the encoder learns a low‑dimensional manifold where phonetic factors—especially same‑syllable identity and continuous acoustic similarity—become the primary drivers of hidden‑state similarity ($R^2 = 0.66$ after PCA). This phonetic alignment initially introduces ambiguity for the decoder (slower early convergence), yet the decoder recovers to baseline accuracy, confirming that the hidden space has traded off character‑specificity for phonetic structure without sacrificing downstream performance. The results demonstrate that a text‑only model can acquire a sound‑based representation when given a suitable grouping basis, a continuous target space, and post‑hoc dimensionality reduction – a principle that generalises beyond Mandarin to any representation learning task that seeks to embed objects according to a continuous similarity metric.

