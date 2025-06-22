# Machine-Translation
# Authors
Linda Sekhoasha\
Wandile Ngobese\
Samukelo Mkhize\
Khonzinkosi Mkhize
# Pip Commands (Requirements)
```batch
pip install pandas
pip install evaluate datasets==2.18.0 transformers==4.39.3 sentencepiece==0.1.99
python -m spacy download en_core_web_sm
pip uninstall -y numpy
pip install numpy==2.3.0
pip uninstall -y torch torchtext torchaudio torchvision
pip install torch==2.2.0 torchaudio==2.2.0 torchvision==0.17.0 torchtext==0.17.0
```
# Xhosa-to-English Neural Machine Translation

This project implements a neural machine translation (NMT) pipeline for Xhosa-to-English translation using sequence-to-sequence (Seq2Seq) models in PyTorch. Both a baseline LSTM encoder-decoder and an attention-based architecture are developed and evaluated, with comparisons to a pretrained multilingual transformer model (M2M100).

## 1. Data Preprocessing

### Dataset Acquisition and Partitioning
The dataset is a publicly available Xhosa-English parallel corpus hosted on Hugging Face: `LindaSekhoasha/xh-en_parallel_corpus`. It is loaded using the Hugging Face `datasets` library and partitioned into training (80%), validation (10%), and test (10%) subsets using a fixed random seed.

### Tokenization and Preprocessing
Text is tokenized using SpaCy (`en_core_web_sm`) for both English and Xhosa. Special tokens `<sos>` and `<eos>` are added, sequences are optionally lowercased, truncated, and padded.

### Vocabulary Construction
Vocabularies are built using `torchtext`, with frequency thresholding and explicit inclusion of `<unk>` and `<pad>` tokens. Token indices are consistent across vocabularies.

### Numericalization and Tensor Formatting
Tokenized sequences are converted to integer indices using `lookup_indices()` and `datasets.map()`. These sequences are stored as PyTorch tensors using `.with_format()`.

### Custom Batching via Collation
A `collate_fn` is defined to pad sequences dynamically within each batch using the `<pad>` index.

### DataLoader Construction
The datasets are wrapped in PyTorch `DataLoader` objects with a batch size of 128 and optional shuffling for the training set.

## 2. Model Architecture

### Encoder-Decoder Structure
The baseline Seq2Seq model consists of:
- LSTM-based encoder producing final hidden and cell states
- LSTM-based decoder generating tokens autoregressively
- Embedding layers and dropout for regularization

### Attention-Based Decoder
The attention decoder enhances the baseline by computing a context vector at each step. The current decoder hidden state is compared with all encoder outputs using an additive attention mechanism. The context vector is combined with the input token embedding and fed into the LSTM.

### Attention Mechanism
The attention module computes alignment scores, normalizes them with softmax to get attention weights, and forms a context vector via a weighted sum. This improves translation of long or complex sequences.

## 3. Attention-Based Sequence-to-Sequence Model
The `AttnSeq2Seq` model wraps the encoder and attention-equipped decoder. The encoder outputs the full sequence of hidden states. The decoder uses these to compute dynamic context vectors at each step.

During decoding, the `<sos>` token is fed initially. Teacher forcing is applied based on a fixed probability. The model is compatible with both CPU and GPU.

## 4. Training the Attention-Based Model

The attention model is trained using the same optimizer (Adam) and loss function (cross-entropy with `ignore_index` for `<pad>`). A custom weight initializer is applied before training.

At each epoch, loss and perplexity are recorded. The best model (lowest validation loss) is saved. Visual inspection shows that the attention model achieves smoother convergence and better generalization than the baseline.

## 5. BLEU Evaluation

BLEU (Bilingual Evaluation Understudy) is used to evaluate translation quality based on n-gram overlap.

| Model                  | BLEU Score | Length Ratio |
|------------------------|------------|---------------|
| LSTM (Encoder-Decoder) | 0.0061     | 1.116         |
| Attention-Based Seq2Seq| 0.0089     | 0.915         |
| M2M100                 | 0.0446     | 1.193         |

The attention model improves BLEU by 46% over the baseline. The M2M100 transformer yields the highest score, outperforming the attention model by 401% and the baseline by 631%, highlighting the advantage of pretrained multilingual transformers.

## References

Bahdanau, D., Cho, K. and Bengio, Y., 2015. Neural machine translation by jointly learning to align and translate. *ICLR*.

Chorowski, J., Bahdanau, D., Serdyuk, D., Cho, K. and Bengio, Y., 2015. Attention-based models for speech recognition. *NeurIPS*, 28.

Jurafsky, D. and Martin, J.H., 2021. *Speech and Language Processing*. 3rd ed. Prentice Hall.

Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic optimization. *ICLR*.

Lhoest, Q., et al., 2021. Datasets: A community library for NLP. *EMNLP System Demonstrations*, pp.175–184. https://doi.org/10.18653/v1/2021.emnlp-demo.21

Luong, M.T., Pham, H. and Manning, C.D., 2015. Effective approaches to attention-based neural machine translation. *EMNLP*.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R., 2014. Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), pp.1929–1958.

Sutskever, I., Vinyals, O. and Le, Q.V., 2014. Sequence to sequence learning with neural networks. *NeurIPS*, 27.

Vaswani, A., et al., 2017. Attention is all you need. *NeurIPS*, 30.

[Sequence to Sequence Learning with Neural Networks](https://github.com/bentrevett/pytorch-seq2seq/blob/main/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)
