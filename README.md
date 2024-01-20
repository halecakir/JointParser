# Joint learning of morphology and syntax with cross-level contextual information flow

## Overview
This repository implements an joint deep learning model for morphological segmentation, morpheme tagging, part-of-speech (POS) tagging, and syntactic parsing onto dependencies. The model leverages cross-level contextual information flow for every word, spanning from segments to dependencies, with an attention mechanism facilitating horizontal flow. This work extends the approach introduced by Nguyen and Verspoor (2018) on joint POS tagging and dependency parsing, now incorporating morphological segmentation and morphological tagging.

## Abstract
We propose an integrated deep learning model that tackles the joint tasks of morphological segmentation, morpheme tagging, POS tagging, and syntactic parsing onto dependencies. Our approach builds upon the base model of Nguyen and Verspoor (2018), which consists of a two-layer BiLSTM for POS tagging and a graph-based dependency parsing component. Our enhanced architecture adds a component for learning morphological segmentation and another for morphological identification to obtain morpheme labels.

Primary focus is on agglutination in morphology, particularly in Turkish morphology, where we demonstrate improved performance compared to models trained for individual tasks. As one of the earlier efforts in joint modeling of syntax and morphology along with dependencies, we discuss prospective guidelines for future comparisons. We report our results on several languages, showcasing the versatility and effectiveness of our proposed model.

## Model Architecture
The base model of Nguyen and Verspoor (2018) comprises two components:

1. **POS Tagging Component:** Based on a two-layer BiLSTM (Hochreiter and Schmidhuber, 1997). Encodes sequential information from each word within a sentence. The encoded information is passed through a multilayer perceptron with a single layer, outputting POS tags.
2. **Graph-Based Dependency Parsing Component:** Also involves a BiLSTM. Learns features of dependency arcs and their labels using predicted POS tags, word embeddings, and character-level word embeddings. 

Our extended architecture, as illustrated in the paper (Can, 2022), introduces two additional components:
1. **Morphological Segmentation Component:** Learns morphological segmentation for each word.

2. **Morphological Identification Component:** Identifies morphemes and assigns labels.

## Usage

```sh
python jPTDP.py [options]
```
### Options
* **--type**: Experiment Type.
* **--train**: Path to annotated CONLL train file.
* **--dev**: Path to annotated CONLL dev file.
* **--test**: Path to CONLL test file.
* **--segmentation**: Path to Morph segmentation file.
* **--output**: File name for predicted output.
* **--prevectors**: Pre-trained vector embeddings.
* **--prevectype**: Pre-trained vector embeddings type.
* **--params**: Parameters file.
* **--model**: Load/Save model file.
Various other options for dimensions, sizes, flags, and configurations.

## Examples
**Training**
```sh
python jPTDP.py --type jointAll --train train.conll --dev dev.conll --test test.conll --output predicted_output.conll --prevectors pre-trained_embeddings.vec
```

**Prediction**
```sh
python jPTDP..py --type jointAll --test test.conll --output predicted_output.conll --model saved_model
```
Please have a look at the `train.sh` for more examples.
## Reference
*Can B, Aleçakır H, Manandhar S, Bozşahin C. Joint learning of morphology and syntax with cross-level contextual information flow. Natural Language Engineering. 2022;28(6):763-795. doi:10.1017/S1351324921000371*
```
@article{can2022joint,
  title={Joint learning of morphology and syntax with cross-level contextual information flow},
  author={Can, Burcu and Ale{\c{c}}ak{\i}r, H{\"u}seyin and Manandhar, Suresh and Boz{\c{s}}ahin, Cem},
  journal={Natural Language Engineering},
  volume={28},
  number={6},
  pages={763--795},
  year={2022},
  publisher={Cambridge University Press}
}
```





















