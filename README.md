## Overview

Implementation of a deep recursive neural network for the task of fine-grained sentiment detection.

See the paper,
>"Deep Recursive Neural Networks for Compositionality in Language"
>Ozan Irsoy, Claire Cardie
>NIPS 2014

for details.

If you use my code, please cite:

	@InProceedings{irsoy-drsv,
	  author = {\.Irsoy, Ozan and Cardie, Claire},
	  title = {Deep Recursive Neural Networks for Compositionality in Language},
	  booktitle = {Advances in Neural Information Processing Systems 27},
	  editor = {Z. Ghahramani and M. Welling and C. Cortes and N.D. Lawrence and K.Q. Weinberger},
	  pages = {2096--2104},
	  year = {2014},
	  publisher = {Curran Associates, Inc.},
	  url = {http://papers.nips.cc/paper/5551-deep-recursive-neural-networks-for-compositionality-in-language.pdf},
	  location = {Montreal, Quebec}
	}

Feel free to ask questions: oirsoy [a] cs [o] cornell [o] edu.
<http://www.cs.cornell.edu/~oirsoy/drsv.htm>

## Getting Started

Assuming you have g++ and the code here, running the bash script as

	bash run.sh

should

1. download small word embeddings (50 dimensional Glove)
2. download the Stanford Sentiment Treebank (in PTB form)
3. download the Eigen library
4. compile and run to train a small model to be saved to disk.

That's it! Once you have a working setup, you can play with the hyperparameters or pick different word embeddings (300d word2vec is used in the experiments in the paper).

##License

Code is released under [the MIT license](http://opensource.org/licenses/MIT).
