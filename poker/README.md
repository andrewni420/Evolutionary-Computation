# Poker

Using evolutionary reinforcement learning to create a superhuman poker AI. Code breakdown is described below:

## ERL.clj

The main function of the repository. Runs the evolutionary loop and outputs a trained neural network

## transformer.clj

Implementation of the transformer neural network. Functions for model creation, weight initialization, weight manipulation, and inference.

## ndarray.clj

Various utility methods for interfacing with the DJL implementation of ndarrays

The following java classes were created to fill in functionality not available in the DJL library:


### FloatTranslator.java
A translator to turn lists of float arrays into NDArray inputs for a model, and then the NDList output of the model into a list of float arrays. Only used for experiments so far.

### TransformerTranslator.java
A translator to turn a list of NDArrays into an NDList for input into a model, and then the output NDList into a list of float arrays to be parsed as an action. Translators are necessary for prediction, to manage the lifecycles of NDArrays created during inference, as these NDArrays are native and unable to be garbage collected.

### LinearEmbedding.java
A Linear Embedding. Adapted from DJL's Linear block instead of DJL's Embedding interface to support multi-hot encoded inputs.

### ParallelEmbedding.java
Applies separate embedding blocks to separate inputs, along with a final function to turn all of the outputs of the separate embedding blocks into a final output. Implements the Embedding interface. See Embedding.java

### PositionalEncoding.java
Implements multiple positional encodings, concatenating the output of each sub positional encoding to produce a final output.

### SinglePositionEncoding.java
Basic instantiation of DJL's Embedding interface. Used for the positional encoding of a single position type, such as the game-number. Note that game numbers are uniformly decreased at inference to start at game 0 to maintain consistency in the encoding.

### SeparateParallelBlocks.java
Based upon DJL's ParallelBlock, but applies separate blocks to separate inputs, instead of separate blocks to a single input. Accepts a List<NDList>->NDList function to turn the outputs of all the blocks into a single output. Used, for example, to combine positional encoding and input embedding.

### SparseAttentionBlock.java
An implementation of sparse attention, in which all attention weights are set to 0 except for the top k attention weights, which are then softmaxed. A hybrid between soft and hard attention. Since this is where the N^2 runtime of the transformer comes from, sparse attention could greatly speed up inference, leaving computational power for greater population sizes or more generations.
### SparseMax.java
An improved implementation of the top-k softmax, which produces 0s upon being passed uniformly large negative weights, whereas the DJL class by the same name would produce nan. See SparseAttentionBlock.java
### TransformerDecoderBlock.java
A Transformer Decoder Block. Applies an attention mask to its inputs, and passes the attention mask along in its output to be used by the next decoder block in the layer
### Embedding.java
Interface specifying a reverse method that implements the output unembedding in weight sharing. See UnembedBlock.java
### UnembedBlock.java
Implementation of weight sharing, as proposed in https://arxiv.org/pdf/1608.05859.pdf. Inputs are multiplied by the transpose of the embedding matrix. 
### Utils.java
I think this is unused and should be deleted. I will check later.

## headsup.clj

Implementation of headsup no limit poker. Maintains constantly updating NDArrays containing the game encoding to avoid having to create massive NDArrays every time an action is needed.

## onehot.clj

Implementation of one-hot encoding to turn features from the game state into one-hot encoded vectors to be concatenated onto the game encoding

## utils.clj

Various utilities and macros used throughout the repository. Most of the code is about calculating statistics about cards for genetic programming, and is no longer used.

## slumbot.clj

Functions for playing against the publicly available slumbot API. Need to update so that it also keeps a running collection of NDArrays, as in headsup.clj


## License

Copyright © 2023 FIXME

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at
http://www.eclipse.org/legal/epl-2.0.

This Source Code may also be made available under the following Secondary
Licenses when the conditions for such availability set forth in the Eclipse
Public License, v. 2.0 are satisfied: GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or (at your
option) any later version, with the GNU Classpath Exception which is available
at https://www.gnu.org/software/classpath/license.html.
