/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package poker;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.nn.transformer.*;
import ai.djl.training.dataset.ArrayDataset;

import java.util.Collections;
import java.util.function.Function;

/** Self-Attention based transformer decoder block. */
public class TransformerDecoderBlock extends AbstractBlock {
    private Object n = new ArrayDataset.Builder();
    /** The attention mechanism. */
    private ScaledDotProductAttentionBlock selfAttentionBlock;
    /** Dropout before residual & layer normalization. */
    private Dropout selfAttentionDropout;
    /** Normalization of attention output and residual. */
    private BatchNorm attentionNorm;
    /** Fully connected pointwise block for output projection. */
    private PointwiseFeedForwardBlock pointWisefullyConnected;
    /** Dropout after fully connected and before last residual & layer normalization. */
    private Dropout fullyConnectedDropout;
    /** Another normalization for the output and residual. */
    private BatchNorm outputNorm;

    /**
     * Creates a transformer decoder block.
     *
     * @param embeddingSize the embedding size for tokens
     * @param headCount number of attention blocks
     * @param hiddenSize the hidden size for fully connected networks
     * @param dropoutProbability dropout probability
     * @param activationFunction activation function
     */
    public TransformerDecoderBlock(
            int embeddingSize,
            int headCount,
            int hiddenSize,
            float dropoutProbability,
            Function<NDList, NDList> activationFunction) {
        this.selfAttentionBlock =
                addChildBlock(
                        "selfAttention",
                        ScaledDotProductAttentionBlock.builder()
                                .setEmbeddingSize(embeddingSize)
                                .setHeadCount(headCount)
                                .optAttentionProbsDropoutProb(dropoutProbability)
                                .build());
        this.selfAttentionDropout = Dropout.builder().optRate(dropoutProbability).build();
        this.attentionNorm = addChildBlock("attentionNorm", BatchNorm.builder().optAxis(2).build());
        this.pointWisefullyConnected =
                addChildBlock(
                        "outputBlock",
                        new PointwiseFeedForwardBlock(
                                Collections.singletonList(hiddenSize),
                                embeddingSize,
                                activationFunction));
        this.fullyConnectedDropout = Dropout.builder().optRate(dropoutProbability).build();
        this.outputNorm = addChildBlock("outputNorm", BatchNorm.builder().optAxis(2).build());
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return inputShapes;
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        selfAttentionBlock.initialize(manager, dataType, inputShapes);
        attentionNorm.initialize(manager, dataType, inputShapes);
        pointWisefullyConnected.initialize(manager, dataType, inputShapes);
        outputNorm.initialize(manager, dataType, inputShapes);
    }

    protected NDList setAttentionMask(NDList inputs){
        NDArray attentionMask = null;
        if ((inputs.size() == 2) || (inputs.size() == 4)) attentionMask = inputs.get(1);
        if (attentionMask != null) {
                // B=batch size
                long B = inputs.head().getShape().get(0);
                // F=from sequence length
                long F = inputs.head().getShape().get(1);
                // T=to sequence length
                long T;

                if (inputs.size() < 3) T = F;
                else T = inputs.get(1).getShape().get(1);

                NDArray maskOffset;
            
                // The input mask is initially given as a list of integers with a 1 for each existing
                // token. In order to apply it to the attention result, it needs to be expanded and the
                // values turned into offsets for the softmax calculation. For stacked models, this
                // can be done once and reused - hence we check for the number of dimensions if we
                // have to do this locally or whether it was done for us.
                if (attentionMask.getShape().dimension() != 4) {
                        // expand mask to be used on all heads at once
                        NDArray expandedMask = attentionMask.reshape(B, 1, T, F);
                        // we turn the mask from ints into floats and turn all 1s into 0s and all
                        // 0s int o a value of -10000. Adding this to the scores will push all unwanted                            // values towards -inf and keep the unmasked values unchanged
                        maskOffset = expandedMask
                                .toType(DataType.FLOAT32, false)
                                .mul(expandedMask.getManager().create(-1f)) // turn 1 into -1
                                .add(expandedMask
                                        .getManager()
                                        .create(1f)) // turn 0s to 1s, -1s to 0s
                                .mul(expandedMask
                                        .getManager()
                                        .create(-100000f)); // turn 1s (original 0s) into
                        // -100000
                } else {
                        maskOffset = attentionMask;
                }
                if (inputs.size()==2){
                        inputs.set(1,maskOffset);
                } else {
                        inputs.set(3, maskOffset);
                }
        }
        return inputs;
    }

    //Still need to include possible attention on encoder output
    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {

        inputs = setAttentionMask(inputs);

        NDArray embedding = inputs.head();
        // perform attention lookup
        Shape shape = embedding.getShape();
        
        NDList attentionOutput = selfAttentionBlock.forward(ps, inputs, training);
        // add dropout to attention Output
        NDList attentionOutputAfterDropout =
                selfAttentionDropout.forward(ps, attentionOutput, training);
        // add input as residual
        NDArray withResidual = attentionOutputAfterDropout.singletonOrThrow().add(embedding);
        // apply normalization
        NDList normalized = attentionNorm.forward(ps, new NDList(withResidual), training);
        // apply pointwise projection
        NDList afterFullyConnected = pointWisefullyConnected.forward(ps, normalized, training);
        // apply dropout to fully connected output
        NDList afterFullyConnectedDropout =
                fullyConnectedDropout.forward(ps, afterFullyConnected, training);
        // add residual again
        NDList outputWithResidual =
                new NDList(afterFullyConnectedDropout.singletonOrThrow().add(embedding));
        // normalize result
        NDList outputWithNorm = outputNorm.forward(ps, new NDList(outputWithResidual), training);

        // add mask back to result
        outputWithNorm.add(inputs.get(1));
        return outputWithNorm;
    }
}