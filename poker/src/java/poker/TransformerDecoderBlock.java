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
import ai.djl.nn.norm.LayerNorm;
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
    private LayerNorm attentionNorm;
    /** Fully connected pointwise block for output projection. */
    private PointwiseFeedForwardBlock pointWisefullyConnected;
    /** Dropout after fully connected and before last residual & layer normalization. */
    private Dropout fullyConnectedDropout;
    /** Another normalization for the output and residual. */
    private LayerNorm outputNorm;

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
        this.attentionNorm = addChildBlock("attentionNorm", LayerNorm.builder().axis(new int[]{2}).build());
        this.pointWisefullyConnected =
                addChildBlock(
                        "outputBlock",
                        new PointwiseFeedForwardBlock(
                                Collections.singletonList(hiddenSize),
                                embeddingSize,
                                activationFunction));
        this.fullyConnectedDropout = Dropout.builder().optRate(dropoutProbability).build();
        this.outputNorm = addChildBlock("outputNorm", LayerNorm.builder().axis(new int[]{2}).build());
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
        attentionNorm.initialize(manager, dataType, new Shape[]{inputShapes[0]});
        pointWisefullyConnected.initialize(manager, dataType, new Shape[]{inputShapes[0]});
        outputNorm.initialize(manager, dataType, new Shape[]{inputShapes[0]});
    }

    //Still need to include possible attention on encoder output
    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {

        NDArray embedding = inputs.head();
        // perform attention lookup
        Shape shape = embedding.getShape();
        
        NDList attentionOutput = selfAttentionBlock.forward(ps, inputs, training);
        //System.out.println(attentionOutput.head());
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