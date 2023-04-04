/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.NDManager
 *  ai.djl.nn.AbstractBlock
 *  ai.djl.nn.Block
 *  ai.djl.nn.norm.BatchNorm
 *  ai.djl.nn.norm.Dropout
 *  ai.djl.util.PairList
 *  java.lang.Integer
 *  java.lang.Object
 *  java.lang.String
 *  java.util.Collection
 *  java.util.Collections
 *  java.util.List
 *  java.util.function.Function
 */
package ai.djl.nn.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.transformer.PointwiseFeedForwardBlock;
import ai.djl.nn.transformer.ScaledDotProductAttentionBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class TransformerEncoderBlock
extends AbstractBlock {
    private ScaledDotProductAttentionBlock selfAttentionBlock;
    private Dropout selfAttentionDropout;
    private BatchNorm attentionNorm;
    private PointwiseFeedForwardBlock pointWisefullyConnected;
    private Dropout fullyConnectedDropout;
    private BatchNorm outputNorm;

    public TransformerEncoderBlock(int embeddingSize, int headCount, int hiddenSize, float dropoutProbability, Function<NDList, NDList> activationFunction) {
        this.selfAttentionBlock = (ScaledDotProductAttentionBlock)this.addChildBlock("selfAttention", (Block)ScaledDotProductAttentionBlock.builder().setEmbeddingSize(embeddingSize).setHeadCount(headCount).optAttentionProbsDropoutProb(dropoutProbability).build());
        this.selfAttentionDropout = Dropout.builder().optRate(dropoutProbability).build();
        this.attentionNorm = (BatchNorm)this.addChildBlock("attentionNorm", (Block)BatchNorm.builder().optAxis(2).build());
        this.pointWisefullyConnected = (PointwiseFeedForwardBlock)this.addChildBlock("outputBlock", (Block)new PointwiseFeedForwardBlock((List<Integer>)Collections.singletonList((Object)hiddenSize), embeddingSize, activationFunction));
        this.fullyConnectedDropout = Dropout.builder().optRate(dropoutProbability).build();
        this.outputNorm = (BatchNorm)this.addChildBlock("outputNorm", (Block)BatchNorm.builder().optAxis(2).build());
    }

    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return inputShapes;
    }

    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape ... inputShapes) {
        this.selfAttentionBlock.initialize(manager, dataType, inputShapes);
        this.attentionNorm.initialize(manager, dataType, inputShapes);
        this.pointWisefullyConnected.initialize(manager, dataType, inputShapes);
        this.outputNorm.initialize(manager, dataType, inputShapes);
    }

    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray embedding = inputs.head();
        NDList attentionOutput = this.selfAttentionBlock.forward(ps, inputs, training);
        NDList attentionOutputAfterDropout = this.selfAttentionDropout.forward(ps, attentionOutput, training);
        NDArray withResidual = attentionOutputAfterDropout.singletonOrThrow().add(embedding);
        NDList normalized = this.attentionNorm.forward(ps, new NDList(withResidual), training);
        NDList afterFullyConnected = this.pointWisefullyConnected.forward(ps, normalized, training);
        NDList afterFullyConnectedDropout = this.fullyConnectedDropout.forward(ps, afterFullyConnected, training);
        NDList outputWithResidual = new NDList(afterFullyConnectedDropout.singletonOrThrow().add(embedding));
        return this.outputNorm.forward(ps, new NDList((Collection<NDArray>)outputWithResidual), training);
    }
}
