/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.NDArray
 *  ai.djl.ndarray.NDManager
 *  ai.djl.ndarray.types.Shape
 *  ai.djl.nn.AbstractBlock
 *  ai.djl.nn.Block
 *  ai.djl.nn.core.Linear
 *  ai.djl.nn.norm.Dropout
 *  ai.djl.nn.transformer.ScaledDotProductAttentionBlock$Builder
 *  ai.djl.training.ParameterStore
 *  ai.djl.util.PairList
 *  java.lang.IllegalArgumentException
 *  java.lang.Math
 *  java.lang.Object
 *  java.lang.String
 *  java.util.Collection
 */
package ai.djl.nn.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.transformer.ScaledDotProductAttentionBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Collection;

/*
 * Exception performing whole class analysis ignored.
 */
public final class ScaledDotProductAttentionBlock
extends AbstractBlock {
    private static final byte VERSION = 1;
    private int embeddingSize;
    private int headCount;
    private Linear keyProjection;
    private Linear queryProjection;
    private Linear valueProjection;
    private Linear resultProjection;
    private Dropout attentionProbsDropout;

    private ScaledDotProductAttentionBlock(Builder builder) {
        super((byte)1);
        this.embeddingSize = Builder.access$000((Builder)builder);
        this.headCount = Builder.access$100((Builder)builder);
        this.keyProjection = (Linear)this.addChildBlock("keyProjection", (Block)this.buildProjection());
        this.queryProjection = (Linear)this.addChildBlock("queryProjection", (Block)this.buildProjection());
        this.valueProjection = (Linear)this.addChildBlock("valueProjection", (Block)this.buildProjection());
        this.resultProjection = (Linear)this.addChildBlock("resultProjection", (Block)this.buildProjection());
        this.attentionProbsDropout = (Dropout)this.addChildBlock("probabilityDropout", (Block)Dropout.builder().optRate(Builder.access$200((Builder)builder)).build());
    }

    private Linear buildProjection() {
        return Linear.builder().setUnits((long)this.embeddingSize).optBias(true).build();
    }

    public Linear getKeyProjection() {
        return this.keyProjection;
    }

    public Linear getQueryProjection() {
        return this.queryProjection;
    }

    public Linear getValueProjection() {
        return this.valueProjection;
    }

    public Linear getResultProjection() {
        return this.resultProjection;
    }

    public Shape[] getOutputShapes(Shape[] inputShapes) {
        if (inputShapes.length == 1 || inputShapes.length == 2) {
            return new Shape[]{inputShapes[0]};
        }
        if (inputShapes.length == 3 || inputShapes.length == 4) {
            return new Shape[]{inputShapes[1]};
        }
        throw new IllegalArgumentException("Invalid number of input shapes: " + inputShapes.length + ", must be 1-4.");
    }

    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape ... inputShapes) {
        Shape projectionShape = new Shape(new long[]{-1L, this.embeddingSize});
        for (Block projection : this.children.values()) {
            projection.initialize(manager, DataType.FLOAT32, new Shape[]{projectionShape});
        }
    }

    private NDArray createAttentionHeadsFromEmbeddings(NDArray projection, long B, long S, long N, long H) {
        NDArray sequenceAndHeads = projection.reshape(new long[]{B, S, N, H});
        return sequenceAndHeads.transpose(new int[]{0, 2, 1, 3});
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList flattenedValueInput;
        NDList flattenedQueryInput;
        NDList flattenedKeyInput;
        long T;
        long F;
        long E = this.embeddingSize;
        long B = inputs.head().getShape().get(0);
        long N = this.headCount;
        long H = E / N;
        if (inputs.size() < 3) {
            T = F = inputs.head().getShape().get(1);
            flattenedQueryInput = flattenedKeyInput = new NDList(inputs.head());
            flattenedValueInput = flattenedKeyInput;
        } else {
            F = ((NDArray)inputs.get(0)).getShape().get(1);
            T = ((NDArray)inputs.get(1)).getShape().get(1);
            flattenedKeyInput = new NDList((NDArray)inputs.get(0));
            flattenedQueryInput = new NDList((NDArray)inputs.get(1));
            flattenedValueInput = new NDList((NDArray)inputs.get(2));
        }
        NDArray attentionMask = inputs.size() == 2 || inputs.size() == 4 ? (NDArray)inputs.get(inputs.size() - 1) : null;
        NDList keys = this.keyProjection.forward(parameterStore, flattenedKeyInput, training, params);
        NDList queries = this.queryProjection.forward(parameterStore, flattenedQueryInput, training, params);
        NDList values = this.valueProjection.forward(parameterStore, flattenedValueInput, training, params);
        NDArray keyHeads = this.createAttentionHeadsFromEmbeddings(keys.head(), B, F, N, H);
        NDArray queryHeads = this.createAttentionHeadsFromEmbeddings(queries.head(), B, T, N, H);
        NDArray valueHeads = this.createAttentionHeadsFromEmbeddings(values.head(), B, F, N, H);
        NDArray attentionScores = queryHeads.matMul(keyHeads.transpose(new int[]{0, 1, 3, 2}));
        NDArray normalizedAttentionScores = attentionScores.mul(attentionScores.getManager().create(1.0f / (float)Math.sqrt((double)H)));
        if (attentionMask != null) {
            NDArray maskOffset;
            if (attentionMask.getShape().dimension() != 4) {
                NDArray expandedMask = attentionMask.reshape(new long[]{B, 1L, T, F});
                maskOffset = expandedMask.toType(DataType.FLOAT32, false).mul(expandedMask.getManager().create(-1.0f)).add(expandedMask.getManager().create(1.0f)).mul(expandedMask.getManager().create(-100000.0f));
            } else {
                maskOffset = attentionMask;
            }
            normalizedAttentionScores = normalizedAttentionScores.add(maskOffset);
        }
        NDArray attentionProbs = normalizedAttentionScores.softmax(3);
        NDArray attentionProbsAfterDropout = this.attentionProbsDropout.forward(parameterStore, new NDList(attentionProbs), training).singletonOrThrow();
        NDArray attentionResult = attentionProbsAfterDropout.matMul(valueHeads);
        NDArray resultEmbeddings = attentionResult.transpose(new int[]{0, 2, 1, 3}).reshape(new long[]{B, T, E});
        NDList projectedEmbeddings = this.resultProjection.forward(parameterStore, new NDList(resultEmbeddings), training);
        return new NDList((Collection<NDArray>)projectedEmbeddings);
    }

    public static Builder builder() {
        return new Builder(null);
    }
}
