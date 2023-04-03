/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.nn.Block
 *  ai.djl.nn.LambdaBlock
 *  java.lang.Integer
 *  java.lang.Object
 *  java.util.Iterator
 *  java.util.List
 *  java.util.function.Function
 */
package ai.djl.nn.transformer;

import ai.djl.ndarray.NDList;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;

public class PointwiseFeedForwardBlock
extends SequentialBlock {
    public PointwiseFeedForwardBlock(List<Integer> hiddenSizes, int outputSize, Function<NDList, NDList> activationFunction) {
        Iterator iterator = hiddenSizes.iterator();
        while (iterator.hasNext()) {
            int hiddenSize = (Integer)iterator.next();
            this.add((Block)Linear.builder().optBias(true).setUnits((long)hiddenSize).build());
            this.add((Block)new LambdaBlock(activationFunction));
        }
        this.add((Block)Linear.builder().optBias(true).setUnits((long)outputSize).build());
    }
}
