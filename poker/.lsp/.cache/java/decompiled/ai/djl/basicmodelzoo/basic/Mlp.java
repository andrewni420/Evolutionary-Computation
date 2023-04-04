/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.nn.Block
 *  ai.djl.nn.Blocks
 *  java.lang.Object
 *  java.util.function.Function
 */
package ai.djl.basicmodelzoo.basic;

import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import java.util.function.Function;

public class Mlp
extends SequentialBlock {
    public Mlp(int input, int output, int[] hidden) {
        this(input, output, hidden, (Function<NDList, NDList>)((Function)Activation::relu));
    }

    public Mlp(int input, int output, int[] hidden, Function<NDList, NDList> activation) {
        this.add(Blocks.batchFlattenBlock((long)input));
        for (int hiddenSize : hidden) {
            this.add((Block)Linear.builder().setUnits((long)hiddenSize).build());
            this.add(activation);
        }
        this.add((Block)Linear.builder().setUnits((long)output).build());
    }
}
