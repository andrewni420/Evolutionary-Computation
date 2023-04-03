/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.MalformedModelException
 *  ai.djl.ndarray.NDManager
 *  ai.djl.nn.AbstractBlock
 *  ai.djl.nn.Block
 *  ai.djl.nn.LambdaBlock
 *  ai.djl.util.PairList
 *  ai.djl.util.Preconditions
 *  java.io.DataInputStream
 *  java.io.IOException
 *  java.lang.Object
 *  java.lang.String
 *  java.util.ArrayList
 *  java.util.Arrays
 *  java.util.Collection
 *  java.util.Collections
 *  java.util.List
 *  java.util.function.Function
 *  java.util.stream.Collectors
 */
package ai.djl.nn;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.util.Preconditions;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ParallelBlock
extends AbstractBlock {
    private static final byte VERSION = 2;
    private Function<List<NDList>, NDList> function;

    public ParallelBlock(Function<List<NDList>, NDList> function) {
        this(function, (List<Block>)Collections.emptyList());
    }

    public ParallelBlock(Function<List<NDList>, NDList> function, List<Block> blocks) {
        super((byte)2);
        this.function = function;
        this.addAll((Collection<Block>)blocks);
    }

    public final ParallelBlock addAll(Block ... blocks) {
        return this.addAll((Collection<Block>)Arrays.asList((Object[])blocks));
    }

    public final ParallelBlock addAll(Collection<Block> blocks) {
        blocks.forEach(this::add);
        return this;
    }

    public final ParallelBlock add(Block block) {
        if (block != null) {
            this.addChildBlock(block.getClass().getSimpleName(), block);
        }
        return this;
    }

    public final ParallelBlock add(Function<NDList, NDList> f) {
        return this.add((Block)new LambdaBlock(f));
    }

    public ParallelBlock add(Function<NDList, NDList> f, String name) {
        return this.add((Block)new LambdaBlock(f, name));
    }

    public ParallelBlock addSingleton(Function<NDArray, NDArray> f) {
        return this.add((Block)LambdaBlock.singleton(f));
    }

    public ParallelBlock addSingleton(Function<NDArray, NDArray> f, String name) {
        return this.add((Block)LambdaBlock.singleton(f, (String)name));
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        return (NDList)((Object)this.function.apply((Object)((List)this.children.values().stream().map(block -> block.forward(parameterStore, inputs, training, params)).collect(Collectors.toList()))));
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList data, NDList labels, PairList<String, Object> params) {
        return (NDList)((Object)this.function.apply((Object)((List)this.children.values().stream().map(block -> block.forward(parameterStore, data, labels, params)).collect(Collectors.toList()))));
    }

    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape ... inputShapes) {
        for (Block child : this.getChildren().values()) {
            child.initialize(manager, dataType, inputShapes);
        }
    }

    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Preconditions.checkArgument((!this.children.isEmpty() ? 1 : 0) != 0, (String)"The parallel block is empty");
        try (NDManager manager = NDManager.newBaseManager();){
            ArrayList inputs = new ArrayList();
            for (Block block : this.children.values()) {
                Shape[] shapes = block.getOutputShapes(inputShapes);
                NDList output = new NDList(shapes.length);
                for (Shape shape : shapes) {
                    output.add(manager.create(shape));
                }
                inputs.add((Object)output);
            }
            NDList output = (NDList)((Object)this.function.apply((Object)inputs));
            Shape[] outputShapes = new Shape[output.size()];
            for (int i = 0; i < output.size(); ++i) {
                outputShapes[i] = ((NDArray)output.get(i)).getShape();
            }
            Shape[] shapeArray = outputShapes;
            return shapeArray;
        }
    }

    public void loadMetadata(byte loadVersion, DataInputStream is) throws IOException, MalformedModelException {
        if (loadVersion == this.version) {
            this.readInputShapes(is);
        } else if (loadVersion != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
    }
}
