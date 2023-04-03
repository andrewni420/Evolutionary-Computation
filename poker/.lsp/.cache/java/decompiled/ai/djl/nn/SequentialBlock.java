/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.MalformedModelException
 *  ai.djl.ndarray.NDArray
 *  ai.djl.ndarray.NDManager
 *  ai.djl.nn.AbstractBlock
 *  ai.djl.nn.Block
 *  ai.djl.nn.LambdaBlock
 *  ai.djl.training.ParameterStore
 *  ai.djl.util.PairList
 *  java.io.DataInputStream
 *  java.io.DataOutputStream
 *  java.io.IOException
 *  java.lang.IllegalArgumentException
 *  java.lang.Object
 *  java.lang.String
 *  java.util.ArrayList
 *  java.util.Arrays
 *  java.util.Collection
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
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.function.Function;
import java.util.stream.Collectors;

public class SequentialBlock
extends AbstractBlock {
    private static final byte VERSION = 3;
    private boolean returnIntermediate;

    public SequentialBlock() {
        super((byte)3);
    }

    public SequentialBlock addAll(Block ... blocks) {
        this.addAll((Collection<Block>)Arrays.asList((Object[])blocks));
        return this;
    }

    public SequentialBlock addAll(Collection<Block> blocks) {
        blocks.forEach(this::add);
        return this;
    }

    public SequentialBlock add(Block block) {
        if (block != null) {
            this.addChildBlock(block.getClass().getSimpleName(), block);
        }
        return this;
    }

    public SequentialBlock add(Function<NDList, NDList> f) {
        this.add((Block)new LambdaBlock(f));
        return this;
    }

    public SequentialBlock add(Function<NDList, NDList> f, String name) {
        this.add((Block)new LambdaBlock(f, name));
        return this;
    }

    public SequentialBlock addSingleton(Function<NDArray, NDArray> f) {
        this.add((Block)LambdaBlock.singleton(f));
        return this;
    }

    public SequentialBlock addSingleton(Function<NDArray, NDArray> f, String name) {
        this.add((Block)LambdaBlock.singleton(f, (String)name));
        return this;
    }

    public void removeLastBlock() {
        this.children.remove(this.children.size() - 1);
    }

    public void replaceLastBlock(Block block) {
        this.removeLastBlock();
        if (block != null) {
            this.add(block);
        }
    }

    public boolean isReturnIntermediate() {
        return this.returnIntermediate;
    }

    public SequentialBlock setReturnIntermediate(boolean returnIntermediate) {
        this.returnIntermediate = returnIntermediate;
        return this;
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        ArrayList past = new ArrayList(this.children.size());
        NDList current = inputs;
        for (Block block : this.children.values()) {
            current = block.forward(parameterStore, current, training);
            past.add((Object)current);
        }
        if (this.returnIntermediate) {
            return new NDList((Collection<NDArray>)((Collection)past.stream().flatMap(Collection::stream).collect(Collectors.toList())));
        }
        return current;
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList data, NDList labels, PairList<String, Object> params) {
        ArrayList past = new ArrayList(this.children.size());
        NDList current = data;
        for (Block block : this.children.values()) {
            current = block.forward(parameterStore, current, labels, params);
            past.add((Object)current);
        }
        if (this.returnIntermediate) {
            return new NDList((Collection<NDArray>)((Collection)past.stream().flatMap(Collection::stream).collect(Collectors.toList())));
        }
        return current;
    }

    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape ... inputShapes) {
        Shape[] shapes = inputShapes;
        DataType[] lastDataTypes = null;
        for (Block child : this.getChildren().values()) {
            child.initialize(manager, dataType, shapes);
            shapes = child.getOutputShapes(shapes, lastDataTypes);
            lastDataTypes = child.getOutputDataTypes();
        }
    }

    public Shape[] getOutputShapes(Shape[] inputs) {
        if (this.children.isEmpty()) {
            throw new IllegalArgumentException("The sequential block is empty");
        }
        ArrayList past = new ArrayList(this.children.size());
        Shape[] current = inputs;
        for (Block block : this.children.values()) {
            current = block.getOutputShapes(current);
            past.add((Object)current);
        }
        if (this.returnIntermediate) {
            return (Shape[])past.stream().flatMap(Arrays::stream).toArray(Shape[]::new);
        }
        return current;
    }

    protected void saveMetadata(DataOutputStream os) throws IOException {
        this.saveInputShapes(os);
        os.writeBoolean(this.returnIntermediate);
    }

    public void loadMetadata(byte loadVersion, DataInputStream is) throws IOException, MalformedModelException {
        if (loadVersion == this.version) {
            this.readInputShapes(is);
            this.returnIntermediate = is.readBoolean();
        } else if (loadVersion == 2) {
            this.readInputShapes(is);
        } else {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
    }
}
