package poker;

import java.util.List;
import java.util.ArrayList;

/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.util.Preconditions;
import ai.djl.nn.Block;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.LambdaBlock;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * {@code ParallelBlock} is a {@link Block} whose children form a parallel branch in the network and
 * are combined to produce a single output.
 *
 * <p>{@code ParallelBlock} has no direct parameters.
 */
public class SeparateParallelBlock extends AbstractBlock {
     //processes multiple inputs in parallel and combines them using a function

    private static final byte VERSION = 2;

    private Function<List<NDList>, NDList> function;
    private List<Integer> numInputs;

    public SeparateParallelBlock setNumInputs(List<Integer> inputs){
        this.numInputs = inputs;
        return this;
    }

    public SeparateParallelBlock setNumInputs(int index, int input){
        this.numInputs.set(index, input);
        return this;
    }

    /**
     * Creates a parallel block whose branches are combined to form a single output by the given
     * function.
     *
     * @param function the function to define how the parallel branches are combined to form a
     *     single output
     */
    public SeparateParallelBlock(Function<List<NDList>, NDList> function) {
        this(function, Collections.emptyList());
    }

    /**
     * Creates a parallel block whose branches are formed by each block in the list of blocks, and
     * are combined to form a single output by the given function.
     *
     * @param function the function to define how the parallel branches are combined
     * @param blocks the blocks that form each of the parallel branches
     */
    public SeparateParallelBlock(Function<List<NDList>, NDList> function, List<Block> blocks) {
        super(VERSION);
        this.function = function;
        addAll(blocks);
    }

    /**
     * Adds an array of blocks, each of which is a parallel branch.
     *
     * @param blocks the array of blocks to add
     * @return this block
     */
    public final SeparateParallelBlock addAll(Block... blocks) {
        return addAll(Arrays.asList(blocks));
    }

    /**
     * Adds a {@link Collection} of blocks, each of which is a parallel branch.
     *
     * @param blocks the {@link Collection} of blocks to add
     * @return this block
     */
    public final SeparateParallelBlock addAll(Collection<Block> blocks) {
        blocks.forEach(this::add);
        return this;
    }

    /**
     * Adds the given {@link Block} to the block, which is one parallel branch.
     *
     * @param block the block to be added as a parallel branch
     * @return this block
     */
    public final SeparateParallelBlock add(Block block) {
        if (block != null) {
            addChildBlock(block.getClass().getSimpleName(), block);
        }
        return this;
    }

    /**
     * Adds a {@link LambdaBlock}, that applies the given function, to the list of parallel
     * branches.
     *
     * @param f the function that forms the {@link LambdaBlock}
     * @return this block
     */
    public final SeparateParallelBlock add(Function<NDList, NDList> f) {
        return add(new LambdaBlock(f));
    }

    /**
     * Adds a {@link LambdaBlock}, that applies the given function, to the list of parallel
     * branches.
     *
     * @param f the function forms the {@link LambdaBlock}
     * @param name the function name
     * @return this block
     */
    public SeparateParallelBlock add(Function<NDList, NDList> f, String name) {
        return add(new LambdaBlock(f, name));
    }

    /**
     * Adds a {@link LambdaBlock#singleton(Function)}, that applies the given function, to the list
     * of parallel branches.
     *
     * @param f the function forms the {@link LambdaBlock}
     * @return this block
     * @see LambdaBlock#singleton(Function)
     */
    public SeparateParallelBlock addSingleton(Function<NDArray, NDArray> f) {
        return add(LambdaBlock.singleton(f));
    }

    /**
     * Adds a {@link LambdaBlock#singleton(Function)}, that applies the given function, to the list
     * of parallel branches.
     *
     * @param f the function forms the {@link LambdaBlock}
     * @param name the function name
     * @return this block
     * @see LambdaBlock#singleton(Function)
     */
    public SeparateParallelBlock addSingleton(Function<NDArray, NDArray> f, String name) {
        return add(LambdaBlock.singleton(f, name));
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        List<NDList> outputs = new ArrayList<NDList>();
        int j=0;
        for (int i = 0;i<children.size();i++){
            NDList input = new NDList();
            for (int k=0;k<numInputs.get(i);k++) {
                input.add(inputs.get(k+j));
            }
            j+=numInputs.get(i);
            outputs.add(children.values().get(i).forward(parameterStore, input, training, params));
        }
        return function.apply(outputs);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList data,
            NDList labels,
            PairList<String, Object> params) {
        return function.apply(
                children.values().stream()
                        .map(block -> block.forward(parameterStore, data, labels, params))
                        .collect(Collectors.toList()));
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        int j=0;
        for (int i=0;i<children.size();i++){
            Shape[] input = new Shape[(int) numInputs.get(i)];
            for (int k=0;k<numInputs.get(i);k++){
                input[k] = inputShapes[k+j];
            }
            j+=numInputs.get(i);
            children.values().get(i).initialize(manager, dataType, input);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Preconditions.checkArgument(!children.isEmpty(), "The parallel block is empty");

        try (NDManager manager = NDManager.newBaseManager()) {
            List<NDList> inputs = new ArrayList<NDList>();
            int j=0;
            for (int i=0;i<children.size();i++){
                Shape[] input = new Shape[(int) numInputs.get(i)];
                for (int k=0;k<numInputs.get(i);k++){
                    input[k] = inputShapes[k+j];
                }
                j+=numInputs.get(i);
                Shape[] shapes = children.values().get(i).getOutputShapes(input);
                NDList output = new NDList(shapes.length);
                for (Shape shape : shapes) {
                    output.add(manager.create(shape));
                }
                inputs.add(output);
            }

            NDList output = function.apply(inputs);
            Shape[] outputShapes = new Shape[output.size()];
            for (int i = 0; i < output.size(); ++i) {
                outputShapes[i] = output.get(i).getShape();
            }
            return outputShapes;
        }
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        if (loadVersion == version) {
            readInputShapes(is);
        } else if (loadVersion != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
    }
}

