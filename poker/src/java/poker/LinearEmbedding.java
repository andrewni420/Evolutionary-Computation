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
package poker;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.util.Preconditions;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;

/**
 * A Linear block applies a linear transformation \(Y = XW^T + b\).
 *
 * <p>It has the following shapes:
 *
 * <ul>
 *   <li>input X: [x1, x2, ..., xn, input_dim]
 *   <li>weight W: [units, input_dim]
 *   <li>Bias b: [units]
 *   <li>output Y: [x1, x2, ..., xn, units]
 * </ul>
 *
 * <p>It is most typically used with a simple batched 1D input. In that case, the shape would be:
 *
 * <ul>
 *   <li>input X: [batch_num, input_dim]
 *   <li>weight W: [units, input_dim]
 *   <li>Bias b: [units]
 *   <li>output Y: [batch_num, units]
 * </ul>
 *
 * <p>The Linear block should be constructed using {@link Linear.Builder}.
 */

 /* A Linear Embedding Block is a modified Linear Block (without bias parameters)
  * that implements the reverse() method in the Embedding interface
  *
  * reverse() multiplies the input by the transpose of the weight matrix
  */
public class LinearEmbedding extends AbstractBlock implements Embedding {

    private static final byte VERSION = 4;

    private long units;
    private long inputFeatures;
    private Shape inputShape;
    private Parameter weight;

    /* Builds a linear embedding block from an instance of the Builder subclass
     * Modified from Linear to not construct bias parameter
     * 
     * Example usage: LinearEmbedding.Builder().setUnits(10).build()
     */
    protected LinearEmbedding(Builder builder) {
        super(VERSION);
        units = builder.units;
        weight =
                addParameter(
                        Parameter.builder()
                                .setName("weight")
                                .setType(Parameter.Type.WEIGHT)
                                .build());
    }

    /* Copied from Linear
     * Uses ParameterStore to account for NDArrays living on different
     * CPU or GPU devices
     */
    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        NDArray weightArr = parameterStore.getValue(weight, device, training);
        return linear(input, weightArr);
    }

    /* Implementation of Embedding interface
     * Same as forward, but multiplies by the transpose of the weight matrix
     * Attaches transpose to the input's NDManager to ensure that the transpose
     * is closed when inference is finished, preventing memory leaks
     */
    /** {@inheritDoc} */
    @Override
    public NDList reverse(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        NDArray weightArr = parameterStore.getValue(weight, device, training);
        NDArray weightTranspose = weightArr.transpose();
        input.getManager().attachAll(weightTranspose);
        return linear(input,weightTranspose);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        return new Shape[] {inputs[0].slice(0, inputs[0].dimension() - 1).add(units)};
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        return new PairList<>(
                Collections.singletonList("linearInput"), Collections.singletonList(inputShape));
    }

    /** {@inheritDoc} */
    @Override
    protected void beforeInitialize(Shape... inputShapes) {
        super.beforeInitialize(inputShapes);
        Preconditions.checkArgument(inputShapes.length == 1, "Linear block only support 1 input");
        Shape input = inputShapes[0];
        inputFeatures = input.get(input.dimension() - 1);
        inputShape = input.slice(0, input.dimension() - 1);
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Shape[] inputShapes) {
        Shape input = inputShapes[0];
        weight.setShape(new Shape(units, input.get(input.dimension() - 1)));
    }

    /** {@inheritDoc} */
    @Override
    protected void saveMetadata(DataOutputStream os) throws IOException {
        os.writeLong(units);
        os.writeLong(inputFeatures);
        os.write(inputShape.getEncoded());
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        switch (loadVersion) {
            case VERSION:
                units = is.readLong();
                inputFeatures = is.readLong();
                break;
            case 3:
                units = is.readLong();
                if (is.readBoolean()) {
                    throw new IllegalArgumentException("flatten is not supported!");
                }
                inputFeatures = is.readLong();
                break;
            case 2:
                if (is.readBoolean()) {
                    throw new IllegalArgumentException("flatten is not supported!");
                }
                inputFeatures = is.readLong();
                break;
            case 1:
                inputFeatures = Shape.decode(is).size();
                break;
            default:
                throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
        inputShape = Shape.decode(is);
        inputShapes = new Shape[] {inputShape};
    }

    /**
     * Applies a linear transformation to the incoming data.
     *
     * @param input input X: [x1, x2, …, xn, input_dim]
     * @param weight weight W: [units, input_dim]
     * @return output Y: [x1, x2, …, xn, units]
     */
    public static NDList linear(NDArray input, NDArray weight) {
        return linear(input, weight, null);
    }

    /**
     * Applies a linear transformation to the incoming data.
     *
     * @param input input X: [x1, x2, …, xn, input_dim]
     * @param weight weight W: [units, input_dim]
     * @param bias bias b: [units]
     * @return output Y: [x1, x2, …, xn, units]
     */
    public static NDList linear(NDArray input, NDArray weight, NDArray bias) {
        return input.getNDArrayInternal().linear(input, weight, bias);
    }

    /**
     * Creates a builder to build a {@code Linear}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link Linear} type of {@link Block}. */
    public static class Builder {

        protected long units;

        protected Builder() {}

        /**
         * Sets the number of output channels.
         *
         * @param units the number of desired output channels
         * @return this Builder
         */
        public Builder setUnits(long units) {
            this.units = units;
            return this;
        }

        /**
         * Returns the constructed {@code Linear}.
         *
         * @return the constructed {@code Linear}
         * @throws IllegalArgumentException if all required parameters (outChannels) have not been
         *     set
         */
        public LinearEmbedding build() {
            Preconditions.checkArgument(units > 0, "You must specify unit");
            return new LinearEmbedding(this);
        }
    }
}