/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.MalformedModelException
 *  ai.djl.ndarray.NDArray
 *  ai.djl.nn.AbstractBlock
 *  ai.djl.nn.Parameter
 *  ai.djl.nn.Parameter$Type
 *  ai.djl.nn.core.Linear$Builder
 *  ai.djl.training.ParameterStore
 *  ai.djl.util.PairList
 *  ai.djl.util.Preconditions
 *  java.io.DataInputStream
 *  java.io.DataOutputStream
 *  java.io.IOException
 *  java.lang.IllegalArgumentException
 *  java.lang.Object
 *  java.lang.String
 *  java.util.Collections
 */
package ai.djl.nn.core;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.util.Preconditions;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;

/*
 * Exception performing whole class analysis ignored.
 */
public class Linear
extends AbstractBlock {
    private static final byte VERSION = 4;
    private long units;
    private long inputFeatures;
    private Shape inputShape;
    private Parameter weight;
    private Parameter bias;

    protected Linear(Builder builder) {
        super((byte)4);
        this.units = builder.units;
        this.weight = this.addParameter(Parameter.builder().setName("weight").setType(Parameter.Type.WEIGHT).build());
        if (Builder.access$000((Builder)builder)) {
            this.bias = this.addParameter(Parameter.builder().setName("bias").setType(Parameter.Type.BIAS).build());
        }
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        NDArray weightArr = parameterStore.getValue(this.weight, device, training);
        NDArray biasArr = parameterStore.getValue(this.bias, device, training);
        return Linear.linear(input, weightArr, biasArr);
    }

    public Shape[] getOutputShapes(Shape[] inputs) {
        return new Shape[]{inputs[0].slice(0, inputs[0].dimension() - 1).add(this.units)};
    }

    public PairList<String, Shape> describeInput() {
        return new PairList(Collections.singletonList((Object)"linearInput"), Collections.singletonList((Object)this.inputShape));
    }

    protected void beforeInitialize(Shape ... inputShapes) {
        super.beforeInitialize(inputShapes);
        Preconditions.checkArgument((inputShapes.length == 1 ? 1 : 0) != 0, (String)"Linear block only support 1 input");
        Shape input = inputShapes[0];
        this.inputFeatures = input.get(input.dimension() - 1);
        this.inputShape = input.slice(0, input.dimension() - 1);
    }

    public void prepare(Shape[] inputShapes) {
        Shape input = inputShapes[0];
        this.weight.setShape(new Shape(this.units, input.get(input.dimension() - 1)));
        if (this.bias != null) {
            this.bias.setShape(new Shape(this.units));
        }
    }

    protected void saveMetadata(DataOutputStream os) throws IOException {
        os.writeLong(this.units);
        os.writeLong(this.inputFeatures);
        os.write(this.inputShape.getEncoded());
    }

    public void loadMetadata(byte loadVersion, DataInputStream is) throws IOException, MalformedModelException {
        switch (loadVersion) {
            case 4: {
                this.units = is.readLong();
                this.inputFeatures = is.readLong();
                break;
            }
            case 3: {
                this.units = is.readLong();
                if (is.readBoolean()) {
                    throw new IllegalArgumentException("flatten is not supported!");
                }
                this.inputFeatures = is.readLong();
                break;
            }
            case 2: {
                if (is.readBoolean()) {
                    throw new IllegalArgumentException("flatten is not supported!");
                }
                this.inputFeatures = is.readLong();
                break;
            }
            case 1: {
                this.inputFeatures = Shape.decode(is).size();
                break;
            }
            default: {
                throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
            }
        }
        this.inputShape = Shape.decode(is);
    }

    public static NDList linear(NDArray input, NDArray weight) {
        return Linear.linear(input, weight, null);
    }

    public static NDList linear(NDArray input, NDArray weight, NDArray bias) {
        return input.getNDArrayInternal().linear(input, weight, bias);
    }

    public static Builder builder() {
        return new Builder();
    }
}
