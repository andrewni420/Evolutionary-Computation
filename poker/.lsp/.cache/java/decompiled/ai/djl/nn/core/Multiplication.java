/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.MalformedModelException
 *  ai.djl.nn.AbstractBlock
 *  ai.djl.nn.Parameter
 *  ai.djl.nn.Parameter$Type
 *  ai.djl.nn.core.Multiplication$Builder
 *  ai.djl.util.PairList
 *  ai.djl.util.Preconditions
 *  java.io.DataInputStream
 *  java.io.DataOutputStream
 *  java.io.IOException
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
import ai.djl.nn.core.Multiplication;
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
public class Multiplication
extends AbstractBlock {
    private static final byte VERSION = 1;
    private long units;
    private long inputFeatures;
    private Shape inputShape;
    private Parameter weight;

    Multiplication(Builder builder) {
        super((byte)1);
        this.units = Builder.access$000((Builder)builder);
        this.weight = this.addParameter(Parameter.builder().setName("weight").setType(Parameter.Type.WEIGHT).build());
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        NDArray weightArr = parameterStore.getValue(this.weight, device, training);
        return this.multiply(input, weightArr);
    }

    public Shape[] getOutputShapes(Shape[] inputs) {
        return new Shape[]{new Shape(this.units).addAll(inputs[0])};
    }

    public PairList<String, Shape> describeInput() {
        return new PairList(Collections.singletonList((Object)"linearInput"), Collections.singletonList((Object)this.inputShape));
    }

    protected void beforeInitialize(Shape ... inputShapes) {
        super.beforeInitialize(inputShapes);
        Preconditions.checkArgument((inputShapes.length == 1 ? 1 : 0) != 0, (String)"Linear block only support 1 input");
        Shape input = inputShapes[0];
        this.inputFeatures = input.slice(1).size();
        this.inputShape = input.slice(0, 1);
    }

    public void prepare(Shape[] inputShapes) {
        Shape input = inputShapes[0];
        this.weight.setShape(new Shape(this.units, 1L).addAll(input.slice(1)));
    }

    protected void saveMetadata(DataOutputStream os) throws IOException {
        os.writeLong(this.units);
        os.writeLong(this.inputFeatures);
        os.write(this.inputShape.getEncoded());
    }

    public void loadMetadata(byte loadVersion, DataInputStream is) throws IOException, MalformedModelException {
        if (loadVersion != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
        this.units = is.readLong();
        this.inputFeatures = is.readLong();
        this.inputShape = Shape.decode(is);
    }

    public NDList multiply(NDArray input, NDArray weight) {
        NDArray resultArr = input.mul(weight);
        return new NDList(resultArr);
    }

    public static Builder builder() {
        return new Builder();
    }
}
