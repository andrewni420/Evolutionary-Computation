/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.MalformedModelException
 *  ai.djl.ndarray.internal.NDArrayEx
 *  ai.djl.nn.AbstractBlock
 *  ai.djl.nn.Parameter
 *  ai.djl.nn.Parameter$Type
 *  ai.djl.nn.norm.LayerNorm$Builder
 *  ai.djl.util.PairList
 *  java.io.DataInputStream
 *  java.io.DataOutputStream
 *  java.io.IOException
 *  java.lang.Object
 *  java.lang.String
 *  java.util.Arrays
 */
package ai.djl.nn.norm;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

/*
 * Exception performing whole class analysis ignored.
 */
public class LayerNorm
extends AbstractBlock {
    protected float epsilon;
    protected Shape normalizedShape;
    protected boolean center;
    protected boolean scale;
    protected int[] axis;
    protected Parameter gamma;
    protected Parameter beta;

    protected LayerNorm(Builder builder) {
        this.epsilon = Builder.access$000((Builder)builder);
        this.scale = Builder.access$100((Builder)builder);
        this.center = Builder.access$200((Builder)builder);
        this.axis = Builder.access$300((Builder)builder);
        this.gamma = this.addParameter(Parameter.builder().setName("gamma").setType(Parameter.Type.GAMMA).optRequiresGrad(this.scale).build());
        this.beta = this.addParameter(Parameter.builder().setName("beta").setType(Parameter.Type.BETA).optRequiresGrad(this.center).build());
    }

    public static NDList layerNorm(NDArray input, Shape normalizedShape, NDArray gamma, NDArray beta, float eps) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.layerNorm(input, normalizedShape, gamma, beta, eps);
    }

    public static Builder builder() {
        return new Builder();
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        NDArray gammaArr = parameterStore.getValue(this.gamma, device, training);
        NDArray betaArr = parameterStore.getValue(this.beta, device, training);
        return LayerNorm.layerNorm(input, this.normalizedShape, gammaArr, betaArr, this.epsilon);
    }

    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{inputShapes[0]};
    }

    protected void beforeInitialize(Shape ... inputShapes) {
        super.beforeInitialize(inputShapes);
        this.normalizedShape = this.axis == null ? inputShapes[0].slice(1) : new Shape(Arrays.stream((int[])this.axis).mapToLong(dim -> inputShapes[0].get(dim)).toArray());
    }

    public void prepare(Shape[] inputShapes) {
        this.gamma.setShape(this.normalizedShape);
        this.beta.setShape(this.normalizedShape);
    }

    protected void saveMetadata(DataOutputStream os) throws IOException {
        this.saveInputShapes(os);
        os.write(this.normalizedShape.getEncoded());
    }

    public void loadMetadata(byte loadVersion, DataInputStream is) throws IOException, MalformedModelException {
        if (loadVersion != this.version) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
        this.readInputShapes(is);
        this.normalizedShape = Shape.decode(is);
    }
}
