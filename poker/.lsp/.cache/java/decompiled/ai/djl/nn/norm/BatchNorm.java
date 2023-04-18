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
 *  ai.djl.nn.norm.BatchNorm$BaseBuilder
 *  ai.djl.nn.norm.BatchNorm$Builder
 *  ai.djl.util.PairList
 *  java.io.DataInputStream
 *  java.io.DataOutputStream
 *  java.io.IOException
 *  java.lang.Object
 *  java.lang.String
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
import ai.djl.nn.norm.BatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class BatchNorm
extends AbstractBlock {
    private static final byte VERSION = 2;
    private int axis;
    private float epsilon;
    private float momentum;
    private long inChannels;
    private boolean center;
    private boolean scale;
    private Parameter gamma;
    private Parameter beta;
    private Parameter runningMean;
    private Parameter runningVar;

    BatchNorm(BaseBuilder<?> builder) {
        super((byte)2);
        this.axis = builder.axis;
        this.epsilon = builder.epsilon;
        this.momentum = builder.momentum;
        this.center = builder.center;
        this.scale = builder.scale;
        this.gamma = this.addParameter(Parameter.builder().setName("gamma").setType(Parameter.Type.GAMMA).optRequiresGrad(this.scale).build());
        this.beta = this.addParameter(Parameter.builder().setName("beta").setType(Parameter.Type.BETA).optRequiresGrad(this.center).build());
        this.runningMean = this.addParameter(Parameter.builder().setName("runningMean").setType(Parameter.Type.RUNNING_MEAN).optRequiresGrad(false).build());
        this.runningVar = this.addParameter(Parameter.builder().setName("runningVar").setType(Parameter.Type.RUNNING_VAR).optRequiresGrad(false).build());
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        NDArray gammaArr = parameterStore.getValue(this.gamma, device, training);
        NDArray betaArr = parameterStore.getValue(this.beta, device, training);
        NDArray runningMeanArr = parameterStore.getValue(this.runningMean, device, training);
        NDArray runningVarArr = parameterStore.getValue(this.runningVar, device, training);
        return BatchNorm.batchNorm(input, runningMeanArr, runningVarArr, gammaArr, betaArr, this.axis, this.momentum, this.epsilon, training);
    }

    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{inputShapes[0]};
    }

    protected void beforeInitialize(Shape ... inputShapes) {
        super.beforeInitialize(inputShapes);
        this.inChannels = inputShapes[0].size(this.axis);
    }

    public void prepare(Shape[] inputShapes) {
        this.gamma.setShape(new Shape(this.inChannels));
        this.beta.setShape(new Shape(this.inChannels));
        this.runningMean.setShape(new Shape(this.inChannels));
        this.runningVar.setShape(new Shape(this.inChannels));
    }

    protected void saveMetadata(DataOutputStream os) throws IOException {
        this.saveInputShapes(os);
        os.writeLong(this.inChannels);
    }

    public void loadMetadata(byte loadVersion, DataInputStream is) throws IOException, MalformedModelException {
        if (loadVersion == 2) {
            this.readInputShapes(is);
        } else if (loadVersion != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
        this.inChannels = is.readLong();
    }

    public static NDList batchNorm(NDArray input, NDArray runningMean, NDArray runningVar) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.batchNorm(input, runningMean, runningVar, null, null, 1, 0.9f, 1.0E-5f, true);
    }

    public static NDList batchNorm(NDArray input, NDArray runningMean, NDArray runningVar, NDArray gamma, NDArray beta) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.batchNorm(input, runningMean, runningVar, gamma, beta, 1, 0.9f, 1.0E-5f, true);
    }

    public static NDList batchNorm(NDArray input, NDArray runningMean, NDArray runningVar, NDArray gamma, NDArray beta, int axis) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.batchNorm(input, runningMean, runningVar, gamma, beta, axis, 0.9f, 1.0E-5f, true);
    }

    public static NDList batchNorm(NDArray input, NDArray runningMean, NDArray runningVar, NDArray gamma, NDArray beta, int axis, float momentum, float eps, boolean training) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.batchNorm(input, runningMean, runningVar, gamma, beta, axis, momentum, eps, training);
    }

    public static BaseBuilder<?> builder() {
        return new Builder();
    }
}
