/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.MalformedModelException
 *  ai.djl.ndarray.NDManager
 *  ai.djl.nn.AbstractBlock
 *  ai.djl.util.PairList
 *  java.io.DataInputStream
 *  java.io.IOException
 *  java.lang.Object
 *  java.lang.String
 *  java.util.function.Function
 */
package ai.djl.nn;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.function.Function;

public class LambdaBlock
extends AbstractBlock {
    public static final String DEFAULT_NAME = "anonymous";
    private static final byte VERSION = 2;
    private Function<NDList, NDList> lambda;
    private String name;

    public LambdaBlock(Function<NDList, NDList> lambda) {
        this(lambda, DEFAULT_NAME);
    }

    public LambdaBlock(Function<NDList, NDList> lambda, String name) {
        super((byte)2);
        this.lambda = lambda;
        this.name = name;
    }

    public String getName() {
        return this.name;
    }

    public static LambdaBlock singleton(Function<NDArray, NDArray> lambda) {
        return new LambdaBlock((Function<NDList, NDList>)((Function)arrays -> new NDList((NDArray)lambda.apply((Object)arrays.singletonOrThrow()))), lambda.getClass().getSimpleName());
    }

    public static LambdaBlock singleton(Function<NDArray, NDArray> lambda, String name) {
        return new LambdaBlock((Function<NDList, NDList>)((Function)arrays -> new NDList((NDArray)lambda.apply((Object)arrays.singletonOrThrow()))), name);
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        return (NDList)((Object)this.lambda.apply((Object)inputs));
    }

    public Shape[] getOutputShapes(Shape[] inputShapes) {
        try (NDManager manager = NDManager.newBaseManager();){
            NDList input = new NDList(inputShapes.length);
            for (Shape shape : inputShapes) {
                input.add(manager.zeros(shape));
            }
            NDList output = (NDList)((Object)this.lambda.apply((Object)input));
            Shape[] outputShapes = new Shape[output.size()];
            DataType[] dataTypes = new DataType[output.size()];
            for (int i = 0; i < output.size(); ++i) {
                outputShapes[i] = ((NDArray)output.get(i)).getShape();
                dataTypes[i] = ((NDArray)output.get(i)).getDataType();
            }
            this.outputDataTypes = dataTypes;
            Shape[] shapeArray = outputShapes;
            return shapeArray;
        }
    }

    public void loadParameters(NDManager manager, DataInputStream is) throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version == 2) {
            this.readInputShapes(is);
        } else if (version != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
    }
}
