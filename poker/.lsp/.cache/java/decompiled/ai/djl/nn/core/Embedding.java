/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.MalformedModelException
 *  ai.djl.ndarray.NDArrays
 *  ai.djl.ndarray.NDManager
 *  ai.djl.ndarray.types.SparseFormat
 *  ai.djl.nn.AbstractBlock
 *  ai.djl.nn.Parameter
 *  ai.djl.nn.Parameter$Type
 *  ai.djl.nn.core.AbstractIndexedEmbedding
 *  ai.djl.nn.core.Embedding$BaseBuilder
 *  ai.djl.nn.core.Embedding$DefaultEmbedding
 *  ai.djl.nn.core.Embedding$DefaultItem
 *  ai.djl.util.PairList
 *  java.io.DataInputStream
 *  java.io.DataOutputStream
 *  java.io.IOException
 *  java.lang.IllegalArgumentException
 *  java.lang.Math
 *  java.lang.Object
 *  java.lang.String
 *  java.util.Arrays
 */
package ai.djl.nn.core;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.AbstractIndexedEmbedding;
import ai.djl.nn.core.Embedding;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

public abstract class Embedding<T>
extends AbstractBlock
implements AbstractIndexedEmbedding<T> {
    private static final byte VERSION = 6;
    protected int numEmbeddings;
    protected int embeddingSize;
    protected SparseFormat sparseFormat;
    protected AbstractIndexedEmbedding<T> fallthroughEmbedding;
    protected Parameter embedding;

    protected Embedding(BaseBuilder<T, ?> baseBuilder) {
        super((byte)6);
        this.embeddingSize = baseBuilder.embeddingSize;
        this.numEmbeddings = baseBuilder.numEmbeddings != 0 ? baseBuilder.numEmbeddings : 1;
        this.sparseFormat = baseBuilder.sparseFormat;
        this.embedding = this.addParameter(Parameter.builder().setName("embedding").setType(Parameter.Type.WEIGHT).build());
        if (baseBuilder.fallthrough != null && baseBuilder.defaultItem != null) {
            throw new IllegalArgumentException("You can not specify both a fallthrough and a defaultItem");
        }
        if (baseBuilder.fallthrough != null) {
            this.fallthroughEmbedding = baseBuilder.fallthrough;
        } else if (baseBuilder.defaultItem != null) {
            this.fallthroughEmbedding = new DefaultItem(this, baseBuilder.defaultItem);
        } else if (baseBuilder.useDefault) {
            this.fallthroughEmbedding = new DefaultEmbedding(this);
        }
        this.inputShapes = new Shape[]{new Shape(-1L)};
    }

    protected Embedding(NDArray embedding) {
        this(embedding, SparseFormat.DENSE);
    }

    protected Embedding(NDArray embedding, SparseFormat format) {
        super((byte)6);
        this.numEmbeddings = Math.toIntExact((long)embedding.getShape().get(0));
        this.embeddingSize = Math.toIntExact((long)embedding.getShape().get(1));
        this.sparseFormat = format;
        this.embedding = this.addParameter(Parameter.builder().setName("embedding").setType(Parameter.Type.WEIGHT).build());
        this.embedding.setArray(embedding);
        this.inputShapes = new Shape[]{new Shape(-1L)};
        this.freezeParameters(true);
    }

    public void prepare(Shape[] inputShapes) {
        this.embedding.setShape(new Shape(this.numEmbeddings, this.embeddingSize));
    }

    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{inputShapes[0].addAll(new Shape(this.embeddingSize))};
    }

    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray input = inputs.head();
        Device device = input.getDevice();
        NDArray weightArr = parameterStore.getValue(this.embedding, device, training);
        return Embedding.embedding(input, weightArr, this.sparseFormat);
    }

    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(6);
        this.saveInputShapes(os);
        os.writeInt(this.sparseFormat.getValue());
        this.embedding.save(os);
    }

    public void loadParameters(NDManager manager, DataInputStream is) throws IOException, MalformedModelException {
        byte version = is.readByte();
        boolean addMissingZero = false;
        if (version >= 3) {
            this.readInputShapes(is);
            if (version == 3) {
                boolean bl = addMissingZero = !is.readBoolean();
            }
            if (version == 6) {
                this.sparseFormat = SparseFormat.fromValue((int)is.readInt());
            } else {
                SparseFormat sparseFormat = this.sparseFormat = is.readBoolean() ? SparseFormat.ROW_SPARSE : SparseFormat.DENSE;
            }
            if (version < 6) {
                is.readUTF();
            }
            if (version == 3 || version == 4) {
                int embedderSize = is.readInt();
                for (int i = 1; i <= embedderSize; ++i) {
                    int encodedKeySize = is.readInt();
                    byte[] encodedKey = new byte[encodedKeySize];
                    if (is.read(encodedKey) != encodedKey.length) {
                        throw new MalformedModelException("Model data is malformed");
                    }
                    is.readInt();
                }
            }
        } else if (version == 2) {
            this.readInputShapes(is);
            addMissingZero = true;
        } else if (version != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        this.embedding.load(manager, is);
        this.numEmbeddings = (int)this.embedding.getArray().getShape().get(0);
        this.embeddingSize = (int)this.embedding.getArray().getShape().get(1);
        if (addMissingZero) {
            ++this.numEmbeddings;
            this.embedding.setArray(NDArrays.concat((NDList)new NDList(manager.zeros(new Shape(1L, this.embeddingSize)), this.embedding.getArray())));
        }
    }

    public NDArray embed(NDManager manager, T[] items) {
        return manager.create(Arrays.stream((Object[])items).mapToLong(arg_0 -> ((Embedding)this).embed(arg_0)).toArray());
    }

    public static NDList embedding(NDArray input, NDArray weight, SparseFormat sparse) {
        return input.getNDArrayInternal().embedding(input, weight, sparse);
    }
}
