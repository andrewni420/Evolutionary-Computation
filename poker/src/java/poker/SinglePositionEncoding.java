package poker;

import ai.djl.nn.core.Embedding;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Optional;

public class SinglePositionEncoding extends AbstractBlock{
    private NDArray PETable;
    private int embeddingSize;
    private int numEmbeddings;

    public SinglePositionEncoding(NDArray embedding){
        this.PETable = embedding;
        inputShapes = new Shape[]{new Shape(-1)};
        numEmbeddings = Math.toIntExact(PETable.getShape().get(0));
        embeddingSize = Math.toIntExact(PETable.getShape().get(1));
        freezeParameters(true);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(ParameterStore parameterStore,
        NDList inputs,
        boolean training,
        PairList<String, Object> params) {
            NDArray input = inputs.head();
            Device device = input.getDevice();
            return embedding(input, PETable);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {inputShapes[0].addAll(new Shape(embeddingSize))};
    }

    public static NDList embedding(NDArray input, NDArray weight) {
        return input.getNDArrayInternal().embedding(input, weight, SparseFormat.DENSE);
    }
}
