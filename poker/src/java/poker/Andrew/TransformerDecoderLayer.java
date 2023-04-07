package poker.Andrew;

import ai.djl.nn.SequentialBlock;

import ai.djl.MalformedModelException;
//import ai.djl.inference.streaming.StreamingBlock;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.nn.*;


import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class TransformerDecoderLayer extends SequentialBlock{
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList data,
            NDList labels,
            PairList<String, Object> params) {
        List<NDList> past = new ArrayList<>(children.size());
        NDList current = data;
        for (Block block : children.values()) {
            current = block.forward(parameterStore, current, labels, params);
            past.add(current);
        }
        if (isReturnIntermediate()) {
            return new NDList(
                    past.stream().flatMap(Collection::stream).collect(Collectors.toList()));
        }
        return current;
    }
}
