package poker;

import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.nn.Block;

public interface Embedding extends Block{
    public NDList reverse(ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params);
}
