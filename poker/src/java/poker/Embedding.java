package poker;

import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.nn.Block;

/* An interface for token embedding blocks that can be reversed
 * for weight sharing
 * Weight sharing involves multiplying by the transpose of the embedding
 * matrix to unembed outputs
 * 
 * Attention is All You Need:
 *  https://arxiv.org/abs/1706.03762
 * Using the Output Embedding to Improve Language Models 
 *  https://arxiv.org/abs/1608.05859
 */

public interface Embedding extends Block{
    public NDList reverse(ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params);
}
