/* 
package poker.Andrew;
import ai.djl.nn.AbstractBlock;
import poker.ParallelEmbedding;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import poker.UnembedBlock;
import ai.djl.ndarray.*;
import ai.djl.util.PairList;

public class PokerAgent extends AbstractBlock{
    //parallel-embedding 1 state-embedding action-embedding
    //positional encoding
    //parallel-inputs positional-encoding parallel-embedding
    //transformer
    //unembed parallel-embedding
    //sequential-block parallel-inputs transformer unembed

    private ParallelEmbedding embedding;
    private Block coreBlock;
    private PositionalEncoding positionalEncoder;
    private UnembedBlock Unembedding;
    

  /** {@inheritDoc} */
  /* --------------------------------------------
   * --------------------TODO-------------------- 
   * -------------------------------------------- 
   */

   /* 
  @Override
  protected NDList forwardInternal(
    ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        return inputs;
    }
}
*/