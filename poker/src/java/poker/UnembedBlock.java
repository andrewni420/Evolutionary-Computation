package poker;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.nn.transformer.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/** 
 * Takes a reversible, weight sharing Embedding block and calls reverse() 
 * on its input instead of forward() 
 * Used for weight sharing
 * 
 * Attention is All You Need:
 *  https://arxiv.org/abs/1706.03762
 * Using the Output Embedding to Improve Language Models 
 *  https://arxiv.org/abs/1608.05859
 * **/

public class UnembedBlock extends AbstractBlock{
    /* UnembedBlock is a modified SequentialBlock that only
     * has one child. Not a very good design choice.
     * 
     * Calls reverse() on the first child in forwardInternal()
     */
    private static final byte VERSION = 3;

    public UnembedBlock() {
        super(VERSION);
    }

    public UnembedBlock setEmbedding(Embedding e){
        replaceLastBlock(e);
        return this;
    }

    public void removeLastBlock() {
        if (children.size() > 0) {
            children.remove(children.size() - 1);
        }
    }

    public void replaceLastBlock(Block block) {
        removeLastBlock();
        if (block != null) {
            add(block);
        }
    }

    public UnembedBlock add(Block block) {
        if (block != null) {
            addChildBlock(block.getClass().getSimpleName(), block);
        }
        return this;
    }

    protected NDList forwardInternal(ParameterStore parameterStore,
        NDList input,
        boolean training,
        PairList<String, Object> params){
        Embedding embedding = (Embedding) children.values().get(0);
        
        return new NDList(embedding.reverse(parameterStore, 
                            input, 
                            training,
                            params));
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        Embedding embedding = (Embedding) children.values().get(0);
        return embedding.getInputShapes();
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        
    }

}
