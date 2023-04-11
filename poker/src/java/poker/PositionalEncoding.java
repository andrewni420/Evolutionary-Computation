package poker;

import java.util.List;

import ai.djl.nn.core.Embedding;
import ai.djl.nn.AbstractBlock;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.nn.Block;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.util.Preconditions;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import javax.swing.text.Position;

public class PositionalEncoding extends AbstractBlock{

    /* Input shapes: (F, numEncodings) */
    /* embeddings[i] must have the signature (F) -> (F, embeddingSizes[i])*/
    /* Overall has the signature (F, numEncodings) -> (F, ΣembeddingSizes)*/
    private List<Integer> embeddingSizes;
    private List<Block> embeddings;

    public PositionalEncoding(){}

    public int getNumEncodings(){ return this.embeddingSizes.size();}

    public PositionalEncoding setEmbeddingSizes(List<Integer> embeddingSizes){
        this.embeddingSizes = embeddingSizes;
        return this;
    }

    public List<Integer> getEmbeddingSizes(){return this.embeddingSizes;}

    public PositionalEncoding setEmbeddings(List<Block> embeddings){
        embeddings.forEach((block) -> addChildBlock(block.getClass().getSimpleName(), block));
        this.embeddings = embeddings;
        return this;
    }

    public List<Block> getEmbeddings(){return this.embeddings;}

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        for (Block child : embeddings) {
            child.initialize(manager, dataType, new Shape[] {inputShapes[0].slice(0,1)});
        }
    }
    
    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        int outputSize = 0;
        for(int i : embeddingSizes){
            outputSize+=i;
        }
        return new Shape[] {inputShapes[0].add(outputSize)};
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
 
        int numEncodings = this.embeddingSizes.size();
        NDArray input = inputs.singletonOrThrow();
        NDArray[] inputArrays = new NDArray[numEncodings];

        for(int i=0;i<numEncodings;i++){
            inputArrays[i] = inputs.singletonOrThrow().get(new NDIndex(":, "+i));
        }

        NDArray[] outputArrays = new NDArray[numEncodings];
        for (int i=0;i<numEncodings;i++){
            outputArrays[i] = this.embeddings.get(i)
                            .forward(ps, new NDList(inputArrays[i]), training, params)
                            .singletonOrThrow();
        }

        NDArray outputArray = outputArrays[0];
        for(int i=1;i<numEncodings;i++){
            outputArray = outputArray.concat(outputArrays[i], -1);
        }

        return new NDList(outputArray);
    }




}