package poker.Andrew;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.List;

import ai.djl.ndarray.index.NDIndex;
import java.lang.StringBuilder;

public class AddInput extends AbstractBlock{
    /* Given a core block and an NDList of additional inputs,
     * takes some inputs, appends the additional inputs, and puts it through the core block.
     */

     private NDList additionalInputs;
     private Block coreBlock;

     public AddInput(){
     }

     public AddInput setAdditionalInputs(NDList inputs){
        this.additionalInputs = inputs;

        return this;
     }

     public NDList getAdditionalInputs(){ return this.additionalInputs;}

     public AddInput setCoreBlock(Block block){
        addChildBlock(block.getClass().getSimpleName(),block);
        this.coreBlock = block;
        return this;
     }

     public Block getCoreBlock(){ return this.coreBlock; }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
        ParameterStore parameterStore,
        NDList inputs,
        boolean training,
        PairList<String, Object> params) {
            return coreBlock.forward(parameterStore, inputs.addAll(additionalInputs), training, params);            
    }

    public Shape[] getTotalInputShapes(Shape[] inputShapes){
        Shape[] inputs = new Shape[inputShapes.length + additionalInputs.size()];
        for (int i=0;i<inputShapes.length;i++) inputs[i] = inputShapes[i];
        for (int i=0;i<additionalInputs.size();i++) inputs[i+inputShapes.length] = additionalInputs.get(i).getShape();
        return inputs;
    }

    /** {@inheritDoc} */
    @Override
    public void initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        coreBlock.initialize(manager, dataType, getTotalInputShapes(inputShapes));
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return coreBlock.getOutputShapes(getTotalInputShapes(inputShapes));
    }

}
