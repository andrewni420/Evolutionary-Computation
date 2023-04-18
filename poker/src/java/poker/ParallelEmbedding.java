package poker;

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
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.List;

import ai.djl.ndarray.index.NDIndex;
import java.lang.StringBuilder;

public class ParallelEmbedding extends AbstractBlock implements Embedding{
    /* Given a core block and multiple embedding blocks,
     * applies the multiple embedding blocks to inputs separately,
     * and interleaves the embedded inputs back together along a given axis.
     * Then passes the interleaved embedded inputs through the central block
     * Then splits the ouput along the given axis, and applies the reverse of the multiple embedding 
     * blocks
     * inputShapes - the shapes for each of the embeddings
     * inputs - the inputs to each of the embeddings
     * The input to the core block is the outputs of the embedding blocks summed along the splitting axis
     * Forward:
     * (B, F1, D1), (B, F2, D2) ... -embed> 
     * (B, F1, E), (B, F2, E) ...   -interleave> 
     * (B, F, E)                    
     * 
     * Reverse:
     * (B, F, E)                    -split>
     * (B, F1, E), (B, F2, E) ...   -unembed>
     * (B, F1, D1), (B, F2, D2) ...
     * 
     * For multiple embeddings of one vector:
     * Forward:
     * (B, F, D1), (B, F, D2) ... -embed> 
     * (B, F, E1), (B, F, E2) ... -interleave> 
     * (B, F, E)                    
     * 
     * Reverse:
     * (B, F, E)                    -split>
     * (B, F, E1), (B, F, E2) ...   -unembed>
     * (B, F, D1), (B, F, D2) ...
     */

     private List<Embedding> embeddings;
     private int axis;
     private String startOfIndex;
     private String endOfIndex;

     public ParallelEmbedding(){
     }

     public ParallelEmbedding setEmbeddings(List<Embedding> embeddings){
        embeddings.forEach((e) -> addChildBlock(e.getClass().getSimpleName(),e));
        this.embeddings = embeddings;
        return this;
     }

     public ParallelEmbedding setEmbeddings(int index, Embedding e){
        this.embeddings.set(index, addChildBlock(e.getClass().getSimpleName(),e));
        return this;
     }


     public ParallelEmbedding setAxis(int axis){
        this.axis = axis;
        return this;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
        ParameterStore parameterStore,
        NDList inputs,
        boolean training,
        PairList<String, Object> params) {
            assert inputs.size() == embeddings.size() : 
                "Input size not equal to Embedding size. Embedding: " +
                embeddings.size() + 
                " Input: " + 
                inputs.size();
            /* (B, F1, D1), (B, F2, D2) ... -embed> 
             * (B, F1, E), (B, F2, E) ...   -interleave> 
             * (B, F, E)
            */
            // System.out.println("ParallelEmbedding");
            // for (int i=0;i<inputs.size();i++){
            //     System.out.println(inputs.get(i));
            // }
            // System.out.println("ParallelEmbeddingDone");

            Shape[] inputShapes = new Shape[inputs.size()];
            for (int i=0;i<inputShapes.length;i++) inputShapes[i] = inputs.get(i).getShape();
        
            
            Shape outputshape = getOutputShapes(inputShapes)[0];
            NDArray firstInput = inputs.head();
            NDArray output = firstInput.getManager().create(outputshape, firstInput.getDataType());

            for (int i=0;i<embeddings.size();i++){
                String index = startOfIndex + i + endOfIndex;
                //System.out.println("" + i + " "+ index);
                //System.out.println(outputshape);
                //System.out.println(embeddings.get(i).forward(parameterStore, new NDList(inputs.get(i)), training, params).singletonOrThrow());
                //System.out.println(output);
                output.set(new NDIndex(index), 
                    embeddings.get(i).forward(parameterStore, new NDList(inputs.get(i)), training, params).singletonOrThrow());
            }

            return new NDList(output);            
    }



    /** {@inheritDoc} */
    @Override
    public NDList reverse(ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params){
            /* (B, F, E) -split> (final outputShape)
             * (B, F1, E), (B, F2, E) ...   -unembed> (outputShapes of embeddings)
             * (B, F1, D1), (B, F2, D2) ...  (inputShapes)
            */
            NDList outputs = new NDList();
            for (int i=0; i<embeddings.size();i++){
                NDArray input = inputs.singletonOrThrow()
                .get(new NDIndex(startOfIndex + ((i+1)%embeddings.size()) + endOfIndex));
                if (input.size()==0) {
                    NDManager manager = input.getManager();
                    Shape shape = inputs.singletonOrThrow().getShape();
                    shape = Shape.update(shape, axis, 0);
                    input = manager.create(new float[]{}, shape);
                }

                outputs.add(embeddings.get(i).reverse(ps, new NDList(input), 
                                                            training, 
                                                            params).singletonOrThrow());
            }

        return outputs;
            
    }

    private void setStartEndIndices(Shape outputshape){
        String startOfIndex = "";// ":, ... ,"
        for (int i=0;i<axis;i++) startOfIndex += ":,";
        String endOfIndex = "::" + embeddings.size();// "i::n"
        this.startOfIndex = startOfIndex;
        this.endOfIndex = endOfIndex;
        //System.out.println("Start " + startOfIndex);
        //System.out.println("End " + endOfIndex);
    }

    /** {@inheritDoc} */
    @Override
    public void initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        for (int i=0;i<embeddings.size();i++) {
            embeddings.get(i).initialize(manager, dataType, inputShapes[i]);
        }
        Shape outputshape = getOutputShapes(inputShapes)[0];
        setStartEndIndices(outputshape);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        int embeddingsLength = embeddings.size();

        Shape[] input = new Shape[] {inputShapes[0]};
        //System.out.println(input);
        //System.out.println("Kek");
        Shape outputShape = embeddings.get(0).getOutputShapes(new Shape[] {inputShapes[0]})[0];
       for (int i=0;i<inputShapes.length;i++){
        //System.out.println (inputShapes[i]);
       }
       //System.out.println("Yee");
        for (int i=1;i<embeddingsLength;i++) {
            input[0] = inputShapes[i];
            //System.out.println(embeddings.get(i).getOutputShapes(input)[0].get(axis));
            //System.out.println(outputShape.get(axis));
            Shape nextShape = embeddings.get(i).getOutputShapes(input)[0];
            if (nextShape.dimension() > axis){
                //System.out.println(nextShape);
                //System.out.println(outputShape);
                outputShape = Shape.update(outputShape, axis, outputShape.get(axis) + nextShape.get(axis));
            }
            
            //System.out.println(outputShape);
        }

        return new Shape[]{outputShape};
    }

}
