package poker;
import ai.djl.ndarray.*;
import ai.djl.translate.*;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;
import java.lang.Number;

/* A translator to convert between Lists (IPersistentVectors) of float arrays and NDLists
 * for neural net inference
 * 
 * Translator converts a list of float arrays to an NDList of NDArrays with shapes
 * given by inputShapes. 
 * The model then runs inference on the NDList of inputs
 * The translator then converts the first NDArray in the output NDList
 * to a float array and returns it in a singleton list (not a very good design choice)
*/

public class FloatTranslator implements Translator<List<float[]>, List<float[]>> {
    private Shape[] inputShapes;

    public FloatTranslator(Shape[] shapes){
        this.inputShapes = shapes;
    }   

    /* Convert a list of float inputs into an NDList of NDArrays
     *
     * Example usage:
     * input = List<>([1 2 3 4 5 6], [1 2 3], [1])
     * inputShapes = [Shape(2,3), Shape(3), Shape(1,1,1)]
     * ->
     * NDList( NDArray [[1 2 3] [4 5 6]],
     *         NDArray [1 2 3],
     *         NDArray [[[1]]])
     */
    @Override
    public NDList processInput(TranslatorContext ctx, List<float[]> input){
        NDManager manager = ctx.getNDManager();
        NDList inputs = new NDList();
        for (int i=0;i<input.size();i++){
            inputs.add(manager.create(input.get(i), inputShapes[i]));
        }
        return inputs;
    }

    /* Converts an NDList output into a singleton list containing the first
     * NDArray in the output as a float array
     * 
     * Example usage:
     * NDList( NDArray [[1 2 3] [4 5 6]],
     *         NDArray [1 2 3],
     *         NDArray [[[1]]])
     * ->
     * List<float[]>([1 2 3 4 5 6])
     */
    @Override 
    public List<float[]> processOutput(TranslatorContext ctx, NDList list){
        List<float[]> output = new ArrayList<float[]>();
        output.add(list.get(0).toFloatArray());
        // output.add(list.get(1).toFloatArray());
        return output;
    }

    @Override
    public Batchifier getBatchifier(){
        return Batchifier.STACK;
    }
}
