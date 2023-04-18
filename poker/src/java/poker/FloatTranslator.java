package poker;
import ai.djl.ndarray.*;
import ai.djl.translate.*;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;
import java.lang.Number;
/* A translator to convert between Lists (IPersistentVectors) of float arrays and NDLists
 * for neural net inference
*/

public class FloatTranslator implements Translator<List<float[]>, List<float[]>> {
    private Shape[] inputShapes;

    public FloatTranslator(Shape[] shapes){
        this.inputShapes = shapes;
    }   

    @Override
    public NDList processInput(TranslatorContext ctx, List<float[]> input){
        NDManager manager = ctx.getNDManager();
        NDList inputs = new NDList();
        for (int i=0;i<input.size();i++){
            inputs.add(manager.create(input.get(i), inputShapes[i]));
        }
        return inputs;
    }

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
