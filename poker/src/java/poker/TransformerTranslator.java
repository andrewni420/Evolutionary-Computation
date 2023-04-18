package poker;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.translate.*;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;
import java.lang.Number;
/* A translator to convert between Lists (IPersistentVectors) of float arrays and NDLists
 * for neural net inference
*/

public class TransformerTranslator implements Translator<List<NDArray>, List<float[]>> {
    private NDManager externalManager;

    public TransformerTranslator(NDManager manager){this.externalManager = manager;}

    @Override
    public NDList processInput(TranslatorContext ctx, List<NDArray> input){
        NDManager manager = ctx.getNDManager();
        NDList inputs = new NDList();
        manager.attachAll(input.toArray(new NDArray[]{}));
        inputs.addAll(input);
        return inputs;
    }

    @Override 
    public List<float[]> processOutput(TranslatorContext ctx, NDList output){
        List<float[]> outList = new ArrayList<float[]>();
        try(NDManager submanager = ctx.getNDManager().newSubManager()){
            submanager.attachAll(output);
            for (int i=0;i<output.size();i++){
                if ((int) output.get(i).size()==0){
                    outList.add(new float[]{});
                } else {
                    outList.add(output.get(i).get(new NDIndex("...,-1,:")).toFloatArray());
                }
            }
        }
        return outList;
    }

    @Override
    public Batchifier getBatchifier(){
        return Batchifier.STACK;
    }
}
