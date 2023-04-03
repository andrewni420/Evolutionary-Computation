package poker;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.nn.transformer.*;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.Model;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.listener.TrainingListener;
import ai.djl.metric.Metrics;
import ai.djl.training.dataset.Batch;
import ai.djl.nn.ParameterList;
import ai.djl.nn.Block;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.nn.core.Linear;
import java.io.IOException;
import ai.djl.translate.TranslateException;




import java.util.Collections;
import java.util.function.Function;

public class Test {
    public static void main() throws IOException, TranslateException{
        NDManager manager = NDManager.newBaseManager();

        NDArray trueW = manager.create(new float[]{2, -3.4f});
        float trueB = 4.2f;
        
        DataPoints dp = syntheticData(manager, trueW, trueB, 1000);
        NDArray features = dp.getX();
        NDArray labels = dp.getY();
        int batchSize = 10;
        ArrayDataset dataset = loadArray(features, labels, batchSize, false);
        
        Model model = Model.newInstance("lin-reg");

        SequentialBlock net = new SequentialBlock();
        Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build();
        net.add(linearBlock);
        
        model.setBlock(net);


        Loss l2loss = Loss.l2Loss();

        Tracker lrt = Tracker.fixed(0.03f);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();
        
        DefaultTrainingConfig config = new DefaultTrainingConfig(l2loss)
            .optOptimizer(sgd) // Optimizer (loss function)
            .optDevices(manager.getEngine().getDevices(1)) // single GPU
            .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Trainer trainer = model.newTrainer(config);

        trainer.initialize(new Shape(batchSize, 2));

        Metrics metrics = new Metrics();
        trainer.setMetrics(metrics);

        int numEpochs = 3;

        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            System.out.printf("Epoch %d\n", epoch);
            // Iterate over dataset
            for (Batch batch : trainer.iterateDataset(dataset)) {
                // Update loss and evaulator
                EasyTrain.trainBatch(trainer, batch);
                
                // Update parameters
                trainer.step();
                
                batch.close();
            }
            // reset training and validation evaluators at end of epoch
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
        }
        Block layer = model.getBlock();
        ParameterList params = layer.getParameters();
        NDArray wParam = params.valueAt(0).getArray();
        NDArray bParam = params.valueAt(1).getArray();
        
        float[] w = trueW.sub(wParam.reshape(trueW.getShape())).toFloatArray();
        System.out.printf("Error in estimating w: [%f %f]\n", w[0], w[1]);
        System.out.printf("Error in estimating b: %f\n", trueB - bParam.getFloat());
        model.close();
    }

    public static DataPoints syntheticData(NDManager manager, NDArray w, float b, int numExamples) {
        NDArray X = manager.randomNormal(new Shape(numExamples, w.size()));
        NDArray y = X.dot(w).add(b);
        // Add noise
        y = y.add(manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32));
        return new DataPoints(X, y);
    }

    public static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle) {
        return new ArrayDataset.Builder()
                      .setData(features) // set the features
                      .optLabels(labels) // set the labels
                      .setSampling(batchSize, shuffle) // set the batch size and random sampling
                      .build();
    }
    
}

class DataPoints {
    private NDArray X, y;
    public DataPoints(NDArray X, NDArray y) {
        this.X = X;
        this.y = y;
    }
    
    public NDArray getX() {
        return X;
    }
    
    public NDArray getY() {
        return y;
    }
}
