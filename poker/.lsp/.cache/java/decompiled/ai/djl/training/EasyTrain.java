/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.training.Trainer
 *  ai.djl.training.dataset.Batch
 *  ai.djl.training.dataset.Dataset
 *  ai.djl.training.listener.TrainingListener$BatchData
 *  ai.djl.translate.TranslateException
 *  ai.djl.util.Preconditions
 *  java.io.IOException
 *  java.lang.IllegalArgumentException
 *  java.lang.Object
 *  java.lang.String
 *  java.lang.System
 *  java.util.ArrayList
 *  java.util.Map
 *  java.util.concurrent.CompletableFuture
 *  java.util.concurrent.ConcurrentHashMap
 *  java.util.concurrent.Executor
 *  java.util.concurrent.ExecutorService
 */
package ai.djl.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.translate.TranslateException;
import ai.djl.util.Preconditions;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;

public final class EasyTrain {
    private EasyTrain() {
    }

    public static void fit(Trainer trainer, int numEpoch, Dataset trainingDataset, Dataset validateDataset) throws IOException, TranslateException {
        for (int epoch = 0; epoch < numEpoch; ++epoch) {
            for (Batch batch : trainer.iterateDataset(trainingDataset)) {
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();
                batch.close();
            }
            EasyTrain.evaluateDataset(trainer, validateDataset);
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
        }
    }

    public static void trainBatch(Trainer trainer, Batch batch) {
        if (trainer.getManager().getEngine() != batch.getManager().getEngine()) {
            throw new IllegalArgumentException("The data must be on the same engine as the trainer. You may need to change one of your NDManagers.");
        }
        Batch[] splits = batch.split(trainer.getDevices(), false);
        TrainingListener.BatchData batchData = new TrainingListener.BatchData(batch, (Map)new ConcurrentHashMap(), (Map)new ConcurrentHashMap());
        try (GradientCollector collector = trainer.newGradientCollector();){
            if (splits.length > 1 && trainer.getExecutorService().isPresent()) {
                ExecutorService executor = (ExecutorService)trainer.getExecutorService().get();
                ArrayList futures = new ArrayList(splits.length);
                for (Batch split : splits) {
                    futures.add((Object)CompletableFuture.supplyAsync(() -> EasyTrain.trainSplit(trainer, collector, batchData, split), (Executor)executor));
                }
                CompletableFuture.allOf((CompletableFuture[])((CompletableFuture[])futures.stream().toArray(CompletableFuture[]::new)));
            } else {
                for (Batch split : splits) {
                    EasyTrain.trainSplit(trainer, collector, batchData, split);
                }
            }
        }
        trainer.notifyListeners(listener -> listener.onTrainingBatch(trainer, batchData));
    }

    private static boolean trainSplit(Trainer trainer, GradientCollector collector, TrainingListener.BatchData batchData, Batch split) {
        NDList data = split.getData();
        NDList labels = split.getLabels();
        NDList preds = trainer.forward(data, labels);
        long time = System.nanoTime();
        NDArray lossValue = trainer.getLoss().evaluate(labels, preds);
        collector.backward(lossValue);
        trainer.addMetric("backward", time);
        time = System.nanoTime();
        batchData.getLabels().put((Object)((NDArray)labels.get(0)).getDevice(), (Object)labels);
        batchData.getPredictions().put((Object)((NDArray)preds.get(0)).getDevice(), (Object)preds);
        trainer.addMetric("training-metrics", time);
        return true;
    }

    public static void validateBatch(Trainer trainer, Batch batch) {
        Preconditions.checkArgument((trainer.getManager().getEngine() == batch.getManager().getEngine() ? 1 : 0) != 0, (String)"The data must be on the same engine as the trainer. You may need to change one of your NDManagers.");
        Batch[] splits = batch.split(trainer.getDevices(), false);
        TrainingListener.BatchData batchData = new TrainingListener.BatchData(batch, (Map)new ConcurrentHashMap(), (Map)new ConcurrentHashMap());
        if (splits.length > 1 && trainer.getExecutorService().isPresent()) {
            ExecutorService executor = (ExecutorService)trainer.getExecutorService().get();
            ArrayList futures = new ArrayList(splits.length);
            for (Batch split : splits) {
                futures.add((Object)CompletableFuture.supplyAsync(() -> EasyTrain.validateSplit(trainer, batchData, split), (Executor)executor));
            }
            CompletableFuture.allOf((CompletableFuture[])((CompletableFuture[])futures.stream().toArray(CompletableFuture[]::new)));
        } else {
            for (Batch split : splits) {
                EasyTrain.validateSplit(trainer, batchData, split);
            }
        }
        trainer.notifyListeners(listener -> listener.onValidationBatch(trainer, batchData));
    }

    private static boolean validateSplit(Trainer trainer, TrainingListener.BatchData batchData, Batch split) {
        NDList data = split.getData();
        NDList labels = split.getLabels();
        NDList preds = trainer.evaluate(data);
        batchData.getLabels().put((Object)((NDArray)labels.get(0)).getDevice(), (Object)labels);
        batchData.getPredictions().put((Object)((NDArray)preds.get(0)).getDevice(), (Object)preds);
        return true;
    }

    public static void evaluateDataset(Trainer trainer, Dataset testDataset) throws IOException, TranslateException {
        if (testDataset != null) {
            for (Batch batch : trainer.iterateDataset(testDataset)) {
                EasyTrain.validateBatch(trainer, batch);
                batch.close();
            }
        }
    }
}
