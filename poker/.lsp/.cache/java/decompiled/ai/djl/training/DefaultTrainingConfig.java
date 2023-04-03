/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.engine.Engine
 *  ai.djl.nn.Parameter
 *  ai.djl.nn.Parameter$Type
 *  ai.djl.training.TrainingConfig
 *  ai.djl.training.evaluator.Evaluator
 *  ai.djl.training.initializer.Initializer
 *  ai.djl.training.listener.TrainingListener
 *  ai.djl.training.loss.Loss
 *  ai.djl.training.optimizer.Adam
 *  ai.djl.util.PairList
 *  java.lang.Object
 *  java.lang.String
 *  java.util.ArrayList
 *  java.util.Arrays
 *  java.util.Collection
 *  java.util.List
 *  java.util.concurrent.ExecutorService
 *  java.util.concurrent.ForkJoinPool
 *  java.util.function.Predicate
 */
package ai.djl.training;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.nn.Parameter;
import ai.djl.training.TrainingConfig;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Predicate;

public class DefaultTrainingConfig
implements TrainingConfig {
    private PairList<Initializer, Predicate<Parameter>> initializers = new PairList();
    private Optimizer optimizer;
    private Device[] devices;
    private Loss loss;
    private ExecutorService executorService;
    private List<Evaluator> evaluators;
    private List<TrainingListener> listeners;

    public DefaultTrainingConfig(Loss loss) {
        this.loss = loss;
        this.optimizer = Adam.builder().build();
        this.evaluators = new ArrayList();
        this.listeners = new ArrayList();
    }

    public DefaultTrainingConfig optInitializer(Initializer initializer, Parameter.Type type) {
        this.initializers.add((Object)initializer, parameter -> parameter.getType().equals((Object)type));
        return this;
    }

    public DefaultTrainingConfig optInitializer(Initializer initializer, String name) {
        this.initializers.add((Object)initializer, parameter -> parameter.getName().equals((Object)name));
        return this;
    }

    public DefaultTrainingConfig optInitializer(Initializer initializer, Predicate<Parameter> predicate) {
        this.initializers.add((Object)initializer, predicate);
        return this;
    }

    public DefaultTrainingConfig optDevices(Device[] devices) {
        this.devices = devices;
        return this;
    }

    public DefaultTrainingConfig optOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    public DefaultTrainingConfig optExecutorService() {
        return this.optExecutorService((ExecutorService)ForkJoinPool.commonPool());
    }

    public DefaultTrainingConfig optExecutorService(ExecutorService executorService) {
        this.executorService = executorService;
        return this;
    }

    public <T extends Evaluator> DefaultTrainingConfig addEvaluators(Collection<T> evaluators) {
        evaluators.forEach(this::addEvaluator);
        return this;
    }

    public DefaultTrainingConfig addEvaluator(Evaluator evaluator) {
        this.evaluators.add((Object)evaluator);
        return this;
    }

    public DefaultTrainingConfig addTrainingListeners(TrainingListener ... listeners) {
        this.listeners.addAll((Collection)Arrays.asList((Object[])listeners));
        return this;
    }

    public Device[] getDevices() {
        if (this.devices == null) {
            return Engine.getInstance().getDevices();
        }
        return this.devices;
    }

    public PairList<Initializer, Predicate<Parameter>> getInitializers() {
        return this.initializers;
    }

    public Optimizer getOptimizer() {
        return this.optimizer;
    }

    public Loss getLossFunction() {
        return this.loss;
    }

    public ExecutorService getExecutorService() {
        return this.executorService;
    }

    public List<Evaluator> getEvaluators() {
        return this.evaluators;
    }

    public List<TrainingListener> getTrainingListeners() {
        return this.listeners;
    }
}
