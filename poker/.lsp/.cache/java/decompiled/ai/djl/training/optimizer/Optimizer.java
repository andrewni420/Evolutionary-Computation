/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.ndarray.NDArray
 *  ai.djl.training.optimizer.Adadelta$Builder
 *  ai.djl.training.optimizer.Adagrad$Builder
 *  ai.djl.training.optimizer.Adam$Builder
 *  ai.djl.training.optimizer.AdamW$Builder
 *  ai.djl.training.optimizer.Nag$Builder
 *  ai.djl.training.optimizer.Optimizer$OptimizerBuilder
 *  ai.djl.training.optimizer.RmsProp$Builder
 *  ai.djl.training.optimizer.Sgd$Builder
 *  java.lang.Integer
 *  java.lang.Math
 *  java.lang.Object
 *  java.lang.String
 *  java.util.Map
 *  java.util.concurrent.ConcurrentHashMap
 *  java.util.function.Function
 */
package ai.djl.training.optimizer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.training.optimizer.Adadelta;
import ai.djl.training.optimizer.Adagrad;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.AdamW;
import ai.djl.training.optimizer.Nag;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.RmsProp;
import ai.djl.training.optimizer.Sgd;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/*
 * Exception performing whole class analysis ignored.
 */
public abstract class Optimizer {
    protected float rescaleGrad;
    protected float clipGrad;
    private float weightDecays;
    private int beginNumUpdate;
    private int numUpdate;
    private Map<String, Integer> updateCounts = new ConcurrentHashMap();

    public Optimizer(OptimizerBuilder<?> builder) {
        this.rescaleGrad = OptimizerBuilder.access$000(builder);
        this.weightDecays = OptimizerBuilder.access$100(builder);
        this.clipGrad = OptimizerBuilder.access$200(builder);
        this.beginNumUpdate = OptimizerBuilder.access$300(builder);
    }

    public static Sgd.Builder sgd() {
        return new Sgd.Builder();
    }

    public static Nag.Builder nag() {
        return new Nag.Builder();
    }

    public static Adam.Builder adam() {
        return new Adam.Builder();
    }

    public static AdamW.Builder adamW() {
        return new AdamW.Builder();
    }

    public static RmsProp.Builder rmsprop() {
        return new RmsProp.Builder();
    }

    public static Adagrad.Builder adagrad() {
        return new Adagrad.Builder();
    }

    public static Adadelta.Builder adadelta() {
        return new Adadelta.Builder();
    }

    protected float getWeightDecay() {
        return this.weightDecays;
    }

    protected int updateCount(String parameterId) {
        int count = (Integer)this.updateCounts.compute((Object)parameterId, (key, val) -> val == null ? this.beginNumUpdate + 1 : val + 1);
        this.numUpdate = Math.max((int)this.numUpdate, (int)count);
        return this.numUpdate;
    }

    public abstract void update(String var1, NDArray var2, NDArray var3);

    protected NDArray withDefaultState(Map<String, Map<Device, NDArray>> state, String key, Device device, Function<String, NDArray> defaultFunction) {
        Map arrayMap = (Map)state.computeIfAbsent((Object)key, k -> {
            ConcurrentHashMap map = new ConcurrentHashMap();
            NDArray s = (NDArray)defaultFunction.apply(k);
            s.detach();
            map.put((Object)device, (Object)s);
            return map;
        });
        return (NDArray)arrayMap.computeIfAbsent((Object)device, k -> ((NDArray)arrayMap.values().iterator().next()).toDevice(device, true));
    }
}
