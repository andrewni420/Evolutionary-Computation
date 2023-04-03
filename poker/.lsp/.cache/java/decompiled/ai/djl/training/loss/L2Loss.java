/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.NDArray
 *  ai.djl.training.loss.Loss
 *  java.lang.Float
 *  java.lang.Number
 *  java.lang.Object
 *  java.lang.String
 */
package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;

public class L2Loss
extends Loss {
    private float weight;

    public L2Loss() {
        this("L2Loss");
    }

    public L2Loss(String name) {
        this(name, 0.5f);
    }

    public L2Loss(String name, float weight) {
        super(name);
        this.weight = weight;
    }

    public NDArray evaluate(NDList label, NDList prediction) {
        NDArray pred = prediction.singletonOrThrow();
        NDArray labelReshaped = label.singletonOrThrow().reshape(pred.getShape());
        NDArray loss = labelReshaped.sub(pred).square().mul((Number)Float.valueOf((float)this.weight));
        return loss.mean();
    }
}
