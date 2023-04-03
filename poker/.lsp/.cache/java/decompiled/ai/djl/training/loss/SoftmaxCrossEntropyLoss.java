/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.NDArray
 *  ai.djl.ndarray.index.NDIndex
 *  ai.djl.training.loss.Loss
 *  java.lang.Float
 *  java.lang.Math
 *  java.lang.Number
 *  java.lang.Object
 *  java.lang.String
 */
package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.training.loss.Loss;

public class SoftmaxCrossEntropyLoss
extends Loss {
    private float weight;
    private int classAxis;
    private boolean sparseLabel;
    private boolean fromLogit;

    public SoftmaxCrossEntropyLoss() {
        this("SoftmaxCrossEntropyLoss");
    }

    public SoftmaxCrossEntropyLoss(String name) {
        this(name, 1.0f, -1, true, true);
    }

    public SoftmaxCrossEntropyLoss(String name, float weight, int classAxis, boolean sparseLabel, boolean fromLogit) {
        super(name);
        this.weight = weight;
        this.classAxis = classAxis;
        this.sparseLabel = sparseLabel;
        this.fromLogit = fromLogit;
    }

    public NDArray evaluate(NDList label, NDList prediction) {
        NDArray loss;
        NDArray pred = prediction.singletonOrThrow();
        if (this.fromLogit) {
            pred = pred.logSoftmax(this.classAxis);
        }
        NDArray lab = label.singletonOrThrow();
        if (this.sparseLabel) {
            NDIndex pickIndex = new NDIndex().addAllDim(Math.floorMod((int)this.classAxis, (int)pred.getShape().dimension())).addPickDim(lab);
            loss = pred.get(pickIndex).neg();
        } else {
            lab = lab.reshape(pred.getShape());
            loss = pred.mul(lab).neg().sum(new int[]{this.classAxis}, true);
        }
        if (this.weight != 1.0f) {
            loss = loss.mul((Number)Float.valueOf((float)this.weight));
        }
        return loss.mean();
    }
}
