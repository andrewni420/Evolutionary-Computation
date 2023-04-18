/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.NDManager
 *  ai.djl.training.initializer.Initializer
 *  java.lang.Object
 */
package ai.djl.training.initializer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.initializer.Initializer;

public class NormalInitializer
implements Initializer {
    private float sigma;

    public NormalInitializer() {
        this.sigma = 0.01f;
    }

    public NormalInitializer(float sigma) {
        this.sigma = sigma;
    }

    public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
        return manager.randomNormal(0.0f, this.sigma, shape, dataType, manager.getDevice());
    }
}
