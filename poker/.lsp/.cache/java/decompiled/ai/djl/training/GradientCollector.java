/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.NDArray
 *  java.lang.AutoCloseable
 *  java.lang.Object
 */
package ai.djl.training;

import ai.djl.ndarray.NDArray;

public interface GradientCollector
extends AutoCloseable {
    public void backward(NDArray var1);

    public void zeroGradients();

    public void close();
}
