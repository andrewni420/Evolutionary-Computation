/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.training.tracker.CosineTracker
 *  ai.djl.training.tracker.CosineTracker$Builder
 *  ai.djl.training.tracker.CyclicalTracker
 *  ai.djl.training.tracker.CyclicalTracker$Builder
 *  ai.djl.training.tracker.FactorTracker
 *  ai.djl.training.tracker.FactorTracker$Builder
 *  ai.djl.training.tracker.FixedTracker
 *  ai.djl.training.tracker.MultiFactorTracker
 *  ai.djl.training.tracker.MultiFactorTracker$Builder
 *  ai.djl.training.tracker.ParameterTracker
 *  ai.djl.training.tracker.WarmUpTracker
 *  ai.djl.training.tracker.WarmUpTracker$Builder
 *  java.lang.Object
 *  java.lang.String
 */
package ai.djl.training.tracker;

import ai.djl.training.tracker.CosineTracker;
import ai.djl.training.tracker.CyclicalTracker;
import ai.djl.training.tracker.FactorTracker;
import ai.djl.training.tracker.FixedTracker;
import ai.djl.training.tracker.MultiFactorTracker;
import ai.djl.training.tracker.ParameterTracker;
import ai.djl.training.tracker.WarmUpTracker;

public interface Tracker
extends ParameterTracker {
    public float getNewValue(int var1);

    default public float getNewValue(String parameterId, int numUpdate) {
        return this.getNewValue(numUpdate);
    }

    public static FactorTracker.Builder factor() {
        return FactorTracker.builder();
    }

    public static WarmUpTracker.Builder warmUp() {
        return WarmUpTracker.builder();
    }

    public static MultiFactorTracker.Builder multiFactor() {
        return MultiFactorTracker.builder();
    }

    public static CosineTracker.Builder cosine() {
        return CosineTracker.builder();
    }

    public static CyclicalTracker.Builder cyclical() {
        return CyclicalTracker.builder();
    }

    public static Tracker fixed(float value) {
        return FixedTracker.builder().setValue(value).build();
    }
}
