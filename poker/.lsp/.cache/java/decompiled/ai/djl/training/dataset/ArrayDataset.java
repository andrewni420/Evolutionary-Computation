/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.NDArray
 *  ai.djl.ndarray.NDManager
 *  ai.djl.ndarray.index.NDIndex
 *  ai.djl.training.dataset.ArrayDataset$Builder
 *  ai.djl.training.dataset.ArrayDataset$SubDataset
 *  ai.djl.training.dataset.ArrayDataset$SubDatasetByIndices
 *  ai.djl.training.dataset.Batch
 *  ai.djl.training.dataset.BulkDataIterable
 *  ai.djl.training.dataset.DataIterable
 *  ai.djl.training.dataset.RandomAccessDataset
 *  ai.djl.training.dataset.RandomAccessDataset$BaseBuilder
 *  ai.djl.training.dataset.Record
 *  ai.djl.training.dataset.Sampler
 *  ai.djl.translate.Batchifier
 *  ai.djl.translate.TranslateException
 *  ai.djl.util.Progress
 *  java.io.IOException
 *  java.lang.IllegalArgumentException
 *  java.lang.Iterable
 *  java.lang.Long
 *  java.lang.Math
 *  java.lang.Object
 *  java.util.List
 *  java.util.concurrent.ExecutorService
 *  java.util.stream.Stream
 */
package ai.djl.training.dataset;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.BulkDataIterable;
import ai.djl.training.dataset.DataIterable;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.training.dataset.Sampler;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.stream.Stream;

/*
 * Exception performing whole class analysis ignored.
 */
public class ArrayDataset
extends RandomAccessDataset {
    protected NDArray[] data;
    protected NDArray[] labels;

    public ArrayDataset(RandomAccessDataset.BaseBuilder<?> builder) {
        super(builder);
        if (builder instanceof Builder) {
            Builder builder2 = (Builder)builder;
            this.data = Builder.access$000((Builder)builder2);
            this.labels = Builder.access$100((Builder)builder2);
            long size = this.data[0].size(0);
            if (Stream.of((Object[])this.data).anyMatch(array -> array.size(0) != size)) {
                throw new IllegalArgumentException("All the NDArray must have the same length!");
            }
            if (this.labels != null && Stream.of((Object[])this.labels).anyMatch(array -> array.size(0) != size)) {
                throw new IllegalArgumentException("All the NDArray must have the same length!");
            }
        }
    }

    ArrayDataset() {
    }

    protected long availableSize() {
        return this.data[0].size(0);
    }

    public Record get(NDManager manager, long index) {
        NDList datum = new NDList();
        NDList label = new NDList();
        for (NDArray array : this.data) {
            datum.add(array.get(manager, new long[]{index}));
        }
        if (this.labels != null) {
            for (NDArray array : this.labels) {
                label.add(array.get(manager, new long[]{index}));
            }
        }
        return new Record(datum, label);
    }

    public Batch getByIndices(NDManager manager, long ... indices) {
        try (NDArray ndIndices = manager.create(indices);){
            NDIndex index = new NDIndex("{}", new Object[]{ndIndices});
            NDList datum = new NDList();
            NDList label = new NDList();
            for (NDArray array : this.data) {
                datum.add(array.get(manager, index));
            }
            if (this.labels != null) {
                for (NDArray array : this.labels) {
                    label.add(array.get(manager, index));
                }
            }
            Batch batch = new Batch(manager, datum, label, indices.length, Batchifier.STACK, Batchifier.STACK, -1L, -1L);
            return batch;
        }
    }

    public Batch getByRange(NDManager manager, long fromIndex, long toIndex) {
        NDIndex index = new NDIndex().addSliceDim(fromIndex, toIndex);
        NDList datum = new NDList();
        NDList label = new NDList();
        for (NDArray array : this.data) {
            datum.add(array.get(manager, index));
        }
        if (this.labels != null) {
            for (NDArray array : this.labels) {
                label.add(array.get(manager, index));
            }
        }
        int size = Math.toIntExact((long)(toIndex - fromIndex));
        return new Batch(manager, datum, label, size, Batchifier.STACK, Batchifier.STACK, -1L, -1L);
    }

    protected RandomAccessDataset newSubDataset(int[] indices, int from, int to) {
        return new SubDataset(this, indices, from, to);
    }

    protected RandomAccessDataset newSubDataset(List<Long> subIndices) {
        return new SubDatasetByIndices(this, subIndices);
    }

    public Iterable<Batch> getData(NDManager manager, Sampler sampler, ExecutorService executorService) throws IOException, TranslateException {
        this.prepare();
        if (this.dataBatchifier == Batchifier.STACK && this.labelBatchifier == Batchifier.STACK) {
            return new BulkDataIterable(this, manager, sampler, this.dataBatchifier, this.labelBatchifier, this.pipeline, this.targetPipeline, executorService, this.prefetchNumber, this.device);
        }
        return new DataIterable((RandomAccessDataset)this, manager, sampler, this.dataBatchifier, this.labelBatchifier, this.pipeline, this.targetPipeline, executorService, this.prefetchNumber, this.device);
    }

    public void prepare(Progress progress) throws IOException {
    }
}
