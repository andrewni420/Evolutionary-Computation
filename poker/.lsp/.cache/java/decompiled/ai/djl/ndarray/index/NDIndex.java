/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.index.dim.NDIndexAll
 *  ai.djl.ndarray.index.dim.NDIndexBooleans
 *  ai.djl.ndarray.index.dim.NDIndexElement
 *  ai.djl.ndarray.index.dim.NDIndexFixed
 *  ai.djl.ndarray.index.dim.NDIndexNull
 *  ai.djl.ndarray.index.dim.NDIndexPick
 *  ai.djl.ndarray.index.dim.NDIndexSlice
 *  ai.djl.ndarray.index.dim.NDIndexTake
 *  java.lang.CharSequence
 *  java.lang.IllegalArgumentException
 *  java.lang.Integer
 *  java.lang.Long
 *  java.lang.Object
 *  java.lang.String
 *  java.util.ArrayList
 *  java.util.List
 *  java.util.regex.Matcher
 *  java.util.regex.Pattern
 *  java.util.stream.Stream
 */
package ai.djl.ndarray.index;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.dim.NDIndexAll;
import ai.djl.ndarray.index.dim.NDIndexBooleans;
import ai.djl.ndarray.index.dim.NDIndexElement;
import ai.djl.ndarray.index.dim.NDIndexFixed;
import ai.djl.ndarray.index.dim.NDIndexNull;
import ai.djl.ndarray.index.dim.NDIndexPick;
import ai.djl.ndarray.index.dim.NDIndexSlice;
import ai.djl.ndarray.index.dim.NDIndexTake;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class NDIndex {
    private static final Pattern ITEM_PATTERN = Pattern.compile((String)"(\\*)|((-?\\d+|\\{\\})?:(-?\\d+|\\{\\})?(:(-?\\d+|\\{\\}))?)|(-?\\d+|\\{\\})|null");
    private int rank = 0;
    private List<NDIndexElement> indices = new ArrayList();
    private int ellipsisIndex = -1;

    public NDIndex() {
    }

    public NDIndex(String indices, Object ... args) {
        this();
        this.addIndices(indices, args);
    }

    public NDIndex(long ... indices) {
        this();
        this.addIndices(indices);
    }

    public static NDIndex sliceAxis(int axis, long min, long max) {
        NDIndex ind = new NDIndex();
        for (int i = 0; i < axis; ++i) {
            ind.addAllDim();
        }
        ind.addSliceDim(min, max);
        return ind;
    }

    public int getRank() {
        return this.rank;
    }

    public int getEllipsisIndex() {
        return this.ellipsisIndex;
    }

    public NDIndexElement get(int dimension) {
        return (NDIndexElement)this.indices.get(dimension);
    }

    public List<NDIndexElement> getIndices() {
        return this.indices;
    }

    public final NDIndex addIndices(String indices, Object ... args) {
        String[] indexItems = indices.split(",");
        this.rank += indexItems.length;
        int argIndex = 0;
        for (int i = 0; i < indexItems.length; ++i) {
            if ("...".equals((Object)indexItems[i].trim())) {
                if (this.ellipsisIndex != -1) {
                    throw new IllegalArgumentException("an index can only have a single ellipsis (\"...\")");
                }
                this.ellipsisIndex = i;
                continue;
            }
            argIndex = this.addIndexItem(indexItems[i], argIndex, args);
        }
        if (this.ellipsisIndex != -1) {
            --this.rank;
        }
        if (argIndex != args.length) {
            throw new IllegalArgumentException("Incorrect number of index arguments");
        }
        return this;
    }

    public final NDIndex addIndices(long ... indices) {
        this.rank += indices.length;
        for (long i : indices) {
            this.indices.add((Object)new NDIndexFixed(i));
        }
        return this;
    }

    public NDIndex addBooleanIndex(NDArray index) {
        this.rank += index.getShape().dimension();
        this.indices.add((Object)new NDIndexBooleans(index));
        return this;
    }

    public NDIndex addAllDim() {
        ++this.rank;
        this.indices.add((Object)new NDIndexAll());
        return this;
    }

    public NDIndex addAllDim(int count) {
        if (count < 0) {
            throw new IllegalArgumentException("The number of index dimensions to add can't be negative");
        }
        this.rank += count;
        for (int i = 0; i < count; ++i) {
            this.indices.add((Object)new NDIndexAll());
        }
        return this;
    }

    public NDIndex addSliceDim(long min, long max) {
        ++this.rank;
        this.indices.add((Object)new NDIndexSlice(Long.valueOf((long)min), Long.valueOf((long)max), null));
        return this;
    }

    public NDIndex addSliceDim(long min, long max, long step) {
        ++this.rank;
        this.indices.add((Object)new NDIndexSlice(Long.valueOf((long)min), Long.valueOf((long)max), Long.valueOf((long)step)));
        return this;
    }

    public NDIndex addPickDim(NDArray index) {
        ++this.rank;
        this.indices.add((Object)new NDIndexPick(index));
        return this;
    }

    public Stream<NDIndexElement> stream() {
        return this.indices.stream();
    }

    private int addIndexItem(String indexItem, int argIndex, Object[] args) {
        Matcher m = ITEM_PATTERN.matcher((CharSequence)(indexItem = indexItem.trim()));
        if (!m.matches()) {
            throw new IllegalArgumentException("Invalid argument index: " + indexItem);
        }
        if ("null".equals((Object)indexItem)) {
            this.indices.add((Object)new NDIndexNull());
            return argIndex;
        }
        String star = m.group(1);
        if (star != null) {
            this.indices.add((Object)new NDIndexAll());
            return argIndex;
        }
        String digit = m.group(7);
        if (digit != null) {
            if ("{}".equals((Object)digit)) {
                Object arg = args[argIndex];
                if (arg instanceof Integer) {
                    this.indices.add((Object)new NDIndexFixed((long)((Integer)arg).intValue()));
                    return argIndex + 1;
                }
                if (arg instanceof Long) {
                    this.indices.add((Object)new NDIndexFixed(((Long)arg).longValue()));
                    return argIndex + 1;
                }
                if (arg instanceof NDArray) {
                    NDArray array = (NDArray)arg;
                    if (array.getDataType().isBoolean()) {
                        this.indices.add((Object)new NDIndexBooleans(array));
                        return argIndex + 1;
                    }
                    if (array.getDataType().isInteger() || array.getDataType().isFloating()) {
                        this.indices.add((Object)new NDIndexTake(array));
                        return argIndex + 1;
                    }
                } else if (arg == null) {
                    this.indices.add((Object)new NDIndexNull());
                    return argIndex + 1;
                }
                throw new IllegalArgumentException("Unknown argument: " + arg);
            }
            this.indices.add((Object)new NDIndexFixed(Long.parseLong((String)digit)));
            return argIndex;
        }
        Long min = null;
        Long max = null;
        Long step = null;
        if (m.group(3) != null) {
            min = this.parseSliceItem(m.group(3), argIndex, args);
            if ("{}".equals((Object)m.group(3))) {
                ++argIndex;
            }
        }
        if (m.group(4) != null) {
            max = this.parseSliceItem(m.group(4), argIndex, args);
            if ("{}".equals((Object)m.group(4))) {
                ++argIndex;
            }
        }
        if (m.group(6) != null) {
            step = this.parseSliceItem(m.group(6), argIndex, args);
            if ("{}".equals((Object)m.group(6))) {
                ++argIndex;
            }
        }
        if (min == null && max == null && step == null) {
            this.indices.add((Object)new NDIndexAll());
        } else {
            this.indices.add((Object)new NDIndexSlice(min, max, step));
        }
        return argIndex;
    }

    private Long parseSliceItem(String sliceItem, int argIndex, Object ... args) {
        if ("{}".equals((Object)sliceItem)) {
            Object arg = args[argIndex];
            if (arg instanceof Integer) {
                return ((Integer)arg).longValue();
            }
            if (arg instanceof Long) {
                return (Long)arg;
            }
            throw new IllegalArgumentException("Unknown slice argument: " + arg);
        }
        return Long.parseLong((String)sliceItem);
    }
}
