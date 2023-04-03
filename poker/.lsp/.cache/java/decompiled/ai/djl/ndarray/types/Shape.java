/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.types.LayoutType
 *  ai.djl.util.Pair
 *  ai.djl.util.PairList
 *  java.io.DataInputStream
 *  java.io.IOException
 *  java.lang.IllegalArgumentException
 *  java.lang.IndexOutOfBoundsException
 *  java.lang.Long
 *  java.lang.Math
 *  java.lang.Object
 *  java.lang.String
 *  java.lang.StringBuilder
 *  java.lang.System
 *  java.nio.ByteBuffer
 *  java.util.Arrays
 *  java.util.List
 *  java.util.function.Function
 *  java.util.function.Predicate
 *  java.util.stream.Collectors
 *  java.util.stream.LongStream
 *  java.util.stream.Stream
 */
package ai.djl.ndarray.types;

import ai.djl.ndarray.types.LayoutType;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.stream.Stream;

public class Shape {
    private long[] shape;
    private LayoutType[] layout;

    public Shape(long ... shape) {
        this(shape, (LayoutType[])Arrays.stream((long[])shape).mapToObj(x -> LayoutType.UNKNOWN).toArray(LayoutType[]::new));
    }

    public Shape(List<Long> shape) {
        this(shape.stream().mapToLong(l -> l).toArray(), (LayoutType[])shape.stream().map(x -> LayoutType.UNKNOWN).toArray(LayoutType[]::new));
    }

    public Shape(PairList<Long, LayoutType> shape) {
        this(shape.keys().stream().mapToLong(l -> l).toArray(), (LayoutType[])shape.values().toArray((Object[])new LayoutType[shape.size()]));
    }

    public Shape(long[] shape, String layout) {
        this(shape, LayoutType.fromValue((String)layout));
    }

    public Shape(long[] shape, LayoutType[] layout) {
        if (Arrays.stream((long[])shape).anyMatch(s -> s < -1L)) {
            throw new IllegalArgumentException("The shape must be >= -1");
        }
        if (shape.length != layout.length) {
            throw new IllegalArgumentException("The shape and layout must have the same length");
        }
        this.shape = shape;
        this.layout = layout;
    }

    public static Shape update(Shape shape, int dimension, long value) {
        long[] newShape = (long[])shape.shape.clone();
        newShape[dimension] = value;
        return new Shape(newShape, shape.layout);
    }

    public long[] getShape() {
        return this.shape;
    }

    public long get(int dimension) {
        return this.shape[dimension];
    }

    public LayoutType getLayoutType(int dimension) {
        return this.layout[dimension];
    }

    public long size(int ... dimensions) {
        long total = 1L;
        int[] nArray = dimensions;
        int n = nArray.length;
        for (int i = 0; i < n; ++i) {
            long d = nArray[i];
            if (d < 0L || d >= (long)this.shape.length) {
                throw new IllegalArgumentException("Invalid dimension " + d);
            }
            if (this.shape[Math.toIntExact((long)d)] == -1L) {
                return -1L;
            }
            total *= this.shape[Math.toIntExact((long)d)];
        }
        return total;
    }

    public long size() {
        long total = 1L;
        for (long v : this.shape) {
            if (v == -1L) {
                return -1L;
            }
            total *= v;
        }
        return total;
    }

    public int dimension() {
        return this.shape.length;
    }

    public long getUnknownValueCount() {
        return Arrays.stream((long[])this.shape).filter(s -> s == -1L).count();
    }

    public Shape slice(int beginIndex) {
        return this.slice(beginIndex, this.shape.length);
    }

    public Shape slice(int beginIndex, int endIndex) {
        int size = (endIndex += endIndex < 0 ? this.shape.length : 0) - (beginIndex += beginIndex < 0 ? this.shape.length : 0);
        long[] out = new long[size];
        System.arraycopy((Object)this.shape, (int)beginIndex, (Object)out, (int)0, (int)size);
        return new Shape(out);
    }

    public Shape filterByLayoutType(Predicate<LayoutType> predicate) {
        return new Shape((PairList<Long, LayoutType>)new PairList((List)this.stream().filter(pair -> predicate.test((Object)((LayoutType)pair.getValue()))).collect(Collectors.toList())));
    }

    public Shape map(Function<Pair<Long, LayoutType>, Pair<Long, LayoutType>> mapper) {
        return new Shape((PairList<Long, LayoutType>)new PairList((List)this.stream().map(mapper).collect(Collectors.toList())));
    }

    public Stream<Pair<Long, LayoutType>> stream() {
        return new PairList((List)Arrays.stream((long[])this.shape).boxed().collect(Collectors.toList()), Arrays.asList((Object[])this.layout)).stream();
    }

    public Shape add(long ... axes) {
        return this.addAll(new Shape(axes));
    }

    public Shape addAll(Shape other) {
        return new Shape(LongStream.concat((LongStream)Arrays.stream((long[])this.shape), (LongStream)Arrays.stream((long[])other.shape)).toArray());
    }

    public long head() {
        if (this.shape.length == 0) {
            throw new IndexOutOfBoundsException("can't get value from scalar shape.");
        }
        return this.shape[0];
    }

    public long tail() {
        if (this.shape.length == 0) {
            throw new IndexOutOfBoundsException("can't get value from scalar shape.");
        }
        return this.shape[this.shape.length - 1];
    }

    public int getTrailingOnes() {
        for (int i = 0; i < this.shape.length; ++i) {
            if (this.shape[this.shape.length - i - 1] == 1L) continue;
            return i;
        }
        return 0;
    }

    public int getLeadingOnes() {
        for (int i = 0; i < this.shape.length; ++i) {
            if (this.shape[i] == 1L) continue;
            return i;
        }
        return 0;
    }

    public boolean isScalar() {
        return this.dimension() == 0;
    }

    public boolean hasZeroDimension() {
        for (int i = 0; i < this.dimension(); ++i) {
            if (this.shape[i] != 0L) continue;
            return true;
        }
        return false;
    }

    public boolean isLayoutKnown() {
        return !Arrays.stream((Object[])this.layout).allMatch(l -> l == LayoutType.UNKNOWN);
    }

    public LayoutType[] getLayout() {
        return this.layout;
    }

    public String toLayoutString() {
        return LayoutType.toString((LayoutType[])this.layout);
    }

    public byte[] getEncoded() {
        int length = 8 + this.shape.length * 8 + this.layout.length * 2;
        ByteBuffer bb = ByteBuffer.allocate((int)length);
        bb.putInt(this.shape.length);
        for (long l : this.shape) {
            bb.putLong(l);
        }
        bb.putInt(this.layout.length);
        for (LayoutType layoutType : this.layout) {
            bb.putChar(layoutType.getValue());
        }
        return bb.array();
    }

    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || this.getClass() != o.getClass()) {
            return false;
        }
        Shape shape1 = (Shape)o;
        return Arrays.equals((long[])this.shape, (long[])shape1.shape);
    }

    public int hashCode() {
        return Arrays.hashCode((long[])this.shape);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append('(');
        for (int i = 0; i < this.shape.length; ++i) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(this.shape[i]);
        }
        sb.append(')');
        return sb.toString();
    }

    public static Shape decode(DataInputStream dis) throws IOException {
        int length = dis.readInt();
        long[] shapeValue = new long[length];
        for (int i = 0; i < length; ++i) {
            shapeValue[i] = dis.readLong();
        }
        length = dis.readInt();
        char[] layout = new char[length];
        for (int i = 0; i < length; ++i) {
            layout[i] = dis.readChar();
        }
        return new Shape(shapeValue, new String(layout));
    }

    public static Shape decode(ByteBuffer bb) {
        int length = bb.getInt();
        long[] shapeValue = new long[length];
        for (int i = 0; i < length; ++i) {
            shapeValue[i] = bb.getLong();
        }
        length = bb.getInt();
        char[] layout = new char[length];
        for (int i = 0; i < length; ++i) {
            layout[i] = bb.getChar();
        }
        return new Shape(shapeValue, new String(layout));
    }

    public boolean isRankOne() {
        int max = 1;
        int ans = 1;
        for (long s : this.shape) {
            int size = Math.toIntExact((long)s);
            max = Math.max((int)max, (int)size);
            if ((ans *= size) >= 0) continue;
            return false;
        }
        return max == ans;
    }
}
