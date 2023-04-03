/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.ndarray.BytesSupplier
 *  ai.djl.ndarray.NDArray$1
 *  ai.djl.ndarray.NDManager
 *  ai.djl.ndarray.NDResource
 *  ai.djl.ndarray.NDSerializer
 *  ai.djl.ndarray.index.NDIndex
 *  ai.djl.ndarray.internal.NDArrayEx
 *  ai.djl.ndarray.internal.NDFormat
 *  ai.djl.ndarray.types.SparseFormat
 *  ai.djl.util.Float16Utils
 *  java.lang.Byte
 *  java.lang.Double
 *  java.lang.Float
 *  java.lang.IllegalArgumentException
 *  java.lang.IllegalStateException
 *  java.lang.Integer
 *  java.lang.Long
 *  java.lang.Math
 *  java.lang.Number
 *  java.lang.Object
 *  java.lang.String
 *  java.nio.Buffer
 *  java.nio.ByteBuffer
 *  java.nio.DoubleBuffer
 *  java.nio.FloatBuffer
 *  java.nio.IntBuffer
 *  java.nio.LongBuffer
 *  java.nio.charset.Charset
 *  java.nio.charset.StandardCharsets
 *  java.util.Arrays
 *  java.util.Collections
 *  java.util.List
 *  java.util.function.Function
 *  java.util.stream.IntStream
 *  java.util.stream.LongStream
 */
package ai.djl.ndarray;

import ai.djl.Device;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDResource;
import ai.djl.ndarray.NDSerializer;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.internal.NDFormat;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.util.Float16Utils;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public interface NDArray
extends NDResource,
BytesSupplier {
    public static NDArray decode(NDManager manager, byte[] byteArray) {
        return manager.decode(byteArray);
    }

    public String getName();

    public void setName(String var1);

    public String getUid();

    public DataType getDataType();

    public Device getDevice();

    public Shape getShape();

    public SparseFormat getSparseFormat();

    default public boolean isSparse() {
        return this.getSparseFormat() != SparseFormat.DENSE;
    }

    default public boolean isScalar() {
        return this.getShape().isScalar();
    }

    default public byte[] encode() {
        return NDSerializer.encode((NDArray)this);
    }

    public NDArray toDevice(Device var1, boolean var2);

    public NDArray toType(DataType var1, boolean var2);

    public void setRequiresGradient(boolean var1);

    public NDArray getGradient();

    public boolean hasGradient();

    public NDArray stopGradient();

    default public NDArray scaleGradient(double scale) {
        return this.mul((Number)Double.valueOf((double)scale)).add(this.stopGradient().mul((Number)Double.valueOf((double)(1.0 - scale))));
    }

    default public long size(int axis) {
        return this.getShape().size(axis);
    }

    default public long size() {
        return this.getShape().size();
    }

    default public double[] toDoubleArray() {
        if (this.getDataType() != DataType.FLOAT64) {
            throw new IllegalStateException("DataType mismatch, Required double Actual " + (Object)((Object)this.getDataType()));
        }
        DoubleBuffer db = this.toByteBuffer().asDoubleBuffer();
        double[] ret = new double[db.remaining()];
        db.get(ret);
        return ret;
    }

    default public float[] toFloatArray() {
        if (this.getDataType() == DataType.FLOAT16) {
            return Float16Utils.fromByteBuffer((ByteBuffer)this.toByteBuffer());
        }
        if (this.getDataType() != DataType.FLOAT32) {
            throw new IllegalStateException("DataType mismatch, Required float, Actual " + (Object)((Object)this.getDataType()));
        }
        FloatBuffer fb = this.toByteBuffer().asFloatBuffer();
        float[] ret = new float[fb.remaining()];
        fb.get(ret);
        return ret;
    }

    default public int[] toIntArray() {
        if (this.getDataType() != DataType.INT32) {
            throw new IllegalStateException("DataType mismatch, Required int Actual " + (Object)((Object)this.getDataType()));
        }
        IntBuffer ib = this.toByteBuffer().asIntBuffer();
        int[] ret = new int[ib.remaining()];
        ib.get(ret);
        return ret;
    }

    default public long[] toLongArray() {
        if (this.getDataType() != DataType.INT64) {
            throw new IllegalStateException("DataType mismatch, Required long Actual " + (Object)((Object)this.getDataType()));
        }
        LongBuffer lb = this.toByteBuffer().asLongBuffer();
        long[] ret = new long[lb.remaining()];
        lb.get(ret);
        return ret;
    }

    default public byte[] toByteArray() {
        ByteBuffer bb = this.toByteBuffer();
        if (bb.hasArray()) {
            return bb.array();
        }
        byte[] buf = new byte[bb.remaining()];
        bb.get(buf);
        return buf;
    }

    default public int[] toUint8Array() {
        ByteBuffer bb = this.toByteBuffer();
        int[] buf = new int[bb.remaining()];
        for (int i = 0; i < buf.length; ++i) {
            buf[i] = bb.get() & 0xFF;
        }
        return buf;
    }

    default public boolean[] toBooleanArray() {
        if (this.getDataType() != DataType.BOOLEAN) {
            throw new IllegalStateException("DataType mismatch, Required boolean Actual " + (Object)((Object)this.getDataType()));
        }
        ByteBuffer bb = this.toByteBuffer();
        boolean[] ret = new boolean[bb.remaining()];
        for (int i = 0; i < ret.length; ++i) {
            ret[i] = bb.get() != 0;
        }
        return ret;
    }

    default public String[] toStringArray() {
        return this.toStringArray(StandardCharsets.UTF_8);
    }

    public String[] toStringArray(Charset var1);

    default public Number[] toArray() {
        switch (1.$SwitchMap$ai$djl$ndarray$types$DataType[this.getDataType().ordinal()]) {
            case 1: 
            case 2: {
                float[] floatArray = this.toFloatArray();
                return (Number[])IntStream.range((int)0, (int)floatArray.length).mapToObj(i -> Float.valueOf((float)floatArray[i])).toArray(Number[]::new);
            }
            case 3: {
                return (Number[])Arrays.stream((double[])this.toDoubleArray()).boxed().toArray(Double[]::new);
            }
            case 4: {
                return (Number[])Arrays.stream((int[])this.toIntArray()).boxed().toArray(Integer[]::new);
            }
            case 5: {
                return (Number[])Arrays.stream((long[])this.toLongArray()).boxed().toArray(Long[]::new);
            }
            case 6: 
            case 7: {
                ByteBuffer bb = this.toByteBuffer();
                Byte[] ret = new Byte[bb.remaining()];
                for (int i2 = 0; i2 < ret.length; ++i2) {
                    ret[i2] = bb.get();
                }
                return ret;
            }
            case 8: {
                return (Number[])Arrays.stream((int[])this.toUint8Array()).boxed().toArray(Integer[]::new);
            }
        }
        throw new IllegalStateException("Unsupported DataType: " + (Object)((Object)this.getDataType()));
    }

    public void set(Buffer var1);

    default public void set(float[] data) {
        this.set((Buffer)FloatBuffer.wrap((float[])data));
    }

    default public void set(int[] data) {
        this.set((Buffer)IntBuffer.wrap((int[])data));
    }

    default public void set(double[] data) {
        this.set((Buffer)DoubleBuffer.wrap((double[])data));
    }

    default public void set(long[] data) {
        this.set((Buffer)LongBuffer.wrap((long[])data));
    }

    default public void set(byte[] data) {
        this.set((Buffer)ByteBuffer.wrap((byte[])data));
    }

    default public void set(NDIndex index, NDArray value) {
        this.getNDArrayInternal().getIndexer(this.getManager()).set(this, index, (Object)value);
    }

    default public void set(NDIndex index, Number value) {
        this.getNDArrayInternal().getIndexer(this.getManager()).set(this, index, (Object)value);
    }

    default public void set(NDIndex index, Function<NDArray, NDArray> function) {
        NDArray array = this.get(index);
        this.set(index, (NDArray)function.apply((Object)array));
    }

    default public void set(NDArray index, Number value) {
        this.set(new NDIndex("{}", new Object[]{index}), value);
    }

    default public void setScalar(NDIndex index, Number value) {
        this.getNDArrayInternal().getIndexer(this.getManager()).setScalar(this, index, value);
    }

    default public NDArray get(NDIndex index) {
        return this.get(this.getManager(), index);
    }

    default public NDArray get(NDManager manager, NDIndex index) {
        return this.getNDArrayInternal().getIndexer(manager).get(this, index);
    }

    default public NDArray get(NDArray index) {
        return this.get(new NDIndex("{}", new Object[]{index}));
    }

    default public NDArray get(String indices, Object ... args) {
        return this.get(new NDIndex(indices, args));
    }

    default public NDArray get(long ... indices) {
        return this.get(new NDIndex(indices));
    }

    default public NDArray get(NDManager manager, long ... indices) {
        return this.get(manager, new NDIndex(indices));
    }

    public NDArray gather(NDArray var1, int var2);

    public NDArray gatherNd(NDArray var1);

    default public NDArray take(NDArray index) {
        return this.take(this.getManager(), index);
    }

    public NDArray take(NDManager var1, NDArray var2);

    public NDArray put(NDArray var1, NDArray var2);

    public NDArray scatter(NDArray var1, NDArray var2, int var3);

    default public NDArray getScalar(long ... indices) {
        NDArray value = this.get(new NDIndex(indices));
        if (value.size() != 1L) {
            throw new IllegalArgumentException("The supplied Index does not produce a scalar");
        }
        return value;
    }

    default public long getLong(long ... indices) {
        return this.getScalar(indices).toLongArray()[0];
    }

    default public double getDouble(long ... indices) {
        return this.getScalar(indices).toDoubleArray()[0];
    }

    default public float getFloat(long ... indices) {
        return this.getScalar(indices).toFloatArray()[0];
    }

    default public int getInt(long ... indices) {
        return this.getScalar(indices).toIntArray()[0];
    }

    default public byte getByte(long ... indices) {
        return this.getScalar(indices).toByteArray()[0];
    }

    default public int getUint8(long ... indices) {
        return this.getByte(indices) & 0xFF;
    }

    default public boolean getBoolean(long ... indices) {
        return this.getScalar(indices).toBooleanArray()[0];
    }

    public void copyTo(NDArray var1);

    default public NDArray duplicate() {
        NDArray array = this.getManager().create(this.getShape(), this.getDataType(), this.getDevice());
        array.setName(this.getName());
        this.copyTo(array);
        return array;
    }

    default public NDArray booleanMask(NDArray index) {
        return this.booleanMask(index, 0);
    }

    public NDArray booleanMask(NDArray var1, int var2);

    public NDArray sequenceMask(NDArray var1, float var2);

    public NDArray sequenceMask(NDArray var1);

    default public NDArray zerosLike() {
        return this.getManager().zeros(this.getShape(), this.getDataType(), this.getDevice());
    }

    default public NDArray onesLike() {
        return this.getManager().ones(this.getShape(), this.getDataType(), this.getDevice());
    }

    default public NDArray like() {
        return this.getManager().create(this.getShape());
    }

    public boolean contentEquals(Number var1);

    public boolean contentEquals(NDArray var1);

    default public boolean shapeEquals(NDArray other) {
        return this.getShape().equals(other.getShape());
    }

    default public boolean allClose(NDArray other) {
        return this.allClose(other, 1.0E-5, 1.0E-8, false);
    }

    default public boolean allClose(NDArray other, double rtol, double atol, boolean equalNan) {
        if (!this.shapeEquals(other)) {
            return false;
        }
        Number[] actualDoubleArray = this.toArray();
        Number[] expectedDoubleArray = other.toArray();
        for (int i = 0; i < actualDoubleArray.length; ++i) {
            double a = actualDoubleArray[i].doubleValue();
            double b = expectedDoubleArray[i].doubleValue();
            if (equalNan && Double.isNaN((double)a) && Double.isNaN((double)b) || !Double.isNaN((double)a) && !Double.isNaN((double)b) && !(Math.abs((double)(a - b)) > atol + rtol * Math.abs((double)b))) continue;
            return false;
        }
        return true;
    }

    public NDArray eq(Number var1);

    public NDArray eq(NDArray var1);

    public NDArray neq(Number var1);

    public NDArray neq(NDArray var1);

    public NDArray gt(Number var1);

    public NDArray gt(NDArray var1);

    public NDArray gte(Number var1);

    public NDArray gte(NDArray var1);

    public NDArray lt(Number var1);

    public NDArray lt(NDArray var1);

    public NDArray lte(Number var1);

    public NDArray lte(NDArray var1);

    public NDArray add(Number var1);

    public NDArray add(NDArray var1);

    public NDArray sub(Number var1);

    public NDArray sub(NDArray var1);

    public NDArray mul(Number var1);

    public NDArray mul(NDArray var1);

    public NDArray div(Number var1);

    public NDArray div(NDArray var1);

    public NDArray mod(Number var1);

    public NDArray mod(NDArray var1);

    public NDArray pow(Number var1);

    public NDArray pow(NDArray var1);

    public NDArray addi(Number var1);

    public NDArray addi(NDArray var1);

    public NDArray subi(Number var1);

    public NDArray subi(NDArray var1);

    public NDArray muli(Number var1);

    public NDArray muli(NDArray var1);

    public NDArray divi(Number var1);

    public NDArray divi(NDArray var1);

    public NDArray modi(Number var1);

    public NDArray modi(NDArray var1);

    public NDArray powi(Number var1);

    public NDArray powi(NDArray var1);

    public NDArray sign();

    public NDArray signi();

    public NDArray maximum(Number var1);

    public NDArray maximum(NDArray var1);

    public NDArray minimum(Number var1);

    public NDArray minimum(NDArray var1);

    public NDArray neg();

    public NDArray negi();

    public NDArray abs();

    public NDArray square();

    public NDArray sqrt();

    public NDArray cbrt();

    public NDArray floor();

    public NDArray ceil();

    public NDArray round();

    public NDArray trunc();

    public NDArray exp();

    public NDArray gammaln();

    public NDArray log();

    public NDArray log10();

    public NDArray log2();

    public NDArray sin();

    public NDArray cos();

    public NDArray tan();

    public NDArray asin();

    public NDArray acos();

    public NDArray atan();

    public NDArray sinh();

    public NDArray cosh();

    public NDArray tanh();

    public NDArray asinh();

    public NDArray acosh();

    public NDArray atanh();

    public NDArray toDegrees();

    public NDArray toRadians();

    public NDArray max();

    default public NDArray max(int[] axes) {
        return this.max(axes, false);
    }

    public NDArray max(int[] var1, boolean var2);

    public NDArray min();

    default public NDArray min(int[] axes) {
        return this.min(axes, false);
    }

    public NDArray min(int[] var1, boolean var2);

    public NDArray sum();

    default public NDArray sum(int[] axes) {
        return this.sum(axes, false);
    }

    public NDArray sum(int[] var1, boolean var2);

    public NDArray cumProd(int var1);

    public NDArray cumProd(int var1, DataType var2);

    public NDArray prod();

    default public NDArray prod(int[] axes) {
        return this.prod(axes, false);
    }

    public NDArray prod(int[] var1, boolean var2);

    public NDArray mean();

    default public NDArray mean(int[] axes) {
        return this.mean(axes, false);
    }

    public NDArray mean(int[] var1, boolean var2);

    default public NDArray normalize() {
        return this.normalize(2.0, 1L, 1.0E-12);
    }

    default public NDArray normalize(double exponent, long dim) {
        return this.normalize(exponent, dim, 1.0E-12);
    }

    public NDArray normalize(double var1, long var3, double var5);

    public NDArray rotate90(int var1, int[] var2);

    default public NDArray trace() {
        return this.trace(0, 0, 1);
    }

    default public NDArray trace(int offset) {
        return this.trace(offset, 0, 1);
    }

    public NDArray trace(int var1, int var2, int var3);

    default public NDList split(long sections) {
        return this.split(sections, 0);
    }

    default public NDList split(long[] indices) {
        return this.split(indices, 0);
    }

    default public NDList split(long sections, int axis) {
        long axisSize = this.getShape().getShape()[axis];
        if (axisSize % sections != 0L) {
            throw new IllegalArgumentException("array split does not result in an equal division");
        }
        long sectionSize = axisSize / sections;
        long[] indices = LongStream.range((long)0L, (long)sections).map(i -> i * sectionSize).toArray();
        return this.split(indices, axis);
    }

    public NDList split(long[] var1, int var2);

    public NDArray flatten();

    public NDArray flatten(int var1, int var2);

    default public NDArray fft(long length) {
        return this.fft(length, -1L);
    }

    public NDArray fft(long var1, long var3);

    default public NDArray stft(long nFft, long hopLength, boolean center, NDArray window, boolean returnComplex) {
        return this.stft(nFft, hopLength, center, window, false, returnComplex);
    }

    public NDArray stft(long var1, long var3, boolean var5, NDArray var6, boolean var7, boolean var8);

    default public NDArray reshape(long ... newShape) {
        return this.reshape(new Shape(newShape));
    }

    public NDArray reshape(Shape var1);

    public NDArray expandDims(int var1);

    default public NDArray squeeze() {
        long[] shape = this.getShape().getShape();
        return this.squeeze(IntStream.range((int)0, (int)shape.length).filter(i -> shape[i] == 1L).toArray());
    }

    default public NDArray squeeze(int axis) {
        return this.squeeze(new int[]{axis});
    }

    public NDArray squeeze(int[] var1);

    default public NDArray stack(NDArray array) {
        return this.stack(array, 0);
    }

    default public NDArray stack(NDArray array, int axis) {
        return this.getNDArrayInternal().stack(new NDList(array), axis);
    }

    default public NDArray concat(NDArray array) {
        return this.concat(array, 0);
    }

    default public NDArray concat(NDArray array, int axis) {
        return this.getNDArrayInternal().concat(new NDList(array), axis);
    }

    public NDArray logicalAnd(NDArray var1);

    public NDArray logicalOr(NDArray var1);

    public NDArray logicalXor(NDArray var1);

    public NDArray logicalNot();

    default public NDArray argSort() {
        return this.argSort(-1, true);
    }

    default public NDArray argSort(int axis) {
        return this.argSort(axis, true);
    }

    public NDArray argSort(int var1, boolean var2);

    public NDArray sort();

    public NDArray sort(int var1);

    public NDArray softmax(int var1);

    public NDArray logSoftmax(int var1);

    public NDArray cumSum();

    public NDArray cumSum(int var1);

    public void intern(NDArray var1);

    public NDArray isInfinite();

    public NDArray inverse();

    public NDArray isNaN();

    public NDArray tile(long var1);

    public NDArray tile(int var1, long var2);

    public NDArray tile(long[] var1);

    public NDArray tile(Shape var1);

    public NDArray repeat(long var1);

    public NDArray repeat(int var1, long var2);

    public NDArray repeat(long[] var1);

    public NDArray repeat(Shape var1);

    public NDArray dot(NDArray var1);

    public NDArray matMul(NDArray var1);

    public NDArray clip(Number var1, Number var2);

    default public NDArray swapAxes(int axis1, int axis2) {
        int[] dims = IntStream.range((int)0, (int)this.getShape().dimension()).toArray();
        int tmp = dims[axis1];
        dims[axis1] = dims[axis2];
        dims[axis2] = tmp;
        return this.transpose(dims);
    }

    public NDArray flip(int ... var1);

    public NDArray transpose();

    public NDArray transpose(int ... var1);

    public NDArray broadcast(Shape var1);

    default public NDArray broadcast(long ... shape) {
        return this.broadcast(new Shape(shape));
    }

    public NDArray argMax();

    public NDArray argMax(int var1);

    public NDArray argMin();

    public NDArray argMin(int var1);

    public NDArray percentile(Number var1);

    public NDArray percentile(Number var1, int[] var2);

    public NDArray median();

    public NDArray median(int[] var1);

    public NDArray toDense();

    public NDArray toSparse(SparseFormat var1);

    public NDArray nonzero();

    default public boolean isEmpty() {
        return this.getShape().size() == 0L;
    }

    default public NDArray all() {
        return this.toType(DataType.BOOLEAN, false).sum().eq((Number)Long.valueOf((long)this.size()));
    }

    default public NDArray any() {
        return this.toType(DataType.BOOLEAN, false).sum().gt((Number)Integer.valueOf((int)0));
    }

    default public NDArray none() {
        return this.toType(DataType.BOOLEAN, false).sum().eq((Number)Integer.valueOf((int)0));
    }

    default public NDArray countNonzero() {
        return this.toType(DataType.BOOLEAN, false).sum();
    }

    default public NDArray countNonzero(int axis) {
        return this.toType(DataType.BOOLEAN, false).sum(new int[]{axis});
    }

    public NDArray erfinv();

    default public List<NDArray> getResourceNDArrays() {
        return Collections.singletonList((Object)this);
    }

    public NDArrayEx getNDArrayInternal();

    public boolean isReleased();

    default public String toDebugString() {
        if (this.isReleased()) {
            return "This array is already closed";
        }
        if (this.getDataType() == DataType.STRING) {
            return Arrays.toString((Object[])this.toStringArray(StandardCharsets.UTF_8));
        }
        return NDFormat.format((NDArray)this, (int)100, (int)10, (int)10, (int)20);
    }

    default public String toDebugString(boolean withContent) {
        return this.toDebugString(1000, 10, 10, 20, withContent);
    }

    default public String toDebugString(int maxSize, int maxDepth, int maxRows, int maxColumns, boolean withContent) {
        if (this.isReleased()) {
            return "This array is already closed";
        }
        if (this.getDataType() == DataType.STRING) {
            return Arrays.toString((Object[])this.toStringArray(StandardCharsets.UTF_8));
        }
        return NDFormat.format((NDArray)this, (int)maxSize, (int)maxDepth, (int)maxRows, (int)maxColumns, (boolean)withContent);
    }

    public void close();

    default public NDArray norm() {
        return this.norm(false);
    }

    default public NDArray norm(int[] axes) {
        return this.norm(axes, false);
    }

    public NDArray norm(boolean var1);

    default public NDArray norm(int[] axes, boolean keepDims) {
        return this.norm(2, axes, keepDims);
    }

    public NDArray norm(int var1, int[] var2, boolean var3);

    default public NDArray oneHot(int depth) {
        return this.oneHot(depth, 1.0f, 0.0f, DataType.FLOAT32);
    }

    default public NDArray oneHot(int depth, DataType dataType) {
        return this.oneHot(depth, 1.0f, 0.0f, dataType);
    }

    public NDArray oneHot(int var1, float var2, float var3, DataType var4);

    public NDArray batchDot(NDArray var1);

    public NDArray complex();

    public NDArray real();
}
