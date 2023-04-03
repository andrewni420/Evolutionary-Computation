/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.types.DataType$1
 *  ai.djl.ndarray.types.DataType$Format
 *  java.lang.IllegalArgumentException
 *  java.lang.Object
 *  java.lang.String
 *  java.nio.Buffer
 *  java.nio.ByteBuffer
 *  java.nio.ByteOrder
 *  java.nio.DoubleBuffer
 *  java.nio.FloatBuffer
 *  java.nio.IntBuffer
 *  java.nio.LongBuffer
 *  java.nio.ShortBuffer
 */
package ai.djl.ndarray.types;

import ai.djl.ndarray.types.DataType;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

public enum DataType {
    FLOAT32(Format.FLOATING, 4),
    FLOAT64(Format.FLOATING, 8),
    FLOAT16(Format.FLOATING, 2),
    UINT8(Format.UINT, 1),
    INT32(Format.INT, 4),
    INT8(Format.INT, 1),
    INT64(Format.INT, 8),
    BOOLEAN(Format.BOOLEAN, 1),
    COMPLEX64(Format.FLOATING, 4),
    UNKNOWN(Format.UNKNOWN, 0),
    STRING(Format.STRING, -1);

    private Format format;
    private int numOfBytes;

    private DataType(Format format, int numOfBytes) {
        this.format = format;
        this.numOfBytes = numOfBytes;
    }

    public int getNumOfBytes() {
        return this.numOfBytes;
    }

    public Format getFormat() {
        return this.format;
    }

    public boolean isFloating() {
        return this.format == Format.FLOATING;
    }

    public boolean isInteger() {
        return this.format == Format.UINT || this.format == Format.INT;
    }

    public boolean isBoolean() {
        return this.format == Format.BOOLEAN;
    }

    public static DataType fromBuffer(Buffer data) {
        if (data instanceof FloatBuffer) {
            return FLOAT32;
        }
        if (data instanceof ShortBuffer) {
            return FLOAT16;
        }
        if (data instanceof DoubleBuffer) {
            return FLOAT64;
        }
        if (data instanceof IntBuffer) {
            return INT32;
        }
        if (data instanceof LongBuffer) {
            return INT64;
        }
        if (data instanceof ByteBuffer) {
            return INT8;
        }
        throw new IllegalArgumentException("Unsupported buffer type: " + data.getClass().getSimpleName());
    }

    /*
     * Exception decompiling
     */
    public static DataType fromNumpy(String dtype) {
        /*
         * This method has failed to decompile.  When submitting a bug report, please provide this stack trace, and (if you hold appropriate legal rights) the relevant class file.
         * 
         * org.benf.cfr.reader.bytecode.analysis.opgraph.op4rewriters.SwitchStringRewriter$TooOptimisticMatchException
         *     at org.benf.cfr.reader.bytecode.analysis.opgraph.op4rewriters.SwitchStringRewriter.getString(SwitchStringRewriter.java:404)
         *     at org.benf.cfr.reader.bytecode.analysis.opgraph.op4rewriters.SwitchStringRewriter.access$600(SwitchStringRewriter.java:53)
         *     at org.benf.cfr.reader.bytecode.analysis.opgraph.op4rewriters.SwitchStringRewriter$SwitchStringMatchResultCollector.collectMatches(SwitchStringRewriter.java:368)
         *     at org.benf.cfr.reader.bytecode.analysis.opgraph.op4rewriters.matchutil.ResetAfterTest.match(ResetAfterTest.java:24)
         *     at org.benf.cfr.reader.bytecode.analysis.opgraph.op4rewriters.matchutil.KleeneN.match(KleeneN.java:24)
         *     at org.benf.cfr.reader.bytecode.analysis.opgraph.op4rewriters.matchutil.MatchSequence.match(MatchSequence.java:26)
         *     at org.benf.cfr.reader.bytecode.analysis.opgraph.op4rewriters.matchutil.ResetAfterTest.match(ResetAfterTest.java:23)
         *     at org.benf.cfr.reader.bytecode.analysis.opgraph.op4rewriters.SwitchStringRewriter.rewriteComplex(SwitchStringRewriter.java:201)
         *     at org.benf.cfr.reader.bytecode.analysis.opgraph.op4rewriters.SwitchStringRewriter.rewrite(SwitchStringRewriter.java:73)
         *     at org.benf.cfr.reader.bytecode.CodeAnalyser.getAnalysisInner(CodeAnalyser.java:881)
         *     at org.benf.cfr.reader.bytecode.CodeAnalyser.getAnalysisOrWrapFail(CodeAnalyser.java:278)
         *     at org.benf.cfr.reader.bytecode.CodeAnalyser.getAnalysis(CodeAnalyser.java:201)
         *     at org.benf.cfr.reader.entities.attributes.AttributeCode.analyse(AttributeCode.java:94)
         *     at org.benf.cfr.reader.entities.Method.analyse(Method.java:531)
         *     at org.benf.cfr.reader.entities.ClassFile.analyseMid(ClassFile.java:1055)
         *     at org.benf.cfr.reader.entities.ClassFile.analyseTop(ClassFile.java:942)
         *     at org.benf.cfr.reader.Driver.doClass(Driver.java:84)
         *     at org.benf.cfr.reader.CfrDriverImpl.analyse(CfrDriverImpl.java:78)
         *     at clojure_lsp.feature.java_interop$decompile_BANG_$fn__29549.invoke(java_interop.clj:34)
         *     at clojure_lsp.feature.java_interop$decompile_BANG_.invokeStatic(java_interop.clj:33)
         *     at clojure_lsp.feature.java_interop$decompile_BANG_.invoke(java_interop.clj:26)
         *     at clojure_lsp.feature.java_interop$decompile_file.invokeStatic(java_interop.clj:70)
         *     at clojure_lsp.feature.java_interop$decompile_file.invoke(java_interop.clj:61)
         *     at clojure_lsp.feature.java_interop$uri__GT_translated_file_BANG_.invokeStatic(java_interop.clj:112)
         *     at clojure_lsp.feature.java_interop$uri__GT_translated_file_BANG_.invoke(java_interop.clj:99)
         *     at clojure_lsp.feature.java_interop$uri__GT_translated_uri.invokeStatic(java_interop.clj:116)
         *     at clojure_lsp.feature.java_interop$uri__GT_translated_uri.invoke(java_interop.clj:115)
         *     at clojure_lsp.handlers$element__GT_location.invokeStatic(handlers.clj:136)
         *     at clojure_lsp.handlers$element__GT_location.invoke(handlers.clj:135)
         *     at clojure_lsp.handlers$definition.invokeStatic(handlers.clj:251)
         *     at clojure_lsp.handlers$definition.invoke(handlers.clj:245)
         *     at clojure_lsp.server$fn__35332$fn__35333.invoke(server.clj:318)
         *     at promesa.exec$wrap_bindings$fn__31849.invoke(exec.cljc:163)
         *     at promesa.util.Supplier.get(util.cljc:34)
         *     at java.base@11.0.17/java.util.concurrent.CompletableFuture$AsyncSupply.run(CompletableFuture.java:1700)
         *     at java.base@11.0.17/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)
         *     at java.base@11.0.17/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)
         *     at java.base@11.0.17/java.lang.Thread.run(Thread.java:829)
         *     at org.graalvm.nativeimage.builder/com.oracle.svm.core.thread.PlatformThreads.threadStartRoutine(PlatformThreads.java:775)
         *     at org.graalvm.nativeimage.builder/com.oracle.svm.core.posix.thread.PosixPlatformThreads.pthreadStartRoutine(PosixPlatformThreads.java:203)
         */
        throw new IllegalStateException("Decompilation failed");
    }

    public Buffer asDataType(ByteBuffer data) {
        switch (1.$SwitchMap$ai$djl$ndarray$types$DataType[this.ordinal()]) {
            case 1: {
                return data.asShortBuffer();
            }
            case 2: {
                return data.asFloatBuffer();
            }
            case 3: {
                return data.asDoubleBuffer();
            }
            case 4: {
                return data.asIntBuffer();
            }
            case 5: {
                return data.asLongBuffer();
            }
        }
        return data;
    }

    public String asNumpy() {
        char order = ByteOrder.nativeOrder() == ByteOrder.BIG_ENDIAN ? (char)'>' : '<';
        switch (1.$SwitchMap$ai$djl$ndarray$types$DataType[this.ordinal()]) {
            case 2: {
                return order + "f4";
            }
            case 3: {
                return order + "f8";
            }
            case 1: {
                return order + "f2";
            }
            case 6: {
                return "|u1";
            }
            case 4: {
                return order + "i4";
            }
            case 7: {
                return "|i1";
            }
            case 5: {
                return "<i8";
            }
            case 10: {
                return "|b1";
            }
            case 11: {
                return "|S1";
            }
        }
        throw new IllegalArgumentException("Unsupported dataType: " + (Object)((Object)this));
    }

    public String toString() {
        return this.name().toLowerCase();
    }
}
