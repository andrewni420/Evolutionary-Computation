/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.MalformedModelException
 *  ai.djl.inference.Predictor
 *  ai.djl.ndarray.NDManager
 *  ai.djl.nn.Block
 *  ai.djl.training.Trainer
 *  ai.djl.training.TrainingConfig
 *  ai.djl.translate.Translator
 *  ai.djl.util.PairList
 *  java.io.IOException
 *  java.io.InputStream
 *  java.lang.AutoCloseable
 *  java.lang.Object
 *  java.lang.String
 *  java.lang.UnsupportedOperationException
 *  java.net.URL
 *  java.nio.file.Path
 *  java.util.Map
 *  java.util.function.Function
 */
package ai.djl;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import ai.djl.util.PairList;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.Function;

public interface Model
extends AutoCloseable {
    public static Model newInstance(String name) {
        return Model.newInstance(name, (Device)null);
    }

    public static Model newInstance(String name, Device device) {
        return Engine.getInstance().newModel(name, device);
    }

    public static Model newInstance(String name, String engineName) {
        Engine engine = Engine.getEngine(engineName);
        return engine.newModel(name, null);
    }

    public static Model newInstance(String name, Device device, String engineName) {
        if (engineName == null || engineName.isEmpty()) {
            return Model.newInstance(name, device);
        }
        return Engine.getEngine(engineName).newModel(name, device);
    }

    default public void load(Path modelPath) throws IOException, MalformedModelException {
        this.load(modelPath, null, null);
    }

    default public void load(Path modelPath, String prefix) throws IOException, MalformedModelException {
        this.load(modelPath, prefix, null);
    }

    public void load(Path var1, String var2, Map<String, ?> var3) throws IOException, MalformedModelException;

    default public void load(InputStream is) throws IOException, MalformedModelException {
        this.load(is, null);
    }

    public void load(InputStream var1, Map<String, ?> var2) throws IOException, MalformedModelException;

    public void save(Path var1, String var2) throws IOException;

    public Path getModelPath();

    public Block getBlock();

    public void setBlock(Block var1);

    public String getName();

    public String getProperty(String var1);

    default public String getProperty(String key, String defValue) {
        String value = this.getProperty(key);
        if (value == null || value.isEmpty()) {
            return defValue;
        }
        return value;
    }

    public void setProperty(String var1, String var2);

    public NDManager getNDManager();

    public Trainer newTrainer(TrainingConfig var1);

    default public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        return this.newPredictor(translator, this.getNDManager().getDevice());
    }

    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> var1, Device var2);

    public PairList<String, Shape> describeInput();

    public PairList<String, Shape> describeOutput();

    public String[] getArtifactNames();

    public <T> T getArtifact(String var1, Function<InputStream, T> var2) throws IOException;

    public URL getArtifact(String var1) throws IOException;

    public InputStream getArtifactAsStream(String var1) throws IOException;

    public void setDataType(DataType var1);

    public DataType getDataType();

    default public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    default public void quantize() {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    public void close();
}
