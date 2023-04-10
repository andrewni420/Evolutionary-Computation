/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.engine.EngineException
 *  ai.djl.mxnet.engine.MxGradientCollector
 *  ai.djl.mxnet.engine.MxModel
 *  ai.djl.mxnet.engine.MxNDManager
 *  ai.djl.mxnet.engine.MxParameterServer
 *  ai.djl.mxnet.engine.MxSymbolBlock
 *  ai.djl.mxnet.jna.JnaUtils
 *  ai.djl.mxnet.jna.JnaUtils$NumpyMode
 *  ai.djl.mxnet.jna.LibUtils
 *  ai.djl.ndarray.NDManager
 *  ai.djl.nn.SymbolBlock
 *  ai.djl.training.LocalParameterServer
 *  ai.djl.training.ParameterServer
 *  ai.djl.util.Utils
 *  java.io.FileNotFoundException
 *  java.lang.Boolean
 *  java.lang.Object
 *  java.lang.Override
 *  java.lang.Runtime
 *  java.lang.String
 *  java.lang.StringBuilder
 *  java.lang.Thread
 *  java.lang.Throwable
 *  java.nio.file.Files
 *  java.nio.file.LinkOption
 *  java.nio.file.Path
 *  java.nio.file.Paths
 */
package ai.djl.mxnet.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.mxnet.engine.MxGradientCollector;
import ai.djl.mxnet.engine.MxModel;
import ai.djl.mxnet.engine.MxNDManager;
import ai.djl.mxnet.engine.MxParameterServer;
import ai.djl.mxnet.engine.MxSymbolBlock;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.mxnet.jna.LibUtils;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.GradientCollector;
import ai.djl.training.LocalParameterServer;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.Utils;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class MxEngine
extends Engine {
    public static final String ENGINE_NAME = "MXNet";
    static final int RANK = 0;
    private static final String MXNET_EXTRA_LIBRARY_VERBOSE = "MXNET_EXTRA_LIBRARY_VERBOSE";

    private MxEngine() {
    }

    static Engine newInstance() {
        try {
            JnaUtils.getAllOpNames();
            JnaUtils.setNumpyMode((JnaUtils.NumpyMode)JnaUtils.NumpyMode.GLOBAL_ON);
            Runtime.getRuntime().addShutdownHook(new Thread(JnaUtils::waitAll));
            String paths = Utils.getEnvOrSystemProperty((String)"MXNET_EXTRA_LIBRARY_PATH");
            boolean extraLibVerbose = Boolean.parseBoolean((String)Utils.getEnvOrSystemProperty((String)MXNET_EXTRA_LIBRARY_VERBOSE));
            if (paths != null) {
                String[] files;
                for (String file : files = paths.split(",")) {
                    Path path = Paths.get((String)file, (String[])new String[0]);
                    if (Files.notExists((Path)path, (LinkOption[])new LinkOption[0])) {
                        throw new FileNotFoundException("Extra Library not found: " + file);
                    }
                    JnaUtils.loadLib((String)path.toAbsolutePath().toString(), (boolean)extraLibVerbose);
                }
            }
            return new MxEngine();
        }
        catch (Throwable t) {
            throw new EngineException("Failed to load MXNet native library", t);
        }
    }

    @Override
    public Engine getAlternativeEngine() {
        return null;
    }

    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    @Override
    public int getRank() {
        return 0;
    }

    @Override
    public String getVersion() {
        int version = JnaUtils.getVersion();
        int major = version / 10000;
        int minor = version / 100 - major * 100;
        int patch = version % 100;
        return major + "." + minor + '.' + patch;
    }

    @Override
    public boolean hasCapability(String capability) {
        return JnaUtils.getFeatures().contains((Object)capability);
    }

    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        return new MxSymbolBlock(manager);
    }

    @Override
    public Model newModel(String name, Device device) {
        return new MxModel(name, device);
    }

    @Override
    public NDManager newBaseManager() {
        return MxNDManager.getSystemManager().newSubManager();
    }

    @Override
    public NDManager newBaseManager(Device device) {
        return MxNDManager.getSystemManager().newSubManager(device);
    }

    @Override
    public GradientCollector newGradientCollector() {
        return new MxGradientCollector();
    }

    @Override
    public ParameterServer newParameterServer(Optimizer optimizer) {
        return Boolean.getBoolean((String)"ai.djl.use_local_parameter_server") ? new LocalParameterServer(optimizer) : new MxParameterServer(optimizer);
    }

    @Override
    public void setRandomSeed(int seed) {
        super.setRandomSeed(seed);
        JnaUtils.randomSeed((int)seed);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append(this.getEngineName()).append(':').append(this.getVersion()).append(", capabilities: [\n");
        for (String feature : JnaUtils.getFeatures()) {
            sb.append("\t").append(feature).append(",\n");
        }
        sb.append("]\nMXNet Library: ").append(LibUtils.getLibName());
        return sb.toString();
    }
}
