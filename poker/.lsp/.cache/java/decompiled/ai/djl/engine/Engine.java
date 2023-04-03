/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.engine.EngineException
 *  ai.djl.engine.EngineProvider
 *  ai.djl.ndarray.NDManager
 *  ai.djl.nn.SymbolBlock
 *  ai.djl.training.GradientCollector
 *  ai.djl.training.LocalParameterServer
 *  ai.djl.training.ParameterServer
 *  ai.djl.util.Ec2Utils
 *  ai.djl.util.RandomUtils
 *  ai.djl.util.Utils
 *  ai.djl.util.cuda.CudaUtils
 *  java.io.IOException
 *  java.io.InputStream
 *  java.lang.AssertionError
 *  java.lang.IllegalArgumentException
 *  java.lang.Integer
 *  java.lang.Math
 *  java.lang.Object
 *  java.lang.String
 *  java.lang.System
 *  java.lang.Throwable
 *  java.lang.UnsupportedOperationException
 *  java.lang.management.MemoryUsage
 *  java.nio.file.Files
 *  java.nio.file.Path
 *  java.nio.file.Paths
 *  java.nio.file.attribute.FileAttribute
 *  java.util.Map
 *  java.util.Properties
 *  java.util.ServiceLoader
 *  java.util.Set
 *  java.util.concurrent.ConcurrentHashMap
 *  org.slf4j.Logger
 *  org.slf4j.LoggerFactory
 */
package ai.djl.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.EngineException;
import ai.djl.engine.EngineProvider;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.GradientCollector;
import ai.djl.training.LocalParameterServer;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.Ec2Utils;
import ai.djl.util.RandomUtils;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;
import java.io.IOException;
import java.io.InputStream;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileAttribute;
import java.util.Map;
import java.util.Properties;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class Engine {
    private static final Logger logger = LoggerFactory.getLogger(Engine.class);
    private static final Map<String, EngineProvider> ALL_ENGINES = new ConcurrentHashMap();
    private static final String DEFAULT_ENGINE = Engine.initEngine();
    private Device defaultDevice;
    private Integer seed;

    private static synchronized String initEngine() {
        ServiceLoader loaders = ServiceLoader.load(EngineProvider.class);
        for (EngineProvider provider : loaders) {
            Engine.registerEngine(provider);
        }
        if (ALL_ENGINES.isEmpty()) {
            logger.debug("No engine found from EngineProvider");
            return null;
        }
        String def = System.getProperty((String)"ai.djl.default_engine");
        String defaultEngine = Utils.getenv((String)"DJL_DEFAULT_ENGINE", (String)def);
        if (defaultEngine == null || defaultEngine.isEmpty()) {
            int rank = Integer.MAX_VALUE;
            for (EngineProvider provider : ALL_ENGINES.values()) {
                if (provider.getEngineRank() >= rank) continue;
                defaultEngine = provider.getEngineName();
                rank = provider.getEngineRank();
            }
        } else if (!ALL_ENGINES.containsKey((Object)defaultEngine)) {
            throw new EngineException("Unknown default engine: " + defaultEngine);
        }
        logger.debug("Found default engine: {}", (Object)defaultEngine);
        Ec2Utils.callHome((String)defaultEngine);
        return defaultEngine;
    }

    public abstract Engine getAlternativeEngine();

    public abstract String getEngineName();

    public abstract int getRank();

    public static String getDefaultEngineName() {
        return System.getProperty((String)"ai.djl.default_engine", (String)DEFAULT_ENGINE);
    }

    public static Engine getInstance() {
        if (DEFAULT_ENGINE == null) {
            throw new EngineException("No deep learning engine found." + System.lineSeparator() + "Please refer to https://github.com/deepjavalibrary/djl/blob/master/docs/development/troubleshooting.md for more details.");
        }
        return Engine.getEngine(Engine.getDefaultEngineName());
    }

    public static boolean hasEngine(String engineName) {
        return ALL_ENGINES.containsKey((Object)engineName);
    }

    public static void registerEngine(EngineProvider provider) {
        logger.debug("Registering EngineProvider: {}", (Object)provider.getEngineName());
        ALL_ENGINES.putIfAbsent((Object)provider.getEngineName(), (Object)provider);
    }

    public static Set<String> getAllEngines() {
        return ALL_ENGINES.keySet();
    }

    public static Engine getEngine(String engineName) {
        EngineProvider provider = (EngineProvider)ALL_ENGINES.get((Object)engineName);
        if (provider == null) {
            throw new IllegalArgumentException("Deep learning engine not found: " + engineName);
        }
        return provider.getEngine();
    }

    public abstract String getVersion();

    public abstract boolean hasCapability(String var1);

    public Device defaultDevice() {
        if (this.defaultDevice == null) {
            this.defaultDevice = this.hasCapability("CUDA") && CudaUtils.getGpuCount() > 0 ? Device.gpu() : Device.cpu();
        }
        return this.defaultDevice;
    }

    public Device[] getDevices() {
        return this.getDevices(Integer.MAX_VALUE);
    }

    public Device[] getDevices(int maxGpus) {
        int count = this.getGpuCount();
        if (maxGpus <= 0 || count <= 0) {
            return new Device[]{Device.cpu()};
        }
        count = Math.min((int)maxGpus, (int)count);
        Device[] devices = new Device[count];
        for (int i = 0; i < count; ++i) {
            devices[i] = Device.gpu((int)i);
        }
        return devices;
    }

    public int getGpuCount() {
        if (this.hasCapability("CUDA")) {
            return CudaUtils.getGpuCount();
        }
        return 0;
    }

    public SymbolBlock newSymbolBlock(NDManager manager) {
        throw new UnsupportedOperationException("Not supported.");
    }

    public abstract Model newModel(String var1, Device var2);

    public abstract NDManager newBaseManager();

    public abstract NDManager newBaseManager(Device var1);

    public GradientCollector newGradientCollector() {
        throw new UnsupportedOperationException("Not supported.");
    }

    public ParameterServer newParameterServer(Optimizer optimizer) {
        return new LocalParameterServer(optimizer);
    }

    public void setRandomSeed(int seed) {
        this.seed = seed;
        RandomUtils.RANDOM.setSeed((long)seed);
    }

    public Integer getSeed() {
        return this.seed;
    }

    public static String getDjlVersion() {
        String string;
        block9: {
            String version = Engine.class.getPackage().getSpecificationVersion();
            if (version != null) {
                return version;
            }
            InputStream is = Engine.class.getResourceAsStream("api.properties");
            try {
                Properties prop = new Properties();
                prop.load(is);
                string = prop.getProperty("djl_version");
                if (is == null) break block9;
            }
            catch (Throwable throwable) {
                try {
                    if (is != null) {
                        try {
                            is.close();
                        }
                        catch (Throwable throwable2) {
                            throwable.addSuppressed(throwable2);
                        }
                    }
                    throw throwable;
                }
                catch (IOException e) {
                    throw new AssertionError("Failed to open api.properties", (Throwable)e);
                }
            }
            is.close();
        }
        return string;
    }

    public String toString() {
        return this.getEngineName() + ':' + this.getVersion();
    }

    public static void debugEnvironment() {
        System.out.println("----------- System Properties -----------");
        System.getProperties().forEach((k, v) -> System.out.println(k + ": " + v));
        System.out.println();
        System.out.println("--------- Environment Variables ---------");
        Utils.getenv().forEach((k, v) -> System.out.println(k + ": " + v));
        System.out.println();
        System.out.println("-------------- Directories --------------");
        try {
            Path temp = Paths.get((String)System.getProperty((String)"java.io.tmpdir"), (String[])new String[0]);
            System.out.println("temp directory: " + temp);
            Path tmpFile = Files.createTempFile((String)"test", (String)".tmp", (FileAttribute[])new FileAttribute[0]);
            Files.delete((Path)tmpFile);
            Path cacheDir = Utils.getCacheDir();
            System.out.println("DJL cache directory: " + cacheDir.toAbsolutePath());
            Path path = Utils.getEngineCacheDir();
            System.out.println("Engine cache directory: " + path.toAbsolutePath());
            Files.createDirectories((Path)path, (FileAttribute[])new FileAttribute[0]);
            if (!Files.isWritable((Path)path)) {
                System.out.println("Engine cache directory is not writable!!!");
            }
        }
        catch (Throwable e) {
            e.printStackTrace(System.out);
        }
        System.out.println();
        System.out.println("------------------ CUDA -----------------");
        int gpuCount = CudaUtils.getGpuCount();
        System.out.println("GPU Count: " + gpuCount);
        if (gpuCount > 0) {
            System.out.println("CUDA: " + CudaUtils.getCudaVersionString());
            System.out.println("ARCH: " + CudaUtils.getComputeCapability((int)0));
        }
        for (int i = 0; i < gpuCount; ++i) {
            Device device = Device.gpu((int)i);
            MemoryUsage mem = CudaUtils.getGpuMemory((Device)device);
            System.out.println("GPU(" + i + ") memory used: " + mem.getCommitted() + " bytes");
        }
        System.out.println();
        System.out.println("----------------- Engines ---------------");
        System.out.println("DJL version: " + Engine.getDjlVersion());
        System.out.println("Default Engine: " + Engine.getInstance());
        System.out.println("Default Device: " + Engine.getInstance().defaultDevice());
        for (EngineProvider provider : ALL_ENGINES.values()) {
            System.out.println(provider.getEngineName() + ": " + provider.getEngineRank());
        }
    }
}
