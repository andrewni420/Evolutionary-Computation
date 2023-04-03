/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.ndarray.NDManager
 *  ai.djl.nn.Parameter
 *  ai.djl.training.ParameterServer
 *  ai.djl.training.ParameterStore$ParameterData
 *  java.lang.IllegalArgumentException
 *  java.lang.Integer
 *  java.lang.Object
 *  java.lang.String
 *  java.util.Map
 *  java.util.Map$Entry
 *  java.util.concurrent.ConcurrentHashMap
 */
package ai.djl.training;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterServer;
import ai.djl.training.ParameterStore;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/*
 * Exception performing whole class analysis ignored.
 */
public class ParameterStore {
    private NDManager manager;
    private Map<String, ParameterData> parameterMap;
    private Map<Device, Integer> deviceMap;
    private boolean copy;
    private ParameterServer parameterServer;

    public ParameterStore() {
        this(NDManager.newBaseManager(), false);
    }

    public ParameterStore(NDManager manager, boolean copy) {
        this.manager = manager;
        this.copy = copy;
        this.parameterMap = new ConcurrentHashMap();
        this.deviceMap = new ConcurrentHashMap();
        this.deviceMap.put((Object)manager.getDevice(), (Object)0);
    }

    public void setParameterServer(ParameterServer parameterServer, Device[] devices) {
        this.parameterServer = parameterServer;
        this.deviceMap.clear();
        for (int i = 0; i < devices.length; ++i) {
            if (this.deviceMap.put((Object)devices[i], (Object)i) == null) continue;
            throw new IllegalArgumentException("Duplicated devices are not allowed.");
        }
    }

    public void updateAllParameters() {
        for (Map.Entry entry : this.parameterMap.entrySet()) {
            String parameterId = (String)entry.getKey();
            ParameterData data = (ParameterData)entry.getValue();
            if (!ParameterData.access$000((ParameterData)data)) continue;
            NDArray[] params = ParameterData.access$100((ParameterData)data);
            this.parameterServer.update(parameterId, params);
        }
    }

    public NDArray getValue(Parameter parameter, Device device, boolean training) {
        if (parameter == null) {
            return null;
        }
        String parameterId = parameter.getId();
        int index = (Integer)this.deviceMap.get((Object)device);
        ParameterData data = (ParameterData)this.parameterMap.computeIfAbsent((Object)parameterId, k -> new ParameterData(this, parameter, null));
        if (ParameterData.access$200((ParameterData)data)) {
            NDArray array = parameter.getArray();
            if (this.parameterServer != null) {
                this.parameterServer.init(parameterId, new NDArray[]{array});
                NDArray[] arrays = new NDArray[this.deviceMap.size()];
                for (Map.Entry entry : this.deviceMap.entrySet()) {
                    Device dev = (Device)entry.getKey();
                    int i = (Integer)entry.getValue();
                    if (i == index && array.getDevice().equals((Object)dev)) {
                        arrays[i] = array;
                    } else {
                        arrays[i] = array.toDevice(dev, true);
                        arrays[i].attach(this.manager);
                        if (parameter.requiresGradient()) {
                            arrays[i].setRequiresGradient(true);
                        }
                    }
                    ParameterData.access$300((ParameterData)data, (NDArray)arrays[i]);
                }
            } else {
                if (this.copy || !array.getDevice().equals((Object)device)) {
                    array = array.toDevice(device, true);
                    array.attach(this.manager);
                    if (parameter.requiresGradient() && training) {
                        array.setRequiresGradient(true);
                    }
                }
                ParameterData.access$300((ParameterData)data, (NDArray)array);
            }
        }
        return ParameterData.access$400((ParameterData)data, (int)index);
    }

    public NDManager getManager() {
        return this.manager;
    }

    public void sync() {
        for (ParameterData data : this.parameterMap.values()) {
            ParameterData.access$500((ParameterData)data);
        }
    }

    static /* synthetic */ Map access$600(ParameterStore x0) {
        return x0.deviceMap;
    }
}
