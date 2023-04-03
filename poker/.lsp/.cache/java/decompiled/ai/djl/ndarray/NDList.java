/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.Device
 *  ai.djl.ndarray.BytesSupplier
 *  ai.djl.ndarray.NDArray
 *  ai.djl.ndarray.NDManager
 *  ai.djl.ndarray.NDResource
 *  ai.djl.ndarray.NDSerializer
 *  java.io.ByteArrayInputStream
 *  java.io.ByteArrayOutputStream
 *  java.io.DataInputStream
 *  java.io.DataOutputStream
 *  java.io.IOException
 *  java.io.InputStream
 *  java.io.OutputStream
 *  java.io.PushbackInputStream
 *  java.lang.AssertionError
 *  java.lang.IllegalArgumentException
 *  java.lang.IndexOutOfBoundsException
 *  java.lang.Object
 *  java.lang.String
 *  java.lang.StringBuilder
 *  java.lang.Throwable
 *  java.nio.BufferUnderflowException
 *  java.nio.ByteBuffer
 *  java.util.ArrayList
 *  java.util.Arrays
 *  java.util.Collection
 *  java.util.Iterator
 *  java.util.List
 *  java.util.zip.ZipEntry
 *  java.util.zip.ZipInputStream
 *  java.util.zip.ZipOutputStream
 */
package ai.djl.ndarray;

import ai.djl.Device;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDResource;
import ai.djl.ndarray.NDSerializer;
import ai.djl.ndarray.types.Shape;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PushbackInputStream;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class NDList
extends ArrayList<NDArray>
implements NDResource,
BytesSupplier {
    private static final long serialVersionUID = 1L;

    public NDList() {
    }

    public NDList(int initialCapacity) {
        super(initialCapacity);
    }

    public NDList(NDArray ... arrays) {
        super((Collection)Arrays.asList((Object[])arrays));
    }

    public NDList(Collection<NDArray> other) {
        super(other);
    }

    public static NDList decode(NDManager manager, byte[] byteArray) {
        if (byteArray.length < 4) {
            throw new IllegalArgumentException("Invalid input length: " + byteArray.length);
        }
        try {
            if (byteArray[0] == 80 && byteArray[1] == 75) {
                return NDList.decodeNumpy(manager, (InputStream)new ByteArrayInputStream(byteArray));
            }
            if (byteArray[0] == 57 && byteArray[1] == 78 && byteArray[2] == 85 && byteArray[3] == 77) {
                return new NDList(NDSerializer.decode((NDManager)manager, (InputStream)new ByteArrayInputStream(byteArray)));
            }
            ByteBuffer bb = ByteBuffer.wrap((byte[])byteArray);
            int size = bb.getInt();
            if (size < 0) {
                throw new IllegalArgumentException("Invalid NDList size: " + size);
            }
            NDList list = new NDList();
            for (int i = 0; i < size; ++i) {
                list.add(i, NDSerializer.decode((NDManager)manager, (ByteBuffer)bb));
            }
            return list;
        }
        catch (IOException | BufferUnderflowException e) {
            throw new IllegalArgumentException("Invalid NDArray input", e);
        }
    }

    public static NDList decode(NDManager manager, InputStream is) {
        try {
            DataInputStream dis = new DataInputStream(is);
            byte[] magic = new byte[4];
            dis.readFully(magic);
            PushbackInputStream pis = new PushbackInputStream(is, 4);
            pis.unread(magic);
            if (magic[0] == 80 && magic[1] == 75) {
                return NDList.decodeNumpy(manager, (InputStream)pis);
            }
            if (magic[0] == 57 && magic[1] == 78 && magic[2] == 85 && magic[3] == 77) {
                return new NDList(NDSerializer.decode((NDManager)manager, (InputStream)pis));
            }
            dis = new DataInputStream((InputStream)pis);
            int size = dis.readInt();
            if (size < 0) {
                throw new IllegalArgumentException("Invalid NDList size: " + size);
            }
            NDList list = new NDList();
            for (int i = 0; i < size; ++i) {
                list.add(i, manager.decode((InputStream)dis));
            }
            return list;
        }
        catch (IOException e) {
            throw new IllegalArgumentException("Malformed data", (Throwable)e);
        }
    }

    private static NDList decodeNumpy(NDManager manager, InputStream is) throws IOException {
        ZipEntry entry;
        NDList list = new NDList();
        ZipInputStream zis = new ZipInputStream(is);
        while ((entry = zis.getNextEntry()) != null) {
            String name = entry.getName();
            NDArray array = NDSerializer.decodeNumpy((NDManager)manager, (InputStream)zis);
            if (!name.startsWith("arr_") && name.endsWith(".npy")) {
                array.setName(name.substring(0, name.length() - 4));
            }
            list.add(array);
        }
        return list;
    }

    public NDArray get(String name) {
        Iterator iterator = this.iterator();
        while (iterator.hasNext()) {
            NDArray array = (NDArray)iterator.next();
            if (!name.equals((Object)array.getName())) continue;
            return array;
        }
        return null;
    }

    public NDArray remove(String name) {
        int index = 0;
        Iterator iterator = this.iterator();
        while (iterator.hasNext()) {
            NDArray array = (NDArray)iterator.next();
            if (name.equals((Object)array.getName())) {
                this.remove(index);
                return array;
            }
            ++index;
        }
        return null;
    }

    public boolean contains(String name) {
        Iterator iterator = this.iterator();
        while (iterator.hasNext()) {
            NDArray array = (NDArray)iterator.next();
            if (!name.equals((Object)array.getName())) continue;
            return true;
        }
        return false;
    }

    public NDArray head() {
        return (NDArray)this.get(0);
    }

    public NDArray singletonOrThrow() {
        if (this.size() != 1) {
            throw new IndexOutOfBoundsException("Incorrect number of elements in NDList.singletonOrThrow: Expected 1 and was " + this.size());
        }
        return (NDArray)this.get(0);
    }

    public NDList addAll(NDList other) {
        Iterator iterator = other.iterator();
        while (iterator.hasNext()) {
            NDArray array = (NDArray)iterator.next();
            this.add(array);
        }
        return this;
    }

    public NDList subNDList(int fromIndex) {
        return new NDList((Collection<NDArray>)this.subList(fromIndex, this.size()));
    }

    public NDList toDevice(Device device, boolean copy) {
        if (!copy && this.stream().allMatch(array -> array.getDevice() == device)) {
            return this;
        }
        NDList newNDList = new NDList(this.size());
        this.forEach(a -> newNDList.add(a.toDevice(device, copy)));
        return newNDList;
    }

    public NDManager getManager() {
        return this.head().getManager();
    }

    public List<NDArray> getResourceNDArrays() {
        return this;
    }

    public void attach(NDManager manager) {
        this.forEach(array -> array.attach(manager));
    }

    public void tempAttach(NDManager manager) {
        this.forEach(array -> array.tempAttach(manager));
    }

    public void detach() {
        this.forEach(NDResource::detach);
    }

    public byte[] encode() {
        return this.encode(false);
    }

    public byte[] encode(boolean numpy) {
        byte[] byArray;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try {
            this.encode((OutputStream)baos, numpy);
            byArray = baos.toByteArray();
        }
        catch (Throwable throwable) {
            try {
                try {
                    baos.close();
                }
                catch (Throwable throwable2) {
                    throwable.addSuppressed(throwable2);
                }
                throw throwable;
            }
            catch (IOException e) {
                throw new AssertionError("NDList is not writable", (Throwable)e);
            }
        }
        baos.close();
        return byArray;
    }

    public void encode(OutputStream os) throws IOException {
        this.encode(os, false);
    }

    public void encode(OutputStream os, boolean numpy) throws IOException {
        if (numpy) {
            ZipOutputStream zos = new ZipOutputStream(os);
            int i = 0;
            Iterator iterator = this.iterator();
            while (iterator.hasNext()) {
                NDArray nd = (NDArray)iterator.next();
                String name = nd.getName();
                if (name == null) {
                    zos.putNextEntry(new ZipEntry("arr_" + i + ".npy"));
                    ++i;
                } else {
                    zos.putNextEntry(new ZipEntry(name + ".npy"));
                }
                NDSerializer.encodeAsNumpy((NDArray)nd, (OutputStream)zos);
            }
            zos.finish();
            zos.flush();
            return;
        }
        DataOutputStream dos = new DataOutputStream(os);
        dos.writeInt(this.size());
        Iterator iterator = this.iterator();
        while (iterator.hasNext()) {
            NDArray nd = (NDArray)iterator.next();
            NDSerializer.encode((NDArray)nd, (OutputStream)dos);
        }
        dos.flush();
    }

    public byte[] getAsBytes() {
        return this.encode();
    }

    public ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap((byte[])this.encode());
    }

    public Shape[] getShapes() {
        return (Shape[])this.stream().map(NDArray::getShape).toArray(Shape[]::new);
    }

    public void close() {
        this.forEach(NDArray::close);
        this.clear();
    }

    public String toString() {
        StringBuilder builder = new StringBuilder(200);
        builder.append("NDList size: ").append(this.size()).append('\n');
        int index = 0;
        Iterator iterator = this.iterator();
        while (iterator.hasNext()) {
            NDArray array = (NDArray)iterator.next();
            String name = array.getName();
            builder.append(index++).append(' ');
            if (name != null) {
                builder.append(name);
            }
            builder.append(": ").append((Object)array.getShape()).append(' ').append((Object)array.getDataType()).append('\n');
        }
        return builder.toString();
    }
}
