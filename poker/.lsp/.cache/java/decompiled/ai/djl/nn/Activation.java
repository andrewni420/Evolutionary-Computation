/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.ndarray.NDArrays
 *  ai.djl.nn.Block
 *  ai.djl.nn.LambdaBlock
 *  ai.djl.nn.core.Prelu
 *  java.lang.Integer
 *  java.lang.Number
 *  java.lang.Object
 */
package ai.djl.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.core.Prelu;

public final class Activation {
    private Activation() {
    }

    public static NDArray relu(NDArray array) {
        return array.getNDArrayInternal().relu();
    }

    public static NDList relu(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().relu());
    }

    public static NDArray relu6(NDArray array) {
        return NDArrays.minimum((Number)Integer.valueOf((int)6), (NDArray)array.getNDArrayInternal().relu());
    }

    public static NDList relu6(NDList arrays) {
        return new NDList(Activation.relu6(arrays.singletonOrThrow()));
    }

    public static NDArray sigmoid(NDArray array) {
        return array.getNDArrayInternal().sigmoid();
    }

    public static NDList sigmoid(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().sigmoid());
    }

    public static NDArray tanh(NDArray array) {
        return array.getNDArrayInternal().tanh();
    }

    public static NDList tanh(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().tanh());
    }

    public static NDArray softPlus(NDArray array) {
        return array.getNDArrayInternal().softPlus();
    }

    public static NDList softPlus(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().softPlus());
    }

    public static NDArray softSign(NDArray array) {
        return array.getNDArrayInternal().softSign();
    }

    public static NDList softSign(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().softSign());
    }

    public static NDArray leakyRelu(NDArray array, float alpha) {
        return array.getNDArrayInternal().leakyRelu(alpha);
    }

    public static NDList leakyRelu(NDList arrays, float alpha) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().leakyRelu(alpha));
    }

    public static NDArray elu(NDArray array, float alpha) {
        return array.getNDArrayInternal().elu(alpha);
    }

    public static NDList elu(NDList arrays, float alpha) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().elu(alpha));
    }

    public static NDArray selu(NDArray array) {
        return array.getNDArrayInternal().selu();
    }

    public static NDList selu(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().selu());
    }

    public static NDArray gelu(NDArray array) {
        return array.getNDArrayInternal().gelu();
    }

    public static NDList gelu(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().gelu());
    }

    public static NDArray swish(NDArray array, float beta) {
        return array.getNDArrayInternal().swish(beta);
    }

    public static NDList swish(NDList arrays, float beta) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().swish(beta));
    }

    public static NDArray mish(NDArray array) {
        return array.getNDArrayInternal().mish();
    }

    public static NDList mish(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().mish());
    }

    public static Block reluBlock() {
        return new LambdaBlock(Activation::relu, "ReLU");
    }

    public static Block relu6Block() {
        return new LambdaBlock(Activation::relu6, "ReLU6");
    }

    public static Block sigmoidBlock() {
        return new LambdaBlock(Activation::sigmoid, "sigmoid");
    }

    public static Block tanhBlock() {
        return new LambdaBlock(Activation::tanh, "Tanh");
    }

    public static Block softPlusBlock() {
        return new LambdaBlock(Activation::softPlus, "softPlus");
    }

    public static Block softSignBlock() {
        return new LambdaBlock(Activation::softSign, "softSign");
    }

    public static Block leakyReluBlock(float alpha) {
        return new LambdaBlock(arrays -> Activation.leakyRelu(arrays, alpha), "LeakyReLU");
    }

    public static Block eluBlock(float alpha) {
        return new LambdaBlock(arrays -> Activation.elu(arrays, alpha), "ELU");
    }

    public static Block seluBlock() {
        return new LambdaBlock(Activation::selu, "SELU");
    }

    public static Block geluBlock() {
        return new LambdaBlock(Activation::gelu, "GELU");
    }

    public static Block swishBlock(float beta) {
        return new LambdaBlock(arrays -> Activation.swish(arrays, beta), "Swish");
    }

    public static Block mishBlock() {
        return new LambdaBlock(Activation::mish, "Mish");
    }

    public static Block preluBlock() {
        return new Prelu();
    }
}
