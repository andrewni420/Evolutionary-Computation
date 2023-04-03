/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.djl.metric.Metric
 *  ai.djl.metric.Unit
 *  java.lang.Double
 *  java.lang.IllegalArgumentException
 *  java.lang.Number
 *  java.lang.Object
 *  java.lang.String
 *  java.util.ArrayList
 *  java.util.Collection
 *  java.util.Collections
 *  java.util.Comparator
 *  java.util.List
 *  java.util.Map
 *  java.util.Set
 *  java.util.concurrent.ConcurrentHashMap
 *  java.util.function.BiConsumer
 *  java.util.stream.Collectors
 */
package ai.djl.metric;

import ai.djl.metric.Metric;
import ai.djl.metric.Unit;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

public class Metrics {
    private Map<String, List<Metric>> metrics = new ConcurrentHashMap();
    private int limit;
    private BiConsumer<Metrics, String> onLimit;

    public void setLimit(int limit) {
        this.limit = limit;
    }

    public void setOnLimit(BiConsumer<Metrics, String> onLimit) {
        this.onLimit = onLimit;
    }

    public void addMetric(Metric metric) {
        List list = (List)this.metrics.computeIfAbsent((Object)metric.getMetricName(), v -> Collections.synchronizedList((List)new ArrayList()));
        if (this.limit > 0 && list.size() >= this.limit) {
            if (this.onLimit != null) {
                this.onLimit.accept((Object)this, (Object)metric.getMetricName());
            }
            list.clear();
        }
        list.add((Object)metric);
    }

    public void addMetric(String name, Number value) {
        this.addMetric(new Metric(name, value));
    }

    public void addMetric(String name, Number value, Unit unit) {
        this.addMetric(new Metric(name, value, unit));
    }

    public boolean hasMetric(String name) {
        return this.metrics.containsKey((Object)name);
    }

    public List<Metric> getMetric(String name) {
        List list = (List)this.metrics.get((Object)name);
        if (list == null) {
            return Collections.emptyList();
        }
        return list;
    }

    public Set<String> getMetricNames() {
        return this.metrics.keySet();
    }

    public Metric latestMetric(String name) {
        List list = (List)this.metrics.get((Object)name);
        if (list == null || list.isEmpty()) {
            throw new IllegalArgumentException("Could not find metric: " + name);
        }
        return (Metric)list.get(list.size() - 1);
    }

    public Metric percentile(String metricName, int percentile) {
        List metric = (List)this.metrics.get((Object)metricName);
        if (metric == null || this.metrics.isEmpty()) {
            throw new IllegalArgumentException("Metric name not found: " + metricName);
        }
        ArrayList list = new ArrayList((Collection)metric);
        list.sort(Comparator.comparingDouble(Metric::getValue));
        int index = metric.size() * percentile / 100;
        return (Metric)list.get(index);
    }

    public double mean(String metricName) {
        List metric = (List)this.metrics.get((Object)metricName);
        if (metric == null || this.metrics.isEmpty()) {
            throw new IllegalArgumentException("Metric name not found: " + metricName);
        }
        return (Double)metric.stream().collect(Collectors.averagingDouble(Metric::getValue));
    }
}
