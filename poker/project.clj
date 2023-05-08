(defproject poker "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [cheshire "5.11.0"]
                 [clj-python/libpython-clj "2.024"]
                 [net.mikera/core.matrix "0.63.0"]
                 [cnuernber/dtype-next "10.000-beta-39"]
                 [techascent/tech.ml.dataset "7.000-beta-31"]
                 [ai.djl/api "0.21.0"]
                 [ai.djl.pytorch/pytorch-engine "0.22.1"]
                 [ai.djl.pytorch/pytorch-jni "2.0.0-0.22.1"]
                 [ai.djl.mxnet/mxnet-engine "0.22.1"]
                 [org.slf4j/slf4j-api "2.0.7"]
                 [org.slf4j/slf4j-simple "2.0.7"]
                 [org.apache.commons/commons-math3 "3.6.1"]]
  :exclusions []
  :main ^:skip-aot poker.core
  :source-paths      ["src/clojure"]
  :java-source-paths ["src/java"]
  :target-path "target/%s"
  :jvm-opts ["-Dai.djl.pytorch.graph_optimizer=false"
             "-Dai.djl.pytorch.num_interop_threads=116"
             "-Dai.djl.pytorch.num_intraop_threads=1"
             "-Dai.djl.disable_close_resource_on_finalize=true"
             "-Xmx100g"
             "-XX:MaxGCPauseMillis=100"
             "-Dcom.sun.management.jmxremote"
             "-Dcom.sun.management.jmxremote.port=1089"
             "-Dcom.sun.management.jmxremote.ssl=false"
             "-Dcom.sun.management.jmxremote.authenticate=false"
             ]
  :profiles {:precomp {:source-paths ["src/pre/clojure"]
                       :aot [poker.headsup
                             poker.utils
                             poker.transformer
                             poker.ndarray
                             poker.onehot]}})




