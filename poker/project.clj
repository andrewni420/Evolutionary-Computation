(defproject poker "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [net.clojars.lspector/propeller "0.3.0"]
                 [cheshire "5.11.0"]
                 [clj-python/libpython-clj "2.024"]
                 [net.mikera/core.matrix "0.63.0"]
                 [cnuernber/dtype-next "10.000-beta-39"]
                 [techascent/tech.ml.dataset "7.000-beta-31"]
                 [clj-djl "0.1.9"]
                 [clj-djl/dataframe "0.1.2"]
                 [ai.djl/api "0.21.0"]
                 [ai.djl.pytorch/pytorch-engine "0.21.0"]
                 [ai.djl.pytorch/pytorch-native-auto "1.9.1"]
                 [ai.djl.mxnet/mxnet-engine "0.21.0"]
                 [ai.djl.mxnet/mxnet-native-mkl "1.9.1"]
                 [org.slf4j/slf4j-api "2.0.7"]
                 [org.slf4j/slf4j-simple "2.0.7"]]
  :main ^:skip-aot poker.core
  :source-paths      ["src/clojure"]
  :java-source-paths ["src/java"]
  :target-path "target/%s"
  :profiles {:precomp {:prep-tasks ^:replace ["beaver" "compile"]
                       :source-paths ["src/pre/clojure"]
                       :aot [parser.ast]}})
