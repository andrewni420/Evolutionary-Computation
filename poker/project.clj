(defproject poker "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [net.clojars.lspector/propeller "0.3.0"]
                 [clj-http "3.12.3"]
                 [cheshire "5.11.0"]
                 [clj-python/libpython-clj "2.024"]]
  :main ^:skip-aot poker.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all
                       :jvm-opts ["--add-modules" 
                                  "jdk.incubator.foreign"
                                  "--enable-native-access=ALL-UNNAMED"
                                  "-Dclojure.compiler.direct-linking=true"]}})
