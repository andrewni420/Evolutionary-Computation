(ns poker.core
  (:require [poker.ERL :as ERL]
            [poker.MPI :as MPI]
            [clojure.pprint :as pprint]
            [poker.utils :as utils]
            [poker.concurrent :as concurrent]
            [poker.transformer :as transformer]
            [poker.headsup :as headsup]
            [poker.onehot :as onehot]
            [poker.ndarray :as ndarray]
            [clojure.core.matrix :as m]
            [poker.slumbot :as slumbot]
            [poker.Andrew.processresult :as processresult]
            [clojure.string :as s])
  (:import ai.djl.Device
           ai.djl.engine.Engine
           poker.SparseAttentionBlock
           java.lang.Thread
           java.lang.Runtime
           ai.djl.nn.core.SparseMax
           ai.djl.ndarray.types.Shape
           java.lang.System))


(defn benchmark
  "Given a population and a set of benchmark individuals possibly drawn
   from the population, plays each individual in the population against each
   of the benchmark individuals, and updates them with the match results\\
   If symmetrical? is true, then initializes a set of shared decks to be played
   for each matchup. Then each matchup is played twice, once \"normal\" and once
   with players in reversed positions, to reduce variance as much as possible\\
   -> updated population"
  [pop benchmark max-seq-length num-games & {:keys [manager decks symmetrical?]}]
  (let [decks (if symmetrical?
                (or decks
                    (repeatedly num-games
                                #(shuffle utils/deck)))
                decks)
        vs (if manager
             #(ERL/versus-other %1 %2 max-seq-length num-games :reverse? %3 :decks decks :manager manager)
             #(ERL/versus-other %1 %2 max-seq-length num-games :reverse? %3 :decks decks))
        res1 (doall
              (for [ind1 pop
                    ind2 benchmark :when (not (= ind1 ind2))]
                (concurrent/msubmit (vs ind1 ind2 false))))
        res2 (when symmetrical?
               (doall
                (for [ind1 pop
                      ind2 benchmark :when (not (= ind1 ind2))]
                  (concurrent/msubmit (vs ind1 ind2 true)))))]
    (reduce (fn [p res]
              (map #(ERL/update-individual % res :merge-fn conj) p))
            pop
            (map deref (concat res1 res2)))))


(def parent-seeds [{:seeds [-1155869325], :id :p0}
                   {:seeds [431529176], :id :p1}
                   {:seeds [1761283695], :id :p2}
                   {:seeds [1749940626], :id :p3}
                   {:seeds [892128508], :id :p4}
                   {:seeds [-2003437247], :id :p5}
                   {:seeds [1487394176], :id :p6}
                   {:seeds [1049991269], :id :p7}
                   {:seeds [-1224600590], :id :p8}
                   {:seeds [-1437495699], :id :p9}])


(def children (let [r (utils/random 1682255299119)
                    res (into [] (mapcat :seeds parent-seeds))]
                (into [] (for [s res]
                           (mapv (fn [id] {:seeds [s (.nextInt r)]
                                           :id (keyword (str "p" (.indexOf res s) "-" id))})
                                 (range 10))))))


#_(with-open [m (nd/new-base-manager)]
    (let [arr (ndarray/ndarray m [[[[-100000.1,  -99999.9],
                                    [-99999.6,  -99999.7]]]])]
      (println (into [] (transformer/forward (poker.SparseMax. -1 2)
                                             (nd/ndlist arr))))
      #_(println (.exp arr))))


#_(with-open [m (nd/new-base-manager)]
    (ERL/versus-other {:seeds [-1155869325], :id :p0}
                      (utils/init-player utils/rule-agent :rule)
                      10
                      10
                      :manager m
                      :symmetrical? false)
    (println (count (.getManagedArrays m))))

(defmacro print-as-vector
  "Prints [, then prints the body, then prints ]"
  [& body]
  `(try (println "[")
        (println ~@body)
        (finally (println "]"))))

(defn do-erl []
  (ERL/ERL :pop-size 25
           :num-generations 25
           :num-games 500
           :benchmark-count 6
           :random-seed -3057099454162971707
           :max-seq-length 100
           :stdev 0.005))


(defn read-file
  "Reads in a clojure data structure from a file. Automatically prepends the path
   src/clojure/poker/Andrew/results/ \\
   Use make-vector to read all data structures in the file, not just the first, into a 
   vector."
  [name & {:keys [make-vector?]}]
  (read-string
   (if make-vector?
     (str "[" (slurp (str "src/clojure/poker/Andrew/results/" name)) "]")
     (slurp (str "src/clojure/poker/Andrew/results/" name)))))


(defn hot-test
  [& {:keys [block-size next-gen-method num-games gen-output gen-input]}]
  (println gen-output)
  (println gen-input))

(defn hot-start
  [& {:keys [hof-output hof-input gen-output gen-input param-output param-input default-pmap override-params?]
      :as argmap}]
  (let [params (try (read-string (slurp param-input))
                    (catch Exception _ {}))]
    (apply MPI/ERL
           (mapcat identity
                   (into []
                         (merge (if override-params?
                                  (merge params default-pmap)
                                  (merge default-pmap params))
                                (dissoc argmap
                                        :default-pmap)))))))

;;hot-start editing: :max-actions 2198.9734734734734, :time-ms 2154080.222549}

#_(def challenger {:seeds [1214870254 -599077455 1389138710 1985198127 -1196029346 1419764478 -1011071022 344414086 839058045 -808557128 569981436 2141209152 -886498085 -333498454 -665775348 -1877394347 1385959825 669837050 1742082696 342463688 1977105854 -1588884343 775375154 -1015789197 301259794 -74722191 -602854656 780026697 876536406 -2094359061 1346730382 -586561425 -1330699011 -1144666484 1038533690 1603090449 977826193 587534028 -225244400 -1170317260 -319860625 -1914716520 1527872707 1103455723 -737931723 882570405 1149152104 807505075 -401238177 -683322400 336795851 -1558784814 -1469434853 102388824 -1780263778 -24926424 1205584016 130414921 -1934000314 1553138048 1429560218 258005691 -286366515 240703996 173961293 -1212181469 -1780174093 -668870061 574023453 1464857067 -632904208 537720295 542981329 -1074007060 -1935814956 -2114576480 -1633447819 473098796 -714965007 -1795511096 -191769306 -556377078 -812145397 1808738635 2098925459 -232657773 437076651 895912599 559998331 -525209603 -180078282 598128726 508846068 -2097530563 154195222 1793543679 1937995078 1172127731 -1354005376 816419957],
                   :id :client
                   :stdev 0.005})

;:hof-output "src/clojure/poker/Andrew/results/__hof.out"
               ;:hof-input "src/clojure/poker/Andrew/results/__hof.out"
               ;:gen-output "src/clojure/poker/Andrew/results/__gen.out"
               ;:gen-input "src/clojure/poker/Andrew/results/__gen.out"
               ;:param-output "src/clojure/poker/Andrew/results/__param.out"
               ;:param-input "src/clojure/poker/Andrew/results/__param.out"

(def default-pmap
  {:pop-size 100
   :num-generations 50
   :num-games 500
   :benchmark-count 3
   :random-seed -8411666870417163767
   :max-seq-length 100
   :stdev 0.005
   :from-block? true
   :next-gen-method :k-best
   :bench-method :k-best
   :bench-exp 2
   :prop-hof 1.0
   :block-size 1e9})

(def default-transformer-map
  {:d-model 256;;
   :d-ff 1024;;
   :num-layers 12;;
   :num-heads 8
   :d-pe [64 64 64 64];;
   :max-seq-length 512})


#_(MPI/multi-ERL
   :ERL-argmaps (for [stdev [0.001 0.002 0.004 0.008]]
                  (assoc default-pmap
                         :stdev stdev
                         :transformer-parameters default-transformer-map))
   :intra-run? true
   :inter-run? true)

#_(MPI/multi-ERL
   :ERL-argmaps (for [bench-method [:k-best]
                      benchmark-count [2 4 6 8]]
                  (assoc default-pmap
                         :bench-method bench-method
                         :benchmark-count benchmark-count
                         :transformer-parameters default-transformer-map))
   :intra-run? true
   :inter-run? true
   :num-games 5000)

(defn -main
  [& args]
  #_(slumbot/slumbot-rollout challenger "vsSlumbot-1.txt" 10 2000
                             :transformer? true
                             :from-block? true
                             :random-seed -5907454322436654
                             :block-size 1e8)
  (MPI/multi-ERL
   :ERL-argmaps (for [stdev [0.0001]]
                  (assoc default-pmap
                         :stdev stdev
                         :pop-size 25
                         :num-generations 25
                         :num-games 125
                         :benchmark-count 10
                         :bench-exp 1.5
                         :next-gen-method :parents
                         :bench-method :hardexp
                         :transformer-parameters default-transformer-map))
   :intra-run? true
   :inter-run? false
   :num-games 5000)
  #_(MPI/with-MPI
      (doall (for [bench-method [:k-best]
                   benchmark-count [2 4 6 8]]
               (println (hot-start :default-pmap (assoc default-pmap
                                                        :pop-size 10
                                                        :num-generations 3
                                                        :num-games 10
                                                        :bench-method bench-method
                                                        :benchmark-count benchmark-count)
                                   :transformer-parameters default-transformer-map)))))

  #_(println (slumbot/net-gain "slumbot-history-random.txt" :as-list? true :mapcat? true))
  #_(processresult/generation-versus "ERL-250x100-67545.out"))



#_(-main)


(def hyperparameter-search
  "Keep other parameters small while we range one parameter from
   small to large. e.g. minimal model, pop, num-games, benchmark, max-seq with various
   standard deviations"
  {:model-architecture {:num-layers [6 8 12]
                        :num-heads [4 8 16]
                        :d-model [64 128 256]
                        :d-ff [256 512 1024]
                        :d-pe [16 32 64]
                        :sparse? [true false]}
   :pop-size [25 50 100 200]
   :num-games [250 500 1000 2000]
   :benchmark-count [4 8 12 16]
   :max-seq-length [50 100 200 400]
   :stdev [0.001 0.005 0.01 0.05 0.1]})





