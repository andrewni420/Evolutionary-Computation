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
            [poker.Andrew.processresult :as processresult]
            [clojure.string :as s])
  (:import ai.djl.Device
           ai.djl.engine.Engine
           poker.SparseAttentionBlock
           java.lang.Thread
           java.lang.Runtime
           ai.djl.nn.core.SparseMax
           ai.djl.ndarray.types.Shape))


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

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  #_(MPI/test-mpi)
  (print-as-vector
   (println (MPI/ERL :pop-size 25
                     :num-generations 25
                     :num-games 500
                     :benchmark-count 4
                     :random-seed -4467023281627811388
                     :max-seq-length 100
                     :stdev 0.005
                     :from-block? true)))
  #_(processresult/singlerun-versus
   ["stdev-pretest-0.5.txt"
    "stdev-pretest-0.05.txt"
    "stdev-pretest-0.005.txt"])
  #_(run! processresult/multirun-versus ["ERL-num-games-comparison-63748.out"
                                       "ERL-pop-ablation-63759.out"])
  #_(run! processresult/multirun-versus ["ERL-benchmark-comparison-63742.out"
                                         "ERL-generation-ablation-63760.out"])
  #_(do (processresult/multirun-versus "ERL-2-4-8-16-heads-63899.out" :keep-all? true)
      (processresult/multirun-versus "ERL-seq-length-comparison-63746.out"))
  #_(processresult/multirun-versus ["ERL-pop-gen-300-63782.out"])
  #_(processresult/multirun-versus ["ERL-pop-gen-ablation-63761.out"])
  #_(processresult/multirun-versus ["ERL-pop-gen-1200-15-63786.out"
                                  "ERL-pop-gen-1200-25-63783.out"
                                  "ERL-pop-gen-1200-34-63785.out"])
  #_(processresult/generation-versus "ERL-pop-ablation-63759.out"))



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





(def results
  [{:seeds [-1155869325],
    :id :p0,
    :error
    {:random {:mean -2.299516800789707, :stdev 131.66351464871548},
     :rule {:mean -0.7250685400997252, :stdev 47.708607827318374},
     :wait-and-bet
     {:mean -2.3600650542667894, :stdev 146.6509342787873}}}
   {:seeds [431529176],
    :id :p1,
    :error
    {:random {:mean -1.5521247036917054, :stdev 104.75501236574735},
     :rule {:mean -0.8133563592020555, :stdev 30.194003659552948},
     :wait-and-bet
     {:mean -1.1567215179104555, :stdev 104.62215767774595}}}
   {:seeds [1761283695],
    :id :p2,
    :error
    {:random {:mean 0.008500111910809665, :stdev 112.46387451266838},
     :rule {:mean -1.044665386146007, :stdev 46.24727233974615},
     :wait-and-bet
     {:mean -2.960686570833728, :stdev 122.57392377195217}}}
   {:seeds [1749940626],
    :id :p3,
    :error
    {:random {:mean -0.3168140685832318, :stdev 143.9859819349083},
     :rule {:mean 1.4525024551092394, :stdev 58.25149253360078},
     :wait-and-bet {:mean 0.6060240795430073, :stdev 179.1475424957181}}}
   {:seeds [892128508],
    :id :p4,
    :error
    {:random {:mean -0.15022848106910097, :stdev 46.37055546471076},
     :rule {:mean -0.1436437453508377, :stdev 6.410748139314529},
     :wait-and-bet
     {:mean -0.37940930548223034, :stdev 55.95697681953668}}}
   {:seeds [-2003437247],
    :id :p5,
    :error
    {:random {:mean -2.7354956840615428, :stdev 146.89224490556518},
     :rule {:mean 0.8232784890317009, :stdev 77.17310121116614},
     :wait-and-bet
     {:mean 1.8629400051241978, :stdev 188.14449965400183}}}
   {:seeds [1487394176],
    :id :p6,
    :error
    {:random {:mean -0.8668267477660971, :stdev 102.72117702471415},
     :rule {:mean 0.1882938726159683, :stdev 32.136258063038895},
     :wait-and-bet
     {:mean -3.4200330007826416, :stdev 127.17030915898582}}}
   {:seeds [1049991269],
    :id :p7,
    :error
    {:random {:mean 1.5159529468007968, :stdev 138.913335181375},
     :rule {:mean -1.5669364774871435, :stdev 78.35807147742116},
     :wait-and-bet
     {:mean -10.408253691163873, :stdev 166.17206284223877}}}
   {:seeds [-1224600590],
    :id :p8,
    :error
    {:random {:mean 0.22393526503095718, :stdev 128.38091677413968},
     :rule {:mean 1.0679116695760626, :stdev 62.86526960602778},
     :wait-and-bet
     {:mean -3.4655270480350513, :stdev 151.89216337318504}}}
   {:seeds [-1437495699],
    :id :p9,
    :error
    {:random {:mean -1.0028136983822598, :stdev 32.26840451945378},
     :rule {:mean -0.23953652721040417, :stdev 6.679605727512594},
     :wait-and-bet {:mean -0.953388852098994, :stdev 32.21449974044938}}}])

#_(map #(/ (:mean %) (/ (:stdev %) 100)) (map :wait-and-bet (map :error results)))



