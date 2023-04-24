(ns poker.core
  (:require [poker.ERL :as ERL]
            [clj-djl.ndarray :as nd]
            [clojure.pprint :as pprint]
            [libpython-clj2.python :as py]
            [libpython-clj2.require :as req]
            [poker.utils :as utils]
            [poker.concurrent :as concurrent]
            [poker.transformer :as transformer]
            [poker.headsup :as headsup]
            [poker.ndarray :as ndarray])
  (:import ai.djl.Device
           ai.djl.engine.Engine
           java.lang.Thread
           java.lang.Runtime))


(defn update-individual
  "Update the map of match results of an individual\\
   If the indiviudal already contains match results against this opponent,
   adds the match results together\\
   -> individual"
  [individual net-gain]
  (let [[p1 p2] (keys net-gain)
        [id gain] (condp = (:id individual)
                    p1 [p2 (net-gain p1)]
                    p2 [p1 (net-gain p2)]
                    [nil nil])]
    (if id
      (update individual
              :error
              #(merge-with conj % {id gain}))
      individual)))

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
              (map #(update-individual % res) p))
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
  (ERL/versus-other {:seeds [-1155869325], :id :p0} 
                  (utils/init-player utils/rule-agent :rule)
                  10
                  10
                  :manager m
                  :symmetrical? false)
  (println (count (.getManagedArrays m))))


(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  #_(mapv #(do (println %)
               (println (ERL/versus (parent-seeds %)
                                    (parent-seeds (mod (inc %) (count parent-seeds)))
                                    10
                                    1000
                                    :net-gain? true
                                    :as-list? true)))
          (range 10))
  #_(benchmark [{:seeds [-2003437247 1540470339], :id :p5-0}]
               [(utils/init-player utils/wait-and-bet :wait-and-bet)]
               10
               10
               :symmetrical? false)
  #_(with-open [manager (nd/new-base-manager)]
    (let [mask (ndarray/ndarray manager (ndarray/causal-mask [1 10 10] -2))
          i1 (transformer/model-from-seeds {:seeds [-2003437247 1540470339], :id :p5-0} 10 manager mask)]
      (with-open [_i1 (utils/make-closeable i1 transformer/close-individual)]
        (:net-gain (headsup/iterate-games-reset
                    [(transformer/as-player i1)
                     (utils/init-player utils/wait-and-bet :wait-and-bet)]
                    manager
                    10
                    :as-list? false
                    :decks (repeatedly 10
                                       #(shuffle utils/deck)))))))
  #_(ERL/versus-other {:seeds [-2003437247 1540470339], :id :p5-0}
                      (utils/init-player utils/wait-and-bet :wait-and-bet) 10 10 :reverse? false :decks (repeatedly 10
                                                                                                                    #(shuffle utils/deck)))
  #_(time (let [futures (doall (for [i (range 10)]
                                 (concurrent/msubmit (benchmark (nth children i)
                                                                [(utils/init-player utils/random-agent :random)
                                                                 (utils/init-player utils/rule-agent :rule)
                                                                 (utils/init-player utils/wait-and-bet :wait-and-bet)]
                                                                10
                                                                20
                                                                :symmetrical? false))))]
            (println (with-out-str (run! pprint/pprint (mapcat deref futures)))))
          #_(benchmark
             [{:seeds [-1155869325], :id :p0}
              {:seeds [431529176], :id :p1}
              {:seeds [1761283695], :id :p2}
              {:seeds [1749940626], :id :p3}
              {:seeds [892128508], :id :p4}
              {:seeds [-2003437247], :id :p5}
              {:seeds [1487394176], :id :p6}
              {:seeds [1049991269], :id :p7}
              {:seeds [-1224600590], :id :p8}
              {:seeds [-1437495699], :id :p9}]
             (list (utils/init-player utils/random-agent :random)
                   (utils/init-player utils/rule-agent :rule)
                   (utils/init-player utils/wait-and-bet :wait-and-bet))
             10
             10
             :manager manager
             :symmetrical? false))
  #_(catch Exception e (println (str e (.getCause e) (.getCause (.getCause e)))))
  #_(ERL/ERL :pop-size 10
             :num-generations 2
             :num-games 500
             :benchmark-count 2
             :random-seed 323091568684100223
             :max-seq-length 100))



#_(-main)



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



