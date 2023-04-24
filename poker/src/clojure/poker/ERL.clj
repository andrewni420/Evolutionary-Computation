(ns poker.ERL
  (:require [poker.headsup :as headsup]
            [poker.utils :as utils]
            [poker.transformer :as transformer]
            [poker.ndarray :as ndarray]
            [clojure.set :as set]
            [clj-djl.ndarray :as nd]
            [clojure.pprint :as pprint])
  (:import ai.djl.ndarray.NDManager
           ai.djl.ndarray.NDArray
           ai.djl.ndarray.NDList))


(defn versus
  "Returns the winning individual, the match results of the individuals, or the 
   individuals updated to contain the match results"
  [ind1 ind2 max-seq-length num-games & {:keys [manager net-gain? update-error? as-list? decks]}]
  (with-open [manager (if manager
                        (.newSubManager manager)
                        (nd/new-base-manager))]
    (let [mask (ndarray/ndarray manager (ndarray/causal-mask [1 max-seq-length max-seq-length] -2))
          i1 (transformer/model-from-seeds ind1 max-seq-length manager mask :stdev 0.005)
          i2 (transformer/model-from-seeds ind2 max-seq-length manager mask :stdev 0.005)]
      (with-open [_i1 (utils/make-closeable i1 transformer/close-individual)
                  _i2 (utils/make-closeable i2 transformer/close-individual)]
        (let [{net-gain :net-gain} (apply
                                    headsup/iterate-games-reset
                                    [(transformer/as-player i1) (transformer/as-player i2)]
                                    manager
                                    num-games
                                    (concat (when decks [:decks decks])
                                            (when as-list? [:as-list? as-list?])))]
          (cond update-error? [(update ind1
                                       :error
                                       #(assoc % (:id ind2) ((:id ind1) net-gain)))
                               (update ind2
                                       :error
                                       #(assoc % (:id ind1) ((:id ind2) net-gain)))]
                net-gain? net-gain
                (> ((:id ind2) net-gain) 0) ind2
                :else ind1))))))


#_(time (versus {:seeds [2074038742],
                 :id :p1}
                {:seeds [-888633566],
                 :id :p2}
                20
                1000
                :net-gain? true))

(defn versus-other
  "Plays a transformer seed individual against another individual.\\
   Returns the winning individual, the match results of the individuals, or the 
   individuals updated to contain the match results"
  [individual opponent max-seq-length num-games & {:keys [manager reverse? decks]}]
  (with-open [manager (if manager
                        (.newSubManager manager)
                        (nd/new-base-manager))]
    (let [mask (ndarray/ndarray manager (ndarray/causal-mask [1 max-seq-length max-seq-length] -2))
          i1 (transformer/model-from-seeds individual max-seq-length manager mask)]
      (with-open [_i1 (utils/make-closeable i1 transformer/close-individual)]
        (:net-gain (apply
                    headsup/iterate-games-reset
                    ((if reverse? reverse identity)
                     [(transformer/as-player i1)
                      (if (map? opponent) opponent (utils/init-player opponent :opp))])
                    manager
                    num-games
                    :as-list? true
                    (if decks [:decks decks] [])))))))

#_(versus-other {:seeds [2074038742],
               :id :p1} 
                (utils/init-player utils/random-agent :random)
              10
              10)


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
              #(merge-with + % {id gain}))
      individual)))

(defn benchmark
  "Given a population and a set of benchmark individuals possibly drawn
   from the population, plays each individual in the population against each
   of the benchmark individuals, and updates them with the match results\\
   If symmetrical? is true, then initializes a set of shared decks to be played
   for each matchup. Then each matchup is played twice, once \"normal\" and once
   with players in reversed positions, to reduce variance as much as possible\\
   -> updated population"
  [pop benchmark max-seq-length num-games & {:keys [decks symmetrical? as-list?]}]
  (let [decks (if symmetrical?
                (or decks
                    (repeatedly num-games
                                #(shuffle utils/deck)))
                decks)
        vs #(versus %1 %2 max-seq-length num-games 
                    :net-gain? true 
                    :decks decks
                    :as-list? as-list?)
        fut #(future (vs %1 %2))
        res1 (doall
              (for [ind1 pop
                    ind2 benchmark :when (not (= ind1 ind2))]
                (fut ind1 ind2)))
        res2 (when symmetrical?
               (doall
                (for [ind1 pop
                      ind2 benchmark :when (not (= ind1 ind2))]
                  (fut ind2 ind1))))]
    (reduce (fn [p res]
              (map #(update-individual % res) p))
            pop
            (map deref (concat res1 res2)))))


#_(defn benchmark-pmap
  "Given a population and a set of benchmark individuals possibly drawn
   from the population, plays each individual in the population against each
   of the benchmark individuals, and updates them with the match results\\
   If symmetrical? is true, then initializes a set of shared decks to be played
   for each matchup. Then each matchup is played twice, once \"normal\" and once
   with players in reversed positions, to reduce variance as much as possible\\
   -> updated population"
  [pop benchmark max-seq-length num-games & {:keys [decks symmetrical?]}]
  (let [decks (if symmetrical?
                (or decks
                    (repeatedly num-games
                                #(shuffle utils/deck)))
                decks)
        vs #(versus %1 %2 max-seq-length num-games :net-gain? true :decks decks)
        res1 (for [ind1 pop
                   ind2 benchmark :when (not (= ind1 ind2))]
               [ind1 ind2])
        res2 (when symmetrical?
               (for [ind1 pop
                     ind2 benchmark :when (not (= ind1 ind2))]
                 [ind1 ind2]))]
    (reduce (fn [p res]
              (map #(update-individual % res) p))
            pop
            (doall (pmap #(vs (first %) (second %)) (concat res1 res2))))))

#_(time (benchmark [{:seeds [-1155869325], :id :p0}
                  {:seeds [431529176], :id :p1}
                  {:seeds [1761283695], :id :p2}
                  {:seeds [1749940626], :id :p3}
                  {:seeds [892128508], :id :p4}]
                 (list {:seeds [431529176], :id :p1}
                  {:seeds [1749940626], :id :p3}
                  {:seeds [1761283695], :id :p2})
                 20
                 1000
                 :symmetrical? true))


(defn get-challengers 
  "Get the index of players to compete with each other for round robin 
  tournament. Returns a list of lists, in which the individual at the first
  index must play against all the individuals at the rest of the indices in the 
  list.\\
   This does not mean that an individual only plays against the opponents in its list.
   It could appear as an opponent in another individual's opponent list.\\
  -> ((player1 opp1 opp2 ...) ...) "
  [pop-count round-robin-count]
  (assert (or (even? pop-count) (even? round-robin-count)) "pop*round-robin-count must be even since 2 players participate in each game")
  (let [challengers (take pop-count
                          (partition (int (inc (Math/floor (/ round-robin-count 2))))
                                     1
                                     (cycle (range pop-count))))
        challengers (if (odd? round-robin-count)
                      (concat challengers
                              (map #(vector % (+ % (/ pop-count 2)))
                                   (range (/ pop-count 2))))
                      challengers)]
    challengers))

(defn round-robin
  "Round robin tournament in which each player in the population plays 
   round-robin-count other players. Updates each player with the match 
   results against other players\\
   -> updated population"
  [pop num-games round-robin-count max-seq-length]
  (loop [challengers (get-challengers (count pop) round-robin-count)
         pop pop]
    (if (empty? challengers)
      pop
      (recur (rest challengers)
             (let [[p1 & others] (first challengers)]
               (loop [o others
                      pop pop]
                 (if (empty? o)
                   pop
                   (recur (rest o)
                          (let [p2 (first o)
                                [ind1 ind2] (versus (pop p1) (pop p2) max-seq-length num-games)]
                            (assoc pop 
                                   p1 ind1
                                   p2 ind2))))))))))

(defn lexicase-selection
  "Lexicase selection using the individuals as test cases. Picks a random individual,
   and selects from the population by their match results against that individual\\
   For individuals that haven't played against the test case individual, assume neither won
   money from the other\\
   If a set of benchmarking test case individuals has not been provided, uses population
   as test cases\\
   -> individual"
  [pop & {:keys [benchmark-pop]}]
  (loop [opponents (shuffle (map :id (or benchmark-pop pop)))
         pop pop]
    (cond (empty? opponents) (rand-nth pop)
          (= 1 (count pop)) (first pop)
          :else (recur (rest opponents)
                       (let [errors (map #(or ((first opponents)
                                               (:error %))
                                              0)
                                         pop)
                             avg-error (utils/mean errors)
                             max-error (apply max errors)
                             med-deviation (utils/median (map #(abs (- % avg-error))
                                                              errors))]
                         #_(println "avg-error " avg-error "med-deviation " med-deviation)
                         (map first
                              (filter #(>= (second %)
                                          (- max-error med-deviation))
                                      (zipmap pop errors))))))))

(defn mutate
  "Given a random number generator and an individual,
   removes the individual's errors and adds a random seed to it\\
   -> individual"
  [individual random id]
  (-> individual
      (dissoc :error)
      (update :seeds conj (.nextInt random))
      (update :id #(keyword (str (name %) "-" id)))))

(defn next-generation
  "Selects and mutates indivduals, producing the new generation 
   of individuals\\
   Adds parents to hall of fame\\
   -> pop"
  [pop random]
  (let [l (count pop)]
    (loop [new-pop (transient [])
           hof (transient #{})
           i 0]
      (if (= i l)
        [(persistent! new-pop) (persistent! hof)]
        (let [parent (lexicase-selection pop)]
          (recur (conj! new-pop (mutate parent random i))
                 (conj! hof parent)
                 (inc i)))))))

#_(next-generation [{:id :p0, :error {:p1 -0.75, :p2 -0.3, :p3 -0.79, :p4 0.46}, :seeds [-4964420948893066024]}
                    {:id :p1, :error {:p0 0.75, :p2 0.55, :p3 0.12, :p4 0.72}, :seeds [7564655870752979346]}
                    {:id :p2, :error {:p1 -0.55, :p0 0.3, :p3 -1.35, :p4 1.22}, :seeds [3831662765844904176]}
                    {:id :p3, :error {:p1 1.78, :p0 1.15, :p2 0.93, :p4 -1.84}, :seeds [6137546356583794141]}
                    {:id :p4, :error {:p1 -0.37, :p0 -0.505, :p2 1.36, :p3 -1.7}, :seeds [-594798593157429144]}]
                   (utils/random 1))

(defn select-from-hof
  "Selects n individual from the hall of fame. \\
   method: method used to select individuals\\
   :exp = exponential. Selects 1 individual from the previous ⌊e⌋ generations, 1 from the 
   previous ⌊e^2⌋ generations, and so on\\
   :random = randomly selects n individuals from all individuals so far"
  [hof n & {:keys [method]
            :or {method :exp}}]
  (condp = method
    :exp (loop [i 0
                selected #{}]
           (if (>= i n) selected
               (recur (inc i)
                      (set/union
                       selected
                       (loop [p (shuffle (mapcat identity
                                                 (take-last (int (Math/exp i))
                                                            hof)))]
                         (println (count p))
                         (cond (empty? p) #{}
                               (selected (first p)) (recur (rest p))
                               :else #{(first p)}))))))
    :random (take n (shuffle (mapcat identity hof)))))


#_(select-from-hof 
 [#{{:seeds [1] :id :p0}
    {:seeds [2] :id :p1}
    {:seeds [3] :id :p2}}
  #{{:seeds [2 4] :id :p1-0}
    {:seeds [3 5] :id :p2-0}
    {:seeds [2 6] :id :p1-1}}
  #{{:seeds [3 5 6] :id :p2-0-0}
    {:seeds [3 5 7] :id :p2-0-1}
    {:seeds [3 5 8] :id :p2-0-2}}
  #{{:seeds [3 5 7 9] :id :p2-0-1-0}
    {:seeds [3 5 8 10] :id :p2-0-2-0}
    {:seeds [3 5 8 11] :id :p2-0-2-1}}]
 3)


(def error-functions {:round-robin round-robin
                      :benchmark benchmark})

(defn report-generation
  [pop generation]
  (pprint/pprint {:generation generation
                  :pop pop}))

(defn ERL
  [& {:keys [pop-size num-generations num-games benchmark-count random-seed max-seq-length]
      :or {pop-size 3
           num-generations 1
           num-games 10
           benchmark-count 5
           random-seed 1
           max-seq-length 20}}]
  (let [r (utils/random random-seed)]
    (loop [generation 0
           pop (mapv #(assoc {}
                             :seeds [(.nextInt r)]
                             :id (keyword (str "p" %)))
                     (range pop-size))
           hof []]
      (report-generation pop generation)
      (if (= generation num-generations)
        [pop hof]
        (let [[p h] (time (-> pop
                              (benchmark (concat (take (Math/ceil (/ benchmark-count 2))
                                                       (shuffle pop))
                                                 (select-from-hof hof
                                                                  (Math/floor (/ benchmark-count 2))))
                                         max-seq-length
                                         num-games
                                         :symmetrical? true)
                              (next-generation r)))]
          (recur (inc generation)
                 p
                 (conj hof h)))))))
