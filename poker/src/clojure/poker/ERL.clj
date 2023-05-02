(ns poker.ERL
  (:require [poker.headsup :as headsup]
            [poker.utils :as utils]
            [poker.transformer :as transformer]
            [poker.ndarray :as ndarray]
            [poker.concurrent :as concurrent]
            [clojure.set :as set]
            [clojure.pprint :as pprint]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;   Evolutionary Reinforcement Learning     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Overview:
;;; At its core, this instance of competitive (co)evolutionary reinforcement learning
;;; is just a genetic algorithm in which individuals play other individuals to
;;; determine the fitness function
;;; 
;;; The structure of the coevolutionary loop is as follows:
;;; 1. Initialization of population
;;;        As in https://arxiv.org/abs/1712.06567 and https://arxiv.org/abs/1703.03864 
;;;        an individual is initialized as a single integer that seeds a random number generator.
;;;        This random number generator is then used to initialize the weights of the transformer
;;;        neural net using Xavier initialization, where each parameter is drawn from a normal
;;;        distribution with mean 0 and stdev 1/√N where N is the number of incoming neurons
;;;        This instantiation of Xavier initialization was proposed in https://arxiv.org/abs/1502.01852
;;; 2. Fitness evaluation
;;;        Much of the work on coevolution revolves around this fitness evaluation, as
;;;        the nonstationarity of the environment makes it difficult to design a fitness
;;;        function that pushes generations toward optimal solutions while avoiding common
;;;        pathologies like local minima, cycling, and evolutionary forgetting.
;;;        Common evaluation methods implemented here are as follows:
;;;            Single-elimination tournament: Individuals are randomly paired up, each pair plays games against each other,
;;;                and the winning player moves on to the next round. Unpaired individuals automatically progress. 
;;;                The fitness of an individual is the highest round reached by that individual
;;;                This fitness evaluation method has the advantage of devoting more resources towards
;;;                differentiating between very good individuals, but is difficult to parallelize and is 
;;;                very selective, leading to local optima. Therefore it's used to select an elite "best" individual for each generation
;;;            Benchmarking: A group of individuals, from the current population and/or a hall of fame, are chosen as
;;;                benchmarks. Every individual in the current population plays against every benchmark individual, and the
;;;                fitness function is the amount of money won/lost against each benchmark individual.
;;;                This fitness function is easily parallelizable and highly tunable to different distributions
;;;                of benchmark individuals, and so is used as the main fitness evaluation function.
;;;            K-Random: Each individual plays against K random other individuals in the population, as in http://gpbib.cs.ucl.ac.uk/gecco2002/GA155.pdf
;;;                Matches may contribute to the fitness functions of just one of the players or to both. This is just
;;;                as parallelizable as benchmarking, but makes it difficult to tell which individuals are better due to 
;;;                each individual having a unique set of opponents. This is mostly used in the round-robin extreme during
;;;                hyperparameter search to compare the best individuals evolved by multiple hyperparameter settings
;;; 3. Parent selection
;;;        Generic epsilon lexicase selection, treating the benchmark individuals as test cases, is used here
;;;        When selecting each parent, the benchmarking individuals are randomly shuffled. Then proceeding down
;;;        the sequence of benchmarking individuals, the average amount won against that benchmark is calculated 
;;;        over the whole population. Then the median absolute deviation from the mean is calculated. The cutoff 
;;;        winning amount against this individual is computed by the highest amount won less the median absolute deviation.
;;;        Any individual winning more than this cutoff survives to the next test case. This process is then repeated
;;;        until only one individual is left, or until the test cases are exhausted, in which case a random individual is
;;;        selected from those remaining
;;; 4. Mutation
;;;        As in the two papers cited above, https://arxiv.org/abs/1712.06567 and https://arxiv.org/abs/1703.03864, a mutation is 
;;;        represented as the concatenation of another int to the individual's growing collection of random seeds. This sequence
;;;        of seeds, along with a preset global mutation standard deviation, is a compact representation of the uniform gaussian
;;;        mutations taken to reach this point. This representation lends itself much more easily to distributed computing, as 
;;;        only a couple hundred integers need to be communicated per individual, instead of millions of parameter weights.
;;;        It is also possible to implement an annealing schedule for the mutation strength, as in https://dl.acm.org/doi/abs/10.1145/3205455.3205589
;;; 5. Repeat until the end of the cycle
;;;        As ERL is very computationally intensive, I will try to train mostly at night when no one new is going to need a HPC node.
;;;        To facilitate this, I'll need to be able to pick up and leave off evolution at any time. That functionality will either be
;;;        in this file or in process-result.clj
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Functions: 
;;; versus plays two individuals against each other and returns the match results
;;; benchmark implements the benchmark fitness evaluation described above
;;; single-elim implements the single-elimination tournament fitness evaluation
;;;     described above
;;; round-robin implements the K-random fitness evaluation described above
;;; lexicase-selection implements the epsilon-lexicase selection described above
;;; cull-hof attempts to remove consistently bad, and therefore unhelpful, individuals
;;;    from the hall of fame
;;; select-from-hof implements random selection, k-best as in https://arxiv.org/pdf/2104.05610.pdf, or 
;;;    exponential selection, in which hof individuals from later generations are exponentially more
;;;    likely to be selected as benchmarks to maintain a balance between skilled individuals providing
;;;    selection pressure and earlier individuals preventing evolutionary forgetting and smoothing out convergence.
;;; ERL implements the evolutionary reinforcement learning cycle, reporting out at each generation the
;;;    generation number, the current population, and the time taken for fitness evaluations. I will have
;;;    to implement truncation of game length as in https://arxiv.org/abs/1703.03864 for better CPU utilization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn versus
  "Returns the winning individual, the match results of the individuals, or the 
   individuals updated to contain the match results\\
   -> {(:ind1 :ind2) (:net-gain) (:winner) (:action-count)}"
  [ind1 ind2 max-seq-length num-games & {:keys [manager net-gain? update-error? as-list? action-count? winning-individual? decks stdev max-actions from-block?]
                                         :or {stdev 0.005
                                              max-actions ##Inf}}]
  (with-open [manager (if manager
                        (.newSubManager manager)
                        (ndarray/new-base-manager))]
    (let [mask (ndarray/ndarray manager (ndarray/causal-mask [1 max-seq-length max-seq-length] -2))
          i1 (transformer/model-from-seeds ind1 max-seq-length manager mask :stdev stdev :from-block? from-block?)
          i2 (transformer/model-from-seeds ind2 max-seq-length manager mask :stdev stdev :from-block? from-block?)]
      (with-open [_i1 (utils/make-closeable i1 transformer/close-individual)
                  _i2 (utils/make-closeable i2 transformer/close-individual)]
        (let [{net-gain :net-gain
               action-count :action-count} (apply
                                            headsup/iterate-games-reset
                                            [(transformer/as-player i1) (transformer/as-player i2)]
                                            manager
                                            num-games
                                            :max-actions max-actions
                                            :as-list? as-list?
                                            (concat (when decks [:decks decks])))]
          (merge (when update-error? {:ind1 (update ind1
                                                    :error
                                                    #(assoc % (:id ind2) ((:id ind1) net-gain)))
                                      :ind2 (update ind2
                                                    :error
                                                    #(assoc % (:id ind1) ((:id ind2) net-gain)))})
                 (when net-gain? {:net-gain net-gain})
                 (when winning-individual? {:winner (if (> ((:id ind2) net-gain) 0)
                                                      ind2
                                                      ind1)})
                 (when action-count? {:action-count action-count})))))))



#_(time (versus {:seeds [2074038742],
               :id :p1}
              {:seeds [-888633566],
               :id :p2}
              20
              10
              :net-gain? true
              :decks 1
              :from-block? true))

(defn versus-other
  "Plays a transformer seed individual against another individual.\\
   Returns the winning individual, the match results of the individuals, or the 
   individuals updated to contain the match results"
  [individual opponent max-seq-length num-games & {:keys [manager reverse? decks stdev from-block?]
                                                   :or {stdev 0.005}}]
  (with-open [manager (if manager
                        (.newSubManager manager)
                        (ndarray/new-base-manager))]
    (let [mask (ndarray/ndarray manager (ndarray/causal-mask [1 max-seq-length max-seq-length] -2))
          i1 (transformer/model-from-seeds individual max-seq-length manager mask :stdev stdev :from-block? from-block?)]
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
   Optionally specify a function to use to merge the results of multiple matches
   against a single opponent\\
   -> individual"
  [individual net-gain & {:keys [merge-fn]
                          :or {merge-fn +}}]
  (let [[p1 p2] (keys net-gain)
        [id gain] (condp = (:id individual)
                    p1 [p2 (net-gain p1)]
                    p2 [p1 (net-gain p2)]
                    [nil nil])]
    (if id
      (update individual
              :error
              #(merge-with merge-fn % {id gain}))
      individual)))

(defn process-results
  "Processes the results of benchmarking matches to update the population and 
   benchmark population, as well as return the action counts\\
   -> {:pop :benchmark :results}"
  [pop benchmark results]
  {:pop (reduce (fn [p res]
                  (map #(update-individual % (:net-gain res)) p))
                pop
                results)
   :benchmark (reduce (fn [b res]
                        (map #(update-individual % (:net-gain res)) b))
                      benchmark
                      results)
   :action-counts (mapv :action-count results)})

(defn benchmark
  "Given a population and a set of benchmark individuals possibly drawn
   from the population, plays each individual in the population against each
   of the benchmark individuals, and updates them with the match results\\
   If symmetrical? is true, then initializes a set of shared decks to be played
   for each matchup. Then each matchup is played twice, once \"normal\" and once
   with players in reversed positions, to reduce variance as much as possible\\
   -> {:pop :benchmark :action-counts}"
  [pop bench max-seq-length num-games & {:keys [decks symmetrical? as-list? stdev max-actions from-block?]
                                             :or {stdev 0.005
                                                  symmetrical? true
                                                  max-actions ##Inf}}]
  (let [decks (utils/process-decks decks num-games)
        vs #(concurrent/msubmit
             (versus %1 %2 max-seq-length num-games
                     :net-gain? true
                     :decks decks
                     :as-list? as-list?
                     :stdev stdev
                     :max-actions max-actions
                     :action-count? true
                     :from-block? from-block?))
        res1 (doall
              (for [ind1 pop
                    ind2 bench :when (not (= ind1 ind2))]
                (vs ind1 ind2)))
        res2 (when symmetrical?
               (doall
                (for [ind1 pop
                      ind2 bench :when (not (= ind1 ind2))]
                  (vs ind2 ind1))))
        results (map deref (concat res1 res2))]
    (process-results pop bench results)))


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
                  {:seeds [1761283695], :id :p2}]
                 (list {:seeds [431529176], :id :p1}
                  {:seeds [1749940626], :id :p3}
                  {:seeds [1761283695], :id :p2})
                 20
                 1
                 :symmetrical? true))

(defn single-elim
  "Single elimination tournament of the population. If update-error? is true, then 
   sets the fitness of each individual to the max tournament round reached. Otherwise
   returns all individuals remaining after max-rounds or the only individual left.\\
   Decks are automatically standardized between all games. Provide a list of decks to 
   standardize with other functions\\
   :update-error? - whether to return the last individuals left or the population updated with
   the last round reached as their fitness function\\
   :symmetrical? - whether to standardize games by playing each set of decks from both sides\\
   :stdev - the standard devation used to construct the models"
  [pop max-rounds max-seq-length num-games & {:keys [update-error? symmetrical? stdev decks from-block?]
                                              :or {symmetrical? true
                                                   stdev 0.005}}]
  (loop [pop pop
         cur-pop pop
         i 0]
    (if (or (>= i max-rounds) (= 1 (count cur-pop)))
      (if update-error?
        pop
        cur-pop)
      (let [vs #(versus %1 %2 max-seq-length num-games
                        :net-gain? true
                        :stdev stdev
                        :decks (utils/process-decks decks num-games)
                        :from-block? from-block?)
            matches (partition-all 2 (shuffle cur-pop))
            pass (filter #(= 1 (count %)) matches)
            matches (filter #(= 2 (count %)) matches)
            res-1 (mapv #(concurrent/msubmit (vs (first %) (second %))) matches)
            res-2 (if symmetrical?
                    (mapv #(concurrent/msubmit (vs (second %) (first %))) matches)
                    (repeat (count matches) {}))
            results (apply merge
                           (concat
                            (map #(merge-with +
                                              (:net-gain (deref %1))
                                              (:net-gain ((if symmetrical?
                                                            deref
                                                            identity) %2)))
                                 res-1
                                 res-2)
                            (map #(assoc {} (:id (first %)) ##Inf) pass)))
            next-round (filter #(pos? (get results (:id %) ##-Inf)) cur-pop)]
        (if update-error?
          (recur (mapv (fn [ind]
                         (condp #(%1 %2) (results (:id ind))
                           nil? ind
                           pos? (assoc ind :error (inc i))
                           neg? (assoc ind :error i)
                           ind))
                       pop)
                 next-round
                 (inc i))
          (recur pop next-round (inc i)))))))

#_(single-elim [{:seeds [1] :id :p0}
             {:seeds [2] :id :p2}
             {:seeds [3] :id :p3}
             {:seeds [4] :id :p4}
              {:seeds [5] :id :p5}
              {:seeds [6] :id :p6}
              {:seeds [7] :id :p7}]
             3
             10
             10
             :update-error? true)


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
  [pop num-games round-robin-count max-seq-length & {:keys [decks symmetrical? as-list? stdev from-block?]
                                                     :or {stdev 0.005
                                                          symmetrical? true}}]
  (assert (not (and as-list? symmetrical?)) "Does not currently support reporting standard deviations for symmetrized competition")
  (let [challengers (get-challengers (count pop) round-robin-count)
        matches (mapcat #(map (partial vector (first %)) (rest %)) challengers)
        decks (utils/process-decks decks num-games)
        vs #(versus %1 %2 max-seq-length num-games
                    :net-gain? true
                    :decks decks
                    :stdev stdev
                    :from-block? from-block?
                    :as-list? as-list?)
        res1 (mapv #(concurrent/msubmit (vs (pop (first %)) (pop (second %)))) matches)
        res2 (if symmetrical? 
               (mapv #(concurrent/msubmit (vs (pop (second %)) (pop (first %)))) matches)
                 [])]
    (reduce (fn [p res]
              (map #(update-individual % res) p))
            pop
            (map deref (concat res1 res2)))))

#_(round-robin-parallel [{:seeds [1] :id :p0}
                       {:seeds [2] :id :p1}
                       {:seeds [3] :id :p2}
                       {:seeds [4] :id :p3}
                       {:seeds [5] :id :p4}
                       {:seeds [6] :id :p5}]
                      10
                      5
                      10
                        :as-list? true
                      :symmetrical? false)

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
   :random = randomly selects n individuals from all individuals so far
   :k-best selects the best individual by total chips won from each of the k previous generations"
  [hof n & {:keys [method]
            :or {method :exp}}]
  (condp = method
    :exp (loop [i 0
                selected #{}]
           (if (>= i n) 
             selected
               (recur (inc i)
                      (set/union
                       selected
                       (loop [p (shuffle (mapcat identity
                                                 (take-last (int (Math/exp i))
                                                            hof)))]
                         (cond (empty? p) #{}
                               (selected (first p)) (recur (rest p))
                               :else #{(first p)}))))))
    :random (take n (shuffle (mapcat identity hof)))
    :k-best (mapv (partial apply 
                           max-key 
                           #(transduce (map second) + (:error %))) 
                  (take-last n hof))))

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


(defn merge-errors 
  "Merge the errors of the individuals, adding together results from 
   the same opponents"
  [& individuals]
  (assert (seq individuals) "Cannot merge zero individuals")
  (assoc (first individuals) 
         :error
         (apply merge-with 
                +
                (map :error individuals))))


(defn update-hof
  "Update the hall of fame with the results of individuals serving as benchmarks
   for the population to play against"
  [hof individuals]
  (reduce (fn [h ind]
            (let [find-id (fn [coll]
                            (first (filter #(= (:id ind) (:id %)) coll)))
                  [idx i] (first (keep-indexed
                                  #(if-let [i (find-id %2)]
                                     [%1 i]
                                     nil)
                                  h))]
              (if (and idx i)
                (assoc h
                       idx
                       (-> (h idx)
                           (disj i)
                           (conj (merge-errors i ind))))
                h)))
          hof
          individuals))

#_(update-hof [] [{:seeds [-1666042088],
                 :id :p4,
                 :error
                 {:p2 -60.95,
                  :p3 -61.98817841970013,
                  :p6 18.95}}])

#_(clojure.pprint/pprint 
(update-hof 
 [#{{:seeds [-1666042088],
     :id :p4,
     :error
     {:p2 -60.95,
      :p3 -61.98817841970013,
      :p6 18.95}}
    {:seeds [-155068449],
     :id :p5,
     :error
     {:p2 20.0,
      :p4 42.966722041741015,
      :p6 -59.45748548507691}}}
  #{{:seeds [-155068449 1711817472],
     :id :p5-2,
     :error
     {:p5-7 47.93042122905221,
      :p5-8 -33.193248565138134,}}}]
      [{:seeds [-1666042088],
        :id :p4,
        :error
        {:p2 -100000
         :p1023 0}}
       {:seeds [-155068449 1711817472],
        :id :p5-2,
        :error
        {:p5-7 -100000
         :p1024 0}}]))

(defn winrate 
  "Computes the proportion of matches won by this individual\\
   -> float"
  [individual]
  (/ (count (filter #(> (second %) 0) (:error individual))) 
     (count (:error individual))))

#_(winrate {:seeds [-155068449 1711817472],
          :id :p5-2,
          :error
          {:p5-7 47.93042122905221,
           :p6-0 19.649609374999997,
           :p5-8 -33.193248565138134,
           :p5 -4.095479935856879,
           :p5-9 -0.7113689019172647,
           :p4-1 21.4453125,
           :p8-5 -98.5,
           :p8-6 -19.55,
           :p6-4 -0.05000000000000071,
           :p4-3 -41.98699745027725,
           :p6 -41.626356839400245}})

(defn cull-hof
  "Culls individuals with results that are very consistently poor.
   Probability of getting culled is p(x)=alpha^(x-1) where alpha is a 
   parameterizable value and x is (matches lost)/(total matches) for a specific individual  
   This distribution ensures that individuals that win most of their matches 
   are very unlikely to be culled (lim x→0 p(x) = 1/alpha) and individuals that 
   lose most of their matches are very likely to be culled (lim x→1 p(x) = 1)\\
   -> updated hof"
  [hof & {:keys [alpha cutoff]
          :or {cutoff 0.1}}]
  (assert (or alpha cutoff) "Must choose at least one method")
  (if cutoff
    (let [c (mapv (fn [coll]
                    (into #{}
                          (filter #(> (winrate %) cutoff)
                                  coll)))
                  hof)]
      c)
    (let [prob #(Math/pow alpha (dec (winrate %)))]
      (mapv (fn [coll]
              (into #{} (filter #(> (prob %) (rand))) coll))
            hof))))

#_(clojure.pprint/pprint
 (cull-hof 
 [#{{:seeds [-1666042088],
     :id :p4,
     :error
     {:p2 -60.95,
      :p3 -61.98817841970013,
      :p5 -42.966722041741015,
      :p8 -20.75,
      :p9 -0.8999999999999999,
      :p1 -0.44999999999999996,
      :p0 0.44999999999999996,
      :p7 40.1,
      :p6 18.95}}
    {:seeds [-155068449],
     :id :p5,
     :error
     {:p2 20.0,
      :p4 42.966722041741015,
      :p3 17.46392114125775,
      :p8 20.35,
      :p9 19.95,
      :p1 40.849999999999994,
      :p0 -18.7,
      :p7 2.9862645149230964,
      :p6 -59.45748548507691}}
    {:seeds [1685304270], :id :p8, :error {:p7 40.15, :p5 -20.35, :p4 20.75}}
    {:seeds [-75857613], :id :p6, :error {:p7 1.1, :p5 59.45748548507691, :p4 -18.95}}}
  
  #{{:seeds [-155068449 1711817472],
     :id :p5-2,
     :error
     {:p5-7 47.93042122905221,
      :p6-0 19.649609374999997,
      :p5-8 -33.193248565138134,
      :p5 -4.095479935856879,
      :p5-9 -0.7113689019172647,
      :p4-1 21.4453125,
      :p8-5 -98.5,
      :p8-6 -19.55,
      :p6-4 -0.05000000000000071,
      :p4-3 -41.98699745027725,
      :p6 -41.626356839400245}}
    {:seeds [-1666042088 -1024802974],
     :id :p4-3,
     :error
     {:p8-5 -0.3000000000000007, :p5-7 7.038638102963528, :p5-2 41.98699745027725, :p5 0.04505805969238352, :p6 58.75}}
    {:seeds [1685304270 -1066744347],
     :id :p8-5,
     :error
     {:p5-7 -21.150000000000006,
      :p6-0 -0.44999999999999996,
      :p5-8 39.2,
      :p5 -61.788178419700124,
      :p5-2 98.5,
      :p5-9 -20.6,
      :p4-1 0.5,
      :p8-6 -19.95,
      :p6-4 -20.200000000000003,
      :p4-3 0.3000000000000007,
      :p6 -40.85}}
    {:seeds [-155068449 -1888826919],
     :id :p5-7,
     :error
     {:p6-0 -40.87395850970206,
      :p5-8 9.493824359543936,
      :p5 -30.34968959034503,
      :p5-2 -47.9304212290522,
      :p5-9 35.548996670613114,
      :p4-1 32.092021724855044,
      :p8-5 21.150000000000006,
      :p8-6 20.299999999999997,
      :p6-4 -20.518434188608083,
      :p4-3 -7.0386381029635245,
      :p6 20.400000000000002}}}]
      :cutoff 0.3))


(def error-functions
  "The different evaluation methods for assigning fitnesses to individuals\\
   round robin plays each player against k other players\\
   benchmark plays all players against a set of benchmark players"
  {:round-robin round-robin
   :benchmark benchmark})

(defn report-generation
  "Prints out the generation and the population at that generation"
  [pop generation & {:keys [max-actions time-ms]}]
  (pprint/pprint (merge {:generation generation
                         :pop pop}
                        (when max-actions {:max-actions max-actions})
                        (when time-ms {:time-ms time-ms}))))

(defn round-errors 
  [hof decimal-points]
  (let [map-hof (fn [f hof] (mapv f hof))
        map-coll (fn [f coll] (into #{} (map f coll)))
        update-ind (fn [f ind] (assoc ind :error (f (:error ind))))
        round-errors (fn [f error] (into {} (map #(vector (first %) (f (second %))) error)))
        round (fn [num] (-> num
                            (* (Math/pow 10 decimal-points))
                            (Math/round)
                            (/ (Math/pow 10 decimal-points))))]
    (map-hof (partial map-coll 
                      (partial update-ind 
                               (partial round-errors 
                                        round))) 
             hof)))

#_(round-errors [#{{:seeds [-807793372], 
                    :id :p2, 
                    :error {:p1-6 -0.09999999999999998, 
                            :p6-0 -0.6, 
                            :p4 235.1, }} 
                   {:seeds [807640705], 
                    :id :p1, 
                    :error {:p1-6 0.15, 
                            :p6-0 39.45, 
                            :p2 119.4}} 
                   {:seeds [-75857613], 
                    :id :p6, 
                    :error {:p1-6 0.35, 
                            :p6-0 -39.75, 
                            :p2 2.5000000000000027, 
                            :p4 -3.299999999999997,}}}
                 #{{:seeds [-75857613 -1066744347], 
                    :id :p6-5, 
                    :error {:p1-6 20.2, 
                            :p6-0 -0.3, 
                            :p2 40.25}} 
                   {:seeds [-807793372 -1024802974], 
                    :id :p2-3, 
                    :error {:p1-1 0.04999999999999999, 
                            :p6-9 19.7, 
                            :p6-4 39.8}}}]
              2)

(defn get-benchmark
  "Gets the individuals used for benchmarking the population."
  [n pop hof prop-hof & {:keys [method]
                         :or {method :exp}}]
  (let [n-hof (Math/floor (* prop-hof n))]
    (concat (take (Math/ceil (- n n-hof))
                  (shuffle pop))
            (select-from-hof hof n-hof :method method))))

(defn initialize-pop
  "Given the population size n, a random number generator r, and a standard deviation,
   initializes the population of n individuals"
  [n & {:keys [stdev r]
        :or {stdev 0.005
             r (utils/random 1)}}]
  (mapv #(assoc {}
                :seeds [(.nextInt r)]
                :id (keyword (str "p" %))
                :stdev stdev)
        (range n)))

(defn ERL
  "Main evolutionary reinforcement loop. \\
   First prints the arguments passed to the function. \\
   Then initializes pop-size individuals, and iterates through the generations\\
   At each generation, it selects benchmark individuals from the population and hall of fame,
   and benchmarks the population against the selected individuals to compute their fitness.\\
   Then, parents are selected via lexicase selection and children are generated from parents.\\
   Finally the population, the generation number, and the time taken to benchmark individuals in
   ms is reported out.\\
   Before starting the next cycle, the generation number is incremented, the hall of fame is updated 
   with new parents and culled to remove consistently poor individuals, and the maximum number
   of actions available to evaluate individuals is set to 2x the average number of actions taken to evaluate
   the previous population. This allows for at least 50% CPU utilization.\\
   After the last generation, the final population and the hall of fame are returned\\
   -> {:last-pop :hof}"
  [& {:keys [pop-size num-generations num-games benchmark-count random-seed max-seq-length stdev from-block? block-size]
      :or {pop-size 3
           num-generations 1
           num-games 10
           benchmark-count 5
           random-seed 1
           max-seq-length 20
           stdev 0.005
           block-size 1e8}
      :as argmap}]
  (println argmap)
  (let [r (utils/random random-seed)]
    (when from-block? (utils/initialize-random-block (int block-size) r))
    (loop [generation 0
           pop (initialize-pop pop-size :r r :stdev stdev)
           hof []
           max-actions ##Inf]
      (if (= generation num-generations)
        {:last-pop pop
         :hall-of-fame (round-errors hof 3)}
        (let [{{p :pop
                b :benchmark
                a :action-counts} :result
               t :time} (utils/get-time (benchmark pop
                                                   (get-benchmark benchmark-count pop hof 0.5)
                                                   max-seq-length
                                                   num-games
                                                   :symmetrical? true
                                                   :stdev stdev
                                                   :max-actions max-actions
                                                   :from-block? from-block?))
              [p h] (next-generation p r)]
          (report-generation pop generation
                             :max-actions max-actions
                             :time-ms t)
          (recur (inc generation)
                 p
                 (-> hof
                     (update-hof b)
                     (cull-hof)
                     (conj h))
                 (* 2 (utils/mean a))))))))


#_(ERL :from-block? true)