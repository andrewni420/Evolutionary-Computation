;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Meta-evolutionary algorithm to solve traveling salesman problem ;;;;;
;;;;           Code adapted from evolvesum.clj from moodle           ;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(ns Evo1)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Math and auxiliary functions ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Math
(defn square [n] (* n n))

(defn distance [n1 n2]
  (Math/sqrt
   (+
    (square (- (first n1) (first n2)))
    (square (- (second n1) (second n2))))))

(defn mean [data]
  (/
   (reduce + data)
   (float (count data))))

(defn geomMean [data]
  (Math/pow
   (reduce * data)
   (/ 1 (count data))))

(defn stdev [data]
  (let [x (mean data)]
    (Math/sqrt (/
                (reduce + (map #(square (- % x)) data))
                (count data)))))

;;; Auxiliary
(defn selectN [population n] (repeatedly n #(rand-nth population)))

(defn switch
  "Switches values at positions i and j in vector"
  [vector i j]
  (-> vector (assoc i (vector j)) (assoc j (vector i))))

(defn nodesToDistances
  "Converts list of node locations to array of distances between nodes"
  [nodes]
  (for [n1 nodes]
    (for [n2 nodes]
      (distance n1 n2))))

;;; Random variables
(defn randNormal
  "Marsaglia polar method for normal random variable"
  [m std]
  (loop [u (-> (rand) (* 2) (- 1))
         v (-> (rand) (* 2) (- 1))]
    (let [s (+ (square u) (square v))]
      (if (and (< s 1) (> s 0))
        (+ m
           (* std u
              (Math/sqrt
               (/
                (* (- 2) (Math/log s))
                s))))
        (recur (rand) (rand))))))

(defn logNormal
  "lognormal random variable"
  [m s t]
  (->>
   (randNormal m s)
   (* t)
   (Math/exp)))

(defn randCauchy
  "Cauchy-distributed random variable. Rejects denominators smaller than 10^-10"
  [a b]
  (loop [u (randNormal a b) v (randNormal a b)]
    (if (< (abs v) (Math/pow 10 (- 10)))
      (recur (randNormal a b) (randNormal a b))
      (/ u v))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;Individual Creation and Evaluation;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn addError
  "Adds errors of the form 'a * omega + b' together, where omega is the lowest transfinite ordinal number
   Supports comparison between infinite errors"
  [e i]
  (if
   (= ##Inf i)
    (assoc e :inf (inc (:inf e)))
    (assoc e :val (+ i (:val e)))))

(defn tspError
  "Sum of weights from traversing nodes in order of objective params"
  [objective distances]
  (reduce
   addError
   {:inf 0 :val 0}
   (map
    #(-> distances (nth  (first %)) (nth (second %)))
    (partition 2 1 (concat objective [(first objective)])))))

(defn better [i1 i2]
  (let [{e1 :error} i1
        {e2 :error} i2]
    (if
     (= (:inf e1) (:inf e2))
      (<= (:val e1) (:val e2))
      (< (:inf e1) (:inf e2)))))

(defn  minErrorIndividual [population]
  (reduce
   #(if (better %1 %2) %1 %2)
   {:error {:inf ##Inf :val ##Inf}}
   population))

(defn tspIndividual
  "Creates a new individual for the TSP problem
   Genome initialized to a random permutation
   :maxParents initialized with a normal distribution with mean 1 and stdev 0.3
   :invProb initialized with a normal distribution with mean 0.1 and stdev 0.03"
  [n distances]
  (let [g (shuffle (range n))]
    (assoc {} 
           :genome {:objective g
                    :maxParents (randNormal 1 0.3)
                    :invProb (randNormal 0.1 0.03)}
           :error (tspError g distances))))

(defn genomeToIndividual
  "Converts a genome to a [genome error] pair"
  [genome distances]
  (assoc {} 
         :genome genome 
         :error (tspError (:objective genome) distances)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;       Genome Mutation        ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn invertObjective 
  "Randomly inverts adjacent values in objective parameters
   invProb: probability of switching any given adjacent pair"
  [objective invProb]
  (loop [p objective 
         n 0 
         c (count objective)]
    (if (= n c) 
      p 
      (recur 
       (if (< (rand) invProb) 
         (switch objective n (mod (inc n) c)) 
         p) 
       (inc n) 
       c))))

(defn mutateObjective 
  "Mutates objective parameters using strategy parameters
   :invProb - probability of any given inversion"
  [genome]
  (assoc genome 
         :objective 
         (invertObjective (:objective genome) (:invProb genome))))


(defn mutateStrategy
  "Perturbs strategy parameters using random normal variable
   Stdev of perturbation proportional to 1 / number of objective parameters"
  [genome tau]
  (let [n (count (:objective genome))]
    (-> genome
        (assoc :invProb
               (-> genome (:invProb) (+ (randNormal 0 (/ tau n))) (max 0.001) (min 1)))
        (assoc :maxParents
               (-> genome (:maxParents) (+ (randNormal 0 (/ tau n))) (max 0) (min 10))))))


(defn mutateGenome 
  "Mutate genome by mutating objective and strategy parameters"
  [genome tau] 
  (-> 
   genome
   (mutateObjective)
   (mutateStrategy tau)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;          Cross Over          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn crossSlices
  "Generates k slices to use for crossing over
   Slices take the form [0 a] [a b] ... [b c] [0 n] where n is the length of the genome"
  [k n]
  (concat
   (->>
    #(rand-int n)
    (repeatedly (- k 1))
    (concat [0])
    (sort <)
    (partition 2 1))
   [(list 0 n)]))

(defn crossObjective
  "Crossing over for TSP. 
   Concatenates slices from each parent for objective parameters and takes geometric mean of strategy parameters
   By concatenating all objective parameters of the last parent to the end and 
   taking only distinct values, it ensures that output objective parameters are valid"
  [parents]
  (->>
   (mapv
    #(drop (first %2) (take (second %2) (:objective (:genome %1))))
    parents
    (crossSlices
     (count parents)
     (->> parents
          (first)
          (:genome)
          (:objective)
          (count))))
   (apply concat)
   (distinct)
   (vec)))

(defn crossStrategy [parents]
  (assoc {}
         :invProb
         (mean (mapv #(:invProb (:genome %)) parents))
         :maxParents
         (mean (mapv #(:maxParents (:genome %)) parents))
   ))
  
(defn selectParents 
  "Selects random parent
   Then randomizes number of parents from 1 to :maxParents of that parent
   Then selects maxParents-1 additional parents from population
   Floating points treated as probability of having ceil(maxParents) parents"
  [parents]
  (let [first (rand-nth parents)
        maxParents (- (:maxParents (:genome first)) 1)]
    (conj
     (selectN parents (-> maxParents
                          (max 0)
                          (min 10)
                          (rand)
                          (int)
                          (+ 1)))
     first)))

(defn crossOver
  "Crossing over"
  [parents distances]
  (repeatedly
   (count parents)
   #(genomeToIndividual
     (let [parents (selectParents parents)]
       (merge 
        {:objective (crossObjective parents)} 
        (crossStrategy parents)))
     distances)))

(defn mutatePopulation
  "Mutates each member of the population"
  [parents distances tau]
  (map
   #(genomeToIndividual
     (mutateGenome (:genome %) tau)
     distances)
   parents))

(defn mutate
  "Mutates then crosses over population"
  [parents distances tau]
  (->
   parents
   (mutatePopulation distances tau)
   (crossOver distances)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;          Survival            ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn sortedTournament 
  "Faster tournament for a sorted population"
  [sortedPop size]
  (nth sortedPop 
       (apply min 
              (repeatedly 
               size 
               #(rand-int (count sortedPop))))))

(defn tournament
  "Selects a member of the population via a tournament
   size - the size of the tournament"
  [population size]
  (minErrorIndividual
   (repeatedly
    size
    #(rand-nth population))))


(mutate (repeatedly 3 #(tspIndividual 1 [[0]])) [[0]] 0.1)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;           Evolution          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn evolve
  "Evolves a population of individuals until one sums to the target.
   popsize - size of population
   :n - number of nodes
   :distances - edge weights between nodes
   :mutationProb - probability of flipping any individual bit in a genome
   :propSelected - proportion of population selected to be parents
   :tournamentSize - size of tournament to determine parenthood
   :maxParents - maximum number of parents for one offspring. Actual number 
                 of parents chosen randomly between 0 and maxParents
   :maxGenerations - terminate after this many generations
   :minError - terminate after achieving this error or less. Is of the form {:inf a :val b}
               for a path with 'a untravelable paths and a sum of weights of 'b for the remaining paths
   :propNew - proportion of population created anew at each generation. 
              :propNew=1 is equivalent to random search"
  [popsize & {:keys [n distances tau propSelected tournamentSize maxGenerations minError propNew]
              :or {n 3
                   distances (for [i (range n)] (concat (repeat i 0) [1] (repeat (- n i 1) 0)))
                   tau 0.1
                   propSelected 0.5
                   tournamentSize 10
                   maxGenerations 100
                   minError {:error {:inf 0 :val 0}}
                   propNew 0.1}}]
  (loop [generation 0
         population (repeatedly popsize #(tspIndividual n distances))]
    (let [best (minErrorIndividual population)
          numNew (int (* propNew (count population)))]
    ;;   (println "Generation:" generation
    ;;            ", Best error:" (:error best)
    ;;            ", mean:" (format "%.2f" (mean (map #(:val (:error %)) population)))
    ;;            ", stdev:" (format "%.2f" (stdev (map #(:val (:error %)) population))))
      (if (or (better best minError) (> generation maxGenerations))
        (assoc {} :best best :generation generation)
        (let [parents (repeatedly
                       (int (* popsize propSelected))
                       #(tournament population tournamentSize))]
          (recur
           (inc generation)
           (concat
            (drop (+ numNew 1) (mutate
                   (repeatedly popsize #(rand-nth parents))
                   distances
                   tau))
            [best]
            (repeatedly numNew #(tspIndividual n distances)))))))))


;; Functionality for testing rate of evolution
(defn run-generations 
  "Runs evolution numTimes amount of times and returns a list of the number 
   of generations before termination"
  [numTimes]
  (->>
   #(evolve
     50
     :n 10
     :distances [[0     1     1   1.5 ##Inf ##Inf ##Inf ##Inf ##Inf ##Inf]
                 [1     0     2     4     3 ##Inf ##Inf ##Inf ##Inf ##Inf]
                 [1     2     0     3 ##Inf     3 ##Inf     4 ##Inf     5]
                 [1.5     4     3     0   1.5     3     4 ##Inf ##Inf ##Inf]
                 [##Inf     3 ##Inf   1.5     0     1     1 ##Inf ##Inf ##Inf]
                 [##Inf ##Inf     3     3     1     0     2     4     1     3]
                 [##Inf ##Inf ##Inf     4     1     2     0 ##Inf ##Inf ##Inf]
                 [##Inf ##Inf     4 ##Inf ##Inf     4 ##Inf     0     3     1]
                 [##Inf ##Inf ##Inf ##Inf ##Inf     1 ##Inf     3     0     1]
                 [##Inf ##Inf     5 ##Inf ##Inf     3 ##Inf     1     1     0]]
     :minError {:error {:inf 0 :val 16}}
     :maxGenerations 100
     :tau 0.1
     :propNew 0)
   (repeatedly numTimes)))



(defn test-evolve 
  "Compiles statistics for the number of generations before terminating.
   Runs evolve numTimes number of times to generate data"
  [numTimes]
  (let [evolution (run-generations numTimes)
        generations (map :generation evolution)
        invProb (map #(:invProb (:genome (:best %))) evolution)
        maxParents (map #(:maxParents (:genome (:best %))) evolution)
        bestErrorInf (map #(:inf (:error (:best %))) evolution)
        bestErrorVal (map #(:val (:error (:best %))) evolution)]
    (assoc {} 
           :mean (mean generations) 
           :stdev (stdev generations)
           ;; :unreached (reduce #(if (= 1001 %2) (inc %1) %1) 0 generations)
           ;; :generations generations
           :meanInvProb (mean invProb)
           :stdInvProb (stdev invProb)
           :meanMaxParents (mean maxParents)
           :stdMaxParents (stdev maxParents)
           :meanBestError (assoc {} :inf (mean bestErrorInf) :val (mean bestErrorVal)))))

;; Use 1000 data points
(test-evolve 25)


(let [r (evolve
         50
         :n 10
         :distances [[    0     1     1   1.5 ##Inf ##Inf ##Inf ##Inf ##Inf ##Inf]
                     [    1     0     2     4     3 ##Inf ##Inf ##Inf ##Inf ##Inf]
                     [    1     2     0     3 ##Inf     3 ##Inf     4 ##Inf     5]
                     [  1.5     4     3     0   1.5     3     4 ##Inf ##Inf ##Inf]
                     [##Inf     3 ##Inf   1.5     0     1     1 ##Inf ##Inf ##Inf]
                     [##Inf ##Inf     3     3     1     0     2     4     1     3]
                     [##Inf ##Inf ##Inf     4     1     2     0 ##Inf ##Inf ##Inf]
                     [##Inf ##Inf     4 ##Inf ##Inf     4 ##Inf     0     3     1]
                     [##Inf ##Inf ##Inf ##Inf ##Inf     1 ##Inf     3     0     1]
                     [##Inf ##Inf     5 ##Inf ##Inf     3 ##Inf     1     1     0]]
         :minError {:error {:inf 0 :val 16}}
         :maxGenerations 1000
         :tau 0.1
         :propNew 0.3)]
  (
    clojure.pprint/pprint r
   ))






"City locations for the DJ38 TSP instance containing 38 cities in Djibouti"
(def djNodes
  [[11003.611100 42102.500000]
   [11108.611100 42373.888900]
   [11133.333300 42885.833300]
   [11155.833300 42712.500000]
   [11183.333300 42933.333300]
   [11297.500000 42853.333300]
   [11310.277800 42929.444400]
   [11416.666700 42983.333300]
   [11423.888900 43000.277800]
   [11438.333300 42057.222200]
   [11461.111100 43252.777800]
   [11485.555600 43187.222200]
   [11503.055600 42855.277800]
   [11511.388900 42106.388900]
   [11522.222200 42841.944400]
   [11569.444400 43136.666700]
   [11583.333300 43150.000000]
   [11595.000000 43148.055600]
   [11600.000000 43150.000000]
   [11690.555600 42686.666700]
   [11715.833300 41836.111100]
   [11751.111100 42814.444400]
   [11770.277800 42651.944400]
   [11785.277800 42884.444400]
   [11822.777800 42673.611100]
   [11846.944400 42660.555600]
   [11963.055600 43290.555600]
   [11973.055600 43026.111100]
   [12058.333300 42195.555600]
   [12149.444400 42477.500000]
   [12286.944400 43355.555600]
   [12300.000000 42433.333300]
   [12355.833300 43156.388900]
   [12363.333300 43189.166700]
   [12372.777800 42711.388900]
   [12386.666700 43334.722200]
   [12421.666700 42895.555600]
   [12645.000000 42973.333300]])

;; Distance array for the DJ38 TSP instance
(def djDistances (nodesToDistances djNodes))

(let [r (evolve
         200
         :n (count djNodes)
         :distances djDistances
         :minError {:error {:inf 0 :val 6656}} ;; optimal solution
         :maxGenerations 100
         :tau 0.1
         :propNew 0;;About 2x better than random selection. Mean doesn't really decrease by a lot between populations
         :propSelected 0.25
         :tournamentSize 5
         )]
  r)

#(evolve
  10
  :n 8
  :distances [[0 5 ##Inf 6 ##Inf 4 ##Inf 7]
              [5 0 2 4 3 ##Inf ##Inf ##Inf]
              [##Inf 2 0 1 ##Inf ##Inf ##Inf ##Inf]
              [6 4 1 0 7 ##Inf ##Inf ##Inf]
              [##Inf 3 ##Inf 7 0 ##Inf 6 4]
              [4 ##Inf ##Inf ##Inf ##Inf 0 3 ##Inf]
              [##Inf ##Inf ##Inf ##Inf 6 3 0 2]
              [7 ##Inf ##Inf ##Inf 4 ##Inf 2 0]]
  :minError {:error {:inf 0 :val 25}}
  :maxGenerations 1000
  :tau 0.01
  :propSelected 0.5
  :tournamentSize 5
  :propNew 0)

;; Exercises:
;; - Add parameters for the various hardcoded values. 
;;   Had to change concat to random selection with replacement
;; - Avoid recomputing errors by making individuals pairs of [error genome].
;; - Print more information about the population each generation.
;;   Mean and stdev. Also returns best individual now
;; - Select parents via tournaments.
;; - Add crossover.
;;   Cross over = multi point recombination between parents randomly sampled with replacement
;;   Number of parents randomly chosen from 1 to maxParents
;; - Use a more interesting genome representation, for a more interesting
;;   problem.
;;    - Now solves traveling salesman problem
;;    - Removed unnecessary sorting and implemented elitism
;;    - genome is now a vector because the switch function needs to set values by index
;;   distances: nxn array of edge weights between nodes where n is the number of nodes
;;   genome: path of nodes in which to travel. Actual path starts and ends at
;;           the first node in the genome. Alphabet is integers 0-n
;;   mutation: randomly switches adjacent nodes in the genome
;;   crossover: For k parents, selects slices from the genomes of k-1 parents
;;              and adds the entire genome of the last parent to the end. 
;;              Then removes repeat nodes. This ensures that crossover always 
;;              results in a valid genome that passes through all nodes once

(defn a [] (print "hi"))
(defn b [] (a))
(b)