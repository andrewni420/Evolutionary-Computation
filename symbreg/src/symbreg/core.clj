(ns symbreg.core
  (:gen-class)
  (:require [incanter.core]
            [incanter.stats]
            [incanter.datasets]
            [incanter.charts]
            [clojure.core.matrix :as matrix]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]))

;; (defn -main
;;   "I don't do a whole lot ... yet."
;;   [& args]
;;   (println "Hello, World!"))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;    Genetic Programming Algorithm for Vectorized      ;;;;;;;;;
;;;;;;                  Symbolic Regression                 ;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;       Math and auxiliary functions        ;;;
;;;       Mostly imported from previous       ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;Copied from clojure.data.csv readme
(defn csv-data->maps
  "Turns csv data into a vector of maps"
  [csv-data]
  (map zipmap
       (->> (first csv-data)
            (map keyword)
            repeat)
       (rest csv-data)))

(defn splitNL [s] (clojure.string/split s #"\n"))

(defn splitsemi [s] (clojure.string/split s #";"))

(defn splitcomma [s] (clojure.string/split s #",")) 

(defn combine [l] (str "[" (clojure.string/join " " l) "]"))

(defn combineData [s] 
  (combine 
   (map (fn [s]
          (combine 
           (map (fn [s]
                  (combine (splitcomma s))) 
                (splitsemi s)))) 
        (splitNL s))))


(def testA
  "Test A for symbolic regression"
  (rest (read-string (combineData (slurp "test_A_01.csv")))))

(def testB
  "Test B for symbolic regression"
  (rest (read-string (combineData (slurp "test_B_01.csv")))))


;;copied from stackexchange
(defmacro make-fn
  "Turns a macro into a function"
  [m]
  `(fn [& args#]
     (eval
      (cons '~m args#))))

(defn multiplot
  "Plots multiple y-series against x-series
   ys: [y-values label]"
  [x & ys]
  (incanter.core/view
   (reduce
    #(incanter.charts/add-lines %1 x (first %2) :series-label (second %2))
    (incanter.charts/xy-plot x (first (first ys)) :series-label (second (first ys)))
    (rest ys))))

(defn compareModels
  "Compares different models for predicting timeseries on the same graph
   models: [fn model label] where (fn model ts) produces a list of the model's predictions"
  [ts & models]
  (apply multiplot
         (range (count ts))
         ts
         (map #(vector ((first %) (second %) ts) (nth % 2))
              models)))



(defn length
  "The number of parameters in an expression"
  [expr] (-> expr (flatten) (count)))

(defn dot
  "Dot product.
   [a b] -> a.b
   [a] -> a.a"
  ([a b] (reduce + (map * a b)))
  ([a] (dot a a)))

(defn sigmoid
  "Sigmoid function for neural networks"
  [x]
  (/ 1 (+ 1 (Math/exp (- x)))))

(defn relu
  "Rectified linear for neural networks"
  [x] (max 0 x))

(defn magnitude
  "Magnitude of a vector"
  [a] (Math/sqrt (dot a)))


(defn square
  "Square input with autopromotion"
  [n] (*' n n))

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


(defn pd
  "Protected division"
  [a b]
  (try (if (zero? b)
         0
         (/ a b))
       (catch Exception _ 0)))

(defn mean
  "Takes the average of dataset"
  [data]
  (pd
   (reduce + 0 data)
   (float (count data))))



(defn geomMean
  "Takes the geometric mean of a dataset"
  [data]
  (Math/pow
   (reduce * data)
   (/ 1 (count data))))

(defn median
  "Takes the median of the arguments"
  [data]
  (let [s (sort data)
        c (count data)]
    (if (= 0 c)
      0
      (/ (+ (nth s (Math/ceil (/ (dec c) 2)))
            (nth s (Math/floor (/ (dec c) 2))))
         2))))

(defn cov [x y]
  (let [xbar (mean x)
        ybar (mean y)]
    (/ (reduce + (map #(* (- %1 xbar) (- %2 ybar))
                      x
                      y))
       (count x))))

(defn distance [a b]
  (magnitude (map - a b)))


(defn stdev
  "Takes the standard deviation of the arguments"
  [data]
  (let [x (mean data)]
    (Math/sqrt (pd
                (reduce + 0 (map #(square (- % x)) data))
                (count data)))))

(defn MSE
  "Mean squared error of a predicted time series 
   and the ground truth time series"
  [prediction ts]
  (/ (reduce + (map #(square (- %1 %2))
                    ts
                    prediction))
     (count ts)))

(defn NMSE
  "MSE normalized by the variance of the time series.
   Returns 1 for a model that constantly guesses the mean of the ts"
  [prediction ts]
  (/ (MSE prediction ts)
     (square (stdev ts))))

(defn preorderTraversal
  "Preorder traversal of a tree-based expression"
  [individual]
  (if (seq? individual)
    (apply concat
           [individual]
           (map preorderTraversal (rest individual)))
    [individual]))


(defn postorderTraversal
  "Postorder traversal of a tree-based expression"
  [individual]
  (if (seq? individual)
    (concat (apply concat
                   (map postorderTraversal
                        (rest individual)))
            [individual])
    [individual]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;                Evolving GP                ;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;               Expressions                 ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn vectorize-ar2
  "Vectorize arity 2 functions like +/-"
  [f]
  (fn [a b]
    (if (sequential? a)
      (if (sequential? b)
        (map f a b)
        (map #(f % b) a))
      (if (sequential? b)
        (map #(f a %) b)
        (f a b)))))

(defn vectorize-ar1
  "Vectorize arity 1 functions like sin/cos"
  [f]
  (fn [a]
    (if (sequential? a)
      (map f a)
      (f a))))

(defn aggregate-ar1
  "Support scalars for aggregate functions like mean/max"
  [f]
  (fn [a]
    (if (sequential? a)
      (f a)
      (f [a]))))

(defn aggregate-ar2
  "Support scalars for aggregate functions like dot/covariance"
  [f]
  (fn [a b]
    (if (sequential? a)
      (if (sequential? b)
        (f a b)
        (f a (repeat (count a) b)))
      (if (sequential? b)
        (f (repeat (count b) a) b)
        (f [a] [b])))))

(defn sin [a] (Math/sin a))
(defn cos [a] (Math/cos a))
(defn log [a] (Math/log (max 0.00000000001 a)))


(defn iflte
  "If a<b then c else d"
  [a b c d]
  (if (<= a b) c d))

(def v_plus (vectorize-ar2 +))
(def v_minus (vectorize-ar2 -))
(def v_mult (vectorize-ar2 *))
(def v_pd (vectorize-ar2 pd))
(def v_sin (vectorize-ar1 sin))
(def v_cos (vectorize-ar1 cos))
(def v_log (vectorize-ar1 log))
(def v_mean (aggregate-ar1 mean))
(def v_stdev (aggregate-ar1 stdev))
(def v_cov (aggregate-ar2 cov))
(def v_distance (aggregate-ar2 distance))
(def v_dot (aggregate-ar2 dot))

(def function-table
  "List of functions that output symbols corresponding
 to allowed functions in MIPs"
  {'v_plus 2
   'v_minus 2
   'v_mult 2
   'v_pd 2
   'v_sin 1
   'v_cos 1
   'v_log 1
   'v_mean 1
   'v_stdev 1
   'v_cov 2
   'v_distance 2
   'v_dot 2})

(defn rand-aggregate [] (rand-nth '(v_mean v_stdev v_cov v_distance v_dot)))
(defn rand-function [] (rand-nth (keys function-table)))
;;logistic, tanh, relu, step, 

(defn terms
  "list of functions that output symbols or real numbers
   [n0 ... np l0 ... lq N(0,1)]
   for p terms and q lags"
  [numTerms]
  (concat (map
           #(fn [] (symbol (str "n" %)))
           (range numTerms))
          [#(randNormal 0 1)]))

(defn rand-terminal [numTerms] ((rand-nth (terms numTerms))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                Mutations                  ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn replaceExpression
  "Replaces expression at index idx in the parent with expr
   indexes smaller than 0 concat to the front
   indexes larger than (count parent) concat to the end"
  [parent idx expr]
  (concat (take idx parent) (list expr) (drop (inc idx) parent)))

(defn depth
  "Calculates the depth of a tree-based expression"
  [expression]
  (if (seq? expression)
    (+ 1 (apply max (map depth expression)))
    0))


(defn evaluate
  "Evaluates an expression given variable bindings.
   Bindings must be of the form '[a 0]"
  [bindings expr]
  (eval 
   (concat
    '(let)
    (list bindings)
    (list expr)))
  )


(defn bindings
  "Creates bindings to put into evaluate
   (terms lags)"
  ([numTerms vals]
  (vec (mapcat #(list %1 %2) 
               (map #(%) (drop-last (terms numTerms)))
               vals)))
  ([vals] (bindings (count vals) vals)))

(defn randExpr
  "Helper method for makeExpression
   vars: calls random method in vars to generate a symbol or real number"
  [probReal numTerms depth]
  (let [f (rand-function)
        n (function-table f)]
    (if (or (< (rand) probReal) (<= depth 0))
      (rand-terminal numTerms)
      (concat
       (list f)
       (repeatedly n #(randExpr
                       probReal
                       numTerms
                       (dec depth)))))))


(defn makeExpression
  "Constructs a random expression that is not a terminal.
   probTerminal: probabiliy an argument is terminal vs an expression
   maxDepth: maximum depth of expressions"
  [probTerminal numTerms maxDepth]
  (let [f (rand-aggregate)
        n (function-table f)]
    (apply list
           (concat
            (list f)
            (repeatedly n #(randExpr
                            probTerminal
                            numTerms
                            (dec maxDepth)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;      Mutation Operators        ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;more specific to more general
(defn perturbReal
  "Perturbs real numbers in an expression with a normal random
   variable N(0,std) with probability PRprob"
  [expression PRprob std]
  (cond
    (number? expression) (if (< (rand) PRprob)
                           (+ expression (randNormal 0 std))
                           expression)
    (seq? expression) (map #(perturbReal % PRprob std) expression)
    :else expression))


(defn perturbTerms
  "Randomly replaces terms in an expression with probability PTprob"
  [expression PTprob numTerms]
  (if (seq? expression)
    (cons
     (first expression)
     (map #(perturbTerms % PTprob numTerms)
          (rest expression)))
    (if  (< (rand) PTprob)
      (rand-terminal numTerms)
      expression)))

(defn perturbExpr
  "Randomly replaces an expression with a random expression
  with probability PEprob.
   Random expr constrained to be at most one level deeper than replaced expr"
  [expression PEprob numTerms probReal]
  (if (seq? expression)
    (if (< (rand) PEprob)
      (randExpr probReal numTerms (inc (depth expression)))
      (map #(perturbExpr % PEprob numTerms probReal) expression))
    expression))

(defn safePerturbExpr
  "Prevents returning a terminal"
  [expression PEprob numTerms probReal]
  (if (seq? expression)
    (map #(perturbExpr % PEprob numTerms probReal) expression)
    expression))

(defn insert
  "Inserts randomly above an expression. Depth is increased by at most 1"
  [expression Iprob numTerms probTerminal]
  (if (< (rand) Iprob)
    (let [e (randExpr 0 numTerms 1)
          i (inc (rand-int (dec (count e))))]
      (replaceExpression e i expression))
    (if (seq? expression)
      (cons (first expression)
            (map #(insert % Iprob numTerms probTerminal)
                 (rest expression)))
      expression)))



;; Needs to be safe-ified
(defn delete
  "Randomly promotes a child. Removes at most one from depth"
  [expression Dprob]
  (if (seq? expression)
    (if (< (rand) Dprob)
      (let [c (count expression)]
        (nth expression (+ 1 (rand-int (dec c)))))
      (map #(delete % Dprob) expression))
    expression))

(defn safedelete
  "Prevents returning a terminal"
  [expression Dprob]
  (if (seq? expression)
    (cons (first expression)
          (map #(delete % Dprob) (rest expression)))
    expression))

(defn enforceDepth
  "Goes down tree until depth of the remaining tree <= maxDepth"
  [expression maxDepth]
  (if (< maxDepth (depth expression))
    (let [c (count expression)
          i (+ 1 (rand-int (dec c)))]
      (recur (nth expression i) maxDepth))
    expression))

(defn suggestDepth
  "Deletes numTries times or until reaching maxDepth or less"
  [expression Dprob maxDepth numTries]
  (if (or (<= (depth expression) maxDepth)
          (<= numTries 0))
    expression
    (recur (safedelete expression Dprob)
           Dprob
           maxDepth
           (dec numTries))))

(defn perturbOrder
  "Shuffles terms around with probability POprob"
  [expression POprob]
  (if (and (seq? expression) (< (rand) POprob))
    (cons (first expression)
          (shuffle (map #(perturbOrder % POprob) (rest expression))))
    expression))

(defn mutateObjective
  "Always performs perturbReal.
   Chooses randomly to perform one of:
   perturbTerms, perturbExpr, insert, suggestDepth, perturbOrder"
  [individual numTerms probTerminal]
  (let [{strategy :strategy
         objective :objective} individual
        objective (perturbReal objective (first strategy) (second strategy))]
    (condp > (rand)
        1/5 (perturbTerms objective
                          (nth strategy 2)
                          numTerms)
        2/5 (safePerturbExpr objective
                             (nth strategy 3)
                             numTerms
                             probTerminal)
        3/5 (insert objective
                    (nth strategy 4)
                    numTerms
                    probTerminal)
        4/5 (suggestDepth objective
                          (nth strategy 5)
                          (max 1 (dec (depth objective)))
                          1)
        (perturbOrder objective
                      (nth strategy 6)))))

(defn mutateStrategy
  "Multiplies the stdev parameter by a lognormal random variable exp(tN(0,1))
   where t is proportional to 1/sqrt(2n) for an individual with n interacting programs
   Perturbs the other parameters using N(0,tau) and clamps them to the interval [0,1]"
  [individual tau]
  (let [{strategy :strategy} individual
        r (->> individual
               (:objective)
               (count)
               (* 2)
               (Math/sqrt)
               (/ tau)
               (logNormal 0 1))]
    (cons (* r (first strategy))
          (map #(max 0 (min 1 (+ % (randNormal 0 tau)))) (rest strategy)))))

(defn mutate
  "Combines mutation of objective, state, and strategy parameters"
  [individual numTerms probReal tau]
   (assoc {}
          :objective (mutateObjective individual numTerms probReal)
          :strategy (mutateStrategy individual tau)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;      Crossover Operators       ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn randomSubtree
  "chooses a random subtree of an individual"
  [individual]
  (rand-nth (preorderTraversal individual)))

(defn crossTrees
  "Returns a binary function of a random subtree taken
   from each parent"
  [t1 t2]
  (let [f (first (rand-nth (filter #(= 2 (second %))
                                function-table)))]
    (list f
          (randomSubtree t1)
          (randomSubtree t2))))

(defn crossover-single 
  "Produce a child from crossing over parents
   Subtree substitution for objective parameters
   Uniform for strategy parameters"
  [p1 p2]
  (let [newObj (crossTrees (:objective p1) (:objective p2))
        newStrat (map #(if (< (rand) 0.5) %1 %2) (:strategy p1) (:strategy p2))]
    (assoc {}
           :objective newObj
           :strategy newStrat)))



(defn crossover 
  "Produce n children via crossover from population"
  [population n]
  (repeatedly n #(apply crossover-single
                         (take 2 (shuffle population)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;        Initialization          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn individual
  "probReal - probability of terminal vs expression
   numTerms - number of independent variables
   maxDepth - maximum depth of a program"
  [probReal numTerms maxDepth]
  (assoc {}
            :objective (makeExpression probReal
                                         numTerms
                                         maxDepth)
            :strategy (repeatedly 7 rand);;std PRprob PTprob PEprob Iprob Dprob POprob
            ))

(defn tournamentSelect
  "Tournament selection"
  [population tournamentSize n]
  (repeatedly
   n
   #(->> population
         (shuffle)
         (take tournamentSize)
         (apply min-key :Error))))


(defn predict-single [individual input]
  ;; (println (:objective individual))
  (let [e (evaluate (bindings input) (:objective individual))]
    (if (seq? e)
      (first e)
      e)))



(defn error [individual data]
  (transduce (comp (map #(vector (predict-single individual
                                                 (drop-last %))
                                 (first (last %))))
                   (map #(square (- (first %) (second %)))))
             +
             0
             data))

(defn compareData [individual data]
  (map #(vector (predict-single individual
                                (drop-last %))
                (first (last %)))
       data))

(defn lexicaseSelect
  "Downsampled lexicase selection
   Individuals with error closer than epsilon to the best error are kept
   Uses a subset of ts of size numTrials"
  [population numTrials epsilon ts]
  (loop [obs (take numTrials (shuffle ts))
         pop population]
    (if (or (<= (count population) 1) (empty? obs))
      (rand-nth population)
      (recur (rest obs)
             (let [errors (map
                           #(vector (error % (take 1 obs)) %)
                           pop)
                   best (apply min-key first errors)]
               (map second (filter
                            #(< (abs (- (first %) (first best))) epsilon)
                            errors)))))))



(defn evolve
  "Evolution generates excess individuals by mutating and/or crossing over
   existing individuals. Selection is then used to cull the population back
   to popsize.
   Meta evolution controls the probability of removing or adding a term,
   the step size for changing existing terms, and the probability of crossing over."
  [popsize data & {:keys [maxGenerations tau numTrials epsilon probReal numTerms maxDepth]
                 :or {maxGenerations 100
                      tau 0.1
                      numTrials 3
                      epsilon 10
                      probReal 0.8
                      numTerms 1
                      maxDepth 3}}]
  (loop [population (repeatedly popsize #(individual probReal numTerms maxDepth))
         generation 0]
    (println (apply min (map #(error % data) population)))
      (if (>= generation maxGenerations)
        (apply min-key #(error % data) population)
        (recur
         (let [parents (repeatedly popsize #(lexicaseSelect population numTrials epsilon data))]
           (concat parents
                   (map #(mutate % numTerms probReal tau) parents)
                   (crossover parents (count parents))))
         (inc generation)))))

(spit "test.txt"(with-out-str (evolve 20 (take 15 testA) :maxGenerations 50)))

(evolve 10 (take 15 testA) :maxGenerations 2)


(error {:objective '(v_cov n0 n0)} (take 15 testA))

(let [popsize 25 numTerms 1 probReal 0.5 tau 0.1 numTrials 5 epsilon 0.1 data (take 15 testA) maxDepth 2]
  (let [population (repeatedly popsize #(individual probReal numTerms maxDepth))
      generation 0]
  (let [parents (repeatedly popsize #(lexicaseSelect population numTrials epsilon data))]
    (concat parents
            (map #(mutate % numTerms probReal tau) parents)
            (crossover parents (count parents))))))

(map #(vector (:objective (mutate % 1 0.5 0.1)) (:objective %))
     (repeatedly 10 #(individual 0.5 1 2)))
