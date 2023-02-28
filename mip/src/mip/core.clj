(ns mip.core
  (:gen-class)
  (:require [incanter.core]
            [incanter.stats]
            [incanter.datasets]
            [incanter.charts]
            [clojure.core.matrix :as matrix]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;; Three algorithms for time series prediction          ;;;;;;;;;
;;;;;; by evolving autoregressive models, recurrent         ;;;;;;;;;
;;;;;; neural networks, and multiple interacting programs,  ;;;;;;;;;
;;;;;; in order from least to most general.                 ;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;       Math and auxiliary functions        ;;;
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

(def temperatures
  "Minimum daily temperatures from 1981-1990 in Melbourne Aus
   Data from the Austrailian Bureau of Meteorology"
  (with-open [reader (io/reader "resources/temperatures.csv")]
    (doall (map
            (fn [row] (update row :Temp #(Double/parseDouble %)))
            (csv-data->maps
             (csv/read-csv reader))))))

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

(defn mean 
  "Takes the average of dataset"
  [data]
  (/
   (reduce + data)
   (float (count data))))

(defn geomMean 
  "Takes the geometric mean of a dataset"
  [data]
  (Math/pow
   (reduce * data)
   (/ 1 (count data))))

(defn median [data]
  (let [s (sort-by :AIC data)
        c (count data)]
    (nth s (int (/ c 2)))))

(defn stdev 
  "Takes the standard deviation of a dataset"
  [data]
  (let [x (mean data)]
    (Math/sqrt (/
                (reduce + (map #(square (- % x)) data))
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
;;;;;;  Evolving Multiple Interacting Programs   ;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                Expressions                ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn safediv 
  "Protected division"
  [a & args]
  (try (/ a (apply * args)) (catch Exception _ 0)))

(defn iflte 
  "If a<b then c else d"
  [a b c d]
  (if (<= a b) c d))


(def functions
  "List of functions that output symbols corresponding
 to allowed functions in MIPs"
  [[#(identity '+) 2]
   [#(identity '-) 2]
   [#(identity '*) 2]
   [#(identity 'safediv) 2]
   [#(identity 'Math/sin) 1]
   [#(identity 'Math/cos) 1]
   [#(identity 'iflte) 4]])
;;logistic, tanh, relu, step, 


(defn terms 
  "list of functions that output symbols or real numbers
   [n0 ... np l0 ... lq N(0,1)]
   for p terms and q lags"
  [numTerms numLags]
  (concat (map 
         #(fn [] (symbol (str "n" %))) 
         (range numTerms)) 
          (map
           #(fn [] (symbol (str "l" %)))
           (range numLags))
        [#(randNormal 0 1)]))



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
  [vars expr]
  (eval (concat
         '(let)
         (list vars)
         (list expr))))


(defn bindings 
  "Creates bindings to put into evaluate
   (terms lags)"
  [vars vals]
  (vec (mapcat #(list %1 %2) vars vals)))



(defn randExpr
  "Helper method for makeExpression
   vars: calls random method in vars to generate a symbol or real number"
  [probReal vars depth]
  (let [[f n] (rand-nth functions)]
    (if (or (< (rand) probReal) (<= depth 0))
      (rand-nth vars)
      (concat
       (list (f))
       (repeatedly n #(randExpr  
                       (Math/sqrt probReal) 
                       vars
                       (dec depth)))))))


(defn makeExpression
  "Constructs a random expression that is not a terminal.
   probTerminal: probabiliy an argument is terminal vs an expression
   maxDepth: maximum depth of expressions"
  [probTerminal numTerms numLags maxDepth]
  (let [[f n] (rand-nth functions)]
    (apply list
     (concat
     (list (f))
     (repeatedly n #(randExpr 
                     probTerminal 
                     (map (fn [i] (i)) (terms numTerms numLags))
                     (dec maxDepth)))))))

(defn updateMIP
  "Returns the next state as a function of the current state and the lags
  using the MIP expressions "
  [state mip lags vars]
  (let [b (bindings vars (concat state lags))]
    (map (partial evaluate b) mip)))


(defn makeMIP-1ahead 
  "Takes a time series and an individual, and returns a vector of the 
   1-ahead predictions of the individual"
  [individual numLags vars ts]
  (let [{mip :objective 
         state :state } individual]
  (loop [mipseq [] 
         obs (partition numLags 1 ts)
         state state]
    (if (empty? obs)
      (concat (repeat (- numLags 1) 0) mipseq)
      (let [nextState (updateMIP state mip (first obs) vars)]
        (recur
       (conj mipseq (last state))
       (rest obs)
       nextState))))))

(defn mipError 
  "Mean squared error of the 1-ahead predictions"
  [individual numLags vars ts]
  (let [n (count ts)
        l (- n numLags 1)]
    (/ (reduce + (map #(square (- %1 %2))
                   (makeMIP-1ahead individual numLags vars ts)
                   (take l ts)))
       l)))

(defn MIP_AIC 
  "Akaike information criterion of the individual"
  [individual numLags vars ts]
  (let [{objective :objective} individual
        p (reduce + (map length objective))
        l (- (count ts) (dec numLags))]
    (+ (* l (Math/log (mipError individual numLags vars ts)))
       (* 2 p))))

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
  [expression PTprob vars]
  (if (seq? expression)
    (cons
     (first expression)
     (map #(perturbTerms % PTprob vars)
          (rest expression)))
    (if  (< (rand) PTprob)
      (rand-nth vars)
      expression)))

(defn perturbExpr
  "Randomly replaces an expression with a random expression
  with probability PEprob.
   Random expr constrained to be at most one level deeper than replaced expr"
  [expression PEprob vars probReal]
  (if (seq? expression)
    (if (< (rand) PEprob)
      (randExpr probReal vars (inc (depth expression)))
      (map #(perturbExpr % PEprob vars probReal) expression))
    expression))

(defn safePerturbExpr
  "Prevents returning a terminal"
  [expression PEprob vars probReal]
  (if (seq? expression)
    (map #(perturbExpr % PEprob vars probReal) expression)
    expression))

(defn insert
  "Inserts randomly above an expression. Depth is increased by at most 1"
  [expression Iprob vars probTerminal]
  (if (< (rand) Iprob)
    (let [e (randExpr 0 vars 1)
          i (inc (rand-int (dec (count e))))]
      (replaceExpression e i expression))
    (if (seq? expression)
      (cons (first expression)
        (map #(insert % Iprob vars probTerminal) 
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
  "Randomly promotes a child. Removes at most one from depth"
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


(defn MIP_mutateState [individual]
  (map
   #(+ % (randNormal 0 (first (:strategy individual))))
   (:state individual)))

(defn MIP_mutateObjective 
  "Always performs perturbReal.
   Chooses randomly to perform one of:
   perturbTerms, perturbExpr, insert, suggestDepth, perturbOrder"
  [individual vars probTerminal]
  (let [{strategy :strategy
         objective :objective} individual]
    (map
     #(condp > (rand)
        1/5 (perturbTerms %
                          (nth strategy 2)
                          vars)
        2/5 (safePerturbExpr %
                         (nth strategy 3)
                         vars
                         probTerminal)
        3/5 (insert %
                    (nth strategy 4)
                    vars
                    probTerminal)
        4/5 (suggestDepth %
                          (nth strategy 5)
                          (max 1 (dec (depth %)))
                          1)
        (perturbOrder %
                      (nth strategy 6)))
     (perturbReal objective (first strategy) (second strategy)))))

(defn MIP_mutateStrategy 
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
      (map #(max 0 (min 1 (+ % (randNormal 0 tau)))) (rest strategy)))
    ))

     (defn addmipError 
       "Calculates the fitness function of the individual
        and adds it under the key :Error
        Currently does nothing because I was trying out lexicase selection"
       [individual numLags vars ts]
      ;;  (assoc individual
      ;;         :Error (MIP_AIC individual numLags vars ts))
       individual
       )

(defn MIPmutate 
  "Combines mutation of objective, state, and strategy parameters"
  [individual ts numLags vars probReal tau]
  (addmipError
   (assoc {}
          :objective (MIP_mutateObjective individual vars probReal)
          :state (MIP_mutateState individual)
          :strategy (MIP_mutateStrategy individual tau))
   numLags
   vars
   ts))

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
  (let [[f n] (rand-nth (filter #(= 2 (second %))
                                functions))]
    (list (f)
          (randomSubtree t1)
          (randomSubtree t2))))

(defn MIPcrossover-single [p1 p2]
  (let [newMIP (map crossTrees (:objective p1) (:objective p2))
        newStrat (map #(if (< (rand) 0.5) %1 %2) (:strategy p1) (:strategy p2))
        newState (map #(if (< (rand) 0.5) %1 %2) (:state p1) (:state p2))]
    (assoc {}
           :objective newMIP
           :state newState
           :strategy newStrat)))



(defn MIPcrossover [population n numLags vars ts]
  (repeatedly n #(addmipError
                  (apply MIPcrossover-single
                         (take 2 (shuffle population)))
                  numLags
                  vars
                  ts)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;        Initialization          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn MIPIndividual
  "probReal - probability of terminal vs expression
   numTerms - number of MIPs
   numLags - number of input nodes
   maxDepth - maximum depth of a program
   ts - time series"
  [params]
  (let [{probReal :probReal
         numTerms :numTerms
         numLags :numLags
         maxDepth :maxDepth
         ts :ts} params]
    (addmipError
     (assoc {}
            :objective (repeatedly
                        numTerms
                        #(makeExpression probReal
                                         numTerms
                                         numLags
                                         maxDepth))
            :state (repeatedly numTerms rand)
            :strategy (repeatedly 7 rand);;std PRprob PTprob PEprob Iprob Dprob POprob
            )
     numLags
     (map #(%) (terms numTerms numLags))
     ts)))

(defn MIPselect
  "tournament selection"
  [population tournamentSize n]
  (repeatedly
   n
   #(->> population
         (shuffle)
         (take tournamentSize)
         (apply min-key :Error))))

(defn lexicaseSelect
  "tournament selection"
  [population numLags vars numTrials epsilon ts]
  (loop [obs (shuffle (partition (+ 1 numTrials numLags) numTrials ts))
         pop population]
    (if (or (<= (count population) 1) (empty? obs))
      (rand-nth population)
      (recur (rest obs)
             (let [errors (map
                           #(vector (mipError % numLags vars (first obs)) %)
                           pop)
                   best (apply min-key first errors)]
               (map second (filter 
                            #(< (abs (- (first %) (first best))) epsilon) 
                            errors)))
             ))))

(filter #(= 1 %) [1 2 3 4])


(defn mipError [individual numLags vars ts]
  (let [n (count ts)
        l (- n numLags 1)]
    (/ (reduce + (map #(square (- %1 %2))
                      (makeMIP-1ahead individual numLags vars ts)
                      (take l ts)))
       l)))



(defn MIPevolve
  "Evolution generates excess individuals by mutating and/or crossing over
   existing individuals. Selection is then used to cull the population back
   to popsize.
   Meta evolution controls the probability of removing or adding a term,
   the step size for changing existing terms, and the probability of crossing over."
  [popsize ts & {:keys [maxGenerations tau tournamentSize propSelected numTrials epsilon probReal numLags numTerms maxDepth]
                 :or {maxGenerations 100
                      tau 0.1
                      tournamentSize 3
                      propSelected 0.333
                      numTrials 3
                      epsilon 0.0001
                      probReal 0.8
                      numLags 1
                      numTerms 1
                      maxDepth 3}}]
  (loop [population (repeatedly popsize #(MIPIndividual {:probReal probReal
                                                         :numTerms numTerms
                                                         :numLags numLags
                                                         :maxDepth maxDepth
                                                         :ts ts}))
         generation 0]
    (println generation)
    (let [;;best (apply min-key :Error population)
          vars (map #(%) (terms numTerms numLags))]
      (if (>= generation maxGenerations)
        (apply min-key #(mipError % numLags vars ts) population)
        (recur
         (let [parents (repeatedly (* (count population) propSelected) #(lexicaseSelect population 
                                  numLags vars numTrials epsilon ts))]
           (concat parents
                   (map #(MIPmutate % ts numLags vars probReal tau) parents )
                   (MIPcrossover parents (count parents) numLags vars ts)))
         (inc generation))))))

;; (def ind (MIPevolve 50 (makeLogistic 50 3.95 0.5) 
;;            :maxGenerations 20))
;; ind
;; (mipError ind 1 (map #(%) (terms 1 1)) (makeLogistic 50 3.95 0.5))

;; (* 50 (square (stdev (makeLogistic 50 3.95 0.5))))


;; (defn mutateIndividual 
;;   "replace any expression with another expression
;;    given any expression, replace the i>0th with an expression
;;    get all expressions in a list format?"
;;   [individual mutationProb probReal vars]
;;   (if (and (< (rand) mutationProb) (seq? individual))
;;     (let [i (inc (rand-int (count individual)))
;;           f #(mutateIndividual % mutationProb probReal vars)]
;;       (concat
;;        (list (first individual))
;;        (map f (take (dec i) (rest individual)))
;;        (list (randExpr probReal vars 1))
;;        (map f (drop i (rest individual)))))
;;     individual))





;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;Chaotic and Pseudorandom Systems;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn lorentz 
  "helper for makeLorentz"
  [t s r b x y z]
  (lazy-seq
   (cons [x y z]
         (lorentz t s r b
                  (+ x (* t (* s (- y x))))
                  (+ y (* t (- (* x (- r z)) y)))
                  (+ z (* t (- (* x y) (* b z))))))))

(defn makeLorentz
  "Constructs a lazy sequence of n time points according to the Lorentz
   equations for atmospheric convection
   n: number of time steps
   t: size of time step
   s,r,b: parameters for lorentz equation
   x,y,z: initial point
   returns [[x0 y0 z0] [x1 y1 z1]...]"
  [n t s r b x y z]
  (take n (lorentz t s r b x y z)))

(defn logisticMap 
  "helper for makeLogistic"
  [r x]
  (lazy-seq (cons x (logisticMap r (* r x (- 1 x))))))

(defn makeLogistic
  "Constructs a sequence of n points for the equation 
   x = rx(1-x) starting from initial value x.
   r > 3.56995 for chaotic behavior. r ~ 3.82843 for island of stability"
  [n r x]
  (take n (logisticMap r x)))

(defn ARseq 
  "helper for makeAR"
  [ar start]
  (lazy-seq (cons 
             (first start) 
              (ARseq ar (concat 
                         (rest start) 
                         [(dot ar start)])))))

(defn makeAR
  "Extrapolates an AR model to the length of the timeseries
  Uses the first p values of the timeseries to start the AR model"
  [ar ts]
  (let [n (count ts)
        m (count ar)]
    (concat (repeat m 0)
            (drop m
                  (take n
                        (ARseq ar
                               (take m
                                     ts)))))))


(defn makeAR-1ahead
  "AR functioning as an 1-ahead predictor of the timeseries" 
  [ar ts]
  (let [ar (:objective ar)
        m (count ar)
        obs (partition (inc m) 1 ts)]
     (concat 
      (repeat (- (count ts) (count obs)) 0)
      (map #(dot ar (drop-last %)) obs))))



(defn NNupdate
  "returns next state of NN after feeding in lags"
  [nn lags]
  (let [{weights :weights state :state} nn
        nextState
        (flatten
         (matrix/mmul weights
                      (map list
                           (concat state
                                   lags
                                   [1]))))]
    (concat
     (map relu (drop-last nextState))
     (list (last nextState)))))


(defn makeNN-1ahead 
  "NN functioning as a 1-ahead predictor of time series"
  [nn ts]
  (let [{weights :weights state :state} nn
        m (count (first weights))
        n (count weights)]
    (loop [NNseq []
           state state
           obs (partition (- m n 1) 1 ts)]
      (if (empty? obs)
        (concat (vec (repeat (- m n 1) 0)) (rest NNseq))
        (recur
         (conj NNseq 
               (last state))
         (NNupdate {:weights weights :state state} (first obs))
         (rest obs))))))

(defn NNseq 
  "helper for makeNN"
  [nn lags]
  (lazy-seq
   (let [nextState (NNupdate nn lags)]
     (cons (last nextState)
           (NNseq
            (assoc nn :state nextState)
            (concat (rest lags) [(last nextState)]))))))

(defn makeNN 
  "Extrapolates NN to size of ts using the first k terms,
   where k is the size of the NN state"
  [nn ts]
  (let [{weights :weights} nn
        n (count weights)
        m (count (first weights))
        l (count ts)]
    (concat 
     (repeat (- m n 1) 0)
     (take  
     (- l (- m n 1)) 
     (NNseq nn (take (- m n 1) ts))))))



(defn gcd [a b]
  (let [m (mod a b)]
    (if (= m 0)
      b
      (recur b m))))

(defn BBSseq 
  "Helper method for makeBBS"
  [p q x0]
  (lazy-seq (cons x0 (BBSseq p q (mod (square x0) (*' p q))))))

(defn makeBBS 
  "Blum Blum Shub pseudorandom number generator.
   For a large cycle length:
   p and q must be safe primes - (p-1)/2 is also prime
   they should have small gcd[(p-3)/2, (q-3)/2]
   and x0 should be coprime to pq and not 0 or 1"
  [n & {:keys [p q x0]
                  :or {p 100547
                       q 100523
                       x0 50000}}]
  (map #(mod % 10) (take n (BBSseq p q x0))))

(defn smallBBS 
  "50x Smaller defaults avoid integer overflow"
  [n & {:keys [p q x0]
                  :or {p 2963
                       q 2903
                       x0 1000}}]
  (makeBBS n :p p :q q :x0 x0))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;           AR Models          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn ARsseSingle [ar ts]
  (square (- (last ts) (dot ar (drop-last ts)))))

(defn ARmse 
  "Mean squared error for ar as a 1-ahead predictor of ts"
  [ar ts]
  (let [m (count ar)
        n (count ts)]
    (/
     (reduce +
             (map
              (partial ARsseSingle ar)
              (partition (inc m) 1 ts)))
     (- n m))))

(defn ARmseTotal 
  "Mean squared error for ar when extrapolating from first 
   p terms of ts, where p is the lag of the ar"
  [ar ts]
  (let [m (count ar)
        n (count ts)
        ARseq (makeAR ar ts)]
    (reduce + (map #(square (- %1 %2)) ARseq ts))))

(defn AR_AIC
  "Computes the Akaike information criterion for the AR model ar on the 
   time series ts. 
   AIC: nlog(MSE) + 2p"
  [ar ts]
  (let [n (count ts)
        m (count ar)]
    (+
     (* (- n m) (Math/log (ARmse ar ts)))
     (* 2 m))))

(defn ARIndividual
  "Randomizes coefficients for an AR model of lag p, 
   where p is randomly chosen from 1 to maxL"
  [maxL ts]
  (let [p (+ 1 (rand-int maxL))
        ar (repeatedly p #(rand))]
    (assoc {}
           :objective ar
           :strategy [(rand) (rand) (rand)];;stdev changeprob parentprob 
           :AIC (AR_AIC ar ts))))

(defn ARmutate-objective [individual maxL]
  (let [{ar :objective ;;[0.5 0.6 0.7] AR model of 3 lags
         strategy :strategy};; [0.3]
        individual
        objective (map
                   #(+ % (randNormal 0 (first strategy)))
                   ar)]
    (if (< (rand) (second strategy)) 
      (if (and 
           (or 
            (< (rand) 0.5) 
            (< (count ar) 2))
           (< (count ar) maxL))
        (let [obj (conj objective (rand))
              objsum (reduce + obj)]
          (map #(/ % objsum) obj))
        (let [obj (drop-last objective)
              objsum (reduce + obj)]
          (map #(/ % objsum) obj))) 
      objective)
    ))


(defn ARmutate-strategy [individual tau]
  (let [{strategy :strategy} individual]
    (map 
     #(* % (->> individual 
                (:objective) 
                (count) 
                (* 2) 
                (Math/sqrt) 
                (/ tau) 
                (logNormal 0 1))) 
     strategy)))



(defn ARmutate-individual [individual maxL tau ts]
  (let [obj (ARmutate-objective individual maxL)
        strat (ARmutate-strategy individual tau)]
    (assoc individual
           :objective obj
           :AIC (AR_AIC obj ts)
           :strategy strat)))


(defn ARmutate [population maxL tau ts]
  (reduce 
   #(concat 
     %1 
     (set [%2 (ARmutate-individual %2 maxL tau ts)])) 
   {}
   population))

(defn ARcrossover-objective
  "Crossing over between two AR models"
  [ar1 ar2]
  (let [n (rand-int (count ar1))
        obj (concat (take n ar1) (drop n ar2))
        objsum (reduce + obj)]
    (map #(/ % objsum) obj)))

(defn ARcrossover-strategy 
  [s1 s2]
  (map #(/ (+ %1 %2) 2.0) s1 s2))




(defn ARcrossover-individual [p1 p2 ts]
  (let [obj (ARcrossover-objective 
             (:objective p1) 
             (:objective p2))
        strat (ARcrossover-strategy 
               (:strategy p1) 
               (:strategy p2))]
    (assoc {}
           :objective obj
           :strategy strat
           :AIC (AR_AIC obj ts))))

(defn ARcrossover-single
  "choose two parents. If probability smaller than their combined
   probability, then child"
  [population ts]
      (let [[p1 p2] (take 2 (shuffle population))]
    (if (< 
         (rand 2) 
         (+ (nth (:strategy p1) 2) 
            (nth (:strategy p2) 2))) 
      population 
      (conj population (ARcrossover-individual p1 p2 ts)))))




(defn ARcrossover 
  "Apply crossover to population"
  [population ts]
   ((apply comp 
          (repeat 
           (count population) 
           #(ARcrossover-single % ts))) 
   population))




(defn better [i1 i2]
  (< (:AIC i1) (:AIC i2)))

(defn  minAIC [population]
  (reduce
   #(if (better %1 %2) %1 %2)
   {:AIC ##Inf}
   population))


(defn ARselect
  "tournament selection"
  [population tournamentSize n]
  (repeatedly 
   n 
   #(->> population 
         (shuffle) 
         (take tournamentSize) 
         (minAIC))))

(defn ARlexicaseSelect
  "tournament selection"
  [population epsilon ts]
  (let [maxLag (apply max (map #(count (:objective %)) population))]
    (loop [obs (shuffle (partition (+ 1 maxLag) 1 ts))
         pop population]
    (if (or (<= (count population) 1) (empty? obs))
      (rand-nth population)
      (recur (rest obs)
             (let [errors (map ;;[error individual]
                           #(let [p (count (:objective %))]
                              (vector (ARsseSingle 
                                     (:objective %) 
                                     (take-last (inc p) (first obs))) 
                                      %))
                           pop)
                   best (apply min-key first errors)]
               (map second (filter
                            #(< (abs (- (first %) (first best))) epsilon)
                            errors))))))))

(defn ARevolve 
  "Evolution generates excess individuals by mutating and/or crossing over
   existing individuals. Selection is then used to cull the population back
   to popsize.
   Meta evolution controls the probability of removing or adding a term,
   the step size for changing existing terms, and the probability of crossing over."
  [popsize ts & {:keys [maxGenerations tau maxL tournamentSize ]
              :or {maxGenerations 100
                   tau 0.1
                   maxL 6
                   tournamentSize 2}}]
  (loop [population (repeatedly popsize #(ARIndividual maxL ts))
         generation 0]
    (let [best (minAIC population)]
      (println (:AIC best) (median (map :AIC population)))
        (if (>= generation maxGenerations)
      best
      (recur
       (-> population 
           (ARmutate maxL tau ts) 
          ;;  (ARcrossover ts)
           (ARselect-lexicase popsize 0.00001 ts))
       (inc generation))))))

(defn ARselect-lexicase [population popsize epsilon ts]
  (repeatedly popsize #(ARlexicaseSelect population epsilon ts)))





; or concat (repeatedly #(


;;)) population
;; choose two parents. If probability smaller than their combined
;; probability, then no child. else child
;;proportion of extra children = strategy parameter?
(defn least-squares
  "Solves the least-squares regression problem Xb=y"
  [X y]
  (matrix/mmul
   (matrix/inverse (matrix/mmul (matrix/transpose X) X))
   (matrix/transpose X)
   y))

(defn AR-least-squares
  "Computes coefficients for an AR model of lag p using ordinary least-squares"
  [p ts]
  (let [obs (partition (inc p) 1 ts)]
    (least-squares (map drop-last obs) (map last obs))))


;; predict a chaotic system


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;          Main Method         ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; (defn -main
;;   "I don't do a whole lot ... yet."
;;   [& args]
;;   (println "Hello, World!"))

(defn NNWeights [maxL]
  (let [n (+ 3 (rand-int (- maxL 2)))
        m (+ 1 (rand-int (- n 2)))]
    (repeatedly m (fn [] (repeatedly n rand)))))

(defn NNError [weights state ts]
  (let [n (count weights)
        m (count (first weights))
        oneahead (makeNN-1ahead 
                  {:weights weights :state state}
                  ts)]
    (MSE (drop (- m n 1) oneahead) 
         (drop (- m n 1) ts))))



(defn perturbElement [e std prob]
  (if (not (= e 0))
    (if (< (rand) prob)
      0
     (+ e (randNormal 0 std)))
    (if (< (rand) prob)
      (rand)
      0)))

(defn perturbWeights [individual std prob]
  {:weights 
   (map 
    (fn [row] 
      (map 
       #(perturbElement % std prob)
       row))
    (:weights individual))})




;;List of [boolean operator] where the boolean function evaluates
;;whether applying the variation operator to an individual will produce a valid child
(def NNoperators
  ;; Can always increase width of array
  [[#(<= (count (first (:weights %))) 20)
    (fn [i]
      (identity {:weights
                 (map #(concat % [(rand)]) (:weights i))
                 :state
                 (:state i)}))]
   ;; Can decrease height of array if larger than 1
   [#(> (count (:weights %)) 1)
    #(identity {:weights (drop-last (:weights %))
                :state (drop-last (:state %))})]
   ;; Can increase height or decrease width if height < width - 2 (one lag plus bias)
   [#(< (count (:weights %))
        (- (count (first (:weights %))) 2))
    #(identity {:weights (concat
                          (:weights %)
                          [(repeatedly (count (first (:weights %))) rand)])
                :state (concat (:state %) [(rand)])})
    #(identity {:weights (map drop-last (:weights %))
                :state (:state %)})]])

(defn perturbArchitecture [individual prob]
  (let [{weights :weights
         state :state} individual]
    ;; Decide whether to perturb architecture
    (if (>= (rand) prob)
      {:weights weights :state state}
      ;;Get all the valid operators and apply a random one to individual
      (let [operators (reduce
                       #(if ((first %2) individual)
                          (concat %1 (rest %2))
                          %1)
                       [] NNoperators)]
        ((rand-nth operators) individual)))))




(defn perturbStrategy [individual tau]
  {:strategy (map 
              #(->> tau 
                    (logNormal 0 1)
                    (* %)) 
              (:strategy individual))})

(perturbStrategy {:strategy [0.1 0.1 0.1]} 1)

(defn perturbState [individual std]
  {:state (map 
           #(+ % (randNormal 0 std)) 
           (:state individual))})

(defn perturbIndividual [individual tau]
  (let [{strategy :strategy} individual]
    (reduce 
      (fn [i f] (merge i (f i)))
      individual
      [;;#(perturbArchitecture % (second strategy))
       #(perturbWeights % (first strategy) (nth strategy 2))
       #(perturbStrategy % tau)
       #(perturbState % (first strategy))])))

(defn NNmutate [population tau ts]
  (map #(let [individual (perturbIndividual % tau)] 
          (assoc individual 
                 :Error (NNError 
                         (:weights individual) 
                         (:state individual) ts))) 
       population))



;;crossover
(defn pad
  "Pads weights and state with nans until they reach size
   nxm and nx1 respectively"
  [weights state m n]
  (let [m (max m (count (first weights)))
        n (max n (count weights))
        diffM (- m (count (first weights)))
        diffN (- n (count weights))]
    {:weights (concat
               (map #(concat % (repeatedly diffM (constantly ##NaN))) weights)
               (repeatedly diffN
                           (fn []
                             (repeatedly m
                                         (constantly ##NaN)))))
     :state (concat state (repeatedly diffN (constantly ##NaN)))}))

(defn crossElement 
  "if e1 or e2 is nan, returns the other one. Otherwise
   has a 50/50 chance of returning either one"
  [e1 e2]
  (if (Double/isNaN e1)
    e2
    (if (Double/isNaN e2)
      e1
      (if (< (rand) 0.5)
        e1
        e2))))

(defn crossWeights 
  "uniform crossover of the weights matrix of two parents"
  [p1 p2]
  {:weights
   (map (fn [r1 r2]
          (map crossElement
               r1
               r2))
        (:weights p1)
        (:weights p2))})



(defn crossState 
  "uniform crossover of the state vectors of two parents"
  [p1 p2]
  {:state (map crossElement (:state p1) (:state p2))})

(defn crossStrategy 
  "uniform crossover of the strategy vectors of two parents"
  [p1 p2]
  {:strategy (map crossElement (:strategy p1) (:strategy p2))})

(defn crossIndividual 
  "crossover two parents"
  [p1 p2 ts]
  (let [m (max (count (:matrix p1)) (count (:matrix p2)))
        n (apply max 
                 (map 
                  #(-> % (:matrix) (first) (count)) 
                  [p1 p2]))
        p1 (merge p1 (pad (:weights p1) (:state p1) m n))
        p2 (merge p2 (pad (:weights p2) (:state p2) m n))
        child (reduce #(merge %1 (%2 p1 p2)) {}
                      [crossWeights crossState crossStrategy])]
   (merge child {:Error (NNError (:weights child) (:state child) ts)})))

(defn NNcrossover [population ts propCrossed]
  (repeatedly 
   (* propCrossed (count population))
   #(let [[p1 p2] (take 2 (shuffle population))]
      (crossIndividual p1 p2 ts))))

(defn NNIndividual [maxL ts]
  (let [weights (list (concat [0] (repeatedly maxL #(rand))));(NNWeights maxL)
        state (repeatedly (count weights) #(rand))]
    (assoc {}
           :weights weights
           :state state
           :strategy [(rand) 0 0];;std, architecture change prob, addition/deletion prob 
           :Error (NNError weights state ts))))

;;state: h1 h2 out
;;input: h1 h2 out x0 x1
(defn NNIndividual2 [ts]
  (let [weights [[0 0 0 (randNormal 0 1) (randNormal 0 1) (randNormal 0 1)];h0
                 [0 0 0 (randNormal 0 1) (randNormal 0 1) (randNormal 0 1)];h1
                 [(randNormal 0 1) (randNormal 0 1) 0 0 0 (randNormal 0 1)]];out
        state (repeatedly 3 #(randNormal 0 1))]
    (assoc {}
           :weights weights
           :state state
           :strategy [(rand) 0 0];;std, architecture change prob, addition/deletion prob 
           :Error (NNError weights state ts))))


(defn NNselect
  "tournament selection"
  [population tournamentSize n]
  (repeatedly
   n
   #(->> population
         (shuffle)
         (take tournamentSize)
         (apply min-key :Error))))

(defn numParams 
  "Number of nonzero parameters in the neural network"
  [individual]
  (let [nonzero #(if %2 (inc %1) %1)]
   (+
   (reduce 
    #(+ %1 (reduce nonzero 0 %2))
    0
    (:weights individual))
   (reduce nonzero 0 (:state individual)))))

(defn NN_AIC [individual ts]
  (let [n (count ts)
        m (- (count (first (:weights individual)))
             (count (:weights individual)))]
    (+ (* (- n m) (Math/log (:Error individual)))
       (* 2 (numParams individual))
     )))

(defn NNevolve
  "Evolution generates excess individuals by mutating and/or crossing over
   existing individuals. Selection is then used to cull the population back
   to popsize.
   Meta evolution controls the probability of removing or adding a term,
   the step size for changing existing terms, and the probability of crossing over."
  [popsize ts & {:keys [maxGenerations tau maxL tournamentSize propCrossed]
                 :or {maxGenerations 100
                      tau 0.1
                      maxL 6
                      tournamentSize 2
                      propCrossed 0.5}}]
  (loop [population (repeatedly popsize #(NNIndividual2 ts))
         generation 0]
    (let [best (apply min-key :Error population)]
      (if (>= generation maxGenerations)
        best
        (recur
         (conj
          (NNselect
           (concat population
                   (NNmutate population tau ts)
                   (NNcrossover population ts propCrossed))
           tournamentSize
           popsize)
          best)
         (inc generation))))))

(let [ts (makeLogistic 100 3.95 0.5)]
  (compareModels [ts "TS"] [makeNN-1ahead (NNevolve 50 ts) "NN"]
                    [makeAR-1ahead (ARevolve 50 ts) "AR"]))

(spit "test.txt" (with-out-str (ARevolve 50 (take 100 (map :Temp temperatures)))))
(+ 1 2)
(spit "test.txt" (with-out-str (println "hi")))
(ARevolve 50 (take 100 (map :Temp temperatures)))
(def best (ARevolve 50 (take 100 (map :Temp temperatures)) :maxGenerations 50))
(def bestNN (NNevolve 50 (take 100 (map :Temp temperatures)) :maxGenerations 50))

(def ARexample [-0.2641759954634062 0.2874019597966984 -0.28729532625686294 0.3624066158003387 -1.4809555937645422 2.3826183398877743])
(def ARexample2 [0.1031 0.1109 -0.1206 0.1746 -0.1521 0.6216])

(let [ts (take 100 (map :Temp temperatures))
      ar (makeAR-1ahead best ts)
      nn (makeNN bestNN ts)]
  (multiplot (range 100) [ts "TS"] [nn "NN"])
  ;; (ARmse (map #(/ % n) ARexample) ts)
  )

(let [ts (take 100 (map :Temp temperatures))
      ar (makeAR (:objective best) ts)]
  (multiplot (range 100) [ts "TS"] [ar "AR"])
  ;; (ARmse (map #(/ % n) ARexample) ts)
  )

(reduce + (:objective best))
(makeNN-1ahead bestNN (take 100 (map :Temp temperatures)))
(ARmse (:objective (ARevolve 25 (makeLogistic 100 3.95 0.5)
                           :maxL 10
                           :maxGenerations 200)) (makeLogistic 100 3.95 0.5))

(/ (reduce + (map #(square (- % 0.5)) (makeLogistic 100 3.95 0.5))) 100)
;;                            :maxL 15)) (makeLogistic 100 3.95 0.5))
(AR_AIC (AR-least-squares 5 (makeLogistic 100 3.95 0.5)) (makeLogistic 100 3.95 0.5))

;; (ARmse (:objective (evolve 20 (makeLogistic 400 3.95 0.5)
;;                            :maxGenerations 100
;;                   :maxL 10)) 
;;        (makeLogistic 200 3.95 0.5))

(AR-least-squares 6 (take 100 (map :Temp temperatures)))