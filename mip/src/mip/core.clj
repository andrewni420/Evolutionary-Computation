(ns mip.core
  (:gen-class)
  (:require [incanter.core]
            [incanter.stats]
            [incanter.datasets]
            [incanter.charts]
            [clojure.core.matrix :as matrix]))



(defmacro make-fn
  [m]
  `(fn [& args#]
     (eval
      (cons '~m args#))))

(defn multiplot [x & ys]
  (incanter.core/view
   (reduce
    #(incanter.charts/add-lines %1 x %2)
    (incanter.charts/xy-plot x (first ys))
    (rest ys))))

(defn compareModels
  "Compares different models for predicting timeseries on the same graph
   models: [fn model] where (fn model ts) produces a list of the model's predictions"
  [ts & models]
  (apply multiplot
         (range (count ts))
         ts
         (map #((first %) (second %) ts) models)))


(defn dot
  ([a b] (reduce + (map * a b)))
  ([a] (dot a a)))

(defn sigmoid [x]
  (/ 1 (+ 1 (Math/exp (- x)))))

(defn relu [x] (max 0 x))

(defn magnitude [a] (Math/sqrt (dot a)))

(defn square [n] (*' n n))

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

(defn MSE 
  "Mean squared error of a prediction and a time seris"
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

(defn preorderTraversal [individual]
  (if (seq? individual)
    (apply concat
           [individual]
           (map preorderTraversal (rest individual)))
    [individual]))


(defn postorderTraversal [individual]
  (if (seq? individual)
    (concat (apply concat
                   (map postorderTraversal
                        (rest individual)))
            [individual])
    [individual]))


;; multiple interacting programs
;; tree structure
;; + - * /
;; apply function to arguments
;;Each expression is a {:function f, :args }
;; Predictive power vs AIC information

(defn safediv [a & args]
  (try (/ a (apply * args)) (catch Exception _ 0)))

(defn iflte [a b c d]
  (if (<= a b) c d))

"List of functions that output symbols corresponding
 to allowed functions in MIPs"
(def functions [[#(identity '+) 2]
                [#(identity '-) 2]
                [#(identity '*) 2]
                [#(identity 'safediv) 2]
                [#(identity 'Math/sin) 1]
                [#(identity 'Math/cos) 1]
                [#(identity 'iflte) 4]])
;;logistic, tanh, relu, step, 
;;p
;; n lags and m nodes

(defn terms 
  "n0 = input and last ni is the output
   list of functions that output symbols or real numbers"
  [numTerms numLags]
  (concat (map 
         #(fn [] (symbol (str "n" %))) 
         (range numTerms)) 
          (map
           #(fn [] (symbol (str "l" %)))
           (range numLags))
        [rand]))



(defn replaceExpression
  "Replaces expression at index idx in the parent with expr
   indexes smaller than 0 concat to the front
   indexes larger than (count parent) concat to the end"
  [parent idx expr]
  (concat (take idx parent) (list expr) (drop (inc idx) parent)))

(defn depth [individual]
  (if (seq? individual)
    (+ 1 (apply max (map depth individual)))
    0))


(defn evaluate
  "Evaluates an expression given variable bindings.
   Bindings must be of the form '[a 0]"
  [vars individual]
  (eval (concat
         '(let)
         (list vars)
         (list individual))))




(defn randExpr
  "Helper method for makeIndividual
   vars: calls random method in vars to generate a symbol or real number"
  [probReal vars depth]
  (let [[f n] (rand-nth functions)]
    (if (or (< (rand) probReal) (<= depth 0))
      ((rand-nth vars))
      (concat
       (list (f))
       (repeatedly n #(randExpr  
                       (Math/sqrt probReal) 
                       vars
                       (dec depth)))))))

(defn makeIndividual
  "Constructs a random expression.
   probReal: probabiliy an argument is a number or term vs an expression
   maxDepth: maximum depth of expressions"
  [probReal numTerms numLags maxDepth]
  (let [[f n] (rand-nth functions)]
    (apply list
     (concat
     (list (f))
     (repeatedly n #(randExpr 
                     probReal 
                     (terms numTerms numLags) 
                     (dec maxDepth)))))))


(defn length 
  "The number of parameters in an individual"
  [individual] (-> individual (flatten) (count)))


;more specific to more general
(defn perturbReal 
  "Perturbs real numbers in an individual with a normal random
   variable N(0,std) with probability PRprob"
  [individual PRprob std]
  (cond
    (number? individual) (if (< (rand) PRprob)
                           (+ individual (randNormal 0 std))
                           individual)
    (seq? individual) (map #(perturbReal % PRprob std) individual)
    :else individual))

(defn perturbTerms
  "Randomly replaces terms in an individual with probability PTprob"
  [individual PTprob vars]
  (if (seq? individual)
    (cons
     (first individual)
     (map #(perturbTerms % PTprob vars)
          (rest individual)))
    (if  (< (rand) PTprob)
      ((rand-nth vars))
      individual)))

(defn perturbExpr 
  "Randomly replaces an expression with a random expression
   in an individual with probability PEprob.
   Random expr constrained to be at most one level deeper than replaced expr"
  [individual PEprob vars probReal]
  (if (seq? individual)
    (if (< (rand) PEprob)
      (randExpr probReal vars (inc (depth individual)))
      (map #(perturbExpr % PEprob vars probReal) individual))
    individual))

(defn insert 
  "Inserts randomly above an expression. Depth is increased by at most 1"
  [individual Iprob vars probReal]
  (if (< (rand) Iprob)
  (let [e (randExpr probReal vars 1)
        i (inc (rand-int (dec (count e))))]
    (replaceExpression e i individual))
  (if (seq? individual)
    (map #(insert % Iprob vars probReal) individual)
   individual)))


(defn delete 
  "Randomly promotes a child. Removes at most one from depth"
  [individual Dprob]
  (if (seq? individual)
    (if (< (rand) Dprob)
      (let [c (count individual)]
        (nth individual (+ 1 (rand-int (dec c)))))
      (map #(delete % Dprob) individual))
    individual))

(defn enforceDepth [individual maxDepth]
  (if (< maxDepth (depth individual))
    (let [c (count individual)
          i (+ 1 (rand-int (dec c)))]
      (recur (nth individual i) maxDepth))
    individual))

(defn suggestDepth [individual Dprob maxDepth numTries]
  (if (or (<= (depth individual) maxDepth)
          (<= numTries 0))
    individual
    (recur (delete individual Dprob) 
           Dprob 
           maxDepth 
           (dec numTries))))

(defn perturbOrder [individual POprob]
  (if (and (seq? individual) (< (rand) POprob))
    (cons (first individual)
      (shuffle (map #(perturbOrder % POprob) (rest individual))))
    individual))

(defn mutateIndividual 
  "replace any expression with another expression
   given any expression, replace the i>0th with an expression
   get all expressions in a list format?"
  [individual mutationProb probReal vars]
  (if (and (< (rand) mutationProb) (seq? individual))
    (let [i (inc (rand-int (count individual)))
          f #(mutateIndividual % mutationProb probReal vars)]
      (concat
       (list (first individual))
       (map f (take (dec i) (rest individual)))
       (list (randExpr probReal vars 1))
       (map f (drop i (rest individual)))))
    individual))

(defn randomSubtree 
  "chooses a random subtree of an individual"
  [individual]
  (rand-nth (preorderTraversal individual)))

(defn crossTrees 
  "Returns a binary function of a random subtree taken
   from each parent"
  [p1 p2]
  (let [[f n] (rand-nth (filter #(= 2 (second %)) 
                                functions))]
    (list (f)
          (randomSubtree p1)
          (randomSubtree p2))))

(defn lazydec [n]
  (if (>= 0 n)
    [n]
    (lazy-seq (cons n (lazydec (dec n))))))

(lazydec 10)

(defn MIPcrossover-single [p1 p2]
  ())


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
  (let [m (count ar)
        obs (partition (inc m) 1 ts)]
     (concat 
      (repeat (- (count ts) (count obs)) 0)
      (map #(dot ar (drop-last %)) obs))))

(defn NNupdate 
  "returns next state of NN after feeding in lags"
  [nn lags]
  (let [{weights :weights state :state} nn
        nextState
        (matrix/mmul weights
                     (concat state lags [1]))]
    (concat
     (map relu (drop-last nextState))
     (list (last nextState)))))

(defn makeNN-1ahead 
  "NN functioning as a 1-ahead predictor of time series"
  [nn ts]
  (let [{weights :weights state :state} nn
        n (count weights)
        m (count (first weights))]
    (loop [NNseq []
           state state
           obs (partition (- m n 1) 1 ts)]
      (if (empty? obs)
        (concat (vec (repeat (- m n 1) 0)) (rest NNseq))
        (recur
         (conj NNseq 
               (last state))
         (NNupdate nn (first obs))
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
  (let [{ar :objective 
         strategy :strategy} individual
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
  (let [n (rand-int (count ar1))]
    (concat (take n ar1) (drop n ar2))))

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

(defn evolve 
  "Evolution generates excess individuals by mutating and/or crossing over
   existing individuals. Selection is then used to cull the population back
   to popsize.
   Meta evolution controls the probability of removing or adding a term,
   the step size for changing existing terms, and the probability of crossing over."
  [popsize ts & {:keys [maxGenerations tau maxL tournamentSize ]
              :or {maxGenerations 100
                   tau 0.1
                   maxL 6
                   tournamentSize 5}}]
  (loop [population (repeatedly popsize #(ARIndividual maxL ts))
         generation 0]
    (let [best (minAIC population)]
        (if (>= generation maxGenerations)
      best
      (recur
       (-> population 
           (ARmutate maxL tau ts) 
           (ARcrossover ts)
           (ARselect tournamentSize popsize))
       (inc generation))))))


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

(ARmse (:objective (evolve 25 (makeLogistic 100 3.95 0.5)
        :maxL 10
        :maxGenerations 200)) (makeLogistic 100 3.95 0.5))

(/ (reduce + (map #(square (- % 0.5)) (makeLogistic 100 3.95 0.5))) 100)
;;                            :maxL 15)) (makeLogistic 100 3.95 0.5))
(AR_AIC (AR-least-squares 5 (makeLogistic 100 3.95 0.5)) (makeLogistic 100 3.95 0.5))

;; (ARmse (:objective (evolve 20 (makeLogistic 400 3.95 0.5)
;;                            :maxGenerations 100
;;                   :maxL 10)) 
;;        (makeLogistic 200 3.95 0.5))



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




;; increase/decrease 2 dimensions of w
;; (defn perturbArchitecture[individual prob]
;;   (let [{weights :weights
;;          state :state} individual]
;;     (if (< (rand) prob)
;;       (if (>= (count weights) (dec (count (first weights))))
;;         (condp >= (rand)
;;           0.5 {:weights (map #(concat % [(rand)]) weights)
;;                :state state};increment count first weights
;;           {:weights (drop-last weights)
;;              :state (drop-last state)});decrement count weights
;;         (condp >= (rand)
;;           0.25 {:weights (concat weights [(repeatedly (count (first weights)) #(rand))])
;;                 :state (concat state [(rand)])}
;;           0.5 {:weights (map drop-last weights)
;;                :state state}
;;           0.75 {:weights (drop-last weights)
;;                 :state (drop-last state)}
;;           {:weights (map #(concat % [(rand)]) weights)
;;              :state state}))
;;       {:weights weights :state state})))

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
  (let [weights [[0 0 0 (rand) (rand)];h0
                 [0 0 0 (rand) (rand)];h1
                 [(rand) (rand) 0 0 0]];out
        state (repeatedly 3 #(rand))]
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
                      tournamentSize 5
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
  (compareModels ts [makeNN-1ahead
                  (NNevolve 50 ts :maxGenerations 100)]))
(def ts (makeLogistic 100 3.95 0.5))
(def best (NNevolve 50 (makeLogistic 100 3.95 0.5) :maxGenerations 100))

(multiplot (range 100) ts (makeNN-1ahead best ts))
best


(* 98 (Math/log 
 (/ 
  (reduce + 
          (map 
           #(square (- % 0.5)) 
           (makeLogistic 98 3.95 0.5))) 
  98)))

(+ (* 2 6) (* 98 (Math/log 0.10071839784509326)))
