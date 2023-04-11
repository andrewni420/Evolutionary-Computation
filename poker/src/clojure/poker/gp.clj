(ns poker.Andrew.gp
  (:require [propeller.push.state :as state]
            [poker.utils :as utils] 
            [poker.headsup :as headsup]
            [propeller.push.instructions :as instructions]
            [propeller.push.instructions.numeric :as numeric] 
            [propeller.push.instructions.code :as code] 
            [propeller.push.instructions.bool :as bool] 
            [propeller.tools.math :as math]
            [propeller.genome :as genome]
            [propeller.selection :as selection]
            [propeller.variation :as variation]
            [propeller.gp :as gp]
            [propeller.simplification :as simplification]
            [propeller.utils]
            [propeller.push.interpreter :as interpreter]
            [propeller.push.instructions.input-output :as io]))

#_(instructions/def-instruction
  :nort
  ^{:stacks #{:nort}}
  (fn [state]
    (instructions/make-instruction state #(and %1 %2) [:nort :nort] :nort)))

#_(instructions/get-stack-instructions #{:nort})
#_(interpreter/interpret-program (list :nort) (assoc state/empty-state :nort [true false]) 100)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                Functions List             ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;Query for past games with similar hands and cards
;;Query for past games with similar betting action, such as preflop raise
;;Assign "surprise value" to games? Selectively forget more and more games as time goes on
;;depending on how boring they are
;;Output has to be an "action" 
;;Create actions from legal actions, but those aren't actual "actions"
;;Create actions by taking an action or by taking an action type and a float and coercing it to bounds

;;; Constants
(def empty-state
  "Empty state for initializing a push program.
   
   Stacks - 
   :integer
   :float
   :boolean
   :string
   :vector_integer
   :vector_float
   :vector_boolean
   :vector_string
   :keyword
   :char
   :print
   :code
   :exec
   :input
   :output
   :game-history
   :game-state
   :actions"
  (assoc state/empty-state
         ;;:keyword (list) 
         ;;:input2 nil
         ;;input: game history, game state, allowed actions
         :action (list) ;;actions
         ;;output stack: actions
         ;;Push an action onto the output with default weight 1
         ;;:card (list)
         ;;:vector-vard (list);;hands
         ;;:me nil (goes into input)
         ;;:cache {}
         :money (list)
         :probability (list)
         ))



(def game-state-kws
  "Keys for accessing information about the current game"
  (remove #(utils/in? [:active-players
                       :prev-bettor
                       :num-players
                       :players
                       :game-over
                       :community] %) (keys (headsup/init-game))))
(def game-history-kws
  "Keywords for accessing information about a particular game instance in history"
  [:hands :action-history :community :net-gain])

(instructions/def-instruction
  :total-weight
  ^{:stacks #{:action :weight}}
  (fn [state]
    (let [w (reduce + (map second (:actions state)))]
      (if (zero? w)
        state
        (state/push-to-stack state :weight w)))))

(instructions/def-instruction
  :max-weight
  ^{:stacks #{:input :card}}
  (fn [state]
    (let [w (apply max (map second (:actions state)))]
      (if (zero? w)
        state
        (state/push-to-stack state :weight w)))))

(defn push-to-stacks 
  "Push multiple? values onto multiple? stacks
   Values and stacks must be the same length"
  [state stacks values]
  (if (or (not (seq? stacks)) (not (seq? values)))
    (state/push-to-stack state stacks values)
    (loop [state state
         stacks (reverse stacks)
         values (reverse values)]
    (if (or (empty? stacks) (empty? values))
      state
      (let [cur-stack (first stacks)
            cur-val (first values)]
        (recur (state/push-to-stack state cur-stack cur-val)
               (rest stacks)
               (rest values)))))))

(defn make-instruction
  "instructions/make-instructions but with the possibility of multiple
   return types"
  [state function arg-stacks return-stacks]
  (let [popped-args (state/get-args-from-stacks state arg-stacks)]
    (if (= popped-args :not-enough-args)
      state
      (let [results (apply function (:args popped-args))
            new-state (:state popped-args)]
        (if (= results :ignore-instruction)
          state
          (push-to-stacks new-state return-stacks results))))))

(defn set-type [type literal]
  {type literal})

;;; Game State Functions
(defn state-to-inputs ;;10-20ms
  [game-state]
  (identity #_time (let [{current-player :current-player
         hands :hands
         players :players
         bet-values :bet-values
         current-bet :current-bet
         pot :pot
         visible :visible
         betting-round :betting-round} game-state
        my-bet (bet-values current-player)
        my-hand (hands current-player)
        SC-numbers (utils/S-C-numbers (utils/hand-to-string my-hand))]
    {:hand-strength (first (utils/hand-value my-hand))
     :holecard-winrate (set-type :probability
                                 (utils/rollout-winrate my-hand))
     :SC-num-better (first SC-numbers)
     :SC-robustness (set-type :probability
                              (second SC-numbers))
     :SC-money-needed (do (when (not (nth SC-numbers 2)) (println "SC-money-needed" SC-numbers (utils/hand-to-string my-hand)))(set-type :money
                                (nth SC-numbers 2)))
     :betting-round (.indexOf utils/betting-rounds betting-round)
     :num-cards-to-come (- 5 (count visible))
     :cost-to-call (do (when (not (max 0.0 (- current-bet my-bet))) (println "Cost-to-call")) (set-type :money
                             (max 0.0 (- current-bet my-bet))))
     :my-stack (do (when (not (:money (players current-player))) (println "My-stack"))(set-type :money 
                         (:money (players current-player))))
     :opp-stack (do (when (not (:money (players (- 1 current-player)))) (println "opp-stack"))(set-type :money
                          (:money (players (- 1 current-player)))))
     :pot (do (when (not pot) (println "pot"))(set-type :money pot))
     :in-position? (zero? current-player);;boolean
     #_:prob-better-hand #_(set-type :probability
                                 (utils/prob-better-hand-rough visible my-hand));;boolean
     :prob-catch-outs (set-type :probability
                                (float (/ (count (utils/outs (concat my-hand visible)))
                                (- 52 2 (count visible)))))})))

(defn init-push-state 
  "Uses game-state to initialize inputs of push states\\
   Creates action, probability, and money stacks"
  [game-state]
  (assoc state/empty-state
         :money (list)
         :probability (list)
         :input (merge (state-to-inputs game-state)
                       {})
         :action (list)
         :weight (list 1)))

;;Terminals in rule-agent:
;;;;hand-value (strength) //check//
;;;;hand-quality
;;;;betting-round
;;;;cards-to-come?
;;;;prefer-action -> with values? -> maybe just weights the action if it's legal
;;;;cost-to-call / current-bet
;;;;pre-hand-type (cur strength + num outs)
;;;;legal actions -> where to put them?
;;;;stacks
;;;;pot / pot odds / giving how much pot odds
;;;;stack and pot percentages
;;;;in-position?
;;;;each action - (bet float) (raise float) (call) (fold) (all-in) (check)
;;;;random/randnormal/randlog int/float/bool
;;;;;;;;;;;;;;;;;;;;;
;;stacks:
;;;;boolean 
;;;;int (hand-quality (0-4), hand-strength (0-7), outs (0-15))
;;;;float (odds (0-1), money (0-200))
;;;;outputs (actions +? probability)
;;;;input
;;;;card
;;;;card vector
;;;;int vector - hand-values community-values lex-compare-vec
;;;;literals = {:type type :literal literal}
;;;;gots to round money to nearest 0.01
;;;;make an action with a float

;;aggressiveness = 
;;game-state to push input function with all the terminals

#_(instructions/def-instruction
  :get-key-state
  ^{:stacks #{:game-state :string}}
  (fn [state]
    (instructions/make-instruction state #(%1 %2) [:keyword :game-state] :string)))

(def ERC
  "ERC-producing instructions"
  [;;random probability
   #(set-type :probability (rand))
   ;;lograndom monetary amount from 1 to initial stack size, rounded to 2 decimal places
   #(set-type :money (utils/round 
                      (utils/rand-log 1 utils/initial-stack) 
                      2))
   ;;number of outs
   #(rand-int 20)
   ;;various parts of game architecture
   #(rand-nth [1 2 3 4 5 13 52])
   #(set-type :weight (utils/rand-log 0.01 100))
   ;;random card
   #_#(set-type :card (rand-nth utils/deck))])


(def action-functions
  "Functions to push an action onto the action stack.\\
   Values are omitted for any action type that cannot take on different values\\
   (money) -> [action-type (value)]"
  [[:bet #(vector ["Bet" %1] %2)
    [:money :weight] :action]
   [:raise #(vector ["Raise" %1] %2)
    [:money :weight] :action]
   [:call #(vector ["Call"] %)
    [:weight] :action]
   [:fold #(vector ["Fold"] %)
    [:weight] :action]
   [:check #(vector ["Check"] %)
    [:weight] :action]
   [:all-in #(vector ["All-In"] %)
    [:weight] :action]])

(def math-functions
  "Mathematical functions"
  [#_[:rand-normal utils/rand-normal
    [:float :float] [:float]]
   [:rand-int rand-int
    [:int] [:int]]
   #_[:rand-float rand
    [:float] [:float]]
   #_[:dot-int utils/dot
    [:vector-int :vector-int] [:int]]
   #_[:lex-compare-vec-int utils/lex-compare-vec
    [:vector-int :vector-int] :boolean]
   #_[:dot-float utils/dot
    [:vector-float :vector-float] [:float]]])

(def card-functions
  "Functions involving cards"
  [[:concat-vector-card concat 
    [:vector-card :vector-card] :vector-card]
   [:ERC-card #(vector (rand-nth utils/values) (rand-nth utils/suits))
    [] :card]
   [:hand-value #(let [v (utils/hand-value %)]
                   [(first v) (flatten v)])
    [:vector-card] [:int :vector-int]]
   [:count-cards count
    [:vector-card] :int]
   #_[:straight-outs #(let [o (utils/straight-outs %)] [(count o) o])
    [:vector-card] [:int :vector-card]]
   #_[:flush-outs #(let [o (utils/flush-outs %)] [(count o) o])
    [:vector-card] [:int :vector-card]]
   #_[:multiple-outs #(let [o (utils/multiple-outs %)] [(count o) o])
    [:vector-card] [:int :vector-card]]
   [:outs #(let [o (utils/outs %)] [(count o) o])
    [:vector-card] [:int :vector-card]]
   [:hand-quality #(.indexOf utils/hand-qualities (utils/hand-quality (take 2 %)))
    [:int] [:vector-card]]
   [:current-cards #(utils/current-cards (:game-state %))
    [:input] :vector-card]])

(def money-prob-functions
  "Functions involving money and probability"
  [;;[:rand rand [] :probability]
   [:lte_prob <= [:probability :probability] :boolean]
   [:gte_prob >= [:probability :probability] :boolean]
   [:eq-epsilon-prob #(< (abs (- %1 %2)) utils/epsilon) [:probability :probability] :boolean]
   [:lte_money <= [:money :money] :boolean]
   [:gte_money >= [:money :money] :boolean]
   [:eq-epsilon-money #(< (abs (- %1 %2)) utils/epsilon) [:money :money] :boolean]
   [:sample-prob #(> % (rand)) [:probability] :boolean]
   [:protected-add-prob #(min 1 (+ %1 %2)) [:probability :probability] :probability]
   [:protected-sub-prob #(max 0 (- %1 %2)) [:probability :probability] :probability]
   [:mult-prob * [:probability :probability] :probability]
   [:odds #(utils/pd %1 (+ %1 %2)) [:money :money] :probability]
   [:equity * [:money :probability] :money]
   [:rev-equity #(if (< (abs %2) utils/epsilon) ##Inf (/ %1 %2)) [:money :probability] :money]
   [:add-money + [:money :money] :money]
   [:protected-sub-money #(max 0 (- %1 %2)) [:money :money] :money]
   ;;div-prob -> float?
   ;;protected-prob just clamping?
   ])

(def weight-functions
  [[:percent-of * [:probability :weight] :weight]
   [:inc-log (partial * Math/E) [:weight] :weight]
   [:dec-log #(/ % Math/E) [:weight] :weight]])

(def predefined-instructions
  "Predefined multi-stack instructions"
  [;;Integer
   :integer_gte
   :integer_lte
   ;;boolean
   :boolean_from_integer
   :exec_while
   'close
   :exec_if])



(defn make-instructions
  [instructions]
  (doseq [[n f i o] instructions]
    (instructions/def-instruction
      n
      ^{:stacks (into #{} (conj i o))
        :name (str n)}
      (fn [state]
        (make-instruction state f i o)))))


(def instructions
  "Instructions for pushgp"
  (do
    (make-instructions (concat money-prob-functions
                               action-functions
                               weight-functions))
    (concat
     (map vector
          (keys (state-to-inputs (headsup/init-game)))
          (repeat 10))
     (map vector
          (instructions/get-stack-instructions #{:boolean :integer :exec})
          (repeat 1))
     (map vector
          ERC
          (repeat 30))
     (map vector
          (map first money-prob-functions)
          (repeat 2))
     (map vector
          (map first action-functions)
          (repeat 20))
     (map vector
          (map first weight-functions)
          (repeat 20))
     (map vector 
          predefined-instructions
          (repeat 1))
     [['close 100]
      [:exec_if 40]
      [:exec_when 40]
      [:integer_gte 5]
      [:integer_lte 5]
      [:total-weight 20]])))


;;Multiple interacting programs:
;; One program to put the opp on a hand - softmaxed probabilities for hands with filters for suit/values
;; Function to take opp hand with probabilty
;; Program to predict opp response to an action - softmaxed probabilities
;; Function to get prediction with probability
;; Program to decide action
;; Function to get action with probability






;;; Game History Functions
;;;;VPIP voluntarily put money into pot = bet/raise times  / total hands
;;;;PFR preflop raise = times raised at least once preflop / total hands
;;;;Aggression factor = times bet/raise / times call
;;;;Fold facing bet = times folded to bet/raise / times facing bet/raise
;;;;SFSB Saw flop from small blind
;;;;WMSF won money when seeing flop
;;;;WTSD went to showdown
;;;;WMSD won money at showdown
;;;;ATSB Attempt to steal blinds Times opened betting with raise in cut-off or button
;;;;FSBS Folded small blind to steal Times folded in small blind to steal / total steal attempts while in small blind
;;;;FBBS Folded big blind to steal

;;; Filter by conditions
;;; Aggregate probabilities

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                 Mutation                  ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



(defn individual-to-agent [individual argmap]
  (let [program (genome/plushy->push (:plushy individual) argmap)]
    (fn [game-state game-history]
      (let [s (interpreter/interpret-program
               program
               (init-push-state game-state)
               (:step-limit argmap))
            f (s :action)
            f (zipmap (map first f) (utils/softmax (map second f)))]
        (utils/prefer-weighted-action f (utils/legal-actions game-state))))))

(defn error-function [argmap opp individual]
  (let [agent (individual-to-agent individual argmap)
        result (:agent (second (headsup/iterate-games-reset
                                [(utils/init-player agent :agent)
                                 (utils/init-player opp :opp)]
                                (:max-games argmap)
                                []
                                :list? true
                                #_:decks #_(:decks argmap))))]
    (assoc individual
           :total-error (- (:mean result))
           :stdev (:stdev result))))

(defn versus [individuals argmap]
  (if (= 1 (count individuals))
    (first individuals)
    (let [[a1 a2] (map #(individual-to-agent % argmap) individuals)
          win (:mean (:a1 (second (headsup/iterate-games-reset
                                   [(utils/init-player a1 :a1)
                                    (utils/init-player a2 :a2)]
                                   (:max-games argmap)
                                   []
                                   :list? true
                                   :decks (:decks argmap)))))]
      (if (>= win 0)
        (first individuals)
        (second individuals)))))

(defn single-elim-selection [pop argmap]
  (loop [num-tournaments (:num-tournaments argmap)
         pop (take (int (Math/pow 2 num-tournaments))(shuffle pop))]
    (if (or (zero? (dec num-tournaments)) (= 1 (count pop)))
      (rand-nth pop)
      (recur (dec num-tournaments)
             (pmap #(versus % argmap) (partition-all 2 pop))))))


(defn round-robin-tournament 
  "Conducts a round-robin tournament within pop and returns a map from individuals
   to number of wins against other individuals.\\
   -> {ind num-wins, ...}"
  [pop argmap]
  (let [wins (zipmap pop (repeat 0))]
    (for [i1 pop
          i2 pop :while (not= i1 i2)]
      (let [win (:mean (headsup/iterate-games-significantly
                        (individual-to-agent i1 argmap)
                        (individual-to-agent i2 argmap)
                        (:max-games argmap)
                        []))]
        (if (>= win 0)
          (update wins i1 inc)
          (update wins i2 inc))))))

(defn round-robin-selection 
  "Conducts a round-robin tournament and selects one of the individuals with the most wins.\\
   -> ind"
  [pop argmap]
  (let [r-r (round-robin-tournament pop argmap)]
    (first (apply max-key second r-r))))



(defn custom-report [pop generation argmap]
  (if (> 1 (mod generation (:generations-per-report argmap)))
    (let [mapped-pop (map #(error-function argmap utils/rule-agent %) pop)
          best (apply min-key :total-error mapped-pop)]
      (clojure.pprint/pprint {:generation            generation
                              :best-plushy           (:plushy best)
                              :best-program          (genome/plushy->push (:plushy best) argmap)
                              :best-error      (:total-error best)
                              :genotypic-diversity   (float (/ (count (distinct (map :plushy pop))) (count pop)))
                              :average-genome-length (float (/ (reduce + (map count (map :plushy pop))) (count pop)))
                              :average-total-error   (float (/ (reduce + (map :total-error mapped-pop)) (count mapped-pop)))})
      (println))
    (println generation)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                Overrides                  ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn get-literal-type
  "If a piece of data is a literal, return its corresponding type if it has one, or
   default type if it doesn't e.g. `:integer`. Otherwise, return `nil`."
  [data]
  (or (when (map? data)
        (ffirst data))
      (when (vector? data)
        (if (empty? data)
          :generic-vector
          (keyword (str "vector_" (name (get-literal-type (propeller.utils/first-non-nil data)))))))
      #?(:clj  (instructions/cls->type (type data))
         :cljs (loop [remaining pred->type]
                 (let [[pred d-type] (first remaining)]
                   (cond
                     (empty? remaining) nil
                     (pred data) d-type
                     :else (recur (rest remaining))))))))

(defn select-parent
  "Selects a parent from the population using the specified method."
  [pop argmap]
  (case (:parent-selection argmap)
    :tournament (selection/tournament-selection pop argmap)
    :lexicase (selection/lexicase-selection pop argmap)
    :single-elim (single-elim-selection pop argmap)
    :round-robin (round-robin-selection pop argmap)))

(defn random-instruction
  "Returns a random instruction from a supplied pool of instructions with weighted probabilities, evaluating
  ERC-producing functions to a constant literal."
  [instructions]
  (let [instruction (first (utils/random-weighted second instructions))]
    (if (fn? instruction)
      (instruction)
      instruction)))

(defn interpret-one-step
  "Takes a Push state and executes the next instruction on the exec stack."
  [state]
  (let [popped-state (state/pop-stack state :exec)
        instruction (first (:exec state))
        literal-type (instructions/get-literal-type instruction)]     ; nil for non-literals
    (cond
      ;;
      ;; Recognize functional instruction or input instruction
      (keyword? instruction)
      (if-let [function (instruction @instructions/instruction-table)]
        (function popped-state)
        (io/handle-input-instruction popped-state instruction))
      ;;
      ;; Recognize constant literal instruction
      literal-type
      (if (= :generic-vector literal-type)
        ;; Empty vector gets pushed on all vector stacks
        (reduce #(update-in % [%2] conj []) popped-state
                [:vector_boolean :vector_float :vector_integer :vector_string])
        (state/push-to-stack popped-state literal-type (if (map? instruction) (utils/sfirst instruction) instruction)))
      ;;
      ;; Recognize parenthesized group of instructions
      (seq? instruction)
      (update popped-state :exec #(concat %2 %1) instruction)
      ;;
      :else
      (do
        (println instruction)
        (throw #?(:clj  (Exception. (str "Unrecognized Push instruction in program: "
                                       instruction))))))))

(defn gp
  "Main GP loop.
On each iteration, it creates a population of random plushies using a mapper
function and genome/make-random-plushy function,
then it sorts the population by the total error using the error-function
and sort-by function. It then takes the best individual from the sorted population,
and if the parent selection is set to epsilon-lexicase, it adds the epsilons to the argmap.
The function then checks if the custom-report argument is set,
if so it calls that function passing the evaluated population,
current generation and argmap. If not, it calls the report function
passing the evaluated population, current generation and argmap.
Then, it checks if the total error of the best individual is less than or equal
to the solution-error-threshold or if the current generation is greater than or
equal to the max-generations specified. If either is true, the function
exits with the best individual or nil. If not, it creates new individuals
for the next generation using the variation/new-individual function and the
repeatedly function, and then continues to the next iteration of the loop. "
  [{:keys [population-size max-generations error-function instructions
           max-initial-plushy-size solution-error-threshold mapper]
    :or   {solution-error-threshold 0.0
           ;; The `mapper` will perform a `map`-like operation to apply a function to every individual
           ;; in the population. The default is `map` but other options include `mapv`, or `pmap`.
           mapper #?(:clj pmap :cljs map)}
    :as   argmap}]
  (loop [generation 0
         population (mapper
                     (fn [_] {:plushy (genome/make-random-plushy instructions max-initial-plushy-size)})
                     (range population-size))
         argmap argmap]
    (let [evaluated-pop (sort-by :total-error
                                 (mapper
                                  (partial error-function argmap (:training-data argmap))
                                  population))
          #_argmap #_(if (= (:parent-selection argmap) :epsilon-lexicase)
                       (assoc argmap :epsilons (selection/epsilon-list evaluated-pop))
                       argmap)]
      (if (:custom-report argmap)
        ((:custom-report argmap) evaluated-pop generation argmap)
        (gp/report evaluated-pop generation argmap))
      (when (zero? (mod generation (:generations-per-save argmap)))
        (spit "gp-single-elim.txt" (with-out-str (clojure.pprint/pprint [(dissoc argmap :decks) evaluated-pop]))))
      (if
       (< generation max-generations)
         (recur (inc generation)
               (if (:elitism argmap)
                 (conj (repeatedly (dec population-size)
                                   #(variation/new-individual evaluated-pop argmap))
                       (first evaluated-pop))       
                 (repeatedly population-size
                             #(variation/new-individual evaluated-pop argmap)))
               (update argmap :decks (partial drop (:max-games argmap))))
        [evaluated-pop argmap]))))

(defn redef-functions
  [f]
  (with-redefs [selection/select-parent select-parent
                instructions/get-literal-type get-literal-type
                propeller.utils/random-instruction random-instruction
                interpreter/interpret-one-step interpret-one-step]
    (f)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;        Experiments         ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#_(redef-functions
#(error-function {:instructions            instructions
                 :error-function          error-function #_(fn [argmap opp individual] (assoc individual :total-error 1))
                 :training-data           utils/calling-station
                 :testing-data            utils/calling-station
                 :max-generations         30
                 :population-size         3
                 :max-initial-plushy-size 5
                 :generations-per-report 1
          ;;:num-tournaments 2
                 :max-games 200
                 :custom-report custom-report
                 :step-limit              100
                 :parent-selection        :tournament
                 :tournament-size         5
                 :umad-rate               0.1
                 :variation               {:umad      0.2
                                           :crossover 0.02}
                 :elitism                 false
                 :decks (repeatedly (fn [] (shuffle utils/deck)))
                 :solution-error-threshold (- ##Inf)}
                utils/calling-station
                {:plushy
                 [:betting-round 3 :integer_eq
                  :exec_if :holecard-winrate {:type :probability :literal 0.5} :gte_prob
                  :exec_if :all-in 'close
                  :fold :total-weight :inc-log :inc-log :check 'close
                  'close
                  :fold :total-weight :inc-log :inc-log :call :total-weight :inc-log :inc-log :check]}))

(take 2 (headsup/iterate-games-reset [(utils/init-player utils/wait-and-bet :wait)
                              (utils/init-player utils/calling-station :call)]
                             20000
                             []
                             :list? true))

(let [[pop argmap] (redef-functions
 #(gp {:instructions            instructions
       :error-function          #_error-function (fn [argmap opp individual] (assoc individual :total-error 1))
       :training-data           utils/rule-agent
       :testing-data            utils/rule-agent
       :max-generations         70
       :generations-per-save 10
       :population-size         30
       :max-initial-plushy-size 150
       :generations-per-report 5
       :num-tournaments 2
       :max-games 20000
       :custom-report custom-report
       :step-limit              300
       :parent-selection        :single-elim
       :tournament-size         5
       :umad-rate               0.2
       :variation               {:umad      0.2
                                 :crossover 0.02}
       :elitism                 true
       :decks (repeatedly (fn [] (shuffle utils/deck)))
       :solution-error-threshold (- ##Inf)}))]
       #_(spit "gp-vs-rule.txt" (with-out-str (clojure.pprint/pprint (dissoc argmap :decks))
                                            (clojure.pprint/pprint pop)
                                            (custom-report pop (:max-generations argmap) argmap))))

(+ 1 2)
;;;;;;;;;;;;;;;;;;
;;; Whaaaat? Rule agent?
;;;;;;;;;;;;;;;;;;
(redef-functions
#(error-function {:instructions            instructions
                 :error-function          error-function #_(fn [argmap opp individual] (assoc individual :total-error 1))
                 :training-data           utils/rule-agent
                 :testing-data            utils/rule-agent
                 :max-generations         50
                 :generations-per-save 10
                 :population-size         30
                 :max-initial-plushy-size 100
                 :generations-per-report 1
          ;;:num-tournaments 2
                 :max-games 100000
                 :custom-report custom-report
                 :step-limit              200
                 :parent-selection        :tournament
                 :tournament-size         5
                 :umad-rate               0.1
                 :variation               {:umad      0.2
                                           :crossover 0.02}
                 :elitism                 false
                 :decks (repeatedly (fn [] (shuffle utils/deck)))
                 :solution-error-threshold (- ##Inf)}
                 utils/rule-agent
                 {:plushy (list {:type :weight, :literal 0.023758307398480123}
                           {:type :weight, :literal 0.023758307398480123}
                           :inc-log
                           :all-in
                           :all-in
                           :inc-log
                           :inc-log
                           {:type :weight, :literal 0.023758307398480123}
                           {:type :weight, :literal 0.023758307398480123}
                           :call
                           {:type :weight, :literal 0.023758307398480123}
                           {:type :weight, :literal 0.023758307398480123}
                           :all-in
                           :opp-stack
                           :call
                           :all-in
                           :raise)}))
(def beats-rule (individual-to-agent {:plushy (list {:type :weight, :literal 0.023758307398480123}
                                                    {:type :weight, :literal 0.023758307398480123}
                                                    :inc-log
                                                    :all-in
                                                    :all-in
                                                    :inc-log
                                                    :inc-log
                                                    {:type :weight, :literal 0.023758307398480123}
                                                    {:type :weight, :literal 0.023758307398480123}
                                                    :call
                                                    {:type :weight, :literal 0.023758307398480123}
                                                    {:type :weight, :literal 0.023758307398480123}
                                                    :all-in
                                                    :opp-stack
                                                    :call
                                                    :all-in
                                                    :raise)} {:instructions            instructions
                                                              :error-function          error-function #_(fn [argmap opp individual] (assoc individual :total-error 1))
                                                              :training-data           utils/rule-agent
                                                              :testing-data            utils/rule-agent
                                                              :max-generations         50
                                                              :generations-per-save 10
                                                              :population-size         30
                                                              :max-initial-plushy-size 100
                                                              :generations-per-report 1
          ;;:num-tournaments 2
                                                              :max-games 100000
                                                              :custom-report custom-report
                                                              :step-limit              200
                                                              :parent-selection        :tournament
                                                              :tournament-size         5
                                                              :umad-rate               0.1
                                                              :variation               {:umad      0.2
                                                                                        :crossover 0.02}
                                                              :elitism                 false
                                                              :decks (repeatedly (fn [] (shuffle utils/deck)))
                                                              :solution-error-threshold (- ##Inf)}))
(redef-functions
#(take 2 (headsup/iterate-games-reset [(utils/init-player beats-rule :agent)
                              (utils/init-player utils/rule-agent :opp)]
                             100000
                             []
                             :list? true)))