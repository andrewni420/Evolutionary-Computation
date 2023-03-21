(ns poker.gp
  (:require [propeller.push.state :as state]
            [poker.utils :as utils] 
            [poker.headsup :as headsup]
            [propeller.push.instructions :as instructions]
            [propeller.push.instructions.numeric :as numeric] :reload
            [propeller.push.instructions.code :as code] :reload
            [propeller.push.instructions.bool :as bool] :reload
            [propeller.tools.math :as math]
            [propeller.genome :as genome]
            [propeller.gp :as gp]
            [propeller.push.interpreter :as interpreter]))

(instructions/def-instruction
  :nort
  ^{:stacks #{:nort}}
  (fn [state]
    (instructions/make-instruction state #(and %1 %2) [:nort :nort] :nort)))

(instructions/get-stack-instructions #{:nort})
state/empty-state
(interpreter/interpret-program (list :nort) (assoc state/empty-state :nort [true false]) 100)

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
                       :community] %) (keys (headsup/init-game []))))
(def game-history-kws
  "Keywords for accessing information about a particular game instance in history"
  [:hands :action-history :community :net-gain])

(instructions/def-instruction
  :get-state-visible
  ^{:stacks #{:input :card}}
  (fn [state]
    (instructions/make-instruction state #(%1 %2) [:keyword :game-state] :string)))

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
  {:type type
   :literal literal})

;;; Game State Functions
(defn state-to-inputs 
  [game-state]
  (let [{current-player :current-player
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
     :SC-money-needed (set-type :money
                                (nth SC-numbers 2))
     :betting-round (.indexOf utils/betting-rounds betting-round)
     :num-cards-to-come (- 5 visible)
     :cost-to-call (set-type :money
                             (max 0.0 (- current-bet my-bet)))
     :my-stack (set-type :money 
                         (:money (players current-player)))
     :opp-stack (set-type :money
                          (:money (players (- 1 current-player))))
     :pot (set-type :money pot)
     :in-position? (zero? current-player);;boolean
     :prob-better-hand (set-type :probability
                                 (utils/prob-better-hand-rough visible my-hand));;boolean
     :prob-catch-outs (set-type :probability
                                (float (/ (count (utils/outs (concat my-hand visible)))
                                (- 52 2 (count visible)))))}))

(defn init-push-state 
  "Uses game-state to initialize inputs of push states\\
   Creates action, probability, and money stacks"
  [game-state]
  (assoc state/empty-state
         :money (list)
         :probability (list)
         :input (state-to-inputs game-state)
         :action (list)))

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
   (constantly (set-type :probability (rand)))
   ;;lograndom monetary amount from 1 to initial stack size, rounded to 2 decimal places
   (constantly (set-type :money
                         (utils/round
                          (Math/exp
                           (rand
                            (Math/log utils/initial-stack)))
                          2)))
   ;;number of outs
   #(rand-int 20)
   ;;various parts of game architecture
   #(rand-nth [1 2 3 4 5 13 52])
   ;;random card
   #_(constantly (set-type :card (rand-nth utils/deck)))])

(def action-functions
  "Functions to push an action onto the action stack.\\
   Values are omitted for any action type that cannot take on different values\\
   (money) -> [action-type (value)]"
  [[:bet #(vector "Bet" %)
    [:money] :action]
   [:raise #(vector "Raise" %)
    [:money] :action]
   [:call #(vector "Call")
    [] :action]
   [:fold #(vector "Fold")
    [] :action]
   [:check #(vector "Check")
    [] :action]
   [:all-in #(vector "All-In")
    [] :action]])

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
   [:odds #(/ %1 (+ %1 %2)) [:money :money] :probability]
   [:equity * [:money :probability] :money]
   [:rev-equity / [:money :probability] :money]
   [:add-money + [:money :money] :money]
   [:protected-sub-money #(max 0 (- %1 %2)) [:money :money] :money]
   ;;div-prob -> float?
   ;;protected-prob just clamping?
   ])

(def predefined-instructions
  "Predefined multi-stack instructions"
  [;;Integer
   :integer_gte
   :integer_lte
   ;;boolean
   :boolean_from_integer
   :exec_if
   :exec_while])




(defn make-instructions
  [instructions]
  (doseq [[n f i o] instructions]
    (instructions/def-instruction
      n
      ^{:stacks (into #{} (conj i o))}
      (fn [state]
        (make-instruction state f i o)))))


(def instructions
  "Instructions for pushgp"
  (concat (instructions/get-stack-instructions #{:boolean :integer :exec}) 
          ERC
          (make-instructions (concat money-prob-functions 
                                     action-functions))))

;;Multiple interacting programs:
;; One program to put the opp on a hand - softmaxed probabilities for hands with filters for suit/values
;; Function to take opp hand with probabilty
;; Program to predict opp response to an action - softmaxed probabilities
;; Function to get prediction with probability
;; Program to decide action
;; Function to get action with probability




;;override
(defn get-literal-type
  "OVERRIDE for typed inputs\\
   If a piece of data is a literal, return its corresponding stack name
   e.g. `:integer`. Otherwise, return `nil`."
  [data]
  (or (when (and (map? data) (:type data) (:literal data))
        (:type data))
   (when (vector? data)
        (if (empty? data)
          :generic-vector
          (keyword (str "vector_" (name (get-literal-type (u/first-non-nil data)))))))
      #?(:clj  (cls->type (type data))
         :cljs (loop [remaining pred->type]
                 (let [[pred d-type] (first remaining)]
                   (cond
                     (empty? remaining) nil
                     (pred data) d-type
                     :else (recur (rest remaining))))))))


;;; Game History Functions
;;; Filter by conditions
;;; Aggregate probabilities

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                 Mutation                  ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn init-state
  "Initializes empty state with the legal actions"
  [game-state game-history]
  (assoc empty-state
         :legal-actions (headsup/legal-actions game-state)
         :input (merge (state-to-inputs game-state)
                       {})))