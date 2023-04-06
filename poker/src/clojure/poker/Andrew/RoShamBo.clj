(ns poker.Andrew.RoShamBo
  (:require [poker.utils :as utils]
            [propeller.genome :as genome]
            [propeller.push.interpreter :as interpreter]
            [propeller.push.state :as state]
            [propeller.push.instructions :as instructions]
            [propeller.tools.math :as math]
            [propeller.gp :as gp]
            [propeller.selection :as selection]
            [propeller.push.instructions.code :as code]
            [propeller.push.instructions.input-output :as io]))
;;;Rock Paper Scissors

(def actions ["Rock" "Paper" "Scissors"])
(def beats
  "(beats a1) returns the action that beats a1"
  {"Rock" "Paper"
   "Paper" "Scissors"
   "Scissors" "Rock"})

(defn compare-actions 
  "Returns id1 if a1 beats a2, id2 if a2 beats a1, and 0 if they tie"
  [[id1 a1] [id2 a2]]
  (cond (= a1 a2) 0
        (= a1 "Surrender") id2
        (= a2 "Surrender") id1
        (= a2 (beats a1)) id2
        :else id1))

(defn stdev-binary
  "Given the mean of a random variable which can only be 0 or 1,
   computes its standard deviation"
  [mean]
  (Math/sqrt (+ (* (utils/square mean) (- 1 mean)) 
                   (* (utils/square (- 1 mean)) mean))))

(defn init-game [players]
  {:players players
   :action-history []})

(defn init-player 
  "Player composed of an agent and an id
   Agent takes game-history and action-history and outputs an action"
  [id agent]
  {:id id
   :agent agent
   :wins 0})

(defn random-agent 
  "Randomly chooses an action"
  [action-history game-history]
  (rand-nth actions))

(defn beat-player [action-history game-history]
  (if (empty? action-history)
    (rand-nth actions)
    (let [a (second (last (last action-history)))]
      (beats a))))

(defn beat-beat-player [action-history game-history]
  (if (empty? action-history)
    (rand-nth actions)
    (let [a (second (last (last action-history)))]
      (beats (beats a)))))

(defn get-actions [players action-history game-history]
  (map #(vector (:id %)
                ((:agent %)
                 action-history
                 game-history))
       players))

(defn play-game 
  "Returns updated players and game-history"
  [players game-history]
  (let [action-history []]
    (loop [action-history action-history
           i 0]
        (let [actions (get-actions players action-history game-history)
            winner (if (= 10 i)
                     (ffirst actions)
                     (apply compare-actions actions))]
        (if (= 0 winner)
            (recur (conj action-history actions) (inc i))
            [(map #(if (= winner (:id %)) 
                     (update % :wins inc) 
                     %) 
                  players)
             (conj game-history
                  {:action-history (conj action-history actions)
                   :net-gain (into [] (map #(let [id (:id %)] 
                                              (vector id
                                                      (if (= winner id) 1 0)))
                                           players))
                   :winner winner})])))))

(defn iterate-games [players num-games game-history]
  (if (zero? num-games)
    [players game-history]
    (let [[players game-history] (play-game players game-history)]
      (recur players
             (dec num-games)
             game-history))))

(defn stats
  "Returns the mean and stdev of the probability of agent1 beating agent2"
  [agent1 agent2 num-games]
  (let [[[p1]] (iterate-games [(init-player :p1 agent1)
                             (init-player :p2 agent2)]
                            num-games
                            [])
        mean (float (/ (:wins p1) num-games))
        stdev (stdev-binary mean)]
    {:mean mean 
     :stdev stdev
     :CI95 [(- mean (* 1.96 (/ stdev (Math/sqrt num-games))))
            (+ mean (* 1.96 (/ stdev (Math/sqrt num-games))))]}))

(defn better? [p1 p2 num-games]
  (let [[[p1 p2]] (iterate-games [p1 p2] num-games [])]
    (if (> (:wins p1) (:wins p2))
      (assoc p1 :wins 0)
      (assoc p2 :wins 0))))


;;encode input as an integer saying "this is the last action"
;;have literals to put onto the action stack


(def empty-state
  (assoc state/empty-state
         :keyword (list)
         :action (list) ;;actions
         ;;output stack: actions
         ;;Push an action onto the output with default weight 1
         :card (list)
         :vector-vard (list);;hands
         :me nil))

(instructions/def-instruction
  :prev-action
  ^{:stacks #{:action}}
  (fn [state]
    (io/handle-input-instruction state :in1)))

(instructions/def-instruction
  :beats
  ^{:stacks #{:action}}
  (fn [state]
    (instructions/make-instruction 
     state 
      #(beats %) 
      [:string] 
      :string)))

(instructions/def-instruction
  :tied?
  ^{:stacks #{:bool}}
  (fn [state]
    (io/handle-input-instruction state :in2)))

(instructions/def-instruction
  :rand-action
  ^{:stacks #{:bool}}
  (fn [state]
    (instructions/make-instruction
     state
     #(rand-nth actions)
     []
     :string)))

(def instructions 
  (list :in1
        :in2
        :beats
        :exec_if
        :rand-action
        'close))

(defn individual-to-agent [individual argmap]
  (let [program (genome/plushy->push (:plushy individual) argmap)]
      (fn [action-history game-history]
        (let [s 
         (interpreter/interpret-program
          program
          (assoc state/empty-state 
                 :input 
                 {:in1 (if-let [l (last action-history)]
                         (second (second l))
                         nil)
                  :in2 (< 0 (count action-history))})
          (:step-limit argmap))
              a (state/peek-stack s :string)]
          #_(clojure.pprint/pprint s)
          (if (contains? (set actions) a)
            a
            "Surrender")))))


(defn error-function [argmap opp individual]
  (let [agent (individual-to-agent individual argmap)]
    (assoc individual
           :total-error (:mean (stats opp agent 1000)))))



#_(clojure.pprint/pprint
 (interpreter/interpret-program 
 (list
  :rand-action
  :in2
  :in1
  :beats
  :beats
  :exec_if
  :in2
  :rand-action) 
 (assoc state/empty-state 
        :action (list)
        :integer (list 1 2)
        :input {:in1 "Scissors"
                :in2 true}) 
 100))

#_(clojure.pprint/pprint
 (iterate-games [(init-player :p0 beat-player)
                (init-player :p1 (individual-to-agent {:plushy  (list
                                                                 :rand-action
                                                                 :in2
                                                                 :in1
                                                                 :beats
                                                                 :beats
                                                                 :exec_if
                                                                 :in2
                                                                 :rand-action)}
                                                      {:step-limit 100}))]
               3
               []))

#_(clojure.pprint/pprint (stats beat-player 
                                (individual-to-agent {:plushy (list :rand-action :in1 :beats :beats)}
                                                     {:step-limit 100})
         1000))

(defn opp [& args]
  (condp < (rand)
         0.5 "Rock"
         0.25 "Scissors"
         "Paper"))



(defn versus [individuals argmap]
  (if (= 1 (count individuals))
    (first individuals)
    (let [[a1 a2] (map #(individual-to-agent % argmap) individuals)
          winrate (:mean (stats a1 a2 (:num-games argmap)))]
      (if (>= winrate 0.5)
        (first individuals)
        (second individuals)))))

(defn single-elim [pop argmap] 
  (loop [num-tournaments (:num-tournaments argmap)
         pop pop]
    (println num-tournaments)
    (if (or (zero? (dec num-tournaments)) (= 1 (count pop)))
      (rand-nth pop)
      (recur (dec num-tournaments)
             (map #(versus % argmap) (partition-all 2 pop))))))

(defn round-robin [pop argmap]
  (let [wins (zipmap pop (repeat 0))]
    (for [i1 pop
          i2 pop :while (not= i1 i2)]
      (let [winrate (:mean (stats (individual-to-agent i1 argmap)
                                  (individual-to-agent i2 argmap)
                                  (:num-games argmap)))]
        (if (>= winrate 0.5)
          ()
          ())))))

(defn select-parent
  "Selects a parent from the population using the specified method."
  [pop argmap]
  (case (:parent-selection argmap)
    :tournament (selection/tournament-selection pop argmap)
    :lexicase (selection/lexicase-selection pop argmap)
    #_:epsilon-lexicase #_(selection/epsilon-lexicase-selection pop argmap)
    :single-elim (single-elim pop argmap)
    :round-robin ()))

(defn custom-report [pop generation argmap]
  (when (> 1 (mod generation (:generations-per-report argmap)))
    (let [mapped-pop (map #(error-function argmap beat-player %) pop)
          best (apply min-key :total-error mapped-pop)]
      (println best)
      (clojure.pprint/pprint {:generation            generation
                              :best-plushy           (:plushy best)
                              :best-program          (genome/plushy->push (:plushy best) argmap)
                              :best-error      (:total-error best)
                              :genotypic-diversity   (float (/ (count (distinct (map :plushy pop))) (count pop)))
                              :behavioral-diversity  (float (/ (count (distinct (map :behaviors pop))) (count pop)))
                              :average-genome-length (float (/ (reduce + (map count (map :plushy pop))) (count pop)))
                              :average-total-error   (float (/ (reduce + (map :total-error mapped-pop)) (count mapped-pop)))})
      (println))))


(defn redef-functions 
  [f]
  (with-redefs [selection/select-parent select-parent]
    (f)))

(redef-functions
 #(gp/gp {:instructions            instructions
          :error-function          (fn [argmap opp individual] (assoc individual :total-error 1))
          :training-data           beat-player
          :testing-data            beat-player
          :max-generations         5
          :population-size         10
          :max-initial-plushy-size 7
          :generations-per-report 4
          :num-tournaments 2
          :num-games 1000
          :custom-report custom-report
          :step-limit              10
          :parent-selection        :single-elim
          :tournament-size         5
          :umad-rate               0.1
          :variation               {:umad      1.0
                                    :crossover 0.0}
          :elitism                 false}))