(ns poker.headsup
  (:require [poker.utils :as utils]
            [propeller.genome :as genome]
            [propeller.push.interpreter :as interpreter]
            [propeller.push.state :as state]
            [propeller.tools.math :as math]
            [propeller.gp :as gp]
            [propeller.push.instructions :as instructions]
            [clojure.set :as set])
  (:gen-class))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;        Heads-Up Poker Game Engine         ;;;
;;; Values represented in units of big blinds ;;;
;;;   Multiple random decks played for each   ;;;
;;;              rotation of players          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; repeatedly make deck and shuffle players -> for each rotation play game
;;For betting, continue around until each player has put in either the same amount
;;or has folded. Or is all-in
;;each player :bet = amount or "fold"




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;          Agent Initialization             ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;             Agent Evaluation              ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#_(defn make-move
    "Makes a [type spent] move using the current bet and the agent's intended bet value and tolerance.
   If the current bet is too high, agent checks or folds
   If the current bet plus the reraise limit is too low, agent bets or raises
   Otherwise the bet is good enough and the agent checks or calls
   The value is the value of the bet for the next person
   Amount spent is either bet or value"
    [move bet min-raise]
    (let [[value tolerance] move]
      (cond
        (> bet (+ value tolerance)) (if (= bet 0)
                                      ["Checks" bet]
                                      ["Fold" bet])
        (< (+ bet min-raise) (- value tolerance)) (if (= bet 0)
                                                    ["Bet" value]
                                                    ["Raise" value])
        :else (if (= bet 0)
                ["Check" bet]
                ["Call" bet]))))
;;actions: fold, bet, check, call, raise, all-in
;;Agent takes in action history and current gamestate and returns an action
;;action history: [[1 sb] [2 bb] [3 [fold 0]] [4 [call 1]] [1 [raise 2]]]
;;gamestate: community cards, hand, pot size, betsize (cost to call), minraise
;;state: hands, bb/sb/button, pot size, bet value for each player, fold for each player, min raise, min bet, betting round, players, 
;;In heads-up (1v1) poker, small blind acts first preflop then last for future rounds
;;
(defn init-game
  "Removes players with no money and initializes the poker game. In the context of this game, players are 
   referenced by their index, not their id. In the history, players are referenced by their index.
   Hands - two cards per person
   Community - Five cards revealed as 3 in the flop, 1 at the turn, and 1 at the river
   Visible - Cards already revealed
   Bet-values - the amount which players have bet in this round. Resets every round.
   Current-bet - the amount which players need to match to stay in the round. Resets every round.
   Pot - the total pot size
   Active-players - players who have not yet folded
   Min-raise - The minimum amount by which people must raise
   Min-bet - The minimum bet is 1bb
   Betting-round - The round of betting currently happening
   Players - the players in the game
   Current-player - the player whose turn it is to act
   num-players - the number of players
   game-over - whether the game has terminated
   prev-bettor - the index of the last person to bet"
  [players]
  (let [deal (utils/deal-hands 2)]
    {:hands (vec (:hands deal))
     :community (vec (:community deal))
     :visible []
     :bet-values [0.0 0.0]
     :current-bet 0.0
     :pot 0.0
     :active-players [0 1]
     :min-raise 1.0
     :min-bet 1.0
     :betting-round "Pre-Flop"
     :players players
     :current-player 0
     :num-players 2
     :game-over false
     :prev-bettor -1
     :action-history [[]]}))


(defn pre-flop-bb? [game-state]
  (let [{betting-round :betting-round
         current-player :current-player
         current-bet :current-bet} game-state]
    (and (= betting-round "Pre-Flop")
         (= current-player 1)
         (= current-bet 1.0))))





(defn next-player
  "Returns the next player to take an action"
  [game-state]
  (let [{current-player :current-player
         n :num-players} game-state]
    (assoc game-state
           :current-player (mod (inc current-player) n))))








#_(defn round-over-checkall
  "When all active players have contributed the same amount to the pot,
   the round is over."
  [game-state]
  (let [{bet-values :bet-values
         active-players :active-players} game-state]
    (apply = (map #(nth bet-values %) active-players))))













;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Game Engine from single move to full game ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;Single turn
(defn legal-actions
  "The possible actions and their conditions are as follows:
   Fold - always possible, but not allowed when check is possible
   Check - only possible when no one has betted
   Call - Only possible when at least one person has betted
   Raise - Must raise by at least the previous bet or amount by which bet was raised
   Bet - Only possible when no one has betted. Must be at least 1bb
   All-In - always possible
   Returns a [type least most] vector."
  [game-state]
  (let [{min-bet :min-bet
         min-raise :min-raise
         bet-values :bet-values
         current-bet :current-bet
         current-player :current-player
         players :players} game-state
        p (players current-player)
        {money :money} p
        call-cost (- current-bet (bet-values current-player))]
    (#(cond 
        (utils/in? % ["Check" 0.0 0.0]) %
        (utils/in? % ["All-In" money money]) %
        :else (concat % [["Fold" 0.0 0.0]])) 
     (concat [["All-In" money money]]
            (cond 
              (zero? money) []
              (every? zero? bet-values)
              (concat [["Check" 0.0 0.0]]
                      (when (>= money min-bet)
                        [["Bet" min-bet (dec money)]]))
              :else
              (concat []
                      (if (pre-flop-bb? game-state)
                        [["Check" 0.0 0.0]]
                        (when (>= money call-cost)
                          [["Call" call-cost call-cost]]))
                      (let [amount (- (+ current-bet min-raise)
                                      (bet-values current-player))]
                        (when (>= money amount)
                          [["Raise" amount (dec money)]]))))))))

(defn make-move [game-state game-history]
  (let [{current-player :current-player
         players :players} game-state
        player (players current-player)]
    ((:agent player) game-state
                     game-history
                     (:money player)
                     (legal-actions game-state))))


(defn check-active-players [game-state]
  (let [{players :players
         active-players :active-players
         pot :pot} game-state
        i (first active-players)]
    (if (= 1 (count active-players))
      (let [player (utils/add-money (players i) pot)]
        (assoc game-state
               :players (assoc players i player)
               :game-over true))
      game-state)))

(defn parse-action
  "Updates game state based on action. Does not check for legality of action"
  [player-number action game-state]
  (let [{current-player :current-player
         active-players :active-players
         pot :pot
         bet-values :bet-values
         current-bet :current-bet
         players :players
         action-history :action-history
         min-raise :min-raise} game-state
        [type amount] action
        players (update players current-player (fn [p] (update p :money #(- % amount))))
        new-state (assoc (next-player game-state)
                         :action-history (conj (into [] (drop-last action-history))
                                               (conj (last action-history)
                                                     [(:id (players current-player))
                                                      action]))
                         :players players)]
    (condp utils/in? type
      ["Fold"] (check-active-players (assoc new-state
                                            :active-players (remove (partial = player-number)
                                                                    active-players)))
      ["Check"] new-state
      ["Raise"] (assoc new-state
                       :bet-values (update bet-values current-player (partial + amount))
                       :pot (+ pot amount)
                       :current-bet (+ amount (bet-values current-player))
                       :min-raise (- (+ amount (bet-values current-player)) current-bet))
      ["Bet" "All-In" "Call"] (assoc new-state
                                     :bet-values (update bet-values current-player (partial + amount))
                                     :pot (+ pot amount)
                                     :current-bet (+ amount (bet-values current-player))
                                     :min-raise (max min-raise amount)))))



(defn blind
  "Pays blind or goes all in depending on money available. Assumes nonzero money
   Returns [updated-player action]"
  [player value]
  (let [{money :money} player]
    (if (< money value)
      [(assoc player :money 0) ["All-In" money]]
      [(assoc player :money (- money value)) ["Bet" value]])))

(defn pay-blinds
  "Small blind and big blind bet or go all in if they don't have enough money"
  [game-state]
  (let [{players :players
         num-players :num-players
         bet-values :bet-values} game-state
        sb 0
        bb 1
        [sb-player sb-action] (blind (players sb) 0.5)
        [bb-player bb-action] (blind (players bb) 1.0)]
    (assoc game-state
           :players (assoc players
                           sb sb-player
                           bb bb-player)
           :pot (+ (second sb-action)
                   (second bb-action))
           :current-player (mod (inc bb) num-players)
           :bet-values (assoc bet-values
                              sb (second sb-action)
                              bb (second bb-action))
           :prev-bettor bb
           :current-bet 1.0)))




(defn reset-action
  "Big blind acts first in all rounds except pre-flop"
  [game-state]
  (assoc game-state
         :current-player 1
         :current-bet 0.0
         :prev-bettor -1
         :min-raise 1.0
         :bet-values [0.0 0.0]
         :action-history (conj (:action-history game-state) [])))

(defn all-in? [game-state]
  (and (every? zero? (:bet-values game-state))
       (some zero? (map :money (:players game-state)))))

(defn round-over-checkone
  "When betting goes back to the player who first made the bet,
   the round is over.
   Faster than checkall"
  [game-state]
  (let [{bet-values :bet-values
         current-player :current-player
         current-bet :current-bet} game-state]
    (cond
      (every? zero? bet-values) (= 1 current-player)
      (pre-flop-bb? game-state) false
      (all-in? game-state) true
      :else (= (bet-values current-player) current-bet))))




;;Single round
#_(defn bet-round
  "Runs betting for one round"
  [game-state game-history]
  (if (or (:game-over game-state)
          (round-over-checkone game-state))
    game-state
    (let [{current-player :current-player} game-state
          p1-move (make-move game-state
                             game-history)
          new-state (parse-action current-player
                                  p1-move
                                  game-state)]
      (recur new-state
             game-history))))

(defn bet-round
  "Runs betting for one round"
  [game-state game-history]
  (let [{current-player :current-player} game-state
        p1-move (make-move game-state
                           game-history)
        new-state (parse-action current-player
                                p1-move
                                game-state)]
    (println (:action-history new-state))
    (if (or (:game-over new-state)
            (round-over-checkone new-state))
      new-state
      (recur new-state
             game-history))))

(defn showdown
  "Heads-up showdown. Does not consider side pots.
   Computes the people with the highest hands and divides the pot among them"
  [game-state]
  (let [{hands :hands
         community :community
         pot :pot
         players :players
         active-players :active-players} game-state
        player-hands (map #(vector %
                                   (concat (nth hands %)
                                           community))
                          active-players)
        winners (utils/highest-hand player-hands)
        updated-players (reduce #(update %1
                                         (first %2)
                                         (fn [p]
                                           (utils/add-money p (/ pot (count winners)))))
                                players
                                winners)]
    (assoc game-state
           :game-over true
           :players updated-players
           :betting-round "Showdown")))

(defn next-round
  "Checks to see if all players but one have folded, then runs showdown or
   reveals the next card and proceeds to the next betting round"
  [game-state]
  (let [{betting-round :betting-round
         community :community} game-state
        reset-state (check-active-players (reset-action game-state))]
    (if (:game-over reset-state)
      reset-state
      (condp = betting-round
        "Pre-Flop" (assoc reset-state
                          :betting-round "Flop"
                          :visible (take 3 community))
        "Flop" (assoc reset-state
                      :betting-round "Turn"
                      :visible (take 4 community))
        "Turn" (assoc reset-state
                      :betting-round "River"
                      :visible community)
        "River" (showdown game-state)))))

;;Single game
(defn bet-game 
  "Runs betting from initialization of game until showdown"
  [game-state game-history]
  (let [new-state (bet-round game-state game-history)]
    (if (:game-over new-state)
      new-state
      (let [next-round (next-round new-state)]
        (if (:game-over next-round)
          next-round
          (recur next-round game-history))))))

(defn state-to-history [old-state new-state]
  (let [{action-history :action-history
         hands :hands
         new-players :players
         community :community} new-state
        {old-players :players} old-state]
    (assoc {}
           :hands (mapv #(vector (:id %1) %2) new-players hands)
           :playerIDs (mapv :id new-players)
           :action-history action-history
           :community community
           :net-gain (mapv #(vector (:id %2) (- (:money %1) (:money %2))) new-players old-players))))


(defn play-game 
  "Initializes and plays a game of poker, returning the updated players
   and updated game-history
   NOT YET IMPLEMENTED"
  [players game-history]
  (let [game-state (init-game players)
        new-state (bet-game (pay-blinds game-state) game-history)]
    [(:players new-state) 
     (conj game-history (state-to-history game-state new-state))]))


(def always-fold (constantly ["Fold" 0]))
(def loose-passive (fn [history state money id]
                     (let [{bet-values :bet-values
                            current-bet :current-bet} state
                           diff (- current-bet (bet-values id))]
                       (cond (= 0 current-bet) ["Check" 0]
                             (> diff money) ["All-In" money]
                             :else ["Call" diff]))))
(defn random-agent [game-state game-history money actions]
  (let [[type min max] (rand-nth actions)]
    (if (< (rand) 0.0)
    (let [types (map first actions)
          type (cond (utils/in? types "Call") "Call"
                     (utils/in? types "Check") "Check"
                     (utils/in? types "Fold") "Fold")]
    (into [] (take 2 (first (filter #(= type (first %)) actions)))))
    (vector type min #_(+ min (math/round (rand (- max min))))))))



(defn iterate-games
  "Plays num-games hands of poker with players switching from sb to bb every hand.
   An even number of hands ensures balanced play.
   Returns the players and history after num-games are reached or a player has no more money"
  [players num-games game-history]
  (if (or (zero? num-games)
          (some zero? (map :money players)))
    [players game-history]
    (let [[players history] (play-game players game-history)]
      (recur (into [] (reverse players)) (dec num-games) history))))





(apply (fn [p h] (vector p (take 10 (filter #(and (not= 1 (count (:action-history %)))
                                        (utils/in? (flatten (:action-history %)) "Bet")
                                        (> (abs (second (first (:net-gain %)))) 10)
                                        #_(not (utils/in? (flatten (:action-history %)) "Fold"))) h))))
 (iterate-games [(utils/init-player random-agent :p0) 
                       (utils/init-player utils/rule-agent :p1)]
             10 []))

 (iterate-games [(utils/init-player random-agent :p0)
                 (utils/init-player utils/rule-agent :p1)]
                10 [])

(defn kek [& {:keys [this]
              :or {this 1}}]
  (if (= this 4)
    this
    (recur (inc this))))
(kek)

(loop [i 0] (let [state (bet-game (pay-blinds (init-game [(utils/init-player utils/rule-agent :p0) (utils/init-player random-agent :p1)])) [])]
              (if (or (= 1 (count (:action-history state)))
                      (not (utils/in? (flatten (:action-history state)) "Bet"))
                      (utils/in? (flatten (:action-history state)) "Fold"))
                (recur (inc i))
                (do (println i)
                  state))))

(legal-actions (init-game [{:money 0} {:money 0}]))

;;What follows b is the total bet for the round, not how much a person put in
(def move-map {"Call" "c",
               "Check" "k",
               "Bet" "b",
               "All-In" "b",
               "Raise" "b"})

(defn encode-round [round-history bet-values]
  (let [round-history (map second round-history)]
    (loop [s ""
           bet-values bet-values
           cur-player 0
           round-history round-history]
      (if (empty? round-history)
        s
        (let [new-bets (update bet-values cur-player (partial + (second (first round-history))))
              m (move-map (ffirst round-history))]
          (recur (if (= m "b")
                   (str s m (int (* 100 (new-bets cur-player))))
                   (str s m))
                 new-bets
                 (- 1 cur-player)
                 (rest round-history)))))))

(defn encode-action-history [action-history]
  (clojure.string/join "/" (map-indexed #(encode-round %2 (if (= 0 %1) [0.5 1.0] [0.0 0.0])) action-history)))

(defn decode-round 
  "Not implemented yet"
  [round-actions bet-values]
  ())

(defn decode-action-history [action-history]
  (let [inverse-map (clojure.set/map-invert move-map)
        round-actions (clojure.string/split action-history #"/")]
    (map-indexed #(decode-round %2 (if (= 0 %1) [0.5 1.0] [0.0 0.0])) round-actions)))

(encode-action-history [[[:p0 ["Call" 0.5]] [:p1 ["Check" 0.0]]]
                        [[:p1 ["Check" 0.0]] [:p0 ["Check" 0.0]]]
                        [[:p1 ["Check" 0.0]] [:p0 ["Check" 0.0]]]
                        [[:p1 ["Check" 0.0]] [:p0 ["Bet" 1.0]] [:p1 ["All-In" 95.0]] [:p0 ["Call" 94.0]]]],
)






;; An individual is something which takes in the history, state, money and id and outputs a move
;;takes game-state, current money, current options

#_(defn choose-move-player
    "Player chooses move based on agent, money, and legal bets
   Returns [updated-player [move-type move-value]]"
    [player bet min-raise]
    (let [{money :money agent :agent} player
          [value tolerance] (choose-move-agent agent)]
      (if (or (= 0 money) (>= value money))
        [(assoc player
                :money 0
                :bet "All-In") ["All-In" money]]
        (let [move (make-move [value tolerance] bet min-raise)]
          [(assoc player
                  :money (- money (second move))
                  :bet (second move)) move]))))


#_(defn bet [hands players init]
    (loop [pot 0
           bet-size init
           min-raise 0
           players (map #(assoc % :bet 0) players)]
      (let [p (first players)]
        (if (or (= (:bet p) init) (= (:bet p) "All-In"))
          [pot players]
          (let [[p [movetype value]] (choose-move-player p bet-size min-raise)]
            (recur (+ pot value)
                   (max value bet-size)
                   (max min-raise (- value bet-size))
                   (if (= movetype "Fold")
                     (rest players)
                     (concat (rest players) [p]))))))))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                 Mutation                  ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; (defn -main
;;   "I don't do a whole lot ... yet."
;;   [& args]
;;   (println "Hello, World!"))


