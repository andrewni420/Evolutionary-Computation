(ns poker.core
  (:require [poker.utils :as utils]
            [propeller.genome :as genome]
            [propeller.push.interpreter :as interpreter]
            [propeller.push.state :as state]
            [propeller.tools.math :as math]
            [propeller.gp :as gp]
            [propeller.push.instructions :as instructions])
  (:gen-class))

(instructions/def-instruction
  :nort
  ^{:stacks #{:nort}}
  (fn [state]
    (instructions/make-instruction state #(and %1 %2) [:nort :nort] :nort)))

(instructions/get-stack-instructions #{:nort})

(interpreter/interpret-program (list :nort) (assoc state/empty-state :nort [true false]) 100)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                Poker Core                 ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



#_(let [{h :hands c :community} (deal-hands 5)]
  (clojure.pprint/pprint {:hands h :community c})
  (showdown
   (map vector
        (range)
        (map (partial concat
                      c)
             h))))


(defn maxes 
  "Returns all individuals with maximum value of (key %)"
  ([key coll] 
 (let [m (key (apply max-key key coll))] 
   (filter #(= m (key %)) coll)))
  ([coll] (maxes identity coll)))



#_(hand-value [[7 "Hearts"]
            [6 "Hearts"]
            [8 "Spades"]
            [4 "Hearts"]
            [8 "Hearts"]
            [8 "Clubs"]
            [6 "Clubs"]])

#_(deal-hands 4)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;              Poker Gameplay               ;;;
;;; Values represented in units of big blinds ;;;
;;;   Multiple random decks played for each   ;;;
;;;              rotation of players          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; repeatedly make deck and shuffle players -> for each rotation play game
;;For betting, continue around until each player has put in either the same amount
;;or has folded. Or is all-in
;;each player :bet = amount or "fold"


(defn initialize-player [agent]
  {:agent agent
   :money 100})

(defn play-game 
  "Deck and a vector of players"
  [deck players]
  (let [{hands :hands 
         community :community} (utils/deal-hands (count players))]
    ()))

(defn pre-flop 
  "Preflop play. Returns the size of the pot and the agents still in play"
  [hands players]
  (let [[sb bb] players]
    ()))



(defn flop [hands community players])

(defn turn [hands community players])

(defn river [hands community players])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;          Agent Initialization             ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;             Agent Evaluation              ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn update-state 
  [])

(defn choose-move-agent
  [agent])


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
  "Removes players with no money and initializes the poker game"
  [players]
  (let [players (filter #(> (:money %) 0) players)
        n (count players)
        deal (utils/deal-hands n)]
    {:hands (vec (:hands deal))
     :community (vec (:community deal))
     :visible []
     :bet-values (vec (repeat n 0))
     :current-bet 1
     :pot 0
     :active-players (vec (range n))
     :min-raise 1
     :min-bet 1
     :betting-round "Pre-Flop"
     :players players
     :current-player 1
     :numPlayers n
     :game-over false
     }))

;what goes into 2p history? player moves, player cards, final outcomes
;[game] [game] [game]
;[game] = {:action-sequence [[preflop: [id [action]]... ]... ]
;          :hands
;          :community cards [flop: [] ...]
;          :pot
;          :net outcome [[id amount]...]
;          }
;player = money, id = "p1", agent
;action = [type amount]

(defn legal-actions [game-state]
  (let []
    ()))

(defn next-player
  "Returns the next player to take an action"
  [game-state]
  (let [{active-players :active-players
         current-player :current-player
         n :numPlayers} game-state]
    (assoc game-state
           :current-player (apply min-key #(mod (+ % (- n current-player 1)) n) active-players))))

(defn parse-action 
  "Updates game state based on action. Does not check for legality of action"
  [player-number action game-state]
  (let [{current-player :current-player
         active-players :active-players
         pot :pot
         bet-values :bet-values
         current-bet :current-bet} game-state
        [type amount] action
        new-state (assoc game-state
                         :current-player (next-player game-state))]
    (condp contains? type
      ["Fold"] (assoc new-state
                    :active-players (remove (partial = player-number) active-players))
      ["Check"] new-state
      ["Raise"] (assoc new-state
                     :bet-values (update bet-values current-player (partial + amount))
                     :pot (+ pot amount)
                     :min-raise (- (+ amount (bet-values current-player)) current-bet))
      ["Bet" "All-In"] (assoc new-state
                   :bet-values (update bet-values current-player (partial + amount))
                   :pot (+ pot amount)))))



(defn blind
  "Pays blind or goes all in depending on money available. Assumes nonzero money
   Returns [updated-player bet-value]"
  [player value]
  (let [{money :money} player]
    (if (< money value)
      [(assoc player :money 0) ["All-In" money]]
      [(assoc player :money (- money value)) ["Bet" value]])))


(defn pay-blinds 
  "Small blind and big blind bet or go all in if they don't have enough money"
  [game-state]
  (let [{players :players
         numPlayers :numPlayers
         bet-values :bet-values} game-state
        sb 1
        bb (mod 2 numPlayers)
        [sb-player sb-action] (blind (players sb) 0.5)
        [bb-player bb-action] (blind (players bb) 1)]
    (assoc game-state
           :players (assoc players 
                           sb sb-player
                           bb bb-player)
           :pot (+ (second sb-action)
                   (second bb-action))
           :current-player (mod (inc bb) (count players))
           :bet-values (assoc bet-values 
                              sb (second sb-action)
                              bb (second bb-action)))))

(defn round-over-checkall 
  "When all active players have contributed the same amount to the pot,
   the round is over."
  [game-state]
  (let [{bet-values :bet-values
         active-players :active-players} game-state]
    (apply = (map #(nth bet-values %) active-players))))

(defn round-over-checkone 
  "When betting goes back to the player who first made the bet,
   the round is over.
   Faster than checkall"
  [game-state]
  (let [{bet-values :bet-values
         current-player :current-player
         current-bet :current-bet} game-state]
    (= (bet-values current-player) current-bet)))


(defn reset-action
  "Return action to first active player cw from the dealer
   Where 'cw' is 0 -> 1 -> 2 etc."
  [game-state]
  (let [{active-players :active-players
         n :numPlayers} game-state]
    (assoc game-state
           :current-player (apply min-key #(mod (+ % n -1) n) active-players))))

(defn reset-action-2p
  "Return action to first active player cw from the dealer
   Where 'cw' is 0 -> 1 -> 2 etc."
  [game-state]
  (let [{active-players :active-players} game-state]
    (assoc game-state
           :current-player (apply min active-players))))




(defn add-money [player amount]
  (update player :money (partial + amount)))

(defn showdown-2p
  "2-person showdown. Does not consider side pots.
   Computes the people with the highest hands and divides the pot among them"
  [game-state]
  (let [{hands :hands
         visible :visible
         pot :pot
         active-players :active-players
         players :players} game-state
        player-hands (map #(vector % 
                                   (concat (nth hands %) 
                                           visible)) 
                          active-players)
        winners (utils/highest-hand player-hands)
        updated-players (reduce #(update %1 
                                         (first %2)
                                         (fn [p] 
                                           (add-money p (/ pot (count winners))))) 
                                players 
                                winners)]
    (assoc game-state 
           :game-over true
           :players updated-players)))



(defn check-active-players [game-state]
  (let [{players :players
         active-players :active-players
         pot :pot} game-state
        i (first active-players)]
    (if (= 1 (count active-players))
      (let [player (add-money (players i) pot)]
        (assoc game-state 
               :players (assoc players i player)
               :game-over true))
      game-state)))

(defn next-round-2p
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
      "River" (showdown-2p game-state)))))

(defn bet-round [game-state]
  (let []
    ()))

(defn get-action [player game-state]
  ())

(def always-fold (constantly ["Fold" 0]))
(def loose-passive (fn [history state money id]
                     (let [{bet-values :bet-values
                            current-bet :current-bet} state
                           diff (- current-bet (bet-values id))]
                      (cond (= 0 current-bet) ["Check" 0]
                           (> diff money) ["All-In" money]
                           :else ["Call" diff]))))
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


