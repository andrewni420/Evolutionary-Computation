(ns poker.headsup
  (:require [poker.utils :as utils]
            [clojure.pprint :as pprint]
            [poker.onehot :as onehot]
            [poker.ndarray :as ndarray]
            [poker.transformer :as transformer]
            [clojure.core.matrix :as m]
            [clojure.set :as set])
  (:gen-class))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;        Heads-Up Poker Game Engine         ;;;
;;; Values represented in units of big blinds ;;;
;;;   Multiple random decks played for each   ;;;
;;;              rotation of players          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; verbosity 0 = print nothing
;;; verbosity 1 = print function name
;;; verbosity 2 = print some relevant i/o information
;;; verbosity 3 = print final game state
;;; verbosity 4 = print initial game state

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Overview:
;;; Games are played between two players.
;;; A player is composed of an agent, and ID, and a chip stack
;;; An agent is a function that takes the game state and game-encoding
;;;     and outputs a selected action to take
;;; The game-state is a complete representation of the current state
;;;     of the game, including the cards in the player's hands, the 
;;;     cards in the middle of the table, the amount players have to bet
;;;     in order to stay in the game, and other auxiliary variables
;;; The game-encoding is based off of the decision transformer
;;;     https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html
;;;     and is a map {:state {id0 NDArray id1 NDArray} :actions NDArray :positions NDArray}
;;;     of one/multi-hot encoded states, actions and positions that the transformer model
;;;     will use to make a decision. There is one state NDArray per player because the 
;;;     two players can see different pieces of the game information.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Code structure:
;;; init-game initializes the game state 
;;; pay-blinds forces the players to post the big blind and small blind
;;; make-move asks the player to choose an action
;;; bet-round runs one round of betting
;;; showdown decides who wins the pot in a showdown, and whether hands are revealed or mucked
;;; bet-game runs an entire game of betting after the blinds have been paid
;;; play-game plays a game between two agents or players, returning the result of that
;;;     game
;;;
;;; iterate-games plays multiple games between two agents or players until one of the
;;;     players has no money left or until the maximum number of games has been reached
;;; iterate-games-reset plays n games between two players and returns the result of the games
;;; iterate-games-symmetrical is like iterate-games-reset, but it reduces the variance
;;;     by making the players play each deck twice, once in each position
;;; iterate-games-significantly first runs a small number of games to estimate the
;;;     standard deviation in wins, then plays enough games to reduce the radius of the confidence
;;;     interval to 1.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Game Engine from single move to full game ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;
;;  Initialization   ;;
;;;;;;;;;;;;;;;;;;;;;;;


(defn init-game
  "Removes players with no money and initializes the poker game. In the context of this game, players are 
   referenced by their index, not their id. In the history, players are referenced by their index.\\
   Hands - two cards per person\\
   Community - Five cards revealed as 3 in the flop, 1 at the turn, and 1 at the river\\
   Visible - Cards already revealed\\
   Bet-values - the amount which players have bet in this round. Resets every round.\\
   Current-bet - the amount which players need to match to stay in the round. Resets every round.\\
   Pot - the total pot size\\
   Active-players - players who have not yet folded\\
   Min-raise - The minimum amount by which people must raise\\
   Min-bet - The minimum bet is 1bb\\
   Betting-round - The round of betting currently happening\\
   Players - the players in the game\\
   Current-player - the player whose turn it is to act\\
   num-players - the number of players\\
   game-over - whether the game has terminated\\"
  [& {:keys [players deck verbosity game-num player-ids manager]
      :or {players [(constantly ["Fold" 0.0]) (constantly ["Fold" 0.0])]
           deck (shuffle utils/deck)
           verbosity 0
           game-num 0}}]
  (let [players (utils/process-players players)
        player-ids (if player-ids player-ids (mapv :id players))
        deal (utils/deal-hands 2 deck)]
    ;;Printing functionality to help explain how code works
    (utils/print-verbose verbosity
                         {:fn "init-game"}
                         {:players players
                          :deal deal}
                         {}
                         {:initial-state {}})
    ;;Initial gamet state
    {:hands (vec (:hands deal))
     :community (vec (:community deal))
     :visible []
     :manager manager
     :visible-hands []
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
     :action-history [[]]
     :player-ids player-ids
     :game-num game-num}))

#_(init-game :verbosity 2)

(defn init-game-encoding
  "Initializes the game history given the ids of the players and an NDManager\\
   -> game-history"
  [manager player-ids]
  {:actions (.create manager (ndarray/shape [0 onehot/action-length]))
   ;;Each player has their own state encoding, since the information available
   ;;to each player is different
   :state (zipmap player-ids
                  (repeatedly (count player-ids) #(.create manager (ndarray/shape [0 onehot/state-length]))))
   :position (.create manager (ndarray/shape [0 onehot/position-length]))})

#_(with-open [m (ndarray/new-base-manager)]
    ( (init-game-history m [:p0 :p1])))

;;;;;;;;;;;;;;;;;;;;;;;
;;    Single Move    ;;
;;;;;;;;;;;;;;;;;;;;;;;

(defn update-game-encoding
  "Given the additional state, position, and or actions encodings to be added, updates the game-encoding\\
   by appending these encodings onto the appropriate tensors.\\
   -> game-encoding"
  [game-encoding manager & {:keys [state position actions]}]
  (let [;;If some of state, position, and actions are provided, turn them into NDArrays
        ;;If a map is provided for state, turn it into a map of id to NDArray
        state (if state (into {} (map (fn [[k v]]
                                        [k (ndarray/ndarray manager [v])])
                                      state)) state)
        engine (ai.djl.engine.Engine/getDefaultEngineName)
        position (if position (ndarray/ndarray manager
                                               (if (= "PyTorch" engine)
                                                 int-array
                                                 float-array)
                                               [position]) position)
        actions (if actions (ndarray/ndarray manager [actions]) actions)
        #_[state position actions] #_(map #(cond (map? %) (into {} (map (fn [[k v]]
                                                                          [k (ndarray/ndarray manager [v])])
                                                                        %))
                                                 % (ndarray/ndarray manager [%])
                                                 :else %)
                                          [state position actions])
        add-encoding #(.concat %1 %2)
        state-update #(assoc % :state (merge-with add-encoding (:state %) state))
        position-update #(update % :position add-encoding position)
        actions-update #(update % :actions add-encoding actions)
        updated-encoding ((comp (if state state-update identity)
                                (if position position-update identity)
                                (if actions actions-update identity))
                          game-encoding)]
    ;;Close previous, now unused NDArrays to prevent memory loss
    (when state
      (run! #(.close (second %)) state)
      (run! #(.close (second %)) (:state game-encoding)))
    (when position
      (.close position)
      (.close (:position game-encoding)))
    (when actions
      (.close actions)
      (.close (:actions game-encoding)))
    ;;Return updated encoding
    updated-encoding))

(defn make-move
  "Asks the current player for a move\\
   -> [move-type amount]"
  [game-state game-encoding & {:keys [verbosity]
                              :or {verbosity 0}}]
  (let [{current-player :current-player
         players :players} game-state
        player (players current-player)]
    ;;Printing functionality to help explain how code works
    (utils/print-verbose verbosity
                         {:fn "make-move"}
                         {:current-player current-player
                          :player-id (:id player)}
                         {:final-state (make-move game-state game-encoding :verbosity 0)}
                         {:initial-state game-state})
    ;;The :agent of the player is a function that takes in a game-state
    ;;and a game-encoding and returns a move
    ((:agent player) game-state
                     game-encoding)))


#_(make-move (init-game) [] :verbosity 2)

(defn check-active-players
  "Checks to see if there is only one active player left in the game.
   If so, awards that player the pot and ends the game\\
   -> game-state"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (let [{players :players
         active-players :active-players
         pot :pot} game-state
        i (first active-players)]
    ;;Printing functionality to help explain how code works
    (utils/print-verbose verbosity
                         {:fn "check-active-players"}
                         {:num-active (count active-players)}
                         {:final-state (check-active-players game-state :verbosity 0)}
                         {:initial-state game-state})
    ;;If there's only one player that hasn't folded, game over, and that player gets the pot
    (if (= 1 (count active-players))
      (let [player (utils/add-money (players i) pot)]
        (assoc game-state
               :players (assoc players i player)
               :game-over true))
      game-state)))

#_(check-active-players (init-game) :verbosity 2)

(defn next-player
  "Passes action to the next player\\
   -> game-state"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (let [{current-player :current-player
         n :num-players} game-state]
    ;;Printing functionality to help explain how code works
    (utils/print-verbose verbosity
                         {:fn "next-player"}
                         {:next-player  (mod (inc current-player) n)}
                         {:final-state (next-player game-state :verbosity 0)}
                         {:initial-state game-state})
    ;;Increment the :current-player field to the next player in line
    (assoc game-state
           :current-player (mod (inc current-player) n))))

#_(next-player (init-game) :verbosity 2)
;;updates the game history by appending it the action
;;make-move updates the game history by appending to it the state and position...?


(defn parse-action
  "Updates game state based on action. Does not check for legality of action\\
   action: [move-type amount]\\
   -> [game-state game-encoding]"
  [action game-state game-encoding & {:keys [verbosity]
                                      :or {verbosity 0}}]
  (assert (and (first action) (second action)) 
          (str "Cannot have nil in action "action game-state))
  (let [{current-player :current-player
         active-players :active-players
         pot :pot
         bet-values :bet-values
         current-bet :current-bet
         players :players
         action-history :action-history
         min-raise :min-raise
         manager :manager} game-state
        [type amount] action
        ;;Subtract the amount betted from the player that betted
        players (update players current-player (fn [p] (update p :money #(- % amount))))
        ;;Pass action to the next player,
        ;;add the submitted action to the action history 
        new-state (assoc (next-player game-state :verbosity verbosity)
                         :action-history (conj (into [] (drop-last action-history))
                                               (conj (last action-history)
                                                     [(:id (players current-player))
                                                      action]))
                         :players players)
        ;;Update game encoding with the submitted action
        new-encoding (update-game-encoding game-encoding
                                          manager
                                          :actions (onehot/encode-action action game-state)
                                          :position (onehot/encode-position game-state))]
    ;;Printing functionality to help explain how code works
    (utils/print-verbose verbosity
                         {:fn "parse-action"}
                         {:action action
                          :bet-values bet-values
                          :current-bet current-bet
                          :current-player current-player
                          :current-player-id (:id (players current-player))}
                         {:final-state nil #_(parse-action action game-state game-encoding :verbosity 0)}
                         {:initial-state game-state})
    (condp utils/in? type
      ;;If the action was a fold, remove the player from the active players and 
      ;;check to see if there's only one active player left
      ["Fold"] [(check-active-players (assoc new-state
                                             :active-players (remove (partial = current-player)
                                                                     active-players))
                                      :verbosity verbosity)
                new-encoding]
      ["Check"] [new-state new-encoding]
      ;;Otherwise, we need to update the pot, the amount betted by each player,
      ;;(maybe) the minimum raise, and (maybe) the amount players need to bet to stay in the game
      ["Raise" "Bet" "All-In" "Call"] [(assoc new-state
                                      :bet-values (update bet-values current-player (partial + amount))
                                      :pot (+ pot amount)
                                      :current-bet (max current-bet (+ amount (bet-values current-player)))
                                      :min-raise (max min-raise (- (+ amount (bet-values current-player)) current-bet)))
                               new-encoding])))

#_(with-open [m (ndarray/new-base-manager)]
    (clojure.pprint/pprint (parse-action ["Fold" 0.0] (init-game :manager m)
                                         {:actions (.create m (ndarray/shape [1 0 onehot/action-length]))} :verbosity 2)))

;;;;;;;;;;;;;;;;;;;;;;;
;;    Single Round   ;;
;;;;;;;;;;;;;;;;;;;;;;;

(defn blind
  "Pays blind or goes all in depending on money available. Assumes nonzero money
   Returns [updated-player action]"
  [player value & {:keys [verbosity]
                   :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       {:fn "blind"}
                       {:player-id (:id player)
                        :value value}
                       {:final-state (blind player value :verbosity 0)}
                       {:initial-state {}})
  (let [{money :money} player]
    (if (< money value)
      [(assoc player :money 0) ["All-In" money]]
      [(assoc player :money (- money value)) ["Bet" value]])))

#_(blind (utils/init-player (constantly ["Fold" 0.0]) :p0) 
         1.0 
         :verbosity 2)

(defn pay-blinds
  "Small blind and big blind bet or go all in if they don't have enough money"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (let [{players :players
         num-players :num-players
         bet-values :bet-values} game-state
        sb 0
        bb 1
        [sb-player sb-action] (blind (players sb) 0.5 :verbosity verbosity)
        [bb-player bb-action] (blind (players bb) 1.0 :verbosity verbosity)]
    (utils/print-verbose verbosity
                         {:fn "pay-blinds"}
                         {:small-blind (:id (players sb))
                          :big-blind (:id (players bb))}
                         {:final-state (pay-blinds game-state :verbosity 0)}
                         {:initial-state game-state})
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
           :current-bet 1.0)))

#_(pay-blinds (init-game) :verbosity 2)

(defn reset-action
  "Big blind acts first in all rounds except pre-flop"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       {:fn "reset-action"}
                       {:last-action (last (last (:action-history game-state)))}
                       {:final-state (reset-action game-state :verbosity 0)}
                       {:initial-state game-state})
  (assoc game-state
         :current-player 1
         :current-bet 0.0
         :min-raise 1.0
         :bet-values [0.0 0.0]
         :action-history (conj (:action-history game-state) [])))

#_(reset-action (init-game) :verbosity 2)

(defn all-in?
  "Has someone gone all-in in a previous betting round?"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       {:fn "all-in?"}
                       {:bet-values (:bet-values game-state)
                        :players (:players game-state)}
                       {:final-state (reset-action game-state :verbosity 0)}
                       {:initial-state game-state})
  (and (every? zero? (:bet-values game-state))
       (some zero? (map :money (:players game-state)))))

#_(all-in? (init-game) :verbosity 2)

#_(defn round-over-checkall
    "When all active players have contributed the same amount to the pot,
   the round is over."
    [game-state]
    (let [{bet-values :bet-values
           active-players :active-players} game-state]
      (apply = (map #(nth bet-values %) active-players))))


(defn round-over-checkone
  "When betting goes back to the player who first made the bet,
   the round is over."
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (let [{bet-values :bet-values
         current-player :current-player
         current-bet :current-bet} game-state]
    (utils/print-verbose verbosity
                         {:fn "round-over-checkone"}
                         {:bet-values bet-values
                          :current-player current-player
                          :current-bet current-bet}
                         {:final-state (round-over-checkone game-state :verbosity 0)}
                         {:initial-state game-state})
    (cond
      (all-in? game-state) true
      (every? zero? bet-values) (= 1 current-player)
      (utils/pre-flop-bb? game-state) false
      :else (= (bet-values current-player) current-bet))))

#_(round-over-checkone (init-game) :verbosity 2)


(defn bet-round
  "Runs betting for one round\\
   If the game is over or both players are all-in, does nothing\\
   Otherwise updates the game encoding with the current state and position, 
   asks the current agent for a move, and parses that move to obtain a new state and encoding\\
   Repeats until the round or game is over, and returns the game state and encoding at that point.\\
   -> [game-state game-history]"
  [game-state game-encoding & {:keys [verbosity]
                               :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       {:fn "bet-round"}
                       {:game-over (:game-over game-state)
                        :round (:betting-round game-state)}
                       {:final-state nil #_(bet-round game-state game-history manager :verbosity 0)}
                       {:initial-state game-state})
  (if (or (:game-over game-state)
          (all-in? game-state :verbosity verbosity))
    [game-state, game-encoding]
    (let [{manager :manager} game-state
          game-encoding (update-game-encoding game-encoding
                                              manager
                                              :state (onehot/encode-state game-state)
                                              :position (onehot/encode-position game-state))
          p1-move (make-move game-state
                             game-encoding
                             :verbosity verbosity)
          [new-state new-encoding] (parse-action p1-move
                                                 game-state
                                                 game-encoding
                                                 :verbosity verbosity)]
      (if (or (:game-over new-state)
              (round-over-checkone new-state
                                   :verbosity verbosity))
        [new-state, new-encoding]
        (recur new-state
               new-encoding
               {:verbosity verbosity})))))





#_(with-open [m (ndarray/new-base-manager)]
  ( (bet-round (init-game :players [(utils/init-player utils/random-agent :p0)
                                  (utils/init-player utils/random-agent :p1)]
                        :manager m)
             (init-game-history m [:p0 :p1])
             m
             :verbosity 2)))


;;;;;;;;;;;;;;;;;;;;;;;
;;    Single Game    ;;
;;;;;;;;;;;;;;;;;;;;;;;

(defn last-aggressor
  "Given an action history, returns the id of the player who made
   the last aggressive move
   If everyone is all-in or checks, first player to the left of the button shows
   players = [button/sb bb]"
  [action-history players & {:keys [verbosity]
                             :or {verbosity 0}}]
  (let [actions (mapcat identity action-history)
        last-round (take-last 2 action-history)]
    (utils/print-verbose verbosity
                         {:fn "last-aggressor"}
                         {:last-round last-round}
                         {:final-state (last-aggressor action-history players :verbosity 0)}
                         {:initial-state action-history
                          :players players})
    (if (every? (partial contains? #{"All-In" "Check"})
                (map #(first (second %)) last-round))
      (:id (second players))
      (loop [actions actions]
        (if (empty? actions)
          (:id (second players))
          (if (contains? utils/aggressive-actions (first (second (last actions))))
            (first (last actions))
            (recur (drop-last actions))))))))

#_(let [{action-history :action-history
         players :players} (init-game)]
    (last-aggressor action-history players :verbosity 2))

(defn showdown
  "Heads-up showdown. Does not consider side pots.
   Computes the people with the highest hands and divides the pot among them
   Visible hands are the last aggressor and the winning hands"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (let [{hands :hands
         community :community
         pot :pot
         players :players
         active-players :active-players
         action-history :action-history} game-state
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
                                winners)
        l-agg (last-aggressor action-history players :verbosity verbosity)
        l-idx (first (keep-indexed #(if (= l-agg (:id %2)) %1 nil) players))
        visible-idx (set (conj (map first winners) l-idx))]
    (utils/print-verbose verbosity
                         {:fn "bet-round"}
                         {:community community
                          :hands hands
                          :players players
                          :last-aggressor l-agg}
                         {:final-state (showdown game-state :verbosity 0)}
                         {:initial-state game-state})
    (assoc game-state
           :game-over true
           :players updated-players
           :betting-round "Showdown"
           :visible-hands (keep-indexed #(if (contains? visible-idx %1) (vector (:id (players %1)) %2) nil)
                                        hands))))

#_(showdown (init-game) :verbosity 2)

(defn next-round
  "Checks to see if all players but one have folded, then runs showdown or
   reveals the next card and proceeds to the next betting round"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (let [{betting-round :betting-round
         community :community} game-state
        reset-state (check-active-players (reset-action game-state :verbosity verbosity)
                                          :verbosity verbosity)]
    (utils/print-verbose verbosity
                         {:fn "next-round"}
                         {:prev-round betting-round}
                         {:final-state (next-round game-state :verbosity 0)}
                         {:initial-state game-state})
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
        "River" (showdown game-state :verbosity verbosity)))))

#_(next-round (init-game) :verbosity 2)

(defn bet-game
  "Runs betting from initialization of game until showdown"
  [game-state & {:keys [game-encoding verbosity]
                              :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       {:fn "bet-game"}
                       {}
                       {}
                       {:initial-state game-state})
  (let [game-encoding (or game-encoding (init-game-encoding (:manager game-state) (map :id (:players game-state))))
        [new-state new-encoding] (bet-round game-state game-encoding :verbosity verbosity)]
    (if (:game-over new-state)
      [new-state, new-encoding]
      (let [next-round (next-round new-state :verbosity verbosity)]
        (if (:game-over next-round)
          [next-round, new-encoding]
          (recur next-round
                 {:game-encoding new-encoding
                  :verbosity verbosity}))))))

#_(with-open [m (ndarray/new-base-manager)]
  (let [[game-state game-history] (bet-game (pay-blinds
                                             (init-game :players [(utils/init-player utils/random-agent :p0)
                                                                  (utils/init-player utils/random-agent :p1)]
                                                        :manager m))
                                            (init-game-history  m [:p0 :p1]))]
    (clojure.pprint/pprint game-state)
    ( game-history)))

#_(defn var-reduce 
  "Subtracts expected value of hand from the money earned"
  [net-gain game-state]
  (let [{players :players
         hands :hands} game-state
        ids (zipmap (map :id players) [0 1])]
    (into [] (map #(let [[p g] %
                         {win :win 
                          total :total} (utils/rollout (into #{} (hands (ids p))))]
                     (vector p (- g (* 0 (:pot game-state) (/ win total))))) 
                  net-gain))))

(defn var-reduce 
  "NOT IMPLEMENTED\\
   Reduce variance"
  [net-gain _game-state & {:keys [verbosity]
                           :or {verbosity 0}}]
  net-gain)

(defn state-to-history 
  "Summarizes end-of-game state into fields to put into game history.\\
   hands: cards dealt to each player\\
   visible-hands: cards shown at showdown\\
   playerIDs: ids of each player\\
   action-history: actions taken by each player\\
   visible-cards: revealed community cards\\
   net-gain: the net gain of each player in the game"
  [old-state new-state & {:keys [verbosity]
                          :or {verbosity 0}}]
  (let [{action-history :action-history
         hands :hands
         new-players :players
         visible :visible
         visible-hands :visible-hands} new-state
        {old-players :players} old-state]
    (assoc {}
           :hands (mapv #(vector (:id %1) %2) new-players hands)
           :playerIDs (mapv :id new-players)
           :action-history action-history
           :visible-cards visible
           :visible-hands visible-hands
           :net-gain (var-reduce 
                      (mapv #(vector (:id %2) 
                                     (- (:money %1) 
                                        (:money %2))) 
                            new-players 
                            old-players)
                      new-state))))


(defn play-game
  "Initializes and plays a game of poker, updating players and game-history in the process.
   -> {players game-encoding game-history}"
  ([players manager & {:keys [deck verbosity game-num game-encoding game-history]
                       :or {deck (shuffle utils/deck)
                            verbosity 0
                            game-num 0}}]
   (let [game-encoding (or game-encoding (init-game-encoding manager (map :id players)))
         game-history (or game-history [])
         game-state (init-game :players players
                               :deck deck
                               :verbosity verbosity
                               :game-num game-num
                               :manager manager)
         [new-state new-encoding] (bet-game (pay-blinds game-state :verbosity verbosity)
                                            :game-encoding game-encoding
                                            :verbosity verbosity)
         position-encoding (onehot/encode-position new-state)]
     (utils/print-verbose verbosity
                          {:fn "play-game"}
                          {:players players
                           :deck (take 7 deck)}
                          {:final-state new-state}
                          {:initial-state {}})
     {:players (:players new-state)
      :game-encoding (-> new-encoding
                         (update-game-encoding manager
                                               :state (onehot/encode-state new-state)
                                               :position position-encoding
                                               :actions (onehot/encode-action nil new-state))
                         (update-game-encoding manager
                                               :position position-encoding))
      :game-history (conj game-history (state-to-history game-state new-state :verbosity verbosity))})))

(def h (with-open [m (ndarray/new-base-manager)]
  (play-game [(utils/init-player utils/random-agent :p0)
            (utils/init-player utils/random-agent :p1)]
           m
             :game-history [{:hands [[:p0 [[8 "Clubs"] [9 "Diamonds"]]] [:p1 [[10 "Hearts"] [5 "Spades"]]]],
                             :playerIDs [:p0 :p1],
                             :action-history [[[:p0 ["All-In" 199.5]] [:p1 ["Fold" 0.0]]]],
                             :visible-cards [],
                             :visible-hands [],
                             :net-gain [[:p0 1.0] [:p1 -1.0]]}])))

#_(with-open [manager (ndarray/new-base-manager)]
  (println (play-game [(transformer/as-player (transformer/initialize-individual
                                               :manager manager
                                               :nn-factory #(transformer/current-transformer manager)
                                               :id :p0
                                               :mask (ndarray/ndarray manager (ndarray/causal-mask [1 256 256] -2))
                                               :max-seq-length 256))
                       (utils/init-player utils/random-agent :p1)]
                      manager)))

#_(with-open [manager (ndarray/new-base-manager)]
  (let [{game-encoding :game-encoding
         game-history :game-history} (play-game [(utils/init-player utils/random-agent :p0)
                                                 (utils/init-player utils/random-agent :p1)]
                                                manager)]
    (println game-encoding)
    (clojure.pprint/pprint game-history)))



;;;;;;;;;;;;;;;;;;;;;;;
;;   Multiple Games  ;;
;;;;;;;;;;;;;;;;;;;;;;;


(defn process-net-gain
  "Helper for the iterate-games methods to process the net game into a statistic\\
   -> {id {mean stdev}} or {id gain}"
  [net-gain num-games & {:keys [as-list?] :or {as-list? false}}]
  (into {}
        (map #(vector (first %)
                      (if as-list?
                        {:mean (utils/mean (second %))
                         :stdev (utils/stdev (second %))}
                        (/ (second %) num-games)))
             net-gain)))

(defn update-net-gain
  "Updates the net gain based on the amount of money the player has"
  [net-gain players & {:keys [as-list?] :or {as-list? false}}]
  (reduce (fn [gain p]
            (update gain
                    (:id p)
                    #((if as-list?
                        conj
                        +) %
                           (- (:money p) utils/initial-stack))))
          net-gain
          players))


#_(update-net-gain {:p1 [], :p2 []} 
                 [{:agent 1, :money 200.0, :id :p1} 
                  {:agent 1, :money 200.0, :id :p2}]
                 :as-list? true)

(defn iterate-games
  "Plays num-games hands of poker with players switching from sb to bb every hand.\\
   An even number of hands ensures balanced play.\\
   Returns after num-games are reached or a player has no more money\\
   Do not print the second item (game-history) - it can get very large\\
   -> {players game-encoding game-history}"
  [players manager num-games  & {:keys [verbosity game-encoding game-history]
                                 :or {verbosity 0}}]
  (loop [players (utils/process-players players)
         game-num 0
         game-encoding (or game-encoding (init-game-encoding manager (mapv :id players)))
         game-history (or game-history [])]
    (if (or (<= num-games game-num)
            (some zero? (map :money players)))
      {:players players :game-encoding game-encoding :game-history game-history}
      (let [{players :players
             game-encoding :game-encoding
             game-history :game-history} (play-game players
                                                    manager
                                                    :game-encoding game-encoding
                                                    :game-history game-history
                                                    :game-num game-num)]
        (recur (into [] (reverse players))
               (inc game-num)
               game-encoding
               game-history)))))

#_(with-open [m (ndarray/new-base-manager)]
    (let [{players :players
           game-encoding :game-encoding
           game-history :game-history} (iterate-games [utils/random-agent
                                                       utils/rule-agent]
                                                      m
                                                      3)]
      (println game-encoding)
      (clojure.pprint/pprint game-history)))

;;Plays up to 100 games until a player has no money and returns the history of the "interesting" ones
;;"interesting" determined by more than one action, at least one bet, and significant money exchange (>10bb)
;;Not compatible with current version of headsup
#_(apply (fn [p h] (vector p (take 10 (filter #(and (not= 1 (count (:action-history %)))
                                                    (utils/in? (flatten (:action-history %)) "Bet")
                                                    (> (abs (second (first (:net-gain %)))) 10)
                                                    #_(not (utils/in? (flatten (:action-history %)) "Fold"))) h))))
         (iterate-games [utils/random-agent
                         utils/rule-agent]
                        100 {}))

(utils/process-decks 1 1)

(defn iterate-games-reset
  "Plays num-games hands of poker with players switching from sb to bb every hand and resetting their money values
   An even number of hands ensures balanced play.
   Returns the total gain/loss of players and history after num-games are reached\\
   list: Whether to keep a list of the gains to return as a mean and stdev or to simply return
   the total gain over all games\\
   Do not print out the last item (game-history) - it can get very big\\
   -> {players, net-gain = [gain ...] or {:mean :stdev}, game-encoding, game-history}"
  [players manager num-games & {:keys [as-list? decks game-history game-encoding max-actions]
                                :or {as-list? false
                                     max-actions ##Inf}}]
  (loop [players (utils/process-players players)
         net-gain (zipmap (map :id players) (if as-list? [[] []] [0.0 0.0]))
         game-num 0
         game-encoding (or game-encoding (init-game-encoding manager (mapv :id players)))
         game-history (or game-history [])
         decks (utils/process-decks decks num-games)
         action-count 0]
    (if (or (<= num-games game-num) (<= max-actions action-count))
      {:players players
       :net-gain (process-net-gain net-gain num-games :as-list? as-list?)
       :game-encoding game-encoding
       :game-history game-history
       :action-count action-count}
      (let [{[p1 p2] :players
             game-encoding :game-encoding
             game-history :game-history} (play-game players
                                                    manager
                                                    :game-encoding game-encoding 
                                                    :game-history game-history 
                                                    :game-num game-num 
                                                    :deck (first decks))]
        (recur (into [] (reverse players))
               (update-net-gain net-gain [p1 p2] :as-list? as-list?)
               (inc game-num)
               game-encoding
               game-history
               (rest decks)
               (+ action-count (/ (count (flatten (:action-history (last game-history)))) 3)))))))

#_(def k (with-open [m (ndarray/new-base-manager)]
  (iterate-games-reset [(utils/init-player utils/random-agent :p0)
                       (utils/init-player utils/random-agent :p1)]
                       m
                       10
                       :max-actions 3
                       )))


#_(with-open [m (ndarray/new-base-manager)]
  (let [max-seq-length 20
        param-map transformer/initial-parameter-map
        mask (ndarray/ndarray m (ndarray/causal-mask [1 max-seq-length max-seq-length] -2))
        ind1 (transformer/model-from-seeds {:seeds [2074038742],
                                            :id :p1}
                                           max-seq-length
                                           m
                                           mask)
        ind2 (transformer/model-from-seeds {:seeds [-888633566], 
                                            :id :p2}
                                           max-seq-length 
                                           m 
                                           mask)]
    (with-open [_i1 (utils/make-closeable ind1 transformer/close-individual)
                _i2 (utils/make-closeable ind2 transformer/close-individual)]
      (let [{game-history :game-history
             game-encoding :game-encoding
             net-gain :net-gain}
            (time (iterate-games-reset  [(transformer/as-player ind1)
                                         (transformer/as-player ind2)]
                                        m
                                        10
                                        :as-list? true))]
        (println net-gain)
        (println "actions-per-game:"
                 (float (/ (transduce (map #(/ (count (flatten (:action-history %))) 3.0)) + game-history)
                           (count game-history))))
        #_net-gain
        #_(println game-encoding)
        #_(clojure.pprint/pprint game-history)))))

;;random-agent: 2ms per action
;;transformer: 30ms per action
;;An order of magnitude difference

(def time-per-action
  "Average time per action for a transformer model with different length
   context windows"
  {:datapoints {300 85.9
                250 63.9
                200 48.1
                175 39.3
                150 31.7
                125 19.8
                100 15.4
                75 13.5
                50 10.1
                25 10.3}
   :random-agent "1.67 ms/action"
   :best-fit-line "y = 0.000744x^2 + 0.04594x + 6.3059, R^2 = 0.9921"
   :model-parameters {:state-length 183
                      :action-length 64
                      :d-model 64
                      :d-ff 256
                      :num-layers 6
                      :num-heads 8
                      :d-pe [16 16 16 16]
                      :max-seq-length 512}})


;;27ms per action


(defn iterate-games-significantly
  "Takes 100 games to sample the standard deviation of games between the two players.\\
   Calculates the required number of games to have a confidence interval of length CI.\\
   Plays that many games, or the max number of games, whichever is smaller\\
   -> winning agent or {:mean :stdev} of agent0"
  [agent0 agent1 manager max-games &
   {:keys [verbosity winner? CI manager game-history game-encoding]
    :or {verbosity 0
         winner? false
         CI 1}}]
  (let [Z 1.96
        players [(utils/init-player agent0 :p0)
                 (utils/init-player agent1 :p1)]
        {{{s0 :stdev} :p0} :net-gain
         game-history :game-history
         game-encoding :game-encoding} (iterate-games-reset players manager 100
                                                            :game-history game-history
                                                            :game-encoding game-encoding
                                                            :list? true)
        num-games (min max-games (utils/square (* Z (/ s0 (/ CI 2)))))
        {{{m1 :mean s1 :stdev} :p0} :net-gain} (iterate-games-reset players manager num-games
                                                                    :game-history game-history
                                                                    :game-encoding game-encoding
                                                                    :list? true)]
    (when (>= verbosity 1) (clojure.pprint/pprint {:fn "iterate-games-significantly"
                                                   :agents [agent0 agent1]
                                                   :num-games num-games
                                                   :Z Z
                                                   :desired-CI CI
                                                   :CI [(- m1 (/ (* Z s1) (Math/sqrt num-games)))
                                                        (+ m1 (/ (* Z s1) (Math/sqrt num-games)))]}))
    (if winner?
      (if (>= m1 0) agent0 agent1)
      {:mean m1 :stdev s1})))



(defn iterate-games-symmetrical
  "Plays each deck twice, with the players switching sides, and adds the net-gain up for each deck\\
   Lower the variance in the results\\
   -> {players net-gain game-encoding game-history}"
  [players manager num-games & {:keys [as-list? game-encoding game-history verbosity]
                                :or {as-list? false
                                     verbosity 0}}]
  (loop [players (utils/process-players players)
         net-gain (zipmap (map :id players) (if as-list? [[] []] [0.0 0.0]))
         game-encoding (or game-encoding (init-game-encoding manager (mapv :id players)))
         game-history (or game-history [])
         game-num 0]
    (if (<= num-games game-num)
      {:players players
       :net-gain (process-net-gain net-gain num-games :as-list? as-list?)
       :game-encoding game-encoding
       :game-history game-history}
      (let [deck (shuffle utils/deck)
            {[p11 p21] :players
             h1 :game-history
             e1 :game-encoding} (play-game players game-encoding game-history manager :game-num game-num :deck deck)
            {[p12 p22] :players
             h2 :game-history} (play-game (vec (reverse players)) game-encoding h1 manager :game-num game-num :deck deck)]
        (recur players
               (update-net-gain net-gain
                                [(utils/add-money p11 (- (:money p12) utils/initial-stack))
                                 (utils/add-money p21 (- (:money p22) utils/initial-stack))]
                                :as-list? as-list?)
               e1
               h2
               (inc game-num))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;         Runtime       ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
#_(with-open [m (ndarray/new-base-manager)
            m1 (.newSubManager m)]
  (let [arr1 (ndarray/ndarray m [[[1 2 3] [1 2 3]]])
        arr2 (ndarray/ndarray m1 [[[3 4 5]]])]
    (println m)
    (println m1)
    (println (.getManager (.concat arr2 arr1 1)))))

