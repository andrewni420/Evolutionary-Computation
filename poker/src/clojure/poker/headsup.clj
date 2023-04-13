(ns poker.headsup
  (:require [poker.utils :as utils]
            [clojure.pprint :as pprint]
            [poker.onehot :as onehot])
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
  [& {:keys [players deck verbosity]
      :or {players [(constantly ["Fold" 0.0]) (constantly ["Fold" 0.0])]
           deck (shuffle utils/deck)
           verbosity 0}}]
  (let [players (utils/process-players players)
        deal (utils/deal-hands 2 deck)]
    (utils/print-verbose verbosity
                         {:fn "init-game"}
                         {:players players
                          :deal deal}
                         {:final-state (init-game :players players
                                                  :deck deck
                                                  :verbosity 0)}
                         {:initial-state {}})
    {:hands (vec (:hands deal))
     :community (vec (:community deal))
     :visible []
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
     :action-history [[]]}))

#_(init-game :verbosity 2)

;;;;;;;;;;;;;;;;;;;;;;;
;;    Single Move    ;;
;;;;;;;;;;;;;;;;;;;;;;;


(defn make-move
  "Asks the current player for a move\\
   -> [move-type amount]"
  [game-state game-history & {:keys [verbosity]
                              :or {verbosity 0}}]
  (let [{current-player :current-player
         players :players} game-state
        player (players current-player)]
    (utils/print-verbose verbosity
                         {:fn "make-move"}
                         {:current-player current-player
                          :player-id (:id player)}
                         {:final-state (make-move game-state game-history :verbosity 0)}
                         {:initial-state game-state})
    ((:agent player) game-state
                     game-history)))


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
    (utils/print-verbose verbosity
                         {:fn "check-active-players"}
                         {:num-active (count active-players)}
                         {:final-state (check-active-players game-state :verbosity 0)}
                         {:initial-state game-state})
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
    (utils/print-verbose verbosity
                         {:fn "next-player"}
                         {:next-player  (mod (inc current-player) n)}
                         {:final-state (next-player game-state :verbosity 0)}
                         {:initial-state game-state})
    (assoc game-state
           :current-player (mod (inc current-player) n))))

#_(next-player (init-game) :verbosity 2)

(defn parse-action
  "Updates game state based on action. Does not check for legality of action\\
   action: [move-type amount]\\
   -> game-state"
  [action game-state game-history game-num & {:keys [verbosity]
                        :or {verbosity 0}}]
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
        new-state (assoc (next-player game-state :verbosity verbosity)
                         :action-history (conj (into [] (drop-last action-history))
                                               (conj (last action-history)
                                                     [(:id (players current-player))
                                                      action]))
                         :players players)
        {person :person
         positions :positions
         states :states
         actions :actions
         rewards :rewards} game-history
        round-number 0 ;;temp
        ]
    (utils/print-verbose verbosity
                         {:fn "parse-action"}
                         {:action action
                          :bet-values bet-values
                          :current-bet current-bet
                          :current-player current-player
                          :current-player-id (:id (players current-player))}
                         {:final-state (parse-action action game-state game-history game-num :verbosity 0)}
                         {:initial-state game-state})
    (condp utils/in? type
      ["Fold"] (let [updated-new-state (check-active-players (assoc new-state
                                              :active-players (remove (partial = current-player)
                                                                      active-players))
                                       :verbosity verbosity)
                     updated-game-history (assoc game-history
                                              :person (vec (conj person (onehot/encode-player current-player)))
                                              ;; positions should be a bunch of [game-num round-num action-num] vectors
                                              :positions (vec (conj positions (onehot/encode-position game-num round-number action)))
                                              ;;  states should be a bunch of vectors, where each vector is the one-hot encoded state information
                                              :states (vec (conj states (onehot/encode-state updated-new-state)))
                                              ;;  actions should be a bunch of vectors, where each vector is the one-hot encoded action information
                                              :actions (vec (conj actions (onehot/encode-action-type action)))
                                              ;; Reward is 0 since folded
                                              :rewards (vec (conj rewards (onehot/encode-reward 0))))]
                 [updated-new-state, updated-game-history])
      ["Check"] (let [updated-game-history (assoc game-history
                        :person (vec (conj person (onehot/encode-player current-player)))
                        ;; positions should be a bunch of [game-num round-num action-num] vectors
                        :positions (vec (conj positions (onehot/encode-position game-num round-number action)))
                        ;;  states should be a bunch of vectors, where each vector is the one-hot encoded state information
                        :states (vec (conj states (onehot/encode-state new-state)))
                        ;;  actions should be a bunch of vectors, where each vector is the one-hot encoded action information
                        :actions (vec (conj actions (onehot/encode-action-type action)))
                        ;; Reward is 0 since checked
                        :rewards (vec (conj rewards (onehot/encode-reward 0))))]
                  [new-state, updated-game-history])
      ["Raise"] (let [updated-new-state (assoc new-state
                                               :bet-values (update bet-values current-player (partial + amount))
                                               :pot (+ pot amount)
                                               :current-bet (+ amount (bet-values current-player))
                                               :min-raise (- (+ amount (bet-values current-player)) current-bet))
                      updated-game-history (assoc game-history
                                                :person (vec (conj person (onehot/encode-player current-player)))
                                            ;; positions should be a bunch of [game-num round-num action-num] vectors
                                                :positions (vec (conj positions (onehot/encode-position game-num round-number action)))
                                            ;;  states should be a bunch of vectors, where each vector is the one-hot encoded state information
                                                :states (vec (conj states (onehot/encode-state new-state)))
                                            ;;  actions should be a bunch of vectors, where each vector is the one-hot encoded action information
                                                :actions (vec (conj actions (onehot/encode-action-type action)))
                                            ;; Reward is [!NEED TO ADD!]
                                                :rewards (vec (conj rewards (onehot/encode-reward 0))))]
                  [updated-new-state, updated-game-history])
      ["Bet" "All-In" "Call"] (let [updated-new-state (assoc new-state
                                                          :bet-values (update bet-values current-player (partial + amount))
                                                          :pot (+ pot amount)
                                                          :current-bet (max current-bet (+ amount (bet-values current-player)))
                                                          :min-raise (max min-raise (- (+ amount (bet-values current-player)) current-bet)))
                                    updated-game-history (assoc game-history
                                                          :person (vec (conj person (onehot/encode-player current-player)))
                                                      ;; positions should be a bunch of [game-num round-num action-num] vectors
                                                          :positions (vec (conj positions (onehot/encode-position game-num round-number action)))
                                                      ;;  states should be a bunch of vectors, where each vector is the one-hot encoded state information
                                                          :states (vec (conj states (onehot/encode-state new-state)))
                                                      ;;  actions should be a bunch of vectors, where each vector is the one-hot encoded action information
                                                          :actions (vec (conj actions (onehot/encode-action-type action)))
                                                      ;; Reward is [!NEED TO ADD!]
                                                          :rewards (vec (conj rewards (onehot/encode-reward 0))))]
                                [updated-new-state, updated-game-history]))))

#_(parse-action ["Fold" 0.0] (init-game) {} :verbosity 2)

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
  "Runs betting for one round"
  [game-state game-history game-num & {:keys [verbosity]
                              :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       {:fn "bet-round"}
                       {:game-over (:game-over game-state)
                        :round (:betting-round game-state)}
                       {:final-state nil #_(bet-round game-state game-history game-num :verbosity 0)}
                       {:initial-state game-state})
  (if (or (:game-over game-state)
          (all-in? game-state :verbosity verbosity))
    [game-state, game-history]
    (let [p1-move (make-move game-state
                             game-history
                             :verbosity verbosity)
          parsed-action (parse-action p1-move
                                  game-state
                                  game-history
                                  game-num
                                  :verbosity verbosity)
          new-state (first parsed-action)
          updated-game-history (second parsed-action)]
      (if (or (:game-over new-state)
              (round-over-checkone new-state
                                   :verbosity verbosity))
        [new-state, updated-game-history]
        (recur new-state
               updated-game-history
               game-num
              {:verbosity verbosity})))))

#_(bet-round (init-game :players [(utils/init-player utils/random-agent :p0)
                                  (utils/init-player utils/random-agent :p1)])
             {}
             0
             :verbosity 2)

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
        l (last-aggressor action-history players :verbosity verbosity)
        l-idx (first (keep-indexed #(if (= l (:id %2)) %1 nil) players))
        visible-idx (set (conj (map first winners) l-idx))]
    (utils/print-verbose verbosity
                         {:fn "bet-round"}
                         {:community community
                          :hands hands
                          :players players
                          :last-aggressor l}
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
  [game-state game-history game-num & {:keys [verbosity]
                              :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       {:fn "bet-game"}
                       {}
                       {:final-state (bet-game game-state game-history game-num :verbosity 0)}
                       {:initial-state game-state})
  (let [bet-round-results (bet-round game-state game-history game-num :verbosity verbosity)
        new-state (first bet-round-results)
        updated-game-history (second bet-round-results)]
    (if (:game-over new-state)
      [new-state, updated-game-history]
      (let [next-round (next-round new-state :verbosity verbosity)]
        (if (:game-over next-round)
          [next-round, updated-game-history]
          (recur next-round
                 updated-game-history
                 game-num
                 {:verbosity verbosity}))))))


#_(bet-game (pay-blinds
           (init-game :players [(utils/init-player utils/random-agent :p0)
                                (utils/init-player utils/random-agent :p1)]))
          {}
          0
          :verbosity 2)

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

#_(defn state-to-history 
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
   -> [players game-history]"
  ([players game-history game-num & {:keys [deck verbosity]
                            :or {deck (shuffle utils/deck)
                                 verbosity 0}}]
  (let [game-state (init-game :players players 
                              :deck deck 
                              :verbosity verbosity)
        bet-game-results (bet-game (pay-blinds game-state :verbosity verbosity) 
                            game-history
                            game-num
                            :verbosity verbosity)
        new-state (first bet-game-results)
        updated-game-history (second bet-game-results)]
    (utils/print-verbose verbosity
                         {:fn "play-game"}
                         {:players players
                          :deck (take 7 deck)}
                         {:final-state new-state}
                         {:initial-state {}})
    [(:players new-state) 
     updated-game-history
     #_(conj game-history (state-to-history game-state new-state))]
    (spit "game-history.txt" (with-out-str (prn updated-game-history)) :append true))))


#_(play-game [(utils/init-player utils/random-agent :p0)
              (utils/init-player utils/random-agent :p1)]
             {}
             0
             :verbosity 2)
(init-game)
;;;;;;;;;;;;;;;;;;;;;;;
;;   Multiple Games  ;;
;;;;;;;;;;;;;;;;;;;;;;;

(defn iterate-games
  "Plays num-games hands of poker with players switching from sb to bb every hand.\\
   An even number of hands ensures balanced play.\\
   Returns after num-games are reached or a player has no more money\\
   Do not print the second item (game-history) - it can get very large\\
   -> [players game-history]"
  [players num-games game-history & {:keys [verbosity]
                                     :or {verbosity 0}}]
  (let [players (utils/process-players players)
        num-games (int num-games)
        og-num-games (int num-games)]
    (if (or (zero? num-games)
          (some zero? (map :money players)))
    [players game-history]
    (let [curr-num-game (int (- og-num-games num-games))
          [players history] (play-game players game-history curr-num-game)]
      (recur (into [] (reverse players)) 
             (dec num-games) 
             history
             verbosity)))))

#_(first (iterate-games [utils/random-agent
                         utils/rule-agent]
                        100
                        {}))

;;Plays up to 100 games until a player has no money and returns the history of the "interesting" ones
;;"interesting" determined by more than one action, at least one bet, and significant money exchange (>10bb)
#_(apply (fn [p h] (vector p (take 10 (filter #(and (not= 1 (count (:action-history %)))
                                                    (utils/in? (flatten (:action-history %)) "Bet")
                                                    (> (abs (second (first (:net-gain %)))) 10)
                                                    #_(not (utils/in? (flatten (:action-history %)) "Fold"))) h))))
         (iterate-games [utils/random-agent
                         utils/rule-agent]
                        100 {}))


(defn iterate-games-reset
  "Plays num-games hands of poker with players switching from sb to bb every hand and resetting their money values
   An even number of hands ensures balanced play.
   Returns the total gain/loss of players and history after num-games are reached\\
   list: Whether to keep a list of the gains to return as a mean and stdev or to simply return
   the total gain over all games\\
   Do not print out the last item (game-history) - it can get very big\\
   -> [players, net-gain = [gain ...] or {:mean :stdev}, game-history]"
  [players num-games game-history & {:keys [list? decks verbosity]
                                     :or {list? false
                                          decks nil
                                          verbosity 0}}]
  (let [players (utils/process-players players)
        og-num-games (int num-games)]
    (loop [net-gain (zipmap (map :id players) (if list? [[] []] [0.0 0.0]))
           n (int num-games)
           players players
           history game-history
           decks (if decks decks (repeatedly #(shuffle utils/deck)))]
      (if (zero? n)
        [players
         (into {}
               (map #(vector (first %)
                             (if list?
                               {:mean (utils/mean (second %))
                                :stdev (utils/stdev (second %))}
                               (/ (second %) num-games)))
                    net-gain))
         history]
        (let [curr-num-game (int (- og-num-games n))
              [[p1 p2] h] (play-game players history curr-num-game (first decks))]
          (recur (update (update net-gain
                                 (:id p1)
                                 #((if list? conj +) % (- (:money p1) (:money (first players)))))
                         (:id p2)
                         #((if list? conj +) % (- (:money p2) (:money (first players)))))
                 (dec n)
                 (into [] (reverse players))
                 h
                 (rest decks)))))))

#_(take 2 (iterate-games-reset  [utils/rule-agent utils/random-agent] 
                                10000 
                                {} 
                                :list? true))

(defn iterate-games-significantly
  "Takes 100 games to sample the standard deviation of games between the two players.\\
   Calculates the required number of games to have a confidence interval of length CI.\\
   Plays that many games, or the max number of games, whichever is smaller\\
   -> winning agent or {:mean :stdev} of agent0"
  [agent0 agent1 max-games game-history &
   {:keys [verbosity winner? CI]
    :or {verbosity 0
         winner? false
         CI 1}}]
  (let [Z 1.96
        players [(utils/init-player agent0 :p0)
                 (utils/init-player agent1 :p1)]
        [_ {{s0 :stdev} :p0}] (iterate-games-reset players 100 []
                                                   :list? true)
        num-games (min max-games (utils/square (* Z (/ s0 (/ CI 2)))))
        [_ {{m1 :mean s1 :stdev} :p0}] (iterate-games-reset players num-games []
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
   Lower the variance in the results"
  [players num-games game-history & {:keys [list? verbosity]
                                     :or {list? false
                                          verbosity 0}}]
  (loop [net-gain (zipmap (map :id players) (if list? [[] []] [0.0 0.0]))
         n (int num-games)
         players players
         history game-history
         curr-game-num 0]
    (if (zero? n)
      [players
       (into {}
             (map #(vector (first %)
                           (if list?
                             {:mean (utils/mean (second %))
                              :stdev (utils/stdev (second %))}
                             (/ (second %) num-games)))
                  net-gain))
       history]
      (let [deck (shuffle utils/deck)
            [[p11 p21] h1] (play-game players history curr-game-num deck)
            [[p22 p12] h2] (play-game (vec (reverse players)) h1 curr-game-num deck)]
        (recur (update (update net-gain
                               (:id p11)
                               #((if list? conj +) % (- (+ (:money p11) (:money p12))
                                                        (* 2 (:money (first players))))))
                       (:id p22)
                       #((if list? conj +) % (- (+ (:money p21) (:money p22))
                                                (* 2 (:money (second players))))))
               (dec n)
               players
               h2
               (inc curr-game-num))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;         Runtime       ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


