(ns poker.headsup
  (:require [poker.utils :as utils]
            [propeller.tools.math :as math]
            [clojure.set :as set]
            [clojure.pprint :as pprint])
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
      :or {players [{:money utils/initial-stack :id :p0 :agent (constantly ["Fold" 0.0])}
                    {:money utils/initial-stack :id :p1 :agent (constantly ["Fold" 0.0])}]
           deck (shuffle utils/deck)
           verbosity 0}}]
  (let [deal (utils/deal-hands 2 deck)]
    (utils/print-verbose verbosity
                         [{:fn "init-game"}
                          {:players players
                           :deal deal}
                          {:final-state (init-game :players players
                                                   :deck deck
                                                   :verbosity 0)}
                          {:initial-state {}}])
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

#_(init-game :verbosity 1)

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
                         [{:fn "make-move"}
                          {:current-player current-player
                           :player player}
                          {:final-state (make-move game-state game-history :verbosity 0)}
                          {:initial-state game-state}])
    ((:agent player) game-state
                     game-history)))


#_(make-move (init-game) [] :verbosity 1)


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
                         [{:fn "check-active-players"}
                          {:num-active (count active-players)}
                          {:final-state (check-active-players game-state :verbosity 0)}
                          {:initial-state game-state}])
    (if (= 1 (count active-players))
      (let [player (utils/add-money (players i) pot)]
        (assoc game-state
               :players (assoc players i player)
               :game-over true))
      game-state)))

#_(check-active-players (init-game) :verbosity 1)

(defn next-player
  "Passes action to the next player\\
   -> game-state"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (let [{current-player :current-player
         n :num-players} game-state]
    (when (>= verbosity 1)
      (pprint/pprint (merge {:fn "next-player"
                             :next-player (mod (inc current-player) n)}
                            (when (>= verbosity 2) {:output game-state}))))
    (assoc game-state
           :current-player (mod (inc current-player) n))))

#_(next-player (init-game) :verbosity 1)

(defn parse-action
  "Updates game state based on action. Does not check for legality of action\\
   action: [move-type amount]\\
   -> game-state"
  [action game-state & {:keys [verbosity]
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
        new-state (assoc (next-player game-state)
                         :action-history (conj (into [] (drop-last action-history))
                                               (conj (last action-history)
                                                     [(:id (players current-player))
                                                      action]))
                         :players players)]
    (utils/print-verbose verbosity
                         [{:fn "parse-action"}
                          {:action action
                           :bet-values bet-values
                           :current-bet current-bet
                           :current-player current-player}
                          {:final-state (parse-action action game-state :verbosity 0)}
                          {:initial-state game-state}])
    (condp utils/in? type
      ["Fold"] (check-active-players (assoc new-state
                                            :active-players (remove (partial = current-player)
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
                                     :current-bet (max current-bet (+ amount (bet-values current-player)))
                                     :min-raise (max min-raise (- (+ amount (bet-values current-player)) current-bet))))))

#_(parse-action ["Fold" 0.0] (init-game) :verbosity 1)

;;;;;;;;;;;;;;;;;;;;;;;
;;    Single Round   ;;
;;;;;;;;;;;;;;;;;;;;;;;

(defn blind
  "Pays blind or goes all in depending on money available. Assumes nonzero money
   Returns [updated-player action]"
  [player value & {:keys [verbosity]
                   :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       [{:fn "blind"}
                        {:player player
                         :value value}
                        {:final-state (blind player value :verbosity 0)}
                        {:initial-state {}}])
  (let [{money :money} player]
    (if (< money value)
      [(assoc player :money 0) ["All-In" money]]
      [(assoc player :money (- money value)) ["Bet" value]])))

(defn pay-blinds
  "Small blind and big blind bet or go all in if they don't have enough money"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (let [{players :players
         num-players :num-players
         bet-values :bet-values} game-state
        sb 0
        bb 1
        [sb-player sb-action] (blind (players sb) 0.5)
        [bb-player bb-action] (blind (players bb) 1.0)]
    (utils/print-verbose verbosity
                         [{:fn "pay-blinds"}
                          {:small-blind (players sb)
                           :big-blind (players bb)}
                          {:final-state (pay-blinds game-state :verbosity 0)}
                          {:initial-state game-state}])
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

(defn reset-action
  "Big blind acts first in all rounds except pre-flop"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       [{:fn "reset-action"}
                        {:last-action (last (last (:action-history game-state)))}
                        {:final-state (reset-action game-state :verbosity 0)}
                        {:initial-state game-state}])
  (assoc game-state
         :current-player 1
         :current-bet 0.0
         :min-raise 1.0
         :bet-values [0.0 0.0]
         :action-history (conj (:action-history game-state) [])))

(defn all-in? 
  "Has someone gone all-in in a previous betting round?"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       [{:fn "all-in?"}
                        {:bet-values (:bet-values game-state)
                         :players (:players game-state)}
                        {:final-state (reset-action game-state :verbosity 0)}
                        {:initial-state game-state}])
  (and (every? zero? (:bet-values game-state))
       (some zero? (map :money (:players game-state)))))

#_(defn round-over-checkall
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
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (let [{bet-values :bet-values
         current-player :current-player
         current-bet :current-bet} game-state]
    (utils/print-verbose verbosity
                         [{:fn "round-over-checkone"}
                          {:bet-values bet-values
                           :current-player current-player
                           :current-bet current-bet}
                          {:final-state (round-over-checkone game-state :verbosity 0)}
                          {:initial-state game-state}])
    (cond
      (all-in? game-state) true
      (every? zero? bet-values) (= 1 current-player)
      (utils/pre-flop-bb? game-state) false
      :else (= (bet-values current-player) current-bet))))



(defn bet-round
  "Runs betting for one round"
  [game-state game-history & {:keys [verbosity]
                              :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       [{:fn "bet-round"}
                        {:game-over (:game-over game-state)
                         :round (:betting-round game-state)}
                        {:final-state (bet-round game-state game-history :verbosity 0)}
                        {:initial-state game-state}])
  (if (or (:game-over game-state)
          (all-in? game-state))
    game-state
    (let [p1-move (make-move game-state
                             game-history)
          new-state (parse-action p1-move
                                  game-state)]
      (if (or (:game-over new-state)
              (round-over-checkone new-state))
        new-state
        (recur new-state
               game-history
               verbosity)))))

#_(bet-round (init-game [(utils/init-player random-agent :p0)
                         (utils/init-player random-agent :p1)])
             [])

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
                         [{:fn "bet-round"}
                          {:last-round last-round}
                          {:final-state (last-aggressor action-history players :verbosity 0)}
                          {:initial-state action-history
                           :players players}])
    (if (every? (partial contains? #{"All-In" "Check"})
                (map #(first (second %)) last-round))
      (:id (second players))
      (loop [actions actions]
        (if (empty? actions)
          (:id (second players))
          (if (contains? utils/aggressive-actions (first (second (last actions))))
            (first (last actions))
            (recur (drop-last actions))))))))

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
        l (last-aggressor action-history players)
        l-idx (first (keep-indexed #(if (= l (:id %2)) %1 nil) players))
        visible-idx (set (conj (map first winners) l-idx))]
    (utils/print-verbose verbosity
                         [{:fn "bet-round"}
                          {:community community
                           :hands hands
                           :players players
                           :last-aggressor l}
                          {:final-state (showdown game-state :verbosity 0)}
                          {:initial-state game-state}])
    (assoc game-state
           :game-over true
           :players updated-players
           :betting-round "Showdown"
           :visible-hands (keep-indexed #(if (contains? visible-idx %1) (vector (:id (players %1)) %2) nil) 
                                        hands))))



(defn next-round
  "Checks to see if all players but one have folded, then runs showdown or
   reveals the next card and proceeds to the next betting round"
  [game-state & {:keys [verbosity]
                 :or {verbosity 0}}]
  (let [{betting-round :betting-round
         community :community} game-state
        reset-state (check-active-players (reset-action game-state))]
    (utils/print-verbose verbosity
                         [{:fn "next-round"}
                          {:prev-round betting-round}
                          {:final-state (next-round game-state :verbosity 0)}
                          {:initial-state game-state}])
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


(defn bet-game 
  "Runs betting from initialization of game until showdown"
  [game-state game-history & {:keys [verbosity]
                              :or {verbosity 0}}]
  (utils/print-verbose verbosity
                       [{:fn "next-round"}
                        {:prev-round betting-round}
                        {:final-state (next-round game-state :verbosity 0)}
                        {:initial-state game-state}])
  (let [new-state (bet-round game-state game-history)]
    (if (:game-over new-state)
      new-state
      (let [next-round (next-round new-state)]
        (if (:game-over next-round)
          next-round
          (recur next-round 
                 game-history
                 verbosity))))))

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
   -> [players game-history]"
  ([players game-history deck]
  (let [game-state (init-game players deck)
        new-state (bet-game (pay-blinds game-state) game-history)]
    [(:players new-state) 
     (conj game-history (state-to-history game-state new-state))]))
  ([players game-history] (play-game players game-history (shuffle utils/deck))))


#_(clojure.pprint/pprint (bet-game (pay-blinds (init-game [(utils/init-player random-agent :p0)
                                    (utils/init-player random-agent :p1)]))
            []))

#_(clojure.pprint/pprint (second (play-game [(utils/init-player random-agent :p0)
              (utils/init-player random-agent :p1)]
             [])))
;;visibleHands is a map from playerid to vector of hands
;;;;;;;;;;;;;;;;;;;;;;;
;;   Multiple Games  ;;
;;;;;;;;;;;;;;;;;;;;;;;

(defn iterate-games
  "Plays num-games hands of poker with players switching from sb to bb every hand.
   An even number of hands ensures balanced play.
   Returns after num-games are reached or a player has no more money
   -> [players game-history]"
  [players num-games game-history & {:keys [verbosity]
                                     :or {verbosity 0}}]
  (let [num-games (int num-games)]
    (if (or (zero? num-games)
          (some zero? (map :money players)))
    [players game-history]
    (let [[players history] (play-game players game-history)]
      (recur (into [] (reverse players)) 
             (dec num-games) 
             history
             verbosity)))))


(defn iterate-games-reset
  "Plays num-games hands of poker with players switching from sb to bb every hand and resetting their money values
   An even number of hands ensures balanced play.
   Returns the total gain/loss of players and history after num-games are reached\\
   list: Whether to keep a list of the gains to return as a mean and stdev or to simply return
   the total gain over all games\\
   -> [players net-gain = [gain ...] or {:mean :stdev} game-history]"
  [players num-games game-history & {:keys [list? decks verbosity] 
                                     :or {list? false 
                                          decks nil
                                          verbosity 0}}]
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
      (let [[[p1 p2] h] (play-game players history (first decks))]
        (recur (update (update net-gain
                               (:id p1)
                               #((if list? conj +) % (- (:money p1) (:money (first players)))))
                       (:id p2)
                       #((if list? conj +) % (- (:money p2) (:money (first players)))))
               (dec n)
               (into [] (reverse players))
               h
               (rest decks))))))

(defn iterate-games-significantly
  "Iterate play between players until either statistical significance is reached or max-games are played\\
   -> winning agent or {:mean :stdev} of agent0"
  [agent0 agent1 max-games game-history &
   {:keys [verbosity winner?]
    :or {verbosity 0
         winner? false}}]
  (identity #_time (let [desired-std 0.5
        Z 1.96
        players [(utils/init-player agent0 :p0)
                 (utils/init-player agent1 :p1)]
        [_ {{s0 :stdev} :p0}]
        (iterate-games-reset players 100 []
                             :list? true)
        num-games (min max-games (utils/square (* Z (/ s0 desired-std))))
        [_ {{m1 :mean s1 :stdev} :p0}]
        (iterate-games-reset players num-games []
                             :list? true)]
    (when (>= verbosity 1) (clojure.pprint/pprint {:fn "iterate-games-significantly"
                                           :agents [agent0 agent1]
                                           :num-games num-games
                                           :Z Z
                                           :desired-std desired-std
                                           :CI [(- m1 (/ (* Z s1) (Math/sqrt num-games)))
                                                (+ m1 (/ (* Z s1) (Math/sqrt num-games)))]}))
    (if winner?
      (if (>= m1 0) agent0 agent1)
      {:mean m1 :stdev s1}))))

(defn iterate-games-symmetrical
  [players num-games game-history & {:keys [list? verbosity]
                                     :or {list? false
                                          verbosity 0}}]
  (loop [net-gain (zipmap (map :id players) (if list? [[] []] [0.0 0.0]))
         n (int num-games)
         players players
         history game-history]
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
            [[p11 p21] h1] (play-game players history deck)
            [[p22 p12] h2] (play-game (vec (reverse players)) h1 deck)]
        (recur (update (update net-gain
                               (:id p11)
                               #((if list? conj +) % (- (+ (:money p11) (:money p12))
                                                        (* 2 (:money (first players))))))
                       (:id p22)
                       #((if list? conj +) % (- (+ (:money p21) (:money p22))
                                                (* 2 (:money (second players))))))
               (dec n)
               players
               h2)))))



#_(take 2 (iterate-games-reset  [(utils/init-player utils/rule-agent :p0)
    (utils/init-player utils/random-agent :p1)] 10000 [] :list? true))

;;Plays 1000 (or 1000000) games between a rule-based agent and a random agent
;;Returns the players, the average net gain per hand, and the standard deviations of the wins per hand
#_(clojure.pprint/pprint (take 2 (iterate-games-reset [(utils/init-player wait-and-bet :p0)
                                                     (utils/init-player random-agent :p1)]
                                                    10000 #_1000000
                                                    []
                                                    :list? true)))
;;Plays up to 100 games until a player has no money and returns the history of the "interesting" ones
#_(apply (fn [p h] (vector p (take 10 (filter #(and (not= 1 (count (:action-history %)))
                                                  (utils/in? (flatten (:action-history %)) "Bet")
                                                  (> (abs (second (first (:net-gain %)))) 10)
                                                  #_(not (utils/in? (flatten (:action-history %)) "Fold"))) h))))
       (iterate-games [(utils/init-player random-agent :p0)
                       (utils/init-player utils/rule-agent :p1)]
                      100 []))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;          Slumbot HUNL interface           ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn parse-winnings
  "Takes a response from a finished game against slumbot, and performs final effects
   such as updating the money of each player and revealing hands\\
   w: how much money (in 0.01bb) the client has won\\
   r: response from Slumbot\\
   game-state: current game state\\
   -> game-state"
  [w r game-state]
  (let [{hands :hands
         players :players
         action-history :action-history} game-state
        client-pos (r "client_pos")
        game-state (assoc game-state
                          :hands (assoc hands
                                        client-pos (map utils/parse-card
                                                        (r "bot_hole_cards"))))
        game-state (if (not= "Fold" (utils/fsecond (last (last action-history))))
                     (showdown game-state)
                     game-state)]
    (assoc game-state
           :players (assoc players
                           client-pos (utils/set-money (players client-pos)
                                                       (- utils/slumbot-start-bb
                                                          (/ w utils/slumbot-bb)))
                           (- 1 client-pos) (utils/set-money (players (- 1 client-pos))
                                                             (+ utils/slumbot-start-bb
                                                                (/ w utils/slumbot-bb))))
           :game-over true)))


(defn parse-incr-action
  "Takes a response from slumbot, converts it into the incremental actions occurring
   since the last update, parses these incremental actions, and applies them to the game-state.\\
   If the game has ended, calls parse-winnings to return the final game-state.\\
   r: response from Slumbot\\
   game-state: current game state\\
   -> game-state"
  [r game-state]
  (let [_ (if (r "error_msg") (clojure.pprint/pprint {:error "ERROR"
                                                      :r r
                                                      :game-state game-state}) nil)
        old-action (r "old_action")
        action (r "action")
        incr (try (nth (re-find (re-pattern (str "(" old-action ")(.*)"))
                                action)
                       2)
                  (catch Exception e (println (.toString e) r game-state)))
        board (map utils/parse-card (r "board"))
        client-hand (map utils/parse-card (r "hole_cards"))
        client-pos (r "client_pos")]
    (loop [[_ a rem] (utils/get-action+roundover incr)
           game-state (assoc game-state
                             :community board
                             :visible board
                             :hands (assoc [nil nil] (- 1 client-pos) client-hand)
                             :r r)]
      (if a
        (recur (utils/get-action+roundover rem)
               (if (= a "/")
                 (next-round game-state)
                 (parse-action (utils/decode-action a game-state)
                               game-state)))
        (if-let [w (r "winnings")]
          (parse-winnings w r game-state)
          game-state)))))


(defn init-game-slumbot
  "Initializes and plays a game of poker, returning the updated players
   and updated game-history\\
   token: Login token\\
   agent: agent to play against slumbot\\
   game-history: current history of games played against slumbot\\
   -> [token game-state game-history]"
  [token agent game-history]
  (let [[token r] (utils/slumbot-new-hand token)
        players [(utils/init-player agent :client)
                 (utils/init-player :slumbot :bot)]
        client-pos (- 1 (r "client_pos"))
        game-state (pay-blinds (init-game (if (= 0 client-pos)
                                            players
                                            (into [] (reverse players)))))
        game-state (assoc game-state
                          :hands (assoc [nil nil]
                                        client-pos
                                        (map utils/parse-card
                                             (r "hole_cards")))
                          :community (map utils/parse-card (r "board"))
                          :visible (map utils/parse-card (r "board"))
                          :r r)
        game-state (parse-incr-action r game-state)
        #_(if (= 0 client-pos)
                     game-state
                     (parse-action (utils/decode-action (r "action") game-state)
                                   game-state))]
    [token game-state game-history]))



#_(parse-incr-action {"old_action" "",
                      "action" "",
                      "client_pos" 0,
                      "hole_cards" ["Ks" "4h"],
                      "board" ["Ad", "7d", "5s", "9h", "Tc"],
                      "bot_hole_cards" ["Qh", "Qd"],
                      "winnings" -600}
                     (pay-blinds (init-game [{:money 200 :id :p0} {:money 200 :id :p1}])))

(defn remove-unneeded-r [r]
  (remove #(contains? #{"session_total"
                        "session_baseline_total"
                        "old_action"
                        "session_num_hands"
                        "won_pot"} (first %)) r))

(defn play-game-slumbot 
  "Plays a game of poker from start to finish against Slumbot. When finished, summarizes the important features
   of the game and appends it to game-history.\\
   token: login token\\
   agent: agent to play against slumbot\\
   game-history: current history of agent vs slumbot games\\
   -> [token agent game-history]"
  [token agent game-history]
  (let [[token game-state game-history] (init-game-slumbot token agent game-history)]
    (loop [game-state game-state]
      (if (:game-over game-state)
        [token
         agent
         (conj game-history
               (assoc (state-to-history {:players (map #(utils/set-money % utils/slumbot-start-bb)
                                                (:players game-state))}
                                 game-state)
                      :r (into {}
                          (remove-unneeded-r (utils/recursive-copy 
                                   (:r game-state))))))]
        (let [action (make-move game-state game-history)
              r (utils/slumbot-send-action token
                                           (utils/encode-action action
                                                                game-state))]
          (recur (parse-incr-action r game-state)))))))

(defn iterate-games-slumbot
  "Plays multiple games against Slumbot and appends the results of those games to game-history.\\
   token: login token\\
   agent: agent to play against slumbot\\
   num-games: number of games to play\\
   game-history: current history of games played against slumbot\\
   -> [token game-history]"
  ([token agent num-games game-history]
   (loop [i num-games
          token token
          game-history game-history]
     (if (zero? i)
       [token game-history]
              (let [[token _agent game-history] (play-game-slumbot token 
                                                                   agent 
                                                                   game-history)]
                (recur (dec i)
                       token
                       game-history)))))
  ([token agent num-games] (iterate-games-slumbot token agent num-games []))
  ([agent num-games] (iterate-games-slumbot (utils/slumbot-login)
                                            agent
                                            num-games)))

#_(iterate-games-slumbot utils/random-agent 5)

(defn slumbot-rollout
  "Pits agent against slumbot num-samples*num-iter times and writes their match history to the file txt.\\
   agent: agent to play against slumbot\\
   txt: text file to output history to\\
   num-samples: number of times to sample and print to file\\
   num-iter: number of games played per sample\\
   -> game-history"
  [agent txt num-samples num-iter]
  (loop [i 0
         token (utils/slumbot-login)]
    (if (= i num-samples)
      nil
      (let [[token history] (iterate-games-slumbot token agent num-iter)]
        (spit txt
              (with-out-str (clojure.pprint/pprint history))
              :append true)
        (println i)
        (recur (inc i) token)))))


      

;;computing slumbot statistics
#_(let [l (read-string (slurp "slumbot-history-random.txt"))
      _ (println (count l))
      bot-hands (map #(:bot (into {} (:hands %))) l)
      bot-str (map #(let [{win :win total :total}
                          (utils/rollout (into #{} %))]
                      (try (float (/ win total))
                           (catch Exception e (println %)))) 
                   bot-hands)
      net-gain (map #(:bot (into {} (:net-gain %))) l)
      #_preflop-pot #_(map (fn [g]
                         (+ 1.5 (transduce (map utils/ssecond)
                                           +
                                           (first (:action-history g)))))
                       l)]
  (utils/corr bot-str net-gain))

;;calculating how representative my current slumbot history is
#_(let [l (group-by #(into #{} (first %))
                    (map #(vector
                           ((:r %) "hole_cards")
                           (/ ((:r %) "winnings") utils/slumbot-bb))
                         (read-string (slurp "slumbot-history-random.txt"))))]
    {:mean-games-per-hand (utils/mean (map #(count (second %)) l))})

;;Slumbot rollout
#_(loop [i 0]
    (if (= i 10)
      nil
      (do
        (time (slumbot-rollout random-agent
                               "slumbot-history-random.txt"
                               10
                               240))
        (utils/combine-vectors "slumbot-history-random.txt")
        (flush)
        (recur (inc i)))))





;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;         Runtime       ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;iterated versus competition
#_(time (take 2
            (iterate-games-reset [(utils/init-player utils/lazy-agent :p0)
                                  (utils/init-player utils/random-agent :p1)]
                                 100000
                                 []
                                 :list? true)))

#_(/ (* 155 1.96) (Math/sqrt 100000))

;;I gots to estimate the stdev, and then play enough games for it to be 
;;statistically significant.
;;computing statistically significant win results between two agents
#_(time (let [[_ {{m0 :mean s0 :stdev} :p0}] (iterate-games-reset [(utils/init-player utils/rule-agent :p0)
                                                       (utils/init-player random-agent :p1)]
                                                      100
                                                      []
                                                      :list? true)]
        (print (min 100000 (utils/square (* 2 (* s0 (* 0.5 m0))))))
  (take 2 (iterate-games-reset [(utils/init-player utils/rule-agent :p0)
                                (utils/init-player random-agent :p1)]
                               (min 100000 (utils/square (* 2 (/ s0 1))))
                               []
                               :list? true))))

;;statistics on mean and net gain?
#_(clojure.pprint/pprint
 (let [gains (map :net-gain (nth (iterate-games-reset [(utils/init-player random-agent :p0)
                                             (utils/init-player random-agent :p1)]
                                            1000
                                            []
                                            :list? true) 2))
      gains [(map #(if (= :p0 (ffirst %)) (utils/sfirst %) (utils/ssecond %)) gains)
             (map #(if (= :p1 (ffirst %)) (utils/sfirst %) (utils/ssecond %)) gains)]]
  {:p0 {:mean (utils/mean (first gains)) 
        :stdev (utils/stdev (first gains))}
   :p1 {:mean (utils/mean (second gains))
        :stdev (utils/stdev (second gains))}}))


#_(loop [i 0
       m {:p0 0 :p1 0}]
  (if (= i 50)
    {:p0 (/ (:p0 m) i) :p1 (/ (:p1 m) i)}
    (let [g (iterate-games [(utils/init-player random-agent :p0)
                            (utils/init-player utils/rule-agent :p1)]
                           1000 [])
          [p1 p2] (first g)
          m1 (update m (:id p1) (partial + (:money p1)))
          m2 (update m1 (:id p2) (partial + (:money p2)))]
      (recur (inc i) m2))))


#_(loop [i 0] (let [state (bet-game (pay-blinds (init-game [(utils/init-player utils/rule-agent :p0) (utils/init-player random-agent :p1)])) [])]
              (if (or (= 1 (count (:action-history state)))
                      (not (utils/in? (flatten (:action-history state)) "Bet"))
                      (utils/in? (flatten (:action-history state)) "Fold"))
                (recur (inc i))
                (do (println i)
                  state))))

;;using :r to fill in missing values in game-history
#_(let [l (read-string (slurp "slumbot-history-random.txt"))
        bot-hands (map #(map utils/parse-card ((:r %) "bot_hole_cards")) l)
        hands (map :hands l)
        new-hands (map (fn [hand bot-hand]
                         (mapv #(if (not= :bot (first %))
                                  %
                                  [:bot bot-hand]) hand))
                       hands
                       bot-hands)
        new-l (mapv (fn [g h]
                     (assoc g :hands h)) l new-hands)]
    #_(spit "slumbot-history-random.txt" 
          (with-out-str (clojure.pprint/pprint new-l))))