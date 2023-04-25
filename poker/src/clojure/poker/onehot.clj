(ns poker.onehot
  (:require #_[poker.headsup :refer :all]
   [poker.utils :as utils]
            [clj-djl.ndarray :as nd])
  (:import ai.djl.ndarray.NDArray
           ai.djl.ndarray.index.NDIndex))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;   Functions for converting        ;;;
;;;      from game-related objects    ;;;
;;;        and float vectors          ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;Overview:
;;;
;;;    To interface between the poker game engine
;;;    and the transformer neural net, we need to 
;;;    one/multi-hot encode attributes of the game
;;;    as vectors of floats
;;;
;;;    encode-state encodes the visible game-state for each player: 
;;;    0-5 community cards, 2 hole cards, and possibly 2 opponent hole cards.
;;;        encode-cards multi-hot encodes cards
;;;
;;;    encode-action encodes the actions
;;;        encode-action-amount encodes a monetary amount in terms of the
;;;        big blind, pot size, and stack size
;;;        encode-action-type encodes the type of action
;;;
;;;    encode-position encodes the positional information
;;;    as a vector of [game-num round-num action-num current-player]
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;
;;;    Constants     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;

(def default-action-buckets [(utils/log-scale 1 utils/initial-stack :pow 1.25)
                             (utils/log-scale 0.1 10 :pow 1.25)
                             (utils/log-scale 0.05 1 :pow 1.25)])

;;;;;;;;;;;;;;;;;;;;;;;;
;;;    Utilities     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;

(defn one-hot
  "One-hots the ith feature in a vector of length n\\
   If i is nil or i>=n, then returns a 0 vector\\
   -> one hot vector"
  [i n]
  (if i
    (into [] (concat (repeat (min i n) 0)
                     (repeat (min 1 (- n i)) 1)
                     (repeat (- n i 1) 0)))
    (into [] (repeat n 0))))


(defn multi-hot
  "One-hot with the potential to satisfy multiple categories at once\\
   if i-coll is empty, returns a zero vector\\
   -> multi-hot vector"
  [i-coll n]
  (let [i-coll (into #{} i-coll)]
    (into [] (map #(if (i-coll %) 1 0) (range n)))))

;;;;;;;;;;;;;;;;;;;;;;;;
;;; One-hot Encoding ;;;
;;;;;;;;;;;;;;;;;;;;;;;;

(defn encode-card
  "Encodes a card just like in AlphaHoldem"
  [card]
  (one-hot (utils/card-index card) 52))


#_(encode-card [3 "Clubs"])

#_(defn one-hot-hand
    "This doesn't work because it encodes a hand as one of 52^2 hands, but half of these are identical"
    [hand]
    (let [[[value1 suit1] [value2 suit2]] hand
          [s1 s2] (map (partial .indexOf utils/suits) [suit1 suit2])
          [v1 v2] (map #(- % 2) [value1 value2])
          i (+ (+ v1 (* 13 s1)) (* 52 (+ v2 (* 13 s2))))]
      (into [] (concat (repeat i 0)
                       [1]
                       (repeat (- 52 i 1) 0)))))

(defn encode-hand
  "Encodes hand as one of (52 choose 2) = 1326 possible hands\\
   Cards should be ordered from 2 to 14 and from Clubs to Spades (arbitrarily chosen suit order) as in utils/suits.\\
   Maybe sort cards by ascending order:\\
   [1 2] ... [1 52] [2 3] ... [50 51] [50 52] [51 52]\\
   In terms of cards this would look like:\\
   [2 \"Clubs\", 2\"Diamonds\"] ... [2 \"Clubs\", 14 \"Spades\"], [2 \"Diamonds\", 2\"Hearts\"] ... [14 \"Diamonds\", 14\"Hearts\"], [14 \"Diamonds\", 14\"Spades\"], [14\"Hearts\", 14\"Spades\"]\\
   -> one-hot vector or index of one-hot encoding"
  [hand & {:keys [index?]
           :or {index? false}}]
  (if hand
    (let [[min-card max-card] (sort (map utils/card-index hand))
          idx (+ (/ (* min-card (- 103 min-card)) 2) (- max-card min-card 1))]
      (if index?
        idx
        (one-hot idx (utils/choose 52 2))))
    (into [] (repeat (utils/choose 52 2) 0))))


(defn encode-round
  [round]
  (one-hot (.indexOf utils/betting-rounds round) 4))

#_(encode-round "Pre-Flop")

(defn encode-active-players
  [active-players]
  (multi-hot active-players 10))

#_(encode-active-players [0] 2)

(defn encode-cards
  [cards]
  (multi-hot (map utils/card-index cards) 52))



#_(encode-cards [[3 "Clubs"] [5 "Clubs"] [3 "Spades"]])

(def action-types ["Check" "Call" "Fold" "Bet" "All-In"])

(defn action-type-idx
  "Gets the index of the given action type in action-types\\
   action-types: Check, Call, Fold, Bet, All-In\\
   -> integer"
  [type]
  (.indexOf action-types (if (= "Raise" type) "Bet" type)))

(defn encode-action-type
  "Encodes the given action type as a one-hot encoded vector of length 5\\
   -> one-hot encoded vector"
  [action-type]
  (one-hot (action-type-idx action-type) 5))

(defn encode-money-multiple
  "Specify the buckets for a linear / logscale progression as multiples of a 
   base amount, e.g. 1bb, stack, or pot\\
   One-hot encodes the amount into the closest bucket\\
   -> one-hot vector"
  [amount base & {:keys [buckets logscale?]
                  :or {buckets [0.5 0.75 1 1.5 2]
                       logscale? false}}]
  (let [fn (if logscale? #(Math/log %) identity)
        amount (fn amount)
        buckets (mapv (comp fn (partial * base)) buckets)]
    (one-hot (utils/closest-index amount buckets) (count buckets))))


#_(encode-money-multiple 100 200 :buckets #_[0.3 0.4 0.55 0.7])


(defn buckets-to-money
  "Converts the given buckets in terms of bb, pot, or stack, to a flattened vector where all 
   buckets are in terms of bb\\
   -> vector"
  ([buckets pot stack]
   (assert pot "Must have non-nil pot")
   (assert stack "Must have non-nil stack")
   (assert (= (count buckets) 3) "Must be  1 group of buckets for bb, pot, and stack")
   (into [] (concat (buckets 0)
                    (map (partial * pot) (buckets 1))
                    (map (partial * stack) (buckets 2)))))
  ([buckets game-state]
   (assert (and (:pot game-state)
                (:money ((:players game-state) (:current-player game-state))))
           (str "Incomplete game-state " game-state))
   (buckets-to-money buckets (:pot game-state) (:money ((:players game-state) (:current-player game-state))))))

#_(buckets-to-money [[1 2 3] [0.5 1] [0.05 0.2]] 10 200)

(defn encode-money
  "Encodes an amount either as a multi-hot concatenation of one-hot vectors of that amount in terms of 1bb, the pot size,
   and the player's stack size, or as a one-hot vector of all of the buckets combined.\\
   -> multi-hot vector or one-hot vector"
  [amount pot stack buckets & {:keys [logscale? multi-hot?]
                               :or {logscale? [false false false]
                                    multi-hot? true}}]
  (assert (or multi-hot? (boolean? logscale?))
          "Can only have one scale when multi-hot? is false for accurate comparison of buckets")
  (if multi-hot?
    (into [] (concat (encode-money-multiple amount 1 :buckets (buckets 0) :logscale? (logscale? 0))
                     (encode-money-multiple amount pot :buckets (buckets 1) :logscale? (logscale? 1))
                     (encode-money-multiple amount stack :buckets (buckets 2) :logscale (logscale? 2))))
    (encode-money-multiple amount
                           1
                           :buckets (buckets-to-money buckets pot stack)
                           :logscale? logscale?)))

#_(encode-money  25 12 200
                 [[5 10 20 30]
                  [0.5 1 2 3]
                  [0.05 0.1 0.2]]
                 :multi-hot? false
                 :logscale? true)


(defn encode-id
  "Encodes either a player-number (different from cur-player) or a player-id with a list
   of the player ids as a one-hot vector of length 10, since at most 10 players are usually at a poker table\\
   -> [int ...10]"
  ([player-number]
   (one-hot player-number 10))
  ([player-id ids]
   (one-hot (.indexOf ids player-id) 10)))

(defn encode-action
  "Encodes an action given the current state of the game as a multi-hot encoding with respect to 1bb, the pot, and the player's stack\\
   buckets: [(logscale 1-200 :pow 1.25) (logscale 0.1-10 :pow 1.25) (logscale 0.05-1 :pow 1.25)]\\
   -> multi-hot encoding"
  [action game-state & {:keys [buckets]
                        :or {buckets default-action-buckets}}]
  (if (:game-over game-state)
    (into [] (repeat (+ 5 (count (flatten buckets))) 0));;number of action-types is 5
    (into [] (concat (encode-action-type (first action))
                     (encode-money (second action)
                                   (:pot game-state)
                                   (:money ((:players game-state) (:current-player game-state)))
                                   buckets
                                   :logscale? [true true true])))))

(def action-length 64)

#_(count (encode-action ["Raise" 200.0] (poker.headsup/init-game)))


(defn encode-position
  "Encodes the position of the current player in the game\\
   -> [game-num, round-num, action-num, player-num]"
  [game-state]
  (if (:game-over game-state)
    [(:game-num game-state)
     (count utils/betting-rounds)
     0
     0]
    [(:game-num game-state)
     (.indexOf utils/betting-rounds (:betting-round game-state))
     (count (last (:action-history game-state)))
     (.indexOf (:player-ids game-state) (:id ((:players game-state) (:current-player game-state))))]))

(def position-length 4)

(defn encode-state
  "Encode the state of the game\\
   Returns one encoding for each player due to the hidden nature of the game\\
   Encoding is visible-cards, this player's hand, (if revealed) other player's hand, pot\\
   Currently only encodes cards as length 52 vectors\\
   -> {:id1 player1-state :id2 player2-state}"
  [game-state]
  (let [{hands :hands
         visible :visible
         current-player :current-player
         players :players
         pot :pot
         visible-hands :visible-hands} game-state
        current-id (:id (players current-player))
        other-id (:id (players (- 1 current-player)))]
    (assert (not (= current-id other-id)) "Players cannot have the same id")
    {current-id (into [] (concat (encode-cards visible)
                        ;;my hand
                                 (encode-cards (hands current-player))
                        ;;other player's hand
                                 (if (other-id visible-hands)
                                   (encode-cards (hands (- 1 current-player)))
                                   (encode-cards nil))
                        ;;optional pot encoding
                                 (encode-money-multiple pot 1 :buckets (utils/log-scale 1 (* 2 utils/initial-stack) :pow 1.25))))
     other-id (into [] (concat (encode-cards visible)
                      ;;my hand
                               (encode-cards (hands (- 1 current-player)))
                      ;;other player's hand
                               (if (current-id visible-hands)
                                 (encode-cards (hands current-player))
                                 (encode-cards nil))
                      ;;optional pot encoding
                               (encode-money-multiple pot 1 :buckets (utils/log-scale 1 (* 2 utils/initial-stack) :pow 1.25))))}))

(def state-length 183)

#_(:p0 (encode-state (poker.headsup/init-game)))





#_(utils/benchmark 10000 (play-game [utils/random-agent utils/random-agent]
                                    []))

#_(utils/benchmark 10000
                   (let [{hands :hands community :community} (utils/deal-hands 2 (shuffle utils/deck))]
                     (utils/highest-hand (map #(vector %
                                                       (concat (nth hands %)
                                                               community))
                                              (range 2))))
                   #_(utils/deal-hands 2 (shuffle utils/deck))
                   #_(utils/process-players [utils/random-agent utils/random-agent]))
