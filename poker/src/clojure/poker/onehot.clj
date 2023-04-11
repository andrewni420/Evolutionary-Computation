(ns poker.onehot
  (:require [poker.headsup :refer :all]
            [poker.utils :as utils]))

(defn one-hot 
  "One-hots the ith feature in a vector of length n"
  [i n]
  (into [] (concat (repeat i 0)
                   [1]
                   (repeat (- n i 1) 0))))

(defn multi-hot
  "One-hot with the potential to satisfy multiple categories at once"
  [i-coll n]
  (let [i-coll (into #{} i-coll)]
    (into [] (map #(if (i-coll %) 1 0) (range n)))))


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
   [2 \"Clubs\", 2\"Diamonds\"] ... [2 \"Clubs\", 14 \"Spades\"], [2 \"Diamonds\", 2\"Hearts\"] ... [14 \"Diamonds\", 14\"Hearts\"], [14 \"Diamonds\", 14\"Spades\"], [14\"Hearts\", 14\"Spades\"]"
  [hand])

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

(defn encode-action-type 
  [action-type]
  (one-hot (.indexOf action-types action-type) 5))

(defn encode-money-bb
  "Specify the buckets for a linear / logscale progression\\
   One hot encodes the amount into the closest bucket by distance or distance of logs if logscale? is true"
  [amount & {:keys [logscale? buckets]
                         :or {logscale? false
                              buckets (range 1 10)}}]
  )

(defn encode-money-stack
  "Same as encode-money-bb, just divide by the pot first"
  [amount stack & {:keys [buckets logscale?]
                   :or {buckets [0.5 0.75 1 1.5 2]
                        logscale? false}}])

(defn encode-money-pot
  "Same as encode-money-bb, just divide by the stack first"
  [amount pot & {:keys [buckets logscale?]
                   :or {buckets [0.05 0.1 0.2 0.3 0.4]
                        logscale? false}}])

(defn encode-money
  [amount pot stack & {:keys [buckets logscale? multi-hot?]
                       :or {buckets [nil nil nil]
                            logscale? [false false false]
                            multi-hot? true}}]
  (assert (or multi-hot (boolean? logscale?)) 
          "Can only have one scale when multi-hot? is false for accurate comparison of buckets"))


(defn encode-player
  "Encodes either a player-number (different from cur-player) or a player-id with a list
   of the player ids as a one-hot vector of length 10, since at most 10 players are usually at a poker table\\
   -> [int ...10]"
  ([player]
   (one-hot player 10))
  ([player-id ids]
   (one-hot (.indexOf ids player-id) 10)))

;; positional encoding: [game#, round#, action#]
;; How to encode? inf games, 4 rounds, up to 10 actions

(defn encode-state [game-state]
  (let [{active-players :active-players} game-state]
    ()))



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
