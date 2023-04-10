(ns poker.Andrew.headsupTest
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

(defn encode-card [card]
    (one-hot (utils/card-index card) 52))

#_(encode-card [3 "Clubs"])

#_(defn one-hot-hand 
  [hand]
  (let [[[value1 suit1] [value2 suit2]] hand
        [s1 s2] (map (partial .indexOf utils/suits) [suit1 suit2])
        [v1 v2] (map #(- % 2) [value1 value2])
        i (+ (+ v1 (* 13 s1)) (* 52 (+ v2 (* 13 s2))))]
    (into [] (concat (repeat i 0)
                     [1]
                     (repeat (- 52 i 1) 0)))))

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

(defn encode-position 
  [position]
  (one-hot position 10))

(defn encode-money-bb
  ""
  [amount]
  )

(defn encode-money-stack
  [])

(defn encode-money-pot
  [])

(init-game)

(defn encode-state [game-state]
  (let [{active-players :active-players} game-state]
    ()))



(utils/benchmark 10000 (play-game [utils/random-agent utils/random-agent]
                                 []))

(utils/benchmark 10000
                 (let [{hands :hands community :community} (utils/deal-hands 2 (shuffle utils/deck))]
                   (utils/highest-hand (map #(vector %
                                                     (concat (nth hands %)
                                                             community))
                                            (range 2))))
                 #_(utils/deal-hands 2 (shuffle utils/deck))
                 #_(utils/process-players [utils/random-agent utils/random-agent]))

;; 1000 games per game
;; 50 agents 
;; 1225 or 50 games per generation
;; 10 actions per game
;; 20 cores
;; 100 generations
;;10 hours  = 36000 seconds
;; bottom line = do a lot more sequence modeling than acquisition of information
;; start maybe with random actions for information acquisition

(float (/ (* 5000 60 60 1000) 50 100 1000 10))
(float (/ 210 8))
(float (/ 5000 100))


;;;; 80,000 games for Deepstack vs Humans
;;;; 135,000 games for Deepstack vs LBR
;;;; 405,000 games for Hyperborean vs LBR

;;;; 1,000,000 $0.08-$0.16 HUNL games for $6 
;;;; 5,000,000 $0.08-0.16 6p or less NL games for $6 