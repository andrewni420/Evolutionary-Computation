(ns poker.Andrew.headsupTest
  (:require [poker.headsup :refer :all]
            [poker.utils :as utils]))


(defn one-hot-card [card]
  (let [[value suit] card
        s (.indexOf utils/suits suit)
        v (- value 2)
        i (+ v (* 13 s))]
    (into [] (concat (repeat i 0)
                     [1]
                     (repeat (- 52 i 1) 0)))))

#_(defn one-hot-hand 
  [hand]
  (let [[[value1 suit1] [value2 suit2]] hand
        [s1 s2] (map (partial .indexOf utils/suits) [suit1 suit2])
        [v1 v2] (map #(- % 2) [value1 value2])
        i (+ (+ v1 (* 13 s1)) (* 52 (+ v2 (* 13 s2))))]
    (into [] (concat (repeat i 0)
                     [1]
                     (repeat (- 52 i 1) 0)))))

(defn one-hot-round 
  [round]
  )

(one-hot-card [3 "Clubs"])

(defn get-state [game-state]
  (let [{active-players :active-players} game-state]
    ()))

