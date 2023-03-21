(ns poker.core
  (:require [clojure.set :as set]
            [clojure.pprint :as pp])
  (:gen-class :main true))

(defn square
  "Square input with autopromotion"
  [n] (*' n n))

(defn dot
  "Dot product.
   [a b] -> a.b
   [a] -> a.a"
  ([a b] (reduce + (map * a b)))
  ([a] (dot a a)))

(defn pd
  "Protected division"
  [a b]
  (try (if (zero? b)
         0
         (/ a b))
       (catch Exception _ 0)))

(defn mean
  "Takes the average of dataset"
  [data]
  (pd
   (reduce + 0 data)
   (float (count data))))

(defn stdev
  "Takes the standard deviation of the arguments"
  [data]
  (let [x (mean data)]
    (Math/sqrt (pd
                (reduce + 0 (map #(square (- % x)) data))
                (count data)))))

(defn xor
  "Exclusive or"
  [a b]
  (if a
    (if b
      false
      true)
    (if b
      true
      false)))

(defn clamp [range num]
  (let [[rmin rmax] range]
    (cond (< num rmin) rmin
          (> num rmax) rmax
          :else num)))

(defn rand-normal
  "Marsaglia polar method for normal random variable"
  [m std]
  (loop [u (-> (rand) (* 2) (- 1))
         v (-> (rand) (* 2) (- 1))]
    (let [s (+ (square u) (square v))]
      (if (and (< s 1) (> s 0))
        (+ m
           (* std u
              (Math/sqrt
               (/
                (* (- 2) (Math/log s))
                s))))
        (recur (rand) (rand))))))

(defn in? [coll e]
  (some #(= e %) coll))

(defn keys-from-value [v coll]
  (keep #(if (= v (second %)) (first %) nil) coll))

(defn maxes
  "Returns all individuals with maximum value of (key %)"
  ([key coll]
   (let [m (key (apply max-key key coll))]
     (filter #(= m (key %)) coll)))
  ([coll] (maxes identity coll)))

(defn lex-compare-vec
  "Lexicographical comparison of vectors"
  [vec1 vec2]
  (let [m (min (count vec1) (count vec2))]
    (loop [i 0]
      (if (>= i m)
        (compare (count vec1) (count vec2))
        (let [c (compare (nth vec1 i) (nth vec2 i))]
          (if (= c 0)
            (recur (inc i))
            c))))))

;;;;;;;;;;;;;;;;;;;;;;;
;;     Constants     ;;
;;;;;;;;;;;;;;;;;;;;;;;

(def values
  "Card values in a deck from 2 to 14"
  [2 3 4 5 6 7 8 9 10 11 12 13 14])

(def suits
  "Suits in a deck: Clubs, Diamonds, Hearts, and Spades"
  ["Clubs" "Diamonds" "Hearts" "Spades"])

(def deck
  "Unshuffled Deck of 52 cards
   Ordered by increasing value and from clubs to spades"
  (for [v values s suits] [v s]))

(def 
  card-pairs
  "List of all card pairs as a set of two cards"
  (for [x deck
        y deck :while (not= y x)]
    #{x y}))

(def card-triples
  "List of all 3 card combinations as a set of three cards"
  (for [x deck
        y deck :while (not= y x)
        z deck :while (and (not= z x)
                           (not= z y))]
    #{x y z}))

(def type-rankings {"Straight Flush" 9
                    "Four of a Kind" 8
                    "Full House" 7
                    "Flush" 6
                    "Straight" 5
                    "Three of a Kind" 4
                    "Two Pair" 3
                    "Pair" 2
                    "High Card" 1})

;;;;;;;;;;;;;;;;;;;;;;;
;;  Card Functions   ;;
;;;;;;;;;;;;;;;;;;;;;;;


(defn cards-by-value [value] (map #(vector value %) suits))
(defn cards-by-suit [suit] (map #(vector % suit) values))

(defn deal-hands
  "Deals hands to n players and allocates 5 community cards
   {:hands [hands] :community [cards]}"
  [n]
  (loop [hands []
         deck (shuffle deck)]
    (if (< (count hands) n)
      (recur
       (conj hands (into [] (take 2 deck)))
       (drop 2 deck))
      {:hands hands
       :community (take 5 deck)})))

(defn add-ace
  "Turns an ace into a 14 and a 1 for computing straights"
  [cards]
  (mapcat #(if (= 14 (first %))
             [% (vector 1 (second %))]
             [%]) cards))

(defn straight-sort
  "Group by value, sort by descending order, 
   and then group into straights"
  [cards]
  (eduction (partition-by first)
            (map-indexed vector)
            (partition-by #(+ (first %)
                              (first (first (second %)))))
            (map (fn [s] (map second s)))
            (sort-by first > (add-ace cards))))

(defn straight?
  "Determines whether cards can form a straight.
   Returns longest straight formable or false"
  [cards]
  (let [f (apply max-key count (straight-sort cards))]
    (if (>= (count f) 5)
      f
      false)))

(defn flush?
  "Determines whether cards can form a flush.
   Returns highest flush formable or false"
  [cards]
  (let [sorted (sort-by #(count (second %)) > (group-by second cards))
        f (second (first sorted))]
    (if (>= (count f) 5)
      (take 5 (sort-by first > f))
      false)))

(defn group-sort
  "Group and sort by size first then value"
  [cards]
  (sort-by #(vector (first %) (count (second %)))
           #(if (= (second %1) (second %2))
              (>= (first %1) (first %2))
              (>= (second %1) (second %2)))
           (group-by first cards)))

(defn multiple?
  "Selects best card-repetition hand out of multiple cards.
   Hand types include:
   Four of a Hands
   Full House
   Three of a Kind
   Two Pair
   One Pair
   High Card"
  [cards]
  (let [g (group-sort cards)
        f (second (first g))]
    (condp = (count f)
      4 {:type "Four of a Kind"
         :hand (conj f (apply max-key first
                              (mapcat second
                                      (rest g))))}
      3 (let [h (second (second g))]
          (if (>= (count h) 2)
            {:type "Full House"
             :hand (concat f (take 2 h))}
            {:type "Three of a Kind"
             :hand (concat f (take 2 (mapcat second
                                             (rest g))))}))
      2 (let [h (second (second g))]
          (if (>= (count h) 2)
            {:type "Two Pair"
             :hand (apply conj f
                          (apply max-key first
                                 (mapcat second
                                         (rest g)))
                          h)}
            {:type "Pair"
             :hand (concat f (take 3 (mapcat second (rest g))))}))
      {:type "High Card"
       :hand (mapcat second (take 5 g))})))

(defn rank-hand
  "Given 7 cards, selects the best 5-card hand and
   returns it along with the type of hand formed."
  [cards]
  (if-let [h (straight? cards)]
    (if-let [h (flush? (apply concat h))]
      {:type "Straight Flush"
       :hand h}
      {:type "Straight"
       :hand (map first (take 5 h))})
    (if-let [h (flush? cards)]
      {:type "Flush"
       :hand h}
      (multiple? cards))))

(defn hand-value 
  "Given 7 cards, picks the best hand and returns [value [numbers]]
   for the value of the hand in the ranking of possible hands and the numbers on 
   the cards in the hand."
  [cards]
  (let [{type :type hand :hand} (rank-hand cards)]
    (apply vector
           (type-rankings type)
           (map first hand))))

#_(hand-value [[7 "Hearts"]
               [6 "Hearts"]
               [8 "Spades"]
               [4 "Hearts"]
               [8 "Hearts"]
               [8 "Clubs"]
               [6 "Clubs"]])

#_(rank-hand [[7 "Hearts"]
               [6 "Hearts"]
               [8 "Spades"]
               [4 "Hearts"]
               [8 "Hearts"]
               [8 "Clubs"]
               [6 "Clubs"]])

(defn highest-hand
  "Determines winner of showdown between a list of >=1 [player-number [cards]]
   where cards are concatenated [hand community]"
  [player-cards]
  (let [hands (map #(assoc % 1 (hand-value (second %))) player-cards)
        n (count hands)]
    (loop [i 1
           maxes [(first hands)]
           max (second (first hands))]
      (if (>= i n)
        maxes
        (let [h (nth hands i)
              c (lex-compare-vec (second h) max)]
          (cond
            (= c 0) (recur (inc i) (conj maxes h) max)
            (> c 0) (recur (inc i) [h] (second h))
            :else (recur (inc i) maxes max)))))))

(defn hand-quality
  "Quality as defined in Sam Braids The Intelligent Guide to Texas Holdem Poker
   Premium - Strong - Drawing - Garbage"
  [hand]
  (let [[[v1 s1] [v2 s2]] hand
        ace? #(if (= % 14) 1 %)]
    (cond (= v1 v2)
          (condp <= v1
            10 "Premium"
            7 "Strong"
            "Drawing")
          (< (abs (- v1 v2)) 5) (cond (#{v1 v2} 14) "Premium"
                                      (< (- 14 (min v1 v2)) 5) "Strong"
                                      (and (= s1 s2) (> (min v1 v2) 9)) "Strong"
                                      (and (< (abs (- v1 v2)) 2) (= s1 s2)) "Drawing"
                                      :else "Garbage")
          (< (abs (- (ace? v1) (ace? v2))) 5) (if (and (= s1 s2) (#{v1 v2} 2)) "Drawing" "Garbage")
          (= s1 s2) (if (#{v1 v2} 14) "Drawing" "Garbage")
          :else "Garbage")))

(def hand-qualities ["Garbage" "Drawing" "Strong" "Premium"])

(defn straight-outs
  "Returns the outs to make a straight for cards"
  [cards]
  (let [sorted-cards (straight-sort cards)
        card-vals (set (filter int? (flatten sorted-cards)))
        max-straight (apply max-key count sorted-cards)
        straight-vals (filter int? (flatten max-straight))
        maxval (apply max straight-vals)
        minval (apply min straight-vals)]
    (condp = (count max-straight)
      4 (concat (when (< maxval 14) (cards-by-value (inc maxval)))
                  (when (> minval 2) (cards-by-value (dec minval))))
      3 (concat (when (card-vals (+ maxval 2)) (cards-by-value (inc maxval))) 
                (when (card-vals (- minval 2)) (cards-by-value (dec minval))))
      2 (concat (when (and (card-vals (+ maxval 2)) 
                           (card-vals (+ maxval 3))) (cards-by-value (inc maxval)))
                (when (and (card-vals (- minval 2)) 
                           (card-vals (- minval 3))) (cards-by-value (dec minval))))
           [])))

(defn flush-outs 
  "Returns the outs to make a straight for cards"
  [cards]
  (let [sorted-cards (group-by second cards)
        max-flush (second (apply max-key #(count (second %)) sorted-cards))]
    (if (= 4 (count max-flush))
      (filter #(not (in? max-flush %)) (cards-by-suit (second (first max-flush))))
      [])))

(defn multiple-outs 
  "Returns the outs to make a two pair or better"
  [cards]
  (let [multiple-sort (map second (group-sort cards))
        other-cards (fn [group] (filter (partial (complement in?) group) (cards-by-value (ffirst group))))]
    (condp = (count (first multiple-sort))
      3 (if (> (count (second multiple-sort)) 1)
          []
          (mapcat other-cards multiple-sort))
      2 (let [pairs (filter #(> (count %) 1) multiple-sort)]
          (mapcat other-cards pairs))
      [])))

(defn outs
  "Returns the outs to improve a hand"
  [cards]
    (set/union (into #{} (straight-outs cards))
               (into #{} (multiple-outs cards))
               (into #{} (flush-outs cards))))

;;;;;;;;;;;;;;;;;;;;;;;
;; Player Functions  ;;
;;;;;;;;;;;;;;;;;;;;;;;
(def init-money 200)

(defn init-player
  "Each player is composed of a game agent, a starting amount of money (100bb) and a unique id
   agent is a function that takes a game state, legal moves, and game history, and chooses a legal move"
  [agent id]
  {:agent agent
   :money init-money
   :id id})

(defn add-money [player amount]
  (update player :money (partial + amount)))

;;;;;;;;;;;;;;;;;;;;;;;
;;      Agents       ;;
;;;;;;;;;;;;;;;;;;;;;;;

(defn current-cards [game-state]
  (let [{hands :hands
         current-player :current-player
         community :community} game-state]
    (concat community (hands current-player))))

(defn coerce-legal [legal-actions action]
  (if-let [a (first (filter #(= (first action) (first %)) legal-actions))]
    (vector (first action) (clamp (rest a) (second action)))
    nil))



(defn equity?
  "Determines whether player has enough equity to call"
  [cards call-cost pot-size betting-round]
  (let [num-outs (count (outs cards))
        total-cards (condp = betting-round
                      "Flop" 47
                      "Turn" 46)]
    (> (/ num-outs total-cards)
       (/ call-cost (+ call-cost pot-size)))))

(defn pre-hand-type [cards]
  (let [value (hand-value cards)
        num-outs (count (outs cards))]
    (if (or (> (first value) 2)
            (= (second value)
               (apply max (rest value))))
      ["Made" num-outs]
      ["Drawing" num-outs])))

(defn prefer-action [preferences actions]
  (loop [preferences preferences]
    (if (empty? preferences)
      (first actions)
      (if-let [a (first (filter #(= (first preferences)
                                    (first %))
                                actions))]
        a
        (recur (rest preferences))))))

#_(prefer-action ["Fold" "Bet"] [["Call" 0.0 0.0]
                                 ["Fold" 0.0 0.0]
                                 ["Bet" 1.0 99.0]])


(defn rollout-single [init iter]
  (loop [m init
         i 0]
    (if (> i iter)
      m
      (recur 
       (let [h (deal-hands 2)
             hh (ffirst (highest-hand 
                       (map-indexed #(vector %1 
                                             (concat %2 
                                                     (:community h))) 
                                    (:hands h))))
             won (into #{} ((:hands h) hh))
             lost (into #{} ((:hands h) (- 1 hh)))]
         (assoc m 
                won (let [w (m won)] (assoc w :win (inc (:win w))
                                            :total (inc (:total w))))
                lost (let [l (m lost)] (assoc l :total (inc (:total l))))
                 ))
       (inc i)))))


(defn -main
  "I don't do a whole lot ... yet."
  [iter]
  (pp/pprint 
   (rollout-single 
    (zipmap card-pairs 
            (repeat {:win 0 :draw 0 :total 0})) 
    (read-string iter))))



