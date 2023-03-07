(ns poker.utils
  (:require [clojure.set :as set]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Poker Utilities and Card Calculations    ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn init-player
  "Each player is composed of a game agent, a starting amount of money (100bb) and a unique id
   agent is a function that takes a game state, legal moves, and game history, and chooses a legal move"
  [agent id]
  {:agent agent
   :money 1000
   :id id})

(defn in? [coll e]
  (some #(= e %) coll))

(defn add-money [player amount]
  (update player :money (partial + amount)))

(defn keys-from-value [v coll]
  (keep #(if (= v (second %)) (first %) nil) coll))


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

(defn cards-by-value [value] (map #(vector value %) suits))
(defn cards-by-suit [suit] (map #(vector % suit) values))

(defn deal-hands
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


(straight-sort [[7 "Hearts"]
                [6 "Hearts"]
                [5 "Hearts"]
                [4 "Hearts"]
                [3 "Hearts"]
                [2 "Clubs"]
                [7 "Clubs"]])

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

(def type-rankings {"Straight Flush" 9
                    "Four of a Kind" 8
                    "Full House" 7
                    "Flush" 6
                    "Straight" 5
                    "Three of a Kind" 4
                    "Two Pair" 3
                    "Pair" 2
                    "High Card" 1})


(defn hand-value 
  "Given 7 cards, picks the best hand and returns [value [numbers]]
   for the value of the hand in the ranking of possible hands and the numbers on 
   the cards in the hand."
  [cards]
  (let [{type :type hand :hand} (rank-hand cards)]
    (apply vector
           (type-rankings type)
           (map first hand))))



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

(defn maxes
  "Returns all individuals with maximum value of (key %)"
  ([key coll]
   (let [m (key (apply max-key key coll))]
     (filter #(= m (key %)) coll)))
  ([coll] (maxes identity coll)))


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

;;Construct a set of the outs and then get the count

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

(defn multiple-outs [cards]
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
  [cards]
    (set/union (into #{} (straight-outs cards))
               (into #{} (multiple-outs cards))
               (into #{} (flush-outs cards))))


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
            (= (second value) (apply max (rest value)))) ["Made" num-outs] ["Drawing" num-outs])))

(defn prefer-action [preferences actions]
  (loop [preferences preferences]
    (if (empty? preferences)
      (first actions)
      (if-let [a (first (filter #(= (first preferences) 
                                    (first %)) 
                                actions))]
        a
        (recur (rest preferences))))))

;; approx better, approx equal, approx worse 
;; for betting: how many outs does the opp have? Want to bet them out. But they might have good cards?

;;Mathematically optimal player, only considering their own cards
;; Look at cards. If made, bet/call. If drawing, check/call/fold. 
#_(defn rule-agent
  "Rule-based agent following heuristics for good poker play. Agent defaults to fold if no other preferred moves are available.
   In the Pre-Flop, an agent will call or check on premium or strong hands. On all other hands, it will check.
   Afterwards, if the agent 'made' a straight or higher, it will bet 1bb or call.
   If the agent hasn't made a strong hand, it will calculate its outs and check or call if it has enough equity to do so.
   At the river, the agent will bet or call if it's made a strong hand, and check otherwise"
  [game-state game-history money actions]
  (let [{hands :hands
         visible :visible
         current-player :current-player
         bet-values :bet-values
         current-bet :current-bet
         pot :pot
         betting-round :betting-round} game-state
        call-cost (- current-bet (bet-values current-player))
        [type num-outs] (pre-hand-type (concat (hands current-player)
                                               visible))
        {money :money} ((:players game-state) current-player)]
    (condp = betting-round
      "Pre-Flop" (let [q (hand-quality (hands current-player))]
                   (if (in? ["Premium" "Strong" "Drawing"] q)
                     (cond (and (in? (map first actions) "Call") (< current-bet 10)) (take 2 (first (filter #(= "Call" (first %)) actions)))
                           (in? (map first actions) "Check") ["Check" 0.0]
                           :else ["Fold" 0.0])
                     (if (in? (map first actions) "Check")
                       ["Check" 0.0]
                       ["Fold" 0.0])))
      "River" (if (= type "Made")
                (if (in? (map first actions) "Bet")
                  ["Bet" (float (/ money 10.0))]
                  (if (in? (map first actions) "Call")
                    (take 2 (first (filter #(= "Call" (first %)) actions)))
                    ["Fold" 0.0]))
                (if (in? (map first actions) "Check")
                  (take 2 (first (filter #(= "Check" (first %)) actions)))
                  ["Fold" 0.0]))
      (if (= type "Made")
        (if (in? (map first actions) "Bet")
          ["Bet" (float (/ money 10.0))]
          (if (in? (map first actions) "Call")
            (take 2 (first (filter #(= "Call" (first %)) actions)))
            ["Fold" 0.0]))
        (if (in? (map first actions) "Check")
          ["Check" 0.0]
          (if (> (/ num-outs (if (= betting-round "Flop") 47 46))
                 (/ call-cost (+ call-cost pot)))
            (if (in? (map first actions) "Call")
              (take 2 (first (filter #(= "Call" (first %)) actions)))
              ["Fold" 0.0])
            ["Fold" 0.0]))))))

(defn rule-agent
  "Rule-based agent following heuristics for good poker play. Agent defaults to fold if no other preferred moves are available.
   In the Pre-Flop, an agent will call or check on premium or strong hands. On all other hands, it will check.
   Afterwards, if the agent 'made' a straight or higher, it will bet 1bb or call.
   If the agent hasn't made a strong hand, it will calculate its outs and check or call if it has enough equity to do so.
   At the river, the agent will bet or call if it's made a strong hand, and check otherwise"
  [game-state game-history money actions]
  (let [{hands :hands
         visible :visible
         current-player :current-player
         bet-values :bet-values
         current-bet :current-bet
         pot :pot
         betting-round :betting-round} game-state
        call-cost (- current-bet (bet-values current-player))
        [type num-outs] (pre-hand-type (concat (hands current-player)
                                               visible))
        {money :money} ((:players game-state) current-player)]
    (condp = betting-round
      "Pre-Flop" (let [q (hand-quality (hands current-player))]
                   (if (and (in? ["Premium" "Strong" "Drawing"] q)
                            (< current-bet 10))
                     (take 2 (prefer-action ["Check" "Call" "Fold"] actions))
                     (take 2 (prefer-action ["Check" "Fold"] actions))))
      "River" (if (= type "Made")
                (let [a (prefer-action ["Bet" "Call" "All-In"] actions)]
                  (if (= "Bet" (first a))
                    ["Bet" (float (/ money 10.0))]
                    (take 2 a)))
                (take 2 (prefer-action ["Check" "Fold"] actions)))
      (if (= type "Made")
        (let [a (prefer-action ["Bet" "Call" "All-In"] actions)]
          (if (= "Bet" (first a))
            ["Bet" (float (/ money 10.0))]
            (take 2 a)))
        (if (in? (map first actions) "Check")
          ["Check" 0.0]
          (if (> (/ num-outs (if (= betting-round "Flop") 47 46))
                 (/ call-cost (+ call-cost pot)))
            (if (in? actions ["All-In" 0.0 0.0])
              ["All-In" 0.0]
              (take 2 (prefer-action ["Call" "Fold"] actions)))
            (if (in? actions ["All-In" 0.0 0.0])
              ["All-In" 0.0]
              ["Fold" 0.0])))))))





;;If I'm made, I look at his probability distribution. 1/n chance for each hand.
;;Each hand has a probability of making it. If I bet x, each hand with smaller chance than x/(2x+pot)
;;folds, giving me the pot. For each other hand, he has a p_i chance of making it. He will call. Pot is 2x+pot.
;; p_i * (2x+pot) goes to him (1-pi)*(2x+pot) goes to me.
;;Want to maximize p_lower/n * (x+pot) + sum[(1-pi)(2x+pot)] - x
;;Why don't I just maximize x? Diminishing returns? After all foldable hands, returns go to 0. When are returns equal to 1?
;;What are the hands and what are the outs?
(defn better-hands 
  "How many better 'made' hands are there?"
  [hand community]
  ())

(defn prob-suit-remaining 
  "Given hand and community cards, calculates the probability of a card of a given
   suit showing up"
  [hand community suit]
  (let [total (- 52 (count hand) (count community))
        same-suit? #(= suit (second %))
        suits (- 13 (count (filter same-suit? hand)) 
                 (count (filter same-suit? community)))]
    (float (/ suits total))))
(defn prob-value-remaining 
  "Given hand and community cards, calculates the probability of a card of a given
   value showing up"
  [hand community value]
  (let [total (- 52 (count hand) (count community))
        same-val? #(= value (first %))
        values (- 4 (count (filter same-val? hand))
                 (count (filter same-val? community)))]
(float (/ values total))))

(prefer-action ["Fold" "Bet"] [["Call" 0.0 0.0]
                         ["Fold" 0.0 0.0]
                         ["Bet" 1.0 99.0]])



#_(let [{h :hands c :community} (deal-hands 5)]
    (clojure.pprint/pprint {:hands h :community c})
    (showdown
     (map vector
          (range)
          (map (partial concat
                        c)
               h))))

#_(hand-value [[7 "Hearts"]
               [6 "Hearts"]
               [8 "Spades"]
               [4 "Hearts"]
               [8 "Hearts"]
               [8 "Clubs"]
               [6 "Clubs"]])