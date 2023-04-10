(ns poker.utils
  (:require [clojure.set :as set]
            ;;[clojure.core.matrix :as matrix]
            [clojure.pprint :as pprint]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Poker Utilities and Card Calculations    ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;
;; Math and Auxiliary ;;
;;;;;;;;;;;;;;;;;;;;;;;;

(defmacro print-verbose
  "When verbosity is more than 0, merges and prints the first verbosity maps in maps without
   evaluating the other maps.\\
   -> nil"
  [verbosity & maps]
  `(when (>= ~verbosity 1)
     (pprint/pprint
      (merge ~@(map-indexed (fn [idx# item#]
                              `(when (>= ~verbosity ~(inc idx#)) ~item#))
                            maps)))))

#_(print-verbose 2 
                 {:print1 "First map"}
                 {:print2 "Second map"}
                 {:print3 "Third map"})

(defmacro benchmark
  "Evaluates the body num-times times, prints the elapsed time, and returns nil.\\
   -> nil"
  [num-times body]
  `(time (loop [i# 0]
           (if (= i# ~num-times)
             nil
             (do ~body
                 (recur (inc i#)))))))

#_(benchmark 300000000
             (+ 1 2))

(defn print-return
  "Print x and return it\\
   -> x"
  [x & {:keys [pprint?]
        :or {pprint? true}}]
  (if pprint?
    (pprint/pprint x)
    (println x))
  x)

#_(print-return {:hi "hi"})

(defn make-consumer 
  "Make the given ifn implement java.util.function.Consumer\\
   IFn -> Consumer"
  [f]
  (reify java.util.function.Consumer
    (accept [_ x] (f x))))

#_(instance? java.util.function.Consumer
             (make-consumer #(println %)))

(defn make-supplier 
  "Make the given ifn implement java.util.function.Supplier\\
   IFn -> Supplier"
  [f]
  (reify java.util.function.Supplier
    (get [_] (f))))

#_(instance? java.util.function.Supplier
             (make-supplier (constantly 2)))

(defn make-function 
  "Make the given ifn implement java.util.function.Function\\
   IFn -> Function"
  [f]
  (reify java.util.function.Function
    (apply [_ x] (f x))))

#_(instance? java.util.function.Function
             (make-function inc))

(defn choose
  "Returns number of ways to choose k objects from n distinct objects
   Returns 0 when k>n\\
   integer, integer -> integer"
  [n k]
  (if (< n k)
    0
    (let [diff (min k (- n k))]
      (loop [res 1 n n k 1]
        (if (zero? (- k diff 1))
          res
          (recur (/ (* res n) k)
                 (dec n)
                 (inc k)))))))

#_(choose 6 2)

(defn permute
  "Returns the number of permutations of k objects from n distinct objects\\
   integer, integer-> integer"
  [n k]
  (let [k (- n k)]
    (loop [res 1 n n]
      (if (<= n k)
        res
        (recur (* res n) (dec n))))))

#_(permute 6 2)

(defn square
  "Square input with autopromotion\\
   number -> number"
  [n] (*' n n))

(defn round
  "Rounds x to num-digits number of digits after the decimal place\\
   number, integer -> number"
  [x num-digits]
  (let [e (Math/pow 10 num-digits)]
    (/ (Math/round (* x e)) e)))

(defn dot
  "Dot product.\\
   [a b] -> a.b\\
   [a] -> a.a\\
   [number ...], ([number ...]) -> number"
  ([a b] (reduce + (map * a b)))
  ([a] (dot a a)))

(defn pd
  "Protected division - dividing by 0 returns 0\\
  number, number-> number"
  [a b]
  (try (if (zero? b)
         0
         (/ a b))
       (catch Exception _ 0)))

(defn mean
  "Takes the average of the given collection\\
   [number ...] -> number"
  [data]
  (pd
   (reduce + 0 data)
   (float (count data))))

(defn de-mean 
  "Shifts data so that the mean is 0\\
   [number ...] -> [number ...]"
  [data]
  (let [m (mean data)]
    (map #(- % m) data)))

(defn stdev
  "Takes the standard deviation of the arguments\\
   [number ...] -> number"
  [data]
  (let [x (mean data)]
    (Math/sqrt (pd
                (reduce + 0 (map #(square (- % x)) data))
                (count data)))))

(defn de-std
  "Scales the data to have a stdev of 1.\\
   [number ...] -> [number ...]"
  [data]
  (let [s (stdev data)]
    (if (zero? s)
      data
      (map #(/ % s) data))))

(defn cov
  "Takes the covariance of two sequences\\
   [number ...n] [number ...n] -> number"
  [x y]
  (let [xmean (mean x)
        ymean (mean y)]
    (/ (transduce (map #(* (- (first %) xmean)
                           (- (second %) ymean)))
                  +
                  (map vector x y))
       (count x))))

(defn corr
  "Takes the pearson correlation coefficient of two sequences\\
   [number ...n] [number ...n] -> number"
  [x y]
  (/ (cov x y) (* (stdev x) (stdev y))))

(defn s-corr
  "Spearman correlation coefficient of two sequences. \\
   Assumes x and y have all distinct values\\
   [number ...n] [number ...n] -> number"
  [x y]
  (let [x (sort-by first (map vector x (range)))
        x (sort-by second (map conj x (range)))
        y (sort-by first (map vector y (range)))
        y (sort-by second (map conj y (range)))
        ssd (transduce (map (fn [[x y]]
                              (square (- (last x)
                                         (last y)))))
                       +
                       (map vector x y))
        n (count x)]
    (float (- 1 (/ (* 6 ssd)
                   (* n (- (square n) 1)))))))

(defn sigmoid 
  "Sigmoid function.\\
   number∈(-inf, inf) -> number∈(-1,1)\\"
  [x] (/ 1 (+ 1 (Math/exp (- x)))))

(defn logit 
  "Inverse sigmoid function, aka log odds function.\\
   Returns -inf and inf when x is in (-inf, -1] or [1, +inf), respectively\\
   number∈(0,1) -> number∈(-inf, +inf)\\"
  [x] (if (and (< x 1) (> x 0))
                  (Math/log (/ x (- 1 x)))
                  (* x ##Inf)))

(defn relu 
  "Rectified linear activation\\
   number -> number"
  [x] (if (< x 0) 0.0 x))

(defn softmax
  "Softmax function\\
   [number ...] -> [number(0, 1) ...]"
  [args]
  (let [e (map #(Math/exp %) args)
        s (reduce + e)]
    (map #(/ % s) e)))

(defn log-softmax
  "Log-Softmax function\\
   [number ...] -softmax> [number(0, 1) ...] -log> [number(-∞, 0) ...]"
  [args]
  (map #(Math/log %) (softmax args)))

(defn random-weighted 
  "Randomly chooses an element of coll with probabilities weighted by (f element)"
  [f coll]
  (let [p (map f coll)
        c (rand (reduce + p))]
    (loop [p p
           coll coll
           c c]
      (cond 
        (empty? p) nil
        (< c (first p)) (first coll)
        :else (recur (rest p)
               (rest coll)
               (- c (first p)))))))


(defn protected-prob-addn
  "Addition of probabilities while ensuring that the sum will never exceed 1\\
   Reduces to x + y when x<<1 and y<<1\\
   Reduces to (max x y) when 1-x<<1 or 1-y<<1\\
   x y -> x + y - xy"
  [x y]
  (- (+ x y) (* x y)))

(defn protected-prob-sub
  "Subtraction of probabilities while ensuring that the sum will never exceed 1\\
   Reduces to x - y when x>>y\\
   x y -> x - xy"
  [x y]
  (* x (- 1 y)))

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

(defn clamp
  "Clamps a number to a range\\
   range: [min max] the outputted value must be in this range\\
   num: the number input to be clamped\\
   -> num"
  [range num]
  (let [[rmin rmax] range]
    (cond (< num rmin) rmin
          (> num rmax) rmax
          :else num)))

(defn rand-normal
  "Marsaglia polar method for normal random variable\\
   -> float"
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

(defn rand-log 
  "Lograndom number between min and max"
  [min max]
  (if (zero? min) 
    (rand max)
    (let [d (Math/log (/ max min))]
      (* min (Math/exp (rand d))))))


(defn in?
  "Is e in the collection coll?\\
   -> bool"
  [coll e]
  (some #(= e %) coll))

(defn sfirst
  "sugar for (second (first coll))"
  [coll]
  (second (first coll)))

(defn ssecond
  "sugar for (second (second coll))"
  [coll]
  (second (second coll)))

(defn fsecond
  "sugar for (first (second coll))"
  [coll]
  (first (second coll)))

(defn keys-from-value
  "Get all keys in coll that map to v\\
   -> [k ...]"
  [v coll]
  (keep #(if (= v (second %)) (first %) nil) coll))

(defn recursive-copy
  "Recursively copy over elements of a collection.\\
   Used to convert from python objects to clojure objects."
  [coll]
  (into (empty coll)
        (map #(cond
                (map-entry? %) (recursive-copy (vec %))
                (coll? %) (recursive-copy %)
                :else %) coll)))

(defn maxes
  "Returns all individuals with maximum value of (key %)\\
   -> [individuals ...]"
  ([key coll]
   (let [m (key (apply max-key key coll))]
     (filter #(= m (key %)) coll)))
  ([coll] (maxes identity coll)))

(defn lex-compare-vec
  "Lexicographical comparison of vectors. For vectors with the same initial values, longer ones are considered larger.\\
   vec1 > vec2 -> 1\\
   vec1 < vec2 -> -1\\
   vec1 = vec2 -> 0"
  [vec1 vec2]
  (let [m (min (count vec1) (count vec2))]
    (loop [i 0]
      (if (>= i m)
        (compare (count vec1) (count vec2))
        (let [c (compare (nth vec1 i) (nth vec2 i))]
          (if (= c 0)
            (recur (inc i))
            c))))))

(defn drop-supersets
  "Given a collection of sets, removes any set which is a superset of another set in the collection.\\
   -> #{set ...}"
  [sets]
  (let [sorted (sort-by count sets)]
    (reduce (fn [res set] (if (some #(set/superset? set %)
                                    res)
                            res
                            (conj res set)))
            #{}
            sorted)))


(defn combine-vectors
  "Given a text file of the form [sample ...][sample ...] ...\\
   combines them into a single vector of the form [sample ...]\\
   txt: the file to modify\\
   -> [sample ...]"
  [txt]
  (let [samples (read-string (str "[" (slurp txt) "]"))]
    (spit txt (with-out-str (clojure.pprint/pprint (into [] (mapcat identity samples)))))))

(defn shape
  [arr]
  (loop [s []
         arr arr]
    (if (coll? arr)
      (if (empty? arr)
        (conj s 0)
        (recur
         (conj s (count arr))
         (first arr)))
      s)))

;;;;;;;;;;;;;;;;;;;;;;;
;;     Constants     ;;
;;;;;;;;;;;;;;;;;;;;;;;

(def epsilon
  "Resolution of equals operation"
  (Math/pow 10 -5))

(def initial-stack 
  "Initial amount of money in terms of bb"
  200.0)

(defn pairs
  "Given a collection, returns all sets of distinct pairs from that collection.
   If the collection only has one element, returns a set of that element.\\
   -> [#{e1 e2} ...]"
  [coll]
  (if (= 1 (count coll))
    (list #{(first coll)})
    (for [x coll
          y coll :while (not= y x)]
      #{x y})))

(def values
  "Card values in a deck from 2 to 14"
  [2 3 4 5 6 7 8 9 10 11 12 13 14])

(def suits
  "Suits in a deck: \\
   Clubs, Diamonds, Hearts, and Spades"
  ["Clubs" "Diamonds" "Hearts" "Spades"])

(def deck
  "Unshuffled Deck of 52 cards.
   Ordered by increasing value and from clubs to spades"
  (for [v values s suits] [v s]))

(def card-pairs
  "List of all card pairs as a set of two cards"
  (pairs deck))

(def card-triples
  "List of all 3 card combinations as a set of three cards"
  (for [x deck
        y deck :while (not= y x)
        z deck :while (and (not= z x)
                           (not= z y))]
    #{x y z}))

(def type-rankings
  "Map from the name of a hand type to an integer representing its strength.
   Hands with higher strengths beat hands with lower strengths:\\
   Straight Flush > Four of a Kind > Full House > Flush > 
   Straight > Three of a Kind > Two Pair > Pair > High Card"
  {"Straight Flush" 9
   "Four of a Kind" 8
   "Full House" 7
   "Flush" 6
   "Straight" 5
   "Three of a Kind" 4
   "Two Pair" 3
   "Pair" 2
   "High Card" 1})

(def betting-rounds 
  "The four betting rounds in poker:\\
   Pre-Flop, Flop, Turn, River"
  ["Pre-Flop" "Flop" "Turn" "River"])

(def possible-actions 
  "Possible actions in poker:\\
   Check, Call, Fold, Bet, Raise, All-In"
  ["Check" "Call" "Fold" "Bet" "Raise" "All-In"])

(def hand-qualities
  "List of possible hand qualities from The Intelligent Guide to Texas Holdem Poker"
  ["Garbage" "Drawing" "Strong" "Premium"])

;;Stored files

(def rollout
  "Win statistics for each hand against a uniform distribution of opponent hands all-in preflop.\\
   {hand {:win win :draw draw :total total} ...}"
  (read-string (slurp "rollout.txt")))

#_(def S-C-numbers
  "Sklansky-Chubukov numbers describing the strength of each hand when all-in preflop\\
   {hand [num-better-hands win%-when-underdog money-needed-to-prefer-all-in-to-fold]}"
  (read-string (slurp "Sklansky-Chubukov.txt")))

#_(def new-numbers (into {} (map #(let [[h [a b c]] %]
                                    (if (or (= h "AA") (= h "AAo"))
                                      [(if (= 2 (count h))
                                         (str h "o") h) [a b c]]
                                      [(if (= 2 (count h))
                                         (str h "o")
                                         h) [a b (round (float (/ c 3))
                                                        4)]])) S-C-numbers)))

#_(spit "Sklansky-Chubukov.txt" (with-out-str
                                  (clojure.pprint/pprint new-numbers)))

;;;;;;;;;;;;;;;;;;;;;;;
;;  Card Functions   ;;
;;;;;;;;;;;;;;;;;;;;;;;

(defn card-index
  "Gets the index of the card in an array of cards listed by number then by suit\\
   order of suits: Clubs, Diamonds, Hearts, Spades\\
   -> int"
  [card]
  (let [[value suit] card
        s (.indexOf suits suit)
        v (- value 2)]
    (+ v (* 13 s))))

(defn preflop-win-chance
  "Computes the chance of a hand outperforming a random hand when all-in preflop\\
   -> float"
  [hand]
  (let [{win :win
         draw :draw
         total :total} (rollout (into #{} hand))]
    (float (/ (+ win (/ draw 2)) total))))

(defn facecard-from-value
  "Interprets integer value as letter corresponding to face card\\
   v: integer value
   -> string"
  [v]
  (condp = v
    10 "T"
    11 "J"
    12 "Q"
    13 "K"
    14 "A"
    (str v)))


(defn value-from-facecard
  "Interprets facecard letter as integer value\\
   v: string representation of card value
   -> integer"
  [v]
  (condp = v
    "T" 10
    "J" 11
    "Q" 12
    "K" 13
    "A" 14
    (read-string v)))

(def suit-from-abbr
  "Maps a 1-letter abbreviation to the full name of the suit"
  {"s" "Spades"
   "d" "Diamonds"
   "h" "Hearts"
   "c" "Clubs"})

(defn hand-to-string
  "Compresses a two-card hand into a string of the form
   \"AKo\" for the two hand-values and \"s\" or \"o\" for \"suited\" or \"offsuit\"\\
   The larger value must come first\\
   -> string"
  [h]
  (let [[[v1 s1] [v2 s2]] h
        s (cond #_(= v1 v2) #_nil
           (= s1 s2) "s"
                :else "o")]
    (str (facecard-from-value (max v1 v2))
         (facecard-from-value (min v1 v2))
         s)))

(defn string-to-hands
  "Reverses compress-hand, turning a string of the form \"AKo\"
   into a set of possible hands\\
   -> #{hand ...}"
  [string]
  (let [v1 (value-from-facecard (str (first string)))
        v2 (value-from-facecard (str (second string)))]
    (if (= (last string) \s)
      (into #{} (for [s suits]
                  #{[v1 s]
                    [v2 s]}))
      (into #{} (if (= v1 v2)
                  (for [s (pairs suits)]
                    #{[v1 (first s)]
                      [v2 (second s)]})
                  (for [s1 suits
                        s2 suits :when (not= s1 s2)]
                    #{[v1 s1]
                      [v2 s2]}))))))

(def possible-hands
  "A vector of all 169 possible hands of the form \"AKo\""
  (into []
        (for [v1 values]
          (into []
                (for [v2 values
                      s (if (> v1 v2) "s" "o")]
                  (str (facecard-from-value v1)
                       (facecard-from-value v2)
                       s))))))

#_(defn rollout-winrate [hand]
  (let [{win :win total :total} (rollout (into #{} hand))]
    (float (/ win total))))




;;;;;;;;;;;;;;;;;;;;;;;
;;  Card Functions   ;;
;;;;;;;;;;;;;;;;;;;;;;;


(defn cards-by-value
  "Gets all cards with the same value.\\
   value: value of the cards\\
   -> [card ...]"
  [value]
  (map #(vector value %) suits))

(defn cards-by-suit
  "Gets all card of the same suit\\
   suit: suit of the cards\\
   -> [card ...]"
  [suit]
  (map #(vector % suit) values))

(defn one-hot-card 
  "One-hot encoding of card as one of 52 cards\\
   -> [0... 1 0...]"
  [card]
  (let [[v s] card
        n (+ (* (- v 2) 4) (.indexOf suits s))]
    (into [] (concat (repeat n 0)
                     [1]
                     (repeat (- 52 n 1) 0)))))

(defn one-hot 
  "Returns a matrix of one-hot encodings\\
   -> [[0... 1 0...] ...]"
  [cards]
  (into [] (map one-hot-card cards)))

(defn deal-hands
  "Deals hands and allocates 5 community cards\\
   n: number of players to deal hands to\\
   -> {:hands [hands] :community [cards]}"
  ([n deck]
  (loop [hands []
         deck deck]
    (if (< (count hands) n)
      (recur
       (conj hands (into [] (take 2 deck)))
       (drop 2 deck))
      {:hands hands
       :community (take 5 deck)})))
  ([n] (deal-hands n (shuffle deck))))

#_(deal-hands 2 deck)

(defn add-ace
  "Turns an ace into a card with value 14 and a card with value 1 for computing straights\\
   cards: list of cards to process\\
   -> [card ...]"
  [cards]
  (mapcat #(if (= 14 (first %))
             [% (vector 1 (second %))]
             [%]) cards))

(defn straight-sort
  "Group by value, sort by descending order, 
   and then group into straights\\
   cards: cards to sort\\
   -> (straight = (value = [card = [v s] ...] ...) ...)"
  [cards]
  (into []
        (eduction (partition-by first)
                  (map-indexed vector)
                  (partition-by #(+ (first %)
                                    (first (first (second %)))))
                  (map (fn [s] (map second s)))
                  (sort-by first > (add-ace cards)))))

(defn straight?
  "Determines whether cards can form a straight.
   -> longest straight formable or false"
  [cards]
  (let [f (apply max-key count (straight-sort cards))]
    (if (>= (count f) 5)
      f
      false)))

(defn flush?
  "Determines whether cards can form a flush.
   -> highest flush formable or false"
  [cards]
  (let [sorted (sort-by #(count (second %)) > (group-by second cards))
        f (second (first sorted))]
    (if (>= (count f) 5)
      (take 5 (sort-by first > f))
      false)))

(defn group-sort
  "Group and sort by size first then value\\
   -> [group = [v [card = [v s] ...]] ...]"
  [cards]
  (sort-by #(vector (first %) (count (second %)))
           #(if (= (second %1) (second %2))
              (>= (first %1) (first %2))
              (>= (second %1) (second %2)))
           (group-by first cards)))

(defn multiple?
  "Selects best n-of-a-kind hand out of a collection of cards.
   Hand types include:\\
   Four of a Kind\\
   Full House\\
   Three of a Kind\\
   Two Pair\\
   One Pair\\
   High Card\\
   -> string"
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
             :hand (concat f h
                          [(apply max-key first
                                 (mapcat second
                                         (rest (rest g))))])}
            {:type "Pair"
             :hand (concat f (take 3 (mapcat second (rest g))))}))
      {:type "High Card"
       :hand (mapcat second (take 5 g))})))


(defn rank-hand
  "Given 7 cards, selects the best 5-card hand and
   returns it along with the type of hand formed.\\
   -> {:type hand-type :hand [card ...]}"
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
  "Given 7 cards, picks the best hand and returns [value, numbers...]
   for the value of the hand in the ranking of possible hands and the numbers on 
   the cards in the hand.\\
   -> [hand-value, card-value ...]"
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
   where cards are concatenated [hand community]
   -> [[player-number [card ...]] ...]"
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
  "Hand quality as defined in Sam Braids The Intelligent Guide to Texas Holdem Poker
   Premium - Strong - Drawing - Garbage
   -> string"
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



(defn straight-outs
  "Returns the outs to make a straight\\
   -> [card ...]"
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
  "Returns the outs to make a straight for cards\\
   -> [card ...]"
  [cards]
  (let [sorted-cards (group-by second cards)
        max-flush (second (apply max-key #(count (second %)) sorted-cards))]
    (if (= 4 (count max-flush))
      (filter #(not (in? max-flush %)) (cards-by-suit (second (first max-flush))))
      [])))

(defn multiple-outs
  "Returns the outs to make a two pair or better\\
   -> [card ...]"
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
  "Returns the outs to improve a hand\\
   -> #{card ...}"
  [cards]
  (set/union (into #{} (straight-outs cards))
             (into #{} (multiple-outs cards))
             (into #{} (flush-outs cards))))

;;;;;;;;;;;;;;;;;;;;;;;
;; Player Functions  ;;
;;;;;;;;;;;;;;;;;;;;;;;

(def init-money
  "The initial amount of money each player has, as a multiple of the big blind"
  initial-stack)

(defn init-player
  "Each player is composed of a game agent, a starting amount of money (200bb) and a unique id.\\
   agent: game-state game-history money legal-actions -> action\\
   -> {:agent agent :money money :id id}"
  [agent id]
  {:agent agent
   :money init-money
   :id id})

(defn process-players
  "Given a collection of agents, initializes them as players identified by their position in the collection\\
   Given a collection of players, returns them.\\
   -> players"
  [players]
  (into [] (map-indexed #(if (and (map? %2) (:money %2) (:id %2))
                           %2
                           (init-player %2 (keyword (str "p" %1))))
                        players)))

(defn add-money
  "Adds money to a player\\
   -> player"
  [player amount]
  (update player :money (partial + amount)))

(defn set-money
  "Sets the amount of money a player has\\
   -> player"
  [player amount]
  (assoc player :money amount))

(def aggressive-actions
  "Actions which are considered aggressive: Bet, Raise, and the first All-In"
  #{"Bet" "Raise" "All-In"})

;;;;;;;;;;;;;;;;;;;;;;;
;;      Agents       ;;
;;;;;;;;;;;;;;;;;;;;;;;

(defn current-cards
  "Returns the cards visible to the current player.\\
   -> [card ...]"
  [game-state]
  (let [{hands :hands
         current-player :current-player
         community :community} game-state]
    (concat community (hands current-player))))

(defn coerce-legal
  "Coerces an action to be within legal bounds. \\
   -> action"
  [legal-actions action]
  (if-let [a (first (filter #(= (first action) (first %)) legal-actions))]
    (vector (first action) (clamp (rest a) (second action)))
    nil))



(defn equity?
  "Determines whether player has enough equity to call\\
   -> boolean"
  [cards call-cost pot-size betting-round]
  (let [num-outs (count (outs cards))
        total-cards (condp = betting-round
                      "Flop" 47
                      "Turn" 46)]
    (> (/ num-outs total-cards)
       (/ call-cost (+ call-cost pot-size)))))

(defn pre-hand-type
  "Categorizes a hand into \"made\" or \"drawing\"\\
   -> string"
  [cards]
  (let [value (hand-value cards)
        num-outs (count (outs cards))]
    (if (or (> (first value) 2)
            (= (second value)
               (apply max (rest value))))
      ["Made" num-outs]
      ["Drawing" num-outs])))

(defn prefer-action
  "Loops through preferences = [action-type ...]. If a preference matches the name of a legal action,
   returns that legal action. Otherwise returns the first legal action\\
   -> [action-type min max]"
  [preferences actions]
  (loop [preferences preferences]
    (if (empty? preferences)
      (first actions)
      (if-let [a (first (filter #(= (first preferences)
                                    (first %))
                                actions))]
        a
        (recur (rest preferences))))))





(defn prefer-action-value
  "Loops through preferences [[action-type value] ...]. If a preference matches the name of a legal action,
   returns that legal action. Preferences without a value return the legal action matching the action-type
   Otherwise returns the first legal action\\
   -> [action-type value] or [action-type min max]"
  [preferences actions]
  (loop [preferences preferences]
    (if (empty? preferences)
      (first actions)
      (if-let [a (first (filter #(= (ffirst preferences)
                                    (first %))
                                actions))]
        (cond 
          (not (sfirst preferences)) a 
          (and (<= (sfirst preferences) (nth a 2)) 
                 (>= (sfirst preferences) (second a))) (first preferences)
          :else (recur (rest preferences)))
        (recur (rest preferences))))))


#_(prefer-action-value [["Fold" -1.0] ["Bet" 100.0]] [["Call" 0.0 0.0]
                                                      ["Fold" 0.0 0.0]
                                                      ["Bet" 1.0 99.0]])

(defn prefer-weighted-action
  "Loops through preferences = [[[action-type (value)] weight] ...], choosing a random preference with weighted probability
   and popping it from preferences. If this preference has the name and value of a legal action,
   returns that preference. Otherwise returns the first legal action with the minimum value for that legal action\\
   -> [action-type value]"
  [preferences actions]
  (loop [preferences preferences]
    (if (empty? preferences)
      (first actions)
      ;;choose random with weight
      (let [[[t v] :as p] (random-weighted second preferences)
            f (if v
                #(and (>= v (second %))
                      (<= v (nth % 2)))
                (constantly true))]
        (if-let [a (first (filter #(and (= t (first %))
                                          (f %))
                                    actions))]
          (if v 
            (first p) 
            (take 2 a))
          (recur (remove (partial = p) preferences)))
        ))))


#_(prefer-weighted-action [[["Fold" 0.0] 1] 
                           [["Bet" 20.0] 10] 
                           [["Bet" 10.0] 5]
                           [["Call" 10.0] 100]]
                          [["Call" 0.0 0.0]
                           ["Fold" 0.0 0.0]
                           ["Bet" 1.0 99.0]])



;;If I'm made, I look at his probability distribution. 1/n chance for each hand.
;;Each hand has a probability of making it. If I bet x, each hand with smaller chance than x/(2x+pot)
;;folds, giving me the pot. For each other hand, he has a p_i chance of making it. He will call. Pot is 2x+pot.
;; p_i * (2x+pot) goes to him (1-pi)*(2x+pot) goes to me.
;;Want to maximize p_lower/n * (x+pot) + sum[(1-pi)(2x+pot)] - x
;;Why don't I just maximize x? Diminishing returns? After all foldable hands, returns go to 0. When are returns equal to 1?
;;What are the hands and what are the outs?


(defn missing-values
  "Given a list of card values, returns the set of missing values needed to make a straight or straight-draw\\
   vals: unique values of the cards.\\
   height: required height of straight (draw)\\
   total-size: 5 for straight, 4 for straight draw\\
   -> #{#{(val1) (val2)} ...}"
  [vals height total-size]
  (if (empty? vals)
    0
    (let [min-start (max 1
                       (- height (dec total-size))
                       (- (apply min vals) 2))
        max-end (min 15 (+ (apply max vals) 3))
        val-set (into #{} vals)]
    (drop-supersets (transduce (comp (map #(range % (+ % total-size)))
                                     (map #(set/difference (into #{} %)
                                                           val-set))
                                     (map #(cond (> (count %) 2) #{}
                                                 (empty? %) #{#{}}
                                                 :else #{%})))
                               into
                               #{}
                               (range min-start
                                      (- max-end
                                         (dec total-size))))))))

(defn get-single-handcount
  "Helper method for counting hands\\
   Given a bunch of hands in the form of values #{v1}, returns the number of ways
   a hand could have that value.\\
   values: list of sets of 1 value that must be in the hand\\
   cards-per-value: 1 if the suit matters, 4 if it doesn't\\
   hand-vals: the frequency of each value appearing in my hand, which therefore cannot be in the opponent's hand"
  [values other-cards cards-per-value hand-vals]
  (if-let [num-fixed values]
    (let [num-cards (transduce
                     (map #(- cards-per-value (get hand-vals (first %) 0)))
                     +
                     num-fixed)]
      (- (* (dec other-cards)
            num-cards)
         (choose num-cards 2)))
    0))

(defn get-pair-handcount
  "Helper method for counting hands\\
   Given a bunch of hands in the form of value pairs #{v1 v2}, returns the number of ways
   a hand could have those values. Hands with two of the same value are represented as #{v1 -v1}\\
   values: list of sets of value pairs that are in the hand\\
   cards-per-value: 1 if the suit matters, 4 if it doesn't\\
   hand-vals: the frequency of each value appearing in my hand, which therefore cannot be in the opponent's hand"
  [values cards-per-value hand-vals]
  (if-let [pairs values]
    (transduce
     (map #(/ (* (max 0
                      (- cards-per-value
                         (get hand-vals (abs (first %)) 0)
                         (if (< (first %) 0) 1 0)))
                 (max 0
                      (- cards-per-value
                         (get hand-vals (abs (second %)) 0)
                         (if (< (second %) 0) 1 0))))
              (if (zero? (+ (first %) (second %)))
                2
                1)))
     +
     pairs)
    0))

#_(get-pair-handcount [#{4 -4}]
                      4
                      {4 0 2 1})

;;If it's for a flush, we only care about the ones that are in the suit of the
;;flush, so it's only ever going to be #{card suit} or #{card card suit}
(defn hands-with-values
  "Given a list of sets of values a hand can have and the number of other cards,
   returns the number of hands satisfying any one of those sets.\\
   Values: A list of sets defining the possible values in a hand.
   #{} -> any hand satisfies\\
   #{4} -> hand must have a 4\\
   #{4 -4} -> hand must have two fours\\
   #{4 5} -> hand must have a four and a five\\
   my-hand: Cards in my hand which therefore cannot be in the opponent's hand\\
   other-cards: the number of cards remaining\\
   suit: nil if the suit doesn't matter, or the suit if it does matter.
   -> int"
  ([values my-hand other-cards suit]
   (let [hand-vals (frequencies (map first
                                     (if suit
                                       (filter #(= suit (second %))
                                               my-hand)
                                       my-hand)))
         by-count (group-by count values)
         cards-per-value (if suit 1 4)]
     (+ (if (get by-count 0)
          (choose other-cards 2)
          0)
        (get-single-handcount (get by-count 1) other-cards cards-per-value hand-vals)
        (get-pair-handcount (get by-count 2) cards-per-value hand-vals))))
  ([values my-hand other-cards] (hands-with-values values my-hand other-cards nil)))


#_(time (hands-with-values #{#{6} #{-14 14} #{13 -13} #{-12 12} #{2} #{-10 10} #{5} #{11 -11} #{4}}
                     [[6 "Hearts"] [14 "Spades"]]
                     45))

(defn straight-probability
  "Returns the probability that a random distribution of opponent hands can make a 
   straight (draw) using the community cards.\\
   community: visible community cards\\
   my-hand: my cards\\
   total-size = 5 or 4 for flush (draw)\\
   height = required height of flush (draw)
   -> float"
  [community my-hand total-size height & {:keys [set?] :or {set? false}}]
   (if (empty? community)
     0
     (let [additional-cards (- 52 (count community) 2)
         hands (missing-values (map first (group-by first community))
                               height
                               total-size)]
     (float (/ (hands-with-values hands my-hand additional-cards)
               (choose additional-cards 2))))))

#_(straight-probability [[4 "Spades"]
                         [6 "Spades"]
                         [10 "Spades"]
                         [11 "Hearts"]
                         [12 "Hearts"]]
                        [[14 "Hearts"]
                         [13 "Spades"]] 4 1)

(defn num-gte-card
  "Number of cards in the hand of the same suit as the card
   with greater or equal value\\
   -> int"
  [card hand]
  (count (filter #(and (>= (first %) (first card))
                       (= (second %) (second card)))
                 hand)))

(defn num-lt-card
  "Number of cards in the hand of the same suit as the card
   with smaller value\\
   -> int"
  [card hand]
  (count (filter #(and (< (first %) (first card))
                       (= (second %) (second card)))
                 hand)))

(defn max-value
  "Returns the highest value among the cards\\
   -> int"
  [cards]
  (first (apply max-key first cards)))

(defn count-suit
  "Counts the number of cards of the same suit as the given suit
   -> int"
  [suit cards]
  (count (filter #(= suit (second %)) cards)))

(defn flush-with-height
  "Helper method for flush-probability
   Given suited cards and a target suit and height, returns the number of ways to make
   a flush (draw) of that suit of at least that height using two additional hole cards\\
   -> int"
  [height total-size cards my-hand additional-cards]
  (let [s (sfirst cards)
        h (max-value cards)
        c (count cards)]
    (if (>= h height)
      (condp = (- total-size c)
        0 (choose additional-cards 2)
        1 (* (- 13 c (count-suit s my-hand))
             (dec additional-cards))
        2 (choose (- 13 c (count-suit s my-hand)) 2))
      (let [num-higher (max 0 (- 15
                                 height
                                 (num-gte-card [height s]
                                               my-hand)))]
        (- (* num-higher
              (if (> c (- total-size 2))
                (dec additional-cards)
                (- 13 1 c (count-suit s my-hand))))
           (choose num-higher 2))))))

#_(flush-with-height "Spades"
                     13
                     5
                     [[2 "Spades"] [6 "Spades"] [12 "Spades"]]
                     [[13 "Spades"] [14 "Spades"]]
                     45)


(defn flush-probability
  "Returns the probability that a random distribution of opponent hands can make a 
   flush (draw) using the community cards.\\
   community: visible community cards\\
   my-hand: my cards\\
   total-size = 5 or 4 for flush (draw)\\
   height = required height of flush (draw)\\
   -> float"
  [community my-hand total-size height & {:keys [set?] :or {set? false}}]
  (if (empty? community)
    0
    (let [community-req (- total-size 2)
          suit-group (group-by second community)
          max-suits (filter #(>= (count (second %)) community-req)
                            suit-group)
          additional-cards (- 52 (count community) 2)
          total-hands (choose additional-cards 2)]
      (float (/ (transduce (map #(flush-with-height height
                                                    total-size
                                                    (second %)
                                                    my-hand
                                                    additional-cards));;c
                           +
                           max-suits)
                total-hands)))))


#_(flush-probability [[1 "Hearts"]
                      [9 "Hearts"]
                      [10 "Spades"]
                      [11 "Spades"]
                      [12 "Spades"]]
                     [[13 "Hearts"] [3 "Hearts"]] 4 14)

(defn straight-flush-probability
  "Returns the probability that a random distribution of opponent hands can make a 
   straight flush (draw) using the community cards.\\
   community: visible community cards\\
   my-hand: my cards\\
   total-size = 5 or 4 for straight flush (draw)\\
   height = required height of straight flush (draw)\\
   -> float"
  [community my-hand total-size height & {:keys [set?] :or {set? false}}]
  (if (empty? community)
    0
    (let [community-req (- total-size 2)
          suit-group (group-by second community)
          max-suits (filter #(>= (count (second %)) community-req)
                            suit-group)
          additional-cards (- 52 (count community) 2)
          total-hands (choose additional-cards 2)]
      (float (/ (transduce
                 (map #(let [values (missing-values (map first
                                                         (group-by first
                                                                   (second %)))
                                                    height
                                                    total-size)]
                         (hands-with-values values my-hand additional-cards (first %))))
                 +
                 max-suits)
                total-hands)))))

;;#{2 16 [first Hearts] [second Spades]}


(defn get-possible-hands
  "Gets all hands, represented as sets of values, that can make the given pattern or better
   with the community cards.\\
   cards: visible community cards grouped by size = [[card ...] [card ...]]
   pattern: [[size1 val1] [size2 val2]] for example, \"Aces full of Kings\" is [[3 14] [2 13]]
   as there are 3 aces, which have value 13, and 2 kings, which have value 13.\\
   -> #{hand = #{(val1) (val2)} ...}"
  [cards pattern]
  (let [[[size1 val1] [size2 val2]] pattern
        [c1 c2] cards
        c1-values (if-let [c (first c1)]
                    [(first c)]
                    (range (max 2 val1) 15))
        c2-values (if-let [c (first c2)]
                    [(first c)]
                    (range (max 2 val2) 15))
        c1-size (count c1)
        c2-size (count c2)
        candidates (for [v1 c1-values
                         v2 c2-values
                         fh (range 3)
                         sh (range 3)
                         :when (and (< (+ fh sh) 3)
                                    (<= (+ fh c1-size) 4)
                                    (<= (+ sh c2-size) 4)
                                    (not= v1 v2))]
                     [[(+ fh c1-size) v1] [(+ sh c2-size) v2]])
        candidates (filter #(>= (lex-compare-vec (concat (map first %)
                                                         (map second %))
                                                 [size1 size2 val1 val2])
                                0)
                           candidates)]
    (drop-supersets (into #{}
                          (map #(let [[[c1 v1] [c2 v2]] %
                                      fh (- c1 c1-size)
                                      sh (- c2 c2-size)]
                                  (set/union (condp = fh 0 #{} 1 #{v1} 2 #{v1 (- v1)})
                                             (condp = sh 0 #{} 1 #{v2} 2 #{v2 (- v2)})))
                               candidates)))))



(defn multiple-probability
  "Computes the probability of the opponent having a hand with multiple cards of a value.\\
   community: visible community cards\\
   my-hand: my cards\\
   pattern: [[size1 height1] [size2 height2]], e.g. [[3 14] [2 13]] for \"Aces full of Kings\"\\
   -> float"
  [community my-hand pattern & {:keys [set?] :or {set? false}}]
  (let [other-cards (- 52 2 (count community))
        [[size1] [size2]] pattern
        m (conj (map second (group-by first community)) [])
        candidates (for [c1 m  c2 m
                         :when (or (not= c1 c2) (= c1 []))
                         :when (>= (count c1) (- size1 2))
                         :when (>= (count c2) (- size2 2))
                         :when (>= (+ (count c1) (count c2) 2)
                                   (+ size1 size2))]
                     [c1 c2])]
    (transduce (map #(get-possible-hands % pattern))
               (completing #(drop-supersets (set/union %1 %2)))
               #{}
               candidates)
    #_(float (/ (hands-with-values (transduce (map #(get-possible-hands % pattern))
                                            (completing #(drop-supersets (set/union %1 %2)))
                                            #{}
                                            candidates)
                                 my-hand
                                 other-cards)
              (choose other-cards 2)))))


#_(multiple-probability [[2 "Hearts"]
                         [2 "Spades"]
                         [4 "Hearts"]
                         [5 "Hearts"]
                         [6 "Hearts"]]
                        []
                        [[2 10] [1 10]])

(defn better-hands 
  "Given a hand type like \"Two Pair\" or a hand ranking like 9 for \"Straight Flush\"
   returns the names of the better hand types"
  [type]
  (let [v (if (string? type) (type-rankings type) type)]
    (map first (filter #(> (second %) v) type-rankings))))

(def type-to-pattern
  "Given a hand type that requires multiple of a card to get, returns the
   pattern = [[size1 minval1] [size2 minval2]]"
  {"Four of a Kind" [[4 2] [0 2]]
   "Full House" [[3 2] [2 2]]
   "Three of a Kind" [[3 2] [0 3]]
   "Two Pair" [[2 2] [2 2]]
   "Pair" [[2 2] [0 2]]
   "High Card" [[1 2] [0 2]]})

(defn prob-better-hand-rough 
  "Rough calculation of the probability of a random hand combined with the community cards
   being better than my-hand combined with the community cards.\\ 
   Considers straights, flushes, and multiple of a kind separately.\\
   -> float"
  [community my-hand]
  (let [[v & n] (hand-value (concat community my-hand))]
    (+ (cond (< v 5) (straight-probability community my-hand 5 2)
             (= v 5) (straight-probability community my-hand 5 (first n))
             :else 0)
       (cond (< v 6) (flush-probability community my-hand 5 2)
             (= v 6) (flush-probability community my-hand 5 (first n))
             :else 0)
       (cond (< v 5) (- (straight-flush-probability community my-hand 5 2))
             (= v 5) (- (straight-flush-probability community my-hand 5 (first n)))
             (= v 9) (straight-flush-probability community my-hand 5 (first n))
             :else (straight-flush-probability community my-hand 5 2))
       (cond (#{5 6} v) (multiple-probability community my-hand (type-to-pattern "Full House"))
             (= v 9) 0
             :else (let [[v1 v2] (partition-by identity n)]
                     (multiple-probability community
                                         my-hand
                                         [[(count v1) (get v1 0 0)]
                                          [(count v2) (get v2 0 0)]]))))))


  
;;As this illustration shows, an "overcard" on the flop drastically increases the chances
;;that the opponent has a better hand than me, without even considering the fact that
;;the opponent is likely to be skewed towards better hands if they're seeing the flop.
#_(time (prob-better-hand-rough [[2 "Hearts"] [9 "Spades"] [10 "Clubs"] [3 "Hearts"] [4 "Hearts"]]
                          [[11 "Hearts"] [11 "Clubs"]]))


#_(prob-better-hand-rough [[2 "Hearts"] [9 "Spades"] [12 "Clubs"]]
                          [[11 "Hearts"] [11 "Clubs"]])


#_(let [[v & n] (hand-value [[5 "Hearts"] [5 "Spades"] 
                           [4 "Diamonds"]
                         [2 "Spades"] [2 "Diamonds"]
                         [4 "Spades"] [3 "Spades"]])
      _ (println v n)
      [v1 v2] (partition-by identity n)]
  [[(count v1) (first v1)]
   [(count v2) (first v2)]])
  
#_(defn prob-better-hand [community my-hand {:keys [set?] :or {set? false}}]
  (let [[v n](hand-value (concat community my-hand))
        better (better-hands v)
        additional-cards (- 52 2 community)]
    (loop [better better
           hands #{}]
      (if (empty? better)
        (if set? hands (/ (hands-with-values hands)))))))



(defn max-freq-suit
  "Returns the maximum frequency of any suit in the community cards\\
   -> int"
  [community]
  (apply max (map second (frequencies (map second community)))))



(defn nuts
  "Given a set of community cards, finds the two hole cards that will make the highest
   possible hand.\\
   -> #{card1 card2}"
  [community])

(defn prob-suit-remaining
  "Given hand and community cards, calculates the probability of a card of a given
   suit showing up\\
   -> float"
  [hand community suit]
  (let [total (- 52 (count hand) (count community))
        same-suit? #(= suit (second %))
        suits (- 13 (count (filter same-suit? hand))
                 (count (filter same-suit? community)))]
    (float (/ suits total))))

(defn prob-value-remaining
  "Given hand and community cards, calculates the probability of a card of a given
   value showing up\\
   -> float"
  [hand community value]
  (let [total (- 52 (count hand) (count community))
        same-val? #(= value (first %))
        values (- 4 (count (filter same-val? hand))
                  (count (filter same-val? community)))]
    (float (/ values total))))

(defn rollout-single
  "Iterated comparison of two random hands with 5 random community cards.\\
   init: initial map of hand to win record\\
   iter: number of iterations\\
   -> {hand {:win win :draw draw :total total} ...}"
  [init iter]
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
                lost (let [l (m lost)] (assoc l :total (inc (:total l))))))
       (inc i)))))

(defn compute-highest
  "Given two hands, returns the result of comparing them to each other with 
   5 community cards, where a hand will fold if it isn't strong enough.\\
   -> {:win [...] :draw [...]}"
  [rollout hands]
  (let [[h1 h2] (:hands hands)
        c (:community hands)
        [f1 f2] (map #(let [r (rollout (set %))]
                        (< (:win r)
                           (/ (:total r)
                              2)))
                     [h1 h2])
        #_a #_(println "compute-highest" [f1 f2] [h1 h2] c)]
    (cond (and f1 f2) {:win [0 0] :draw [1 1]}
          f1 {:win [0 1] :draw [0 0]}
          f2 {:win [1 0] :draw [0 0]}
          :else {:win (assoc [0 0]
                             (ffirst (highest-hand
                                      (map-indexed #(vector %1
                                                            (concat %2 c))
                                                   [h1 h2])))
                             1)
                 :draw [0 0]})))

(defn update-map-add
  "Values are 1 or 0.5 or 0 for what to add to the win\\
   -> {hand {:win win :draw draw :total total}}"
  [map keys win draw]
  (loop [m map
         k keys
         win win
         draw draw]
    (if (or (empty? win) (empty? draw) (empty? k))
      m
      (recur (let [{w :win t :total d :draw} (m (first k))]
               (assoc m (first k) {:win (+ w (first win))
                                   :draw (+ d (first draw))
                                   :total (inc t)}))
             (rest k)
             (rest win)
             (rest draw)))))

#_(defn rollout-single-2 [init rollout]
    (loop [m init
           i 0]
      (if (> i 1000000)
        m
        (recur
         (let [h (deal-hands 2)
               #_a #_(println "hands" h)
               hh (compute-highest rollout h)
               #_b #_(println "highest" hh)]
           (update-map-add m
                           (map set (:hands h))
                           (:win hh)
                           (:draw hh)))
         (inc i)))))




#_(def rollout2 (zipmap card-pairs (repeat {:win 0 :draw 0 :total 0})))

(defn initialize-rollout
  "Initializes rollout.txt file with a clean slate.
   -> {hand {:win 0 :draw 0 :total 0} ...}"
  [txt]
  (spit txt
        (with-out-str
          (clojure.pprint/pprint
           (zipmap card-pairs (repeat {:win 0 :draw 0 :total 0}))))))


;;Rollout with 20 million samples
(defn rollout-samples
  "Pits two hands against each other num-samples*num-iter amount of times.
   Writes to file every num-iter times, so can cancel evaluation whenever.\\
   num-samples: number of times to sample and update rollout.txt\\
   num-iter: number of iterations per sample\\
   -> {hand {:win 0 :draw 0 :total 0} ...}"
  [num-samples num-iter]
  (loop [i 0]
    (if (= i num-samples)
      nil
      (do
        (println i)
        (spit "rollout.txt"
              (with-out-str
                (clojure.pprint/pprint
                 (rollout-single
                  (read-string
                   (slurp "rollout.txt"))
                  num-iter))))
        (recur (inc i))))))



(defn rollout-update
  "Combines two text files containing maps where corresponding values are added together.
   Spits result into txt1"
  [txt1 txt2]
  (let [roll1 (read-string (slurp txt1))
        roll2 (read-string (slurp txt2))]
    (spit txt1
          (with-out-str
            (clojure.pprint/pprint (merge-with (partial merge-with +) roll1 roll2))))))

;;;;;;;;;;;;;;;;;;;;;;;
;;       Agents      ;;
;;;;;;;;;;;;;;;;;;;;;;;
(defn pre-flop-bb?
  "Checks to see if action has been passed to the big blind in the preflop round.
   Big blind gets to move after betting preflop
   -> boolean"
  [game-state]
  (let [{betting-round :betting-round
         current-player :current-player
         current-bet :current-bet} game-state]
    (and (= betting-round "Pre-Flop")
         (= current-player 1)
         (= current-bet 1.0))))

(defn legal-actions
  "The possible actions and their conditions are as follows:\\
   Fold - always possible, but not allowed when check is possible\\
   Check - only possible when no one has betted\\
   Call - Only possible when at least one person has betted\\
   Raise - Must raise by at least the previous bet or amount by which bet was raised\\
   Bet - Only possible when no one has betted. Must be at least 1bb\\
   All-In - always possible\\
   -> [[type least most] ...]"
  [game-state]
  (let [{min-bet :min-bet
         min-raise :min-raise
         bet-values :bet-values
         current-bet :current-bet
         current-player :current-player
         players :players} game-state
        p (players current-player)
        {money :money} p
        call-cost (- current-bet (bet-values current-player))]
    (#(cond
        (in? % ["Check" 0.0 0.0]) %
        (in? % ["All-In" 0.0 0.0]) %
        :else (concat % [["Fold" 0.0 0.0]]))
     (concat [["All-In" money money]]
             (cond
               (zero? money) []
               (every? zero? bet-values)
               (concat [["Check" 0.0 0.0]]
                       (when (>= money min-bet)
                         [["Bet" min-bet (dec money)]]))
               :else
               (concat []
                       (if (pre-flop-bb? game-state)
                         [["Check" 0.0 0.0]]
                         (when (>= money call-cost)
                           [["Call" call-cost call-cost]]))
                       (let [amount (- (+ current-bet min-raise)
                                       (bet-values current-player))]
                         (when (>= money amount)
                           [["Raise" amount (dec money)]]))))))))

#_(legal-actions (init-game [{:money 10.0} {:money 0.0}]))


(def always-fold
  "Agent that always folds"
  (constantly ["Fold" 0]))

(defn calling-station
  "Calls all bets, including going all-in if the bet is larger than its stack. Otherwise, checks.\\
   Prefers actions in the order [Call Check All-In]"
  [game-state _game-history]
  (take 2 (prefer-action ["Call" "Check" "All-In"] (legal-actions game-state))))

(defn action-station
  "Constantly prefers an action"
  [action]
  (fn [game-state _game-history]
    (take 2 (prefer-action [action] (legal-actions game-state)))))


(defn wait-and-bet
  "'Optimal' strategy against calling-station:\\
   Check before river, and check/fold or call after depending on whether hand is better
   than average"
  [game-state _game-history]
  (let [{betting-round :betting-round
         current-player :current-player
         hands :hands
         community :community} game-state
        actions (legal-actions game-state)]
    (if (= betting-round "River")
      ;;not quite right - doesn't use community cards
      (if (> (preflop-win-chance (hands current-player)) 0.5)
        (take 2 (prefer-action ["All-In"] actions))
        (take 2 (prefer-action ["Check" "Fold"]
                               actions)))
      (take 2 (prefer-action ["Check" "Call" "Fold"]
                             actions)))))

[:betting-round 3 :integer_eq 
 :exec_if :holecard-winrate {:type :probability :literal 0.5} :probability_gte
     :exec_if :all-in 'close 
     :fold :total-weight :inc-log :check 'close 
 'close
 :fold :total-weight :inc-log :call :total-weight :inc-log :check]

(defn random-agent
  "Agent that chooses randomly between available action types"
  [game-state _game-history]
  (let [actions (legal-actions game-state)
        [type min max] (rand-nth actions)]
    (if (< (rand) 0.0)
      (let [types (map first actions)
            type (cond (in? types "Call") "Call"
                       (in? types "Check") "Check"
                       (in? types "Fold") "Fold")]
        (into [] (take 2 (first (filter #(= type (first %)) actions)))))
      (vector type min #_(+ min (math/round (rand (- max min))))))))

(defn lazy-agent [game-state _game-history]
  (let [actions (legal-actions game-state)]
    (take 2 (prefer-action [] actions))))

(defn rule-agent
  "Rule-based agent following heuristics for good poker play. Agent defaults to fold if no other preferred moves are available.\\
   In the Pre-Flop, an agent will call or check on premium, strong, or drawing hands. On all other hands, it will check.\\
   Afterwards, if the agent 'made' a straight or higher, it will bet 1bb or call.
   If the agent hasn't made a strong hand, it will calculate its outs and check or call if it has enough equity to do so.
   At the river, the agent will bet or call if it's made a strong hand, and check otherwise\\
   Beats random agent by 8.8bb/h on 1,000,000 hands with starting values of 1000bb
   95% Confidence interval: [8.26 9.35]\\
   With smaller, 100bb buy-ins, the 95% interval becomes [0.43 0.54]\\
   Agent wins more money the larger buy-ins are. As the blinds are fixed, this reflects
   the rule-based agent's more careful betting strategy compared to the random agent.\\
   -> action"
  [game-state game-history]
  (let [{hands :hands
         visible :visible
         current-player :current-player
         bet-values :bet-values
         current-bet :current-bet
         pot :pot
         players :players
         betting-round :betting-round} game-state
        money (:money (players current-player))
        actions (legal-actions game-state)
        call-cost (- current-bet (bet-values current-player))
        [type num-outs] (pre-hand-type (concat (hands current-player)
                                               visible))]
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
            ;;this is checking to see if I have no money left. So I can't just prefer it.
            (if (in? actions ["All-In" 0.0 0.0])
              ["All-In" 0.0]
              (take 2 (prefer-action ["Call" "Fold"] actions)))
            (if (in? actions ["All-In" 0.0 0.0])
              ["All-In" 0.0]
              (take 2 (prefer-action ["Fold"] actions)))))))))


(defn -main
  "I don't do a whole lot ... yet."
  [iter &args]
  (clojure.pprint/pprint
   (rollout-single
    (zipmap card-pairs
            (repeat {:win 0 :draw 0 :total 0}))
    iter)))