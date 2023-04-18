(ns poker.core-test
  (:require [clojure.test :refer :all]
            [poker.core :refer :all]
            [poker.transformer :as transformer]
            [poker.headsup :as headsup]
            [poker.utils :as utils]
            [poker.onehot :as onehot]
            [clj-djl.ndarray :as nd]
            [poker.transformerinterface :as interface]
            [clj-djl.nn :as nn])
  (:import poker.TransformerLike
           ai.djl.ndarray.types.DataType
           ai.djl.ndarray.types.Shape
           ai.djl.ndarray.NDArray
           ai.djl.ndarray.NDList
           ai.djl.nn.SequentialBlock
           ai.djl.nn.ParallelBlock
           ai.djl.nn.LambdaBlock
           ai.djl.nn.Activation
           ai.djl.nn.Block))

(deftest action-encoding-opt
  (testing "Optional Action Encoding"
    (is (thrown? AssertionError (onehot/encode-money 25 12 200
                                                     :multi-hot? false
                                                     :logscale? [true false false]))
        "Optional one-hot encoding. Cannot have one-hot encoding with multiple different scales")
    (is (= (onehot/encode-money 25 12 200
                                :buckets [[5 10 20 30]
                                          [0.5 1 2 3]
                                          [0.05 0.1 0.2]]
                                :logscale? true
                                :multi-hot? false)
           [0 0 0 0 0 0 1 0 0 0 0])
        "Optional one-hot encoding. 25 is closest to 2xpot = 24 on the log scale")))

(deftest action-encoding
  (testing "Action"
    (is (= (onehot/encode-action-type "Fold") [0 0 1 0 0])
        "Fold is the 3rd out of 5 actions in onehot/action-types")
    (is (= (onehot/encode-money-bb 40 :buckets [1 10 100] :logscale? true) [0 0 1])
        "40 is 4x 10 but 100 is only 2.5x 40, so 40 is closer on a logscale to 100")
    (is (= (onehot/encode-money-bb 40 :buckets [1 10 100]) [0 1 0])
        "40 is 30 more than 10, but 100 is 60 more than 40, so 40 is closer to 10")
    (is (= (onehot/encode-money-bb 4000 :buckets [1 10 100]) [0 0 1])
        "4000 is closest to 100")
    (is (= (onehot/encode-money-bb 4000 :buckets []) [])
        "No buckets, no encoding")
    (is (= (onehot/encode-money-bb 4 :buckets (range 1 10)) [0 0 0 1 0 0 0 0 0])
        "4=4, so there's a 1 in position 4 of the one-hot encoded vector.")
    (is (= (onehot/encode-money-pot 17.5 10 :buckets [0.5 1 1.5 2] :logscale? true) [0 0 0 1])
        "17.5 is closer on the logscale to 2x pot = 20 than to 1.5x pot = 15")
    (is (= (onehot/encode-money-stack 25 200 :buckets [0.05 0.1 0.2 0.3] :logscale? true) [0 0 0 1])
        "25 is closer on the logscale to 0.1x stack = 20 than to 0.2x stack = 40")
    (is (= (onehot/encode-money 25 12 200
                                :buckets [[5 10 20 30]
                                          [0.5 1 2 3]
                                          [0.05 0.1 0.2]]
                                :logscale? [true false false])
           [0 0 0 1 0 0 1 0 0 1 0])
        "The money given satisfies the 30bb logscale bucket, the 2xpot linear scale bucket, and the 0.1xstack linear scale bucket.
           One-hot encodings are then concatenated together to get the final output")
    (is (= (onehot/encode-money 25 12 200
                                :buckets [nil [] nil]
                                :logscale? [true false false])
           (concat (onehot/encode-money-bb 25 :logscale? true)
                   (onehot/encode-money-stack 25 200)))
        "nil means use default buckets")
    (action-encoding-opt)))

(deftest card-encoding
  (testing "Cards"
    (is (= (utils/card-index [3 "Hearts"]) 27)
        "The index of the 3 of hearts should be 1 + 2x13 = 27")
    (is (= (onehot/encode-card [12 "Diamonds"])
           (concat (repeat 23 0) [1] (repeat 28 0)))
        "The index of the 12 of Diamonds should be 10 + 1x13 = 23")
    (is (= (onehot/encode-cards [[12 "Clubs"] [4 "Diamonds"]])
           (concat (repeat 10 0) [1] (repeat 4 0) [1] (repeat 36 0)))
        "The indices of the 12 of Clubs and the 4 of Diamonds are 10 and 15, so 1s should appear at those positions")
    (is (= (onehot/encode-hand [[2 "Clubs"] [3 "Clubs"]]) (concat [1] (repeat 1325 0)))
        "The indices of the 2 of Clubs and the 3 of Clubs are 1 and 2, so this is the lowest hand possible, so it should appear at position 0")
    (is (= (onehot/encode-hand [[14 "Spades"] [13 "Spades"]]) (concat (repeat 1325 0) [1]))
        "The indices of the 14 of Spades and the 13 of Spades are 52 and 51, so this is the highest hand possible, so it should appear at position 1336. 
             Note that a hand is an unordered set in real life, so the specific spade-heart ordering of this hand doesn't matter.")
    (is (= (onehot/encode-hand [[4 "Hearts"] [6 "Diamonds"]]) (concat (repeat 706 0) [1] (repeat 619 0)))
        "The indices of the 4 of hearts and the 6 of diamonds are 28 and 17, so their position in all possible hands 
         should be equal to (51 + 50 + ...(17) terms... + 35)  + (28 - 17 - 1) = 706")))

(deftest person-encoding
  (testing "Person Encoding"
    (is (= (onehot/encode-id :id2 [:id0 :id1 :id2 :id3]) [0 0 1 0 0 0 0 0 0 0])
        ":id2 is the third person")
    (is (= (onehot/encode-id 1) [0 1 0 0 0 0 0 0 0 0])
        "index 1 is the second person. Do not use cur-player for this, but rather (.indexOf ids player-id)")))

(deftest gra-encoding
  (testing "Game, Round, and Action"
    (is (= (count (onehot/position-encode (headsup/init-game))) 3)
        "There should always be three positional integers for the state of a game: gamenum, roundnum, actionnum")
    (is (= (drop 1 (onehot/position-encode (headsup/init-game))) [0 0])
        "Iterate-games-reset should control the game-num, but init-game should
           set the round-num and action-num both back to 0")))

(deftest one-hot
  (testing "One-Hot Encoding"
    (is (= (onehot/one-hot 2 4) [0 0 1 0]))
    (is (= (onehot/multi-hot [1 3] 4) [0 1 0 1]))
    (action-encoding)
    (testing "State"
      ;;pot encoding is probably just going to be encode-money-bb with maybe a logarithmic scale
      (card-encoding)))
  (testing "Positional Encoding"
    (person-encoding)
    (gra-encoding)))

(deftest game-history
  ;;appending one-hot encoded states and actions, positional encodings, and rewards to the game history
  (testing "Game History changes"
    ())
  (testing "Reward parsing"
    ()))

(deftest probability-parsing
  (testing "Parsing Action Probabilities"
    (is (= (interface/parse-weights-as-actions [["Check" 0.0 0.0]]
                                               (repeat 10 0)
                                               :buckets [1 2 3 4 5])
           ["Check" 0.0])
        "Check is the only possible action-type, and it automatically puts 0.0bb into the pot. \\
         Recall that headsup/legal-actions returns a list of [action-type min-amount-spent max-amount-spent].\\
         Also recall that a pre-softmax 0 isn't really special, and only -infinity goes to post-softmax 0")
    (is (= (interface/parse-weights-as-actions [["Raise" 10.0 200.0]]
                                               (repeat 10 0)
                                               :buckets [1 2 3 4 15])
           ["Raise" 15.0])
        "Raise is the only possible action-type, and with a minimum raise of 10bb, 15 is the only valid amount to raise by")
    (is (= (interface/parse-weights-as-actions [["Raise" 10.0 200.0]]
                                               (repeat 10 0)
                                               :buckets [1 2 3 4 5])
           ["Fold" 0.0])
        "We can always fold, so if nothing matches, then fold. This will never occur in practice, because [\"Fold\" 0.0 0.0] will always be in legal-actions")
    (is (= (interface/parse-weights-as-actions [["Raise" 10.0 200.0]]
                                               [0 0 0 0 0
                                                ##-Inf ##-Inf ##-Inf ##-Inf ##-Inf 0 ##-Inf ##-Inf ##-Inf ##-Inf ##-Inf])
           ["Raise" (Math/sqrt 200)])
        "In the default buckets, only the one at index 5 has a non -inf weight, so that's the one we choose. It corresponds to the 5th power of the 10th root of 200, i.e.
         the square root of 200")
    (is (thrown? AssertionError (interface/parse-weights-as-actions [["Check" 0.0 0.0]]
                                                                    (repeat 10 1)
                                                                    [1 2 3]))
        "Should assert that 5 (number of possible - but not necessarily legal - action types)
         + the number of possible buckets adds up to the number of weights")))
;;takes (legal actions, weights) as input

(deftest transformer-like;;input shapes and output shapes. Mostly just build, then print (get-parameters model)
  (testing "Transformer-like model"
    (is (instance? (with-open [manager (nd/new-base-manager)]
                     (let [batch-size 1
                           sequence-length 4
                           embedding-size 3
                           model-dimension 5
                           model (interface/make-transformer-like
                                  manager
                                  model-dimension
                                  (nd/new-shape [batch-size
                                                 sequence-length
                                                 embedding-size]))]
                       model))
                   Block)
        "The transformer-like model should implement Block")
    (is (= (with-open [manager (nd/new-base-manager)]
             (let [batch-size 1
                   sequence-length 4
                   embedding-size 3
                   model-dimension 5
                   model (interface/make-transformer-like
                          manager
                          model-dimension
                          (nd/new-shape [batch-size
                                         sequence-length
                                         embedding-size]))]
               (.getOutputShapes model (into-array Shape [(nd/new-shape [1 4 3])]))))
           (nd/new-shape [1 4 5]))
        "The transformer-like model should take an input and apply a linear transformation along the last
         dimension to turn it from embedding-size to model-dimension")))


(deftest evolution;;idk how to test this. Maybe just add some print statements and see if it's working as expected
  (testing "Evolutionary Cycle"
    ))

(deftest kek
  (testing "kek"
    (is (= 2 2))))

(kek)

(defn test-ns-hook []
  (one-hot)
  (game-history)
  (probability-parsing)
  (transformer-like)
  (evolution))
