(ns poker.transformerinterface
  (:require [poker.headsup :as headsup]
            [poker.transformer :as transformer]
            [poker.utils :as utils]))

(defn parse-weights-as-actions
  [legal-actions weights buckets]
  (assert (= (count weights) (+ 5 (count buckets))))
  ())

;;see utils for softmax and sampling
utils/softmax
utils/random-weighted

(defn log-distance
  [x y]
  (if (or (zero? x) (zero? y))
    ##Inf
    (abs (Math/log (/ x y)))))