(ns poker.transformerinterface
  (:require [poker.headsup :as headsup]
            [poker.transformer :as transformer]
            [poker.utils :as utils]))

(defn parse-weights-as-actions
  [legal-actions weights & {:keys [buckets]
                            :or {buckets (map #(Math/pow 200 (/ % 10)) (range 11))}}]
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

(defn make-transformer-like
  "Builds and initializes a linear block. If weight is nil then doesn't initialize with preset weights.\\
   weight should be of an appropriate size given the input-shape.\\
   model-dimension: the dimension of the model. Same as \"units\" or \"output channels\" 
   for a linear block\\
   manager: a manager to control the lifecycle of ndarrays. Do something like \\
   (with-open [m (nd/new-base-manager)]\\
      (...code...))\\
   But be sure to print out instead of return NDArrays since they'll be closed when after the end parenthesis of the with-open statement"
  [manager model-dimension input-shape & {:keys [weight]
                                         :or {weight nil}}]
  )