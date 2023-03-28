(ns poker.temp
  (:require
   [clj-djl.ndarray :as nd]
   [clj-djl.model :as m]
   [clj-djl.nn :as nn]
   [clj-djl.training :as t]
   [clj-djl.training.dataset :as ds]
   [clj-djl.training.loss :as loss]
   [clj-djl.training.optimizer :as optimizer]
   [clj-djl.training.tracker :as tracker]
   [clj-djl.training.listener :as listener]
   [tech.v3.dataset :as df]
   [clj-djl.nn.parameter :as param]
   [clj-djl.dataframe.column-filters :as cf]
   [clj-djl.dataframe.functional :as dfn]))



(import poker.transformer)

(.initiated (transformer.))


ai.djl.nn.transformer.ScaledDotProductAttentionBlock

(def train-ds-url
  "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv")
(def test-ds-url
  "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv")


(def train-data (df/->dataset train-ds-url))

(class train-data)
(def test-data  (df/->dataset test-ds-url))

(df/select-by-index train-data (range 4) [0 1 2 -2 -1])
;; http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv [4 5]:
;; | Id | MSSubClass | MSZoning | SaleCondition | SalePrice |
;; |----|------------|----------|---------------|-----------|
;; |  1 |         60 |       RL |        Normal |    208500 |
;; |  2 |         20 |       RL |        Normal |    181500 |
;; |  3 |         60 |       RL |        Normal |    223500 |
;; |  4 |         70 |       RL |       Abnorml |    140000 |

(def all-features (-> train-data
                      (df/drop-columns ["SalePrice"])
                      (df/concat test-data)
                      (df/drop-columns ["Id"])
                      (df/replace-missing cf/numeric 0)
                      (df/replace-missing cf/categorical "None")
                      (df/update-columns cf/numeric
                                         #(dfn// (dfn/- % (dfn/mean %))
                                                 (dfn/standard-deviation %)))
                      (df/categorical->one-hot cf/categorical)))

(def n-train (df/row-count train-data))
(def n-test  (df/row-count test-data))

(def train-features (df/head all-features n-train))
(def test-features (df/tail all-features n-test))
(def train-labels (-> train-data
                      (df/select-columns ["SalePrice"])
                      (df/update-columns cf/numeric
                                         #(dfn// % (dfn/mean %)))))

(def ndm (nd/new-base-manager))
(def train-nd (df/->ndarray ndm train-features))
(def test-nd (df/->ndarray ndm test-features))
(def label-nd (df/->ndarray ndm train-labels))

(defn do-train [nepochs learning-rate weight-decay batchsize]
  (let [train-dataset
        (ds/array-dataset {:data (nd/to-type train-nd :float32 false)
                           :labels (nd/to-type label-nd :float32 false)
                           :batchsize batchsize
                           :shuffle false})
        net (nn/sequential {:blocks (nn/linear {:units 1})
                            :initializer (nn/normal-initializer)
                            :parameter param/weight})
        cfg (t/training-config {:loss (loss/l2-loss)
                                :optimizer (optimizer/sgd
                                            {:tracker (tracker/fixed learning-rate)})
                                :evaluator (t/accuracy)
                                :listeners (listener/logging)})]
    (with-open [model (m/model {:name "mlp" :block net})
                trainer (t/trainer model cfg)]
      (t/initialize trainer (nd/shape 1 310))
      (t/set-metrics trainer (t/metrics))
      (t/fit trainer nepochs train-dataset)
      (t/get-result trainer))))

(do-train 5 0.05 0 64)
;; => {:epochs 5,
;;     :train-accuracy 0.87876713,
;;     :train-loss 0.017618015,
;;     :validate-accuracy ##NaN,
;;     :validate-loss ##NaN}