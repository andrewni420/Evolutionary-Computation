(ns poker.transformer
  (:require [clojure.core.matrix :as matrix]
            [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :as py :refer [py. py.. py.-]]))

(require-python '[torch.nn :as nn]
                '[torch]
                '[numpy :as np]
                '[torch.optim :refer [Adam]])

(require-python "/Users/andrewni/Evolutionary-Computation/poker/src/poker/transformer.py" :reload)

(let [l (nn/Linear 2 2)]
(py.- (py/set-attr! l "_parameters" (py/set-item! (py.- l "_parameters") "weight" (torch/tensor [[1 1] [2 2]]))
) "_parameters"))

(def l (let [batch 128
      dim-in 2000
      dim-h 200
      dim-out 20
      input-X (torch/randn batch, dim-in)
      output-Y (torch/randn batch, dim-out)
      Adam-model (nn/Sequential (nn/Linear dim-in, dim-h)
                                (nn/ReLU)
                                (nn/Linear dim-h, dim-out))
      loss-fn (nn/MSELoss)
      rate-learning 1e-4
      optim (Adam (py/py* Adam-model "parameters" [] {}), :lr rate-learning)
      ]
  (loop [step 0
         pred-y (Adam-model input-X)
         loss (loss-fn pred-y output-Y)]
    
    (do (py. optim "zero_grad")
        (py/py* loss "backward" [] {})
        #_(py/py* optim "step" [] {}))
    #_(when (= 49 (mod step 50)) (println step (py/py* loss "item" [] {})))
    (if (= step 0)
      pred-y
      (recur (inc step)
             (Adam-model input-X)
             (loss-fn pred-y output-Y))))))


;;;Crashes if you try to run it multiple times at once
#_(dotimes [_ 500]
  (let [input (torch/randn 3 5 :requires_grad true)
        target (torch/randn 3 5)
        mse-loss (nn/MSELoss)
        output (mse-loss input target)]
    ((py/get-attr output "backward"))))

(let [input (torch/randn 3 5 :requires_grad true)
      model (nn/Sequential (nn/Linear 5, 5)
                           (nn/ReLU)
                           (nn/Linear 5, 5))
      ;;pred (model input)
      target (torch/randn 3 5)
      mse-loss (nn/MSELoss :reduction "sum")
      output (mse-loss input target)]
  ((py.- model "forward") input))




(defn getQ [input attn-layer])

(defn getV [input attn-layer])

(defn getK [input attn-layer])

(defn attention-head [input attn-layer]
  (let [Q (getQ input attn-layer)
        K (getK input attn-layer)
        V (getV input attn-layer)]
    (matrix/mmul )))

