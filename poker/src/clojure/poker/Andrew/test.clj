(ns poker.Andrew.test
  (:require
   [tech.v3.dataset :as df]
   [poker.ndarray :as ndarray]
   [poker.utils :as utils]
   [poker.headsup :as headsup]
   [clojure.core.matrix :as matrix])
  (:import poker.TransformerDecoderBlock))

(poker.Andrew.Test/main)


(utils/benchmark 10000 (headsup/play-game [utils/random-agent utils/random-agent]
                                  []))

(utils/benchmark 10000
                 (let [{hands :hands community :community} (utils/deal-hands 2 (shuffle utils/deck))
                       player-cards (map #(vector %
                                                  (concat (nth hands %)
                                                          community))
                                         (range 2))]
                   (utils/highest-hand player-cards)
                   #_(let [hands (map #(assoc % 1 (utils/hand-value (second %))) player-cards)
                           #_(utils/hand-value (utils/sfirst player-cards))
                           #_(utils/hand-value (utils/ssecond player-cards))
                           h1 (first hands)
                           h2 (second hands)
                           c (utils/lex-compare-vec h1 h2)]
                       #_(cond
                           (= c 0) 2
                           (> c 0) 1
                           :else 1)))
                 #_(utils/deal-hands 2 (shuffle utils/deck))
                 #_(utils/process-players [utils/random-agent utils/random-agent]))


(def m (ndarray/new-base-manager))

(def mulblock (.build
               (.setUnits
                (ai.djl.nn.core.Multiplication/builder)
                1)))

(def squeeze (ai.djl.nn.LambdaBlock.
              (utils/make-function #(ndarray/ndlist (.squeeze (.singletonOrThrow %) 0)))))

(def mask-block (let [s (ai.djl.nn.SequentialBlock.)]
                  (.addAll s [mulblock squeeze])
                  s))

(def f (ai.djl.nn.transformer.PointwiseFeedForwardBlock.
        []
        2
        (utils/make-function identity)))
(def parblock (ai.djl.nn.ParallelBlock.
               (utils/make-function #(poker.ERL/concat-ndlist % -1))
               [f f]))

(.squeeze (poker.ERL/ndarray m float-array [[[1 2]]])
          0)

(.initializeChildBlocks
 mask-block
 m
 ai.djl.ndarray.types.DataType/FLOAT32
 (into-array ai.djl.ndarray.types.Shape [(ndarray/shape [1 1 2])]))

(.forward
 mask-block
 (ai.djl.training.ParameterStore.)
 (poker.ERL/ndlist m float-array [[[1 1]]])
 false
 nil)

(.close m)

(let [p (.getParameters mask-block)]
  (doseq [k (.keys p)]
    (println k (.getArray (.get p k)))))

(.set (.getArray (.get (.getParameters mask-block) "01Multiplication_weight"))
      (float-array  [1 0]))

(poker.ERL/get-parameters mask-block)

(poker.ERL/set-parameter! mask-block "01Multiplication_weight" (float-array [0 1]))

(matrix/mmul [1 1]
             (matrix/transpose [[-1.0015,  0.846],
                                [0.2271,  1.8841]]))

[-0.1554,  2.1112]

(def d (TransformerDecoderBlock. 10 5 20 0.2 (utils/make-function #(ai.djl.nn.Activation/relu %))))


(def transformer (let [t (ai.djl.nn.SequentialBlock.)]
                   (.addAll t (into-array ai.djl.nn.Block [(TransformerDecoderBlock. 10 5 20 0.2 (utils/make-function #(ai.djl.nn.Activation/relu %)))
                                                           (TransformerDecoderBlock. 10 5 20 0.2 (utils/make-function #(ai.djl.nn.Activation/relu %)))]))
                   t))

(.getParameters transformer)

(.initializeChildBlocks d
                        m
                        ai.djl.ndarray.types.DataType/FLOAT32
                        (into-array ai.djl.ndarray.types.Shape [(ndarray/shape [2 3 10])]))


(println (.head (.forward transformer (ai.djl.training.ParameterStore.)
                          (ndarray/ndlist (.create m
                                              (->> [[[1 0 0 0 0 0 0 0 0 0]
                                                     [0 0 0 0 1 0 0 0 0 0]
                                                     [0 0 0 0 0 0 0 0 1 0]]
                                                    [[0 0 0 0 0 0 1 0 0 0]
                                                     [0 0 1 0 0 0 0 0 0 0]
                                                     [0 0 0 0 0 1 0 0 0 0]]]
                                                   (flatten)
                                                   (map float)
                                                   (float-array))
                                              (ndarray/shape [2 3 10])))
                          false
                          nil)))


(-> d
    (.getParameters)
    (.keys)
    (println))

(def o (.ones m (ndarray/shape [2 2])))
(def s (.create m (float-array (flatten [[0.001 1] [10 0.001]]))
                (ndarray/shape [2 2])))

(.toArray (ndarray/shape (poker.ERL/ndarray m float-array [[1 2] [3 4]])))

(matrix/reshape (.toArray (poker.ERL/ndarray m float-array [[1 2] [3 4]]))
                (.toArray (ndarray/shape (poker.ERL/ndarray m float-array [[1 2] [3 4]]))))

(def e (ai.djl.nn.transformer.TransformerEncoderBlock. 10 5 20 0.2 (reify java.util.function.Function
                                                                     (apply [_ x] (ai.djl.nn.Activation/relu x)))))




(def mlp (-> (ai.djl.nn.core.Linear/builder)
             (.optBias true)
             (.setUnits 4)
             (.build)))

(println mlp)

(def seqblock (-> (ai.djl.nn.SequentialBlock.)
                  (.add (-> (ai.djl.nn.core.Linear/builder)
                            (.optBias true)
                            (.setUnits 4)
                            (.build)))
                  (.add (reify java.util.function.Function
                          (apply [_ x] (ai.djl.nn.Activation/sigmoid x))))
                  (.add (-> (ai.djl.nn.core.Linear/builder)
                            (.optBias true)
                            (.setUnits 10)
                            (.build)))
                  (.add (reify java.util.function.Function
                          (apply [_ x] (ai.djl.nn.Activation/sigmoid x))))
                  (.add (-> (ai.djl.nn.core.Linear/builder)
                            (.optBias true)
                            (.setUnits 4)
                            (.build)))))


(.initializeChildBlocks seqblock m
                        ai.djl.ndarray.types.DataType/FLOAT32
                        (into-array ai.djl.ndarray.types.Shape [(ndarray/shape [2 6])]))

(.getOutputShapes seqblock (into-array ai.djl.ndarray.types.Shape [(ndarray/shape [2 6])]))

(.keys (.getParameters seqblock))

(println (.getArray (.get (.getParameters seqblock) "01Linear_bias")))

(.forward seqblock (ai.djl.training.ParameterStore.)
          (ai.djl.ndarray.NDList. [(.ones m (ndarray/shape [2 6]))])
          false
          nil)





(defn get-array [net name]
  (-> (.getParameters net)
      (.get name)
      (.getArray)
      (println)))


(defn datapoints [m w b numExamples]
  (let [X (.randomNormal m (ai.djl.ndarray.types.Shape. [numExamples (.size w)]))
        y (.add (.dot X w) b)
        y (.add y (.randomNormal m 0 0.01 (.getShape y) ai.djl.ndarray.types.DataType/FLOAT32))]
    {:x X :y y}))

(defn load-array [features labels batch-size shuffle?]
  (-> (ai.djl.training.dataset.ArrayDataset$Builder.)
      (.setData (into-array ai.djl.ndarray.NDArray [features]))
      (.optLabels (into-array ai.djl.ndarray.NDArray [labels]))
      (.setSampling batch-size shuffle?)
      (.build)))

(let [m (ndarray/new-base-manager)
      trueW (.create m (float-array [2,-3.4]))
      trueB 4.2
      {features :x labels :y} (datapoints m trueW trueB 1000)
      batch-size 20
      dataset (load-array features labels batch-size false)
      model (ai.djl.Model/newInstance "lin-reg")
      net (ai.djl.nn.SequentialBlock.)
      linearBlock1 (-> (ai.djl.nn.core.Linear/builder)
                       (.optBias true)
                       (.setUnits 4)
                       (.build))
      linearBlock2 (-> (ai.djl.nn.core.Linear/builder)
                       (.optBias true)
                       (.setUnits 1)
                       (.build))
      _ (do (-> net
                (.add linearBlock1)
                (.add (ai.djl.nn.Activation/sigmoidBlock))
                (.add linearBlock2))
            (.setBlock model net))
      l2loss (ai.djl.training.loss.L2Loss.)
      lrt (ai.djl.training.tracker.Tracker/fixed 0.03)
      sgd (-> (ai.djl.training.optimizer.Optimizer/sgd)
              (.setLearningRateTracker lrt)
              (.build))
      config (-> (ai.djl.training.DefaultTrainingConfig. l2loss)
                 (.optOptimizer sgd)
                 (.optDevices (.getDevices (.getEngine m) 1))
                 (.addTrainingListeners (ai.djl.training.listener.TrainingListener$Defaults/logging)))
      trainer (.newTrainer model config)
      _ (.initialize trainer (into-array ai.djl.ndarray.types.Shape [(ai.djl.ndarray.types.Shape. [batch-size 2])]))
      metrics (ai.djl.metric.Metrics.)
      _ (.setMetrics trainer metrics)]
  (loop [epoch 1]
    (if (> epoch 3) nil
        (do (println "epoch " epoch)
            #_(.forEach (.iterateDataset trainer dataset)
                        (reify java.util.function.Consumer
                          (accept [_ batch] (do (ai.djl.training.EasyTrain/trainBatch trainer batch)
                                                (.step trainer)
                                                (.close batch)))))
            (run! (fn [batch] (ai.djl.training.EasyTrain/trainBatch trainer batch)
                    (.step trainer)
                    (.close batch))
                  (.iterateDataset trainer dataset))
            (.notifyListeners trainer (reify java.util.function.Consumer
                                        (accept [_ l] (.onEpoch l trainer))))
            (recur (inc epoch)))))
  #_(let [paramFile (java.io.File. "models/lin-reg.param")
          os (java.io.DataOutputStream.
              (java.nio.file.Files/newOutputStream
               (.toPath paramFile)
               (into-array java.nio.file.OpenOption [])))]
      (.saveParameters net os))
  #_(let [modelDir (java.nio.file.Paths/get "models/lin-reg" (into-array java.lang.String []))
          _ (do (java.nio.file.Files/createDirectories modelDir (into-array java.nio.file.attribute.FileAttribute []))
                (.setProperty model "Epoch" "3")
                (.save model modelDir "lin-reg"))])
  (.close model))


(def model
  (let [m (ndarray/new-base-manager)
        net (ai.djl.nn.SequentialBlock.)
        linearBlock1 (-> (ai.djl.nn.core.Linear/builder)
                         (.optBias true)
                         (.setUnits 4)
                         (.build))
        linearBlock2 (-> (ai.djl.nn.core.Linear/builder)
                         (.optBias true)
                         (.setUnits 1)
                         (.build))
        _ (-> net
              (.add linearBlock1)
              (.add (ai.djl.nn.Activation/sigmoidBlock))
              (.add linearBlock2))]
    (.loadParameters net
                     m
                     (java.io.DataInputStream.
                      (java.nio.file.Files/newInputStream
                       (.toPath (java.io.File.
                                 "models/lin-reg.param"))
                       (into-array java.nio.file.OpenOption []))))
    (let [mod (ai.djl.Model/newInstance "lin-reg")]
      (.setBlock mod net)
      mod)))

(println (.getArray (.valueAt (.getParameters model) 0)))
(.get (.getParameters model) "01Linear_weight")

(defn linreg [X w b]
  (.add (.dot X w) b))


(defn squaredLoss [yHat y]
  (let [diff  (.sub yHat (.reshape y (.getShape yHat)))]
    (.div (.mul diff diff) 2)))

(defn sgd [params lr batchSize]
  (let [s (.size params)]
    (loop [i 0]
      (if (= i s)
        nil
        (do (let [param (.get params i)]
              (.subi param (-> param
                               (.getGradient)
                               (.mul lr)
                               (.div batchSize))))
            (recur (inc i)))))))

(defn process-batch [batch params lr batch-size]
  (let [X (.head (.getData batch))
        y (.head (.getLabels batch))
        _ (Utils/with_open #(.newGradientCollector (ai.djl.engine.Engine/getInstance))
                           #(let [gc (cast ai.djl.training.GradientCollector %)
                                  l (squaredLoss (linreg X
                                                         (.get params 0)
                                                         (.get params 1))
                                                 y)]
                              (.backward gc l)))]
    (sgd params lr batch-size)
    (.close batch)))

(let [m (ndarray/new-base-manager)
      trueW (.create m (float-array [2,-3.4]))
      trueB 4.2
      {features :x labels :y} (datapoints m trueW trueB 1000)
      batch-size 20
      dataset (load-array features labels batch-size false)
      w (.randomNormal m 0 0.01 (ndarray/shape [2 1]) ai.djl.ndarray.types.DataType/FLOAT32)
      b (.zeros m (ndarray/shape [1]))
      params (ndarray/ndlist w b)
      lr 0.03
      _ (.forEach params (utils/make-consumer #(.setRequiresGradient % true)))]
  (loop [epoch 0]
    (if (= epoch 3)
      nil
      (do #_(.forEach (.getData dataset m)
                      (utils/make-consumer #(process-batch % params lr batch-size)))
       (run! #(process-batch % params lr batch-size)
             (.getData dataset m))
          (let [trainL (squaredLoss (linreg features
                                            (.get params 0)
                                            (.get params 1))
                                    labels)]
            (println "Epoch " (inc epoch) ", loss " (.getFloat (.mean trainL) (long-array []))))
          (recur (inc epoch))))))


;;label = [0 0 1 1]
;;probabilities = [[0.1 0.9] [0.2 0.8] [0.3 0.7] [0.4 0.6]]

(let [sm-loss (ai.djl.training.loss.SoftmaxCrossEntropyLoss.
               "softmax1" 1 -1 false true)
      m (ndarray/new-base-manager)
      label (.create m (float-array (flatten [[1 0] [1 0] [0 1] [0 1]]))
                     (ndarray/shape [4 2]))
      pred (.create m (float-array (flatten [[0.1 0.9] [0.2 0.8] [0.3 0.7] [0.4 0.6]]))
                    (ndarray/shape [4 2]))]
  (* 4 (.getFloat (.evaluate sm-loss
                             (ndarray/ndlist label)
                             (ndarray/ndlist pred)) (long-array []))))


  #_(mapv #(do (println %)
               (println (ERL/versus (parent-seeds %)
                                    (parent-seeds (mod (inc %) (count parent-seeds)))
                                    10
                                    1000
                                    :net-gain? true
                                    :as-list? true)))
          (range 10))
#_(benchmark [{:seeds [-2003437247 1540470339], :id :p5-0}]
             [(utils/init-player utils/wait-and-bet :wait-and-bet)]
             10
             10
             :symmetrical? false)
#_(with-open [manager (nd/new-base-manager)]
    (let [mask (ndarray/ndarray manager (ndarray/causal-mask [1 10 10] -2))
          i1 (transformer/model-from-seeds {:seeds [-2003437247 1540470339], :id :p5-0} 10 manager mask)]
      (with-open [_i1 (utils/make-closeable i1 transformer/close-individual)]
        (:net-gain (headsup/iterate-games-reset
                    [(transformer/as-player i1)
                     (utils/init-player utils/wait-and-bet :wait-and-bet)]
                    manager
                    10
                    :as-list? false
                    :decks (repeatedly 10
                                       #(shuffle utils/deck)))))))
#_(ERL/versus-other {:seeds [-2003437247 1540470339], :id :p5-0}
                    (utils/init-player utils/wait-and-bet :wait-and-bet) 10 10 :reverse? false :decks (repeatedly 10
                                                                                                                  #(shuffle utils/deck)))
#_(time (let [futures (doall (for [i (range 10)]
                               (concurrent/msubmit (benchmark (nth children i)
                                                              [(utils/init-player utils/random-agent :random)
                                                               (utils/init-player utils/rule-agent :rule)
                                                               (utils/init-player utils/wait-and-bet :wait-and-bet)]
                                                              10
                                                              20
                                                              :symmetrical? false))))]
          (println (with-out-str (run! pprint/pprint (mapcat deref futures)))))
        #_(benchmark
           [{:seeds [-1155869325], :id :p0}
            {:seeds [431529176], :id :p1}
            {:seeds [1761283695], :id :p2}
            {:seeds [1749940626], :id :p3}
            {:seeds [892128508], :id :p4}
            {:seeds [-2003437247], :id :p5}
            {:seeds [1487394176], :id :p6}
            {:seeds [1049991269], :id :p7}
            {:seeds [-1224600590], :id :p8}
            {:seeds [-1437495699], :id :p9}]
           (list (utils/init-player utils/random-agent :random)
                 (utils/init-player utils/rule-agent :rule)
                 (utils/init-player utils/wait-and-bet :wait-and-bet))
           10
           10
           :manager manager
           :symmetrical? false))
#_(catch Exception e (println (str e (.getCause e) (.getCause (.getCause e)))))

#_(println (with-redefs
            [transformer/current-transformer
             (fn [manager]
               (transformer/transformer manager
                                        (into-array Shape
                                                    (map ndarray/shape
                                                         [[1 256 onehot/state-length];;state
                                                          [1 256 onehot/action-length];;action
                                                          [1 512 4];;position
                                                          [1 512 512]]));;mask
                                        :d-model 64
                                        :d-ff 256
                                        :num-layers 6
                                        :num-heads 8
                                        :d-pe [32 16 8 8]
                                        :max-seq-length 512))]
             (do-erl)))

;;;        MPI          ;;;
#_(req/require-python '[mpi4py :as mpi])
#_(println (py/get-attr mpi/rc "finalize"))
#_(println (py/get-attr mpi/rc "initialize"))
#_(py/set-attr! mpi/rc "initialize" false)
#_(py/set-attr! mpi/rc "finalize" false)
#_(req/require-python '[mpi4py.MPI :as MPI])

(defn test-mpi []
  #_(MPI/Init)
  #_(let [comm mpi4py.MPI/COMM_WORLD
          rank (py/call-attr comm "Get_rank")]
      "if rank == 0:
    data = {'a': 7, 'b': 3.14}
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)"
      (cond (= 0 rank) (do (println "sending data " rank)
                           (py/py. comm send (py/->py-dict {"a" 7 "b" 3.14}) :dest 1 :tag 11)
                           (println "data sent"))
            (= 1 rank) (do (println "receiving data " rank)
                           (let [d (py/py. comm recv :source 0 :tag 11)]
                             (println "data received " d)))))
  #_(MPI/Finalize))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn test-gpu []
  #_(println (into [] (.getDevices (ai.djl.engine.Engine/getEngine "PyTorch"))))
  #_(println (.getGpuCount (Engine/getInstance)))
  #_(println (ai.djl.pytorch.jni.JniUtils/getFeatures)))

(def run-load "
#! /bin/sh              
cd ~/mxnet/build
module load cuda11.2/toolkit
module load cudnn8.1-cuda11.2
module load nccl2-cuda11.2-gcc9
cmake ..
cmake --build .")

"from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"

(def sftp-command
  "lcd /Users/andrewni/Evolutionary-Computation
   put -r poker ERL")

(def delete-files 
  ;;ls | grep -P "^__" | xargs -d "\n" rm
  )
         