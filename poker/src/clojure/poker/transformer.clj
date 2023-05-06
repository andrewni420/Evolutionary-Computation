(ns poker.transformer
  (:require
   [poker.utils :as utils]
   [poker.onehot :as onehot]
   [poker.ndarray :as ndarray]
   [clojure.pprint :as pprint]
   [clojure.core.matrix :as m]
   [clojure.test :as t])
  (:import poker.TransformerDecoderBlock
           poker.UnembedBlock
           poker.SinglePositionEncoding
           poker.LinearEmbedding
           poker.PositionalEncoding
           poker.SeparateParallelBlock
           poker.TransformerTranslator
           poker.ParallelEmbedding
           ai.djl.engine.Engine
           ai.djl.Model
           ai.djl.ndarray.NDArray
           ai.djl.ndarray.NDList
           ai.djl.ndarray.index.NDIndex
           poker.Indexing
           ai.djl.ndarray.types.DataType
           ai.djl.ndarray.types.Shape
           ai.djl.nn.SequentialBlock
           ai.djl.nn.ParallelBlock
           ai.djl.nn.LambdaBlock
           ai.djl.nn.Activation
           ai.djl.nn.transformer.ScaledDotProductAttentionBlock
           ai.djl.training.initializer.Initializer
           ai.djl.training.initializer.XavierInitializer
           java.util.function.Function
           java.lang.Class
           java.util.Random))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;   Models and auxiliary methods    ;;;
;;;     for decision transformer      ;;;
;;;           poker agents            ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;Structure:                              ;;;
;;;    General/Auxiliary model functions   ;;;
;;;        wrappers around DJL functions   ;;;
;;;    Uninitialized Blocks                ;;;
;;;        wrappers around DJL functions   ;;;
;;;        does not initialize the blocks  ;;;
;;;    Blocks                              ;;;
;;;        wrappers incorporating creation ;;;
;;;        and initialization of blocks    ;;;
;;;    Individuals                         ;;;
;;;       methods for creating and         ;;;
;;;       manipulating individuals         ;;;
;;;    Gameplay                            ;;;
;;;       methods for using the transformer;;;
;;;       to choose an action              ;;;
;;;    Runtime                             ;;;
;;;       sandbox for experimentation      ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;Overview:
;;;Each individual is represented as an id and a list of seeds 
;;;representing the mutations applied to get to this
;;;individual over the course of the genetic algorithm as in:
;;;    Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for
;;;    Training Deep Neural Networks for Reinforcement Learning
;;;    https://arxiv.org/pdf/1712.06567.pdf
;;;(Blocks) transformer creates the transformer neural net
;;;(Individual) model-from-seeds reconstructs the individual's neural net from its seeds
;;;(Gameplay) as-agent uses the individual's neural net to decide upon an action
;;;(Gameplay) casts the individual as an action-performing, money-holding poker player
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Example Usage:                                   ;;;
#_(with-open [manager (nd/new-base-manager)]
    (let [model (model-from-seeds {:seeds [1] :id :p0}
                                  10
                                  manager
                                  1)]
      (println model)
      (.close (:model model))))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



(def self-attention-reference
  "E = embedding size\\
   B = batch size\\
   N = number of attention heads\\
   F = \"from\" sequence length (key/value sequence), the input sequence
   T = \"to\" sequence length (query sequence), e.g. the length of the output sequence\\
   S = a sequence length, either F or T\\
   H = Attention head size (= E / N)\\
   In many use cases F=T. For self attention, the input is equal to the output.\n

   This block can process input in four forms:\\
   size 1: [Values] = [(B, F, E)], only input is used as key, query and value 
       (unmasked self attention), e.g. BERT\\
   size 2: [Values, Mask] = [(B, F, E), (B, F, F)], first input is used as 
       key, query and value, masked self attention\\
   size 3: [Keys, Queries, Values] = [(B, F, E), (B, T, E), (B, F, E)], 
       inputs are interpreted as keys, queries and values, unmasked attention\\
   size 4: [Keys, Queries, Values, Mask] = [(B, F, E), (B, T, E), (B, F, E), (B, T, F)], 
       inputs are interpreted as keys, queries, values and an attention mask, full masked attention.\\
   Attention masks must contain a 1 for positions to keep and a 0 for positions to mask."
  nil)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; General Model Functions ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn initialize-model
  "Initializes a model\\
   childblocks? - whether to initialize self or child blocks"
  [model manager datatype input-shapes & {:keys [childblocks?]
                                          :or {childblocks? false}}]
  (.initialize
   model
   manager
   (ndarray/process-datatype datatype)
   (ndarray/process-shape input-shapes :array? true)))

(defn linear
  "Creates a linear block with a given amount of units\\
   -> Linear"
  [units & {:keys [bias?]
            :or {bias? true}}]
  (-> (ai.djl.nn.core.Linear/builder)
      (.setUnits units)
      (.optBias bias?)
      (.build)))

(defn forward
  "Does a forward pass of the model using the input NDList.\\
   model - Block\\
   inputs - NDList\\
   training - boolean\\
   params - PairList\\
   -> NDList"
  [model inputs & {:keys [param-store training params NDArray?]
                   :or {param-store (ai.djl.training.ParameterStore.)
                        training false
                        params nil
                        NDArray? false}}]
  (let [result (.forward model param-store inputs training params)]
    (if NDArray?
      (.singletonOrThrow result)
      result)))

#_(with-open [m (nd/new-base-manager)]
    (let [model (linear 3)]
      (initialize-model model m "float" [1 2])
      (println (.singletonOrThrow (forward model (ndarray/ndlist m float-array [[1 2]]))))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Uninitialized Blocks   ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn lambda-block
  "Creates a lambda block using a function or ifn: NDList -> NDList\\
   -> block"
  ([function name]
   (LambdaBlock. (ndarray/process-activation function) name))
  ([function]
   (LambdaBlock. (ndarray/process-activation function))))

(defn transformer-decoder-block
  "Creates a transformer decoder block.\\
   Decoder block returns the attention mask along with its output.\\
   embedding-size: length of embedded vectors\\
   head-count: number of attention heads\\
   hidden-size: number of hidden units in positional feed-forward network\\
   activation-function (ifn/Function): activation function of positional feed-forward network\\
   -> TransformerDecoderBlock (B, F, E), (B, F, F) -> (B, F, E), (B, F, F)"
  [embedding-size
   head-count
   hidden-size
   & {:keys [sparse topK activation-function dropout-probability]
      :or {activation-function (utils/make-function #(Activation/relu %))
           dropout-probability 0.1}}]
  (assert (or (not sparse) (> topK 1)) "When sparse, topK must be greater than or equal to 1")
  (let [activation-function (ndarray/process-activation activation-function)]
    (if sparse
      (TransformerDecoderBlock. embedding-size
                                head-count
                                hidden-size
                                dropout-probability
                                sparse
                                topK
                                activation-function)
      (TransformerDecoderBlock. embedding-size
                                head-count
                                hidden-size
                                dropout-probability
                                activation-function))))



#_(with-open [m (nd/new-base-manager)]
    (let [d (transformer-decoder-block 2 1 2 :sparse false :topK 2 :activation-function identity)]
      (.setRandomSeed (ai.djl.engine.Engine/getEngine "MXNet") 1)
      (initialize-model d m "float" [1 2 2])
      (let [result (forward d (ndarray/ndlist m [[[1 2] [1 1]]]
                                              [[[1 1] [0 1]]]))]
        #_(println (get-parameters d))
        (println (.get result 0))
        (println (.get result 1)))))


#_(with-open [m (nd/new-base-manager)]
    (let [attn (-> (ai.djl.nn.transformer.ScaledDotProductAttentionBlock/builder)
                   (.optAttentionProbsDropoutProb 0.1)
                   (.setEmbeddingSize 2)
                   (.setHeadCount 1)
                   (.build))]
      (.setRandomSeed (ai.djl.engine.Engine/getEngine "MXNet") 1)
      (initialize-model attn m "float" [1 2 2])
      (let [result (forward attn (ndlist m [[[1 2] [1 1]]]
                                         [[[1 0] [0 1]]]))]
        (println (get-parameters attn))
        (println (.get result 0)))))


#_(defn embed-block
    "Creates an embedding block that turns a one/multi-hot encoded vector of length dictionary-size
   into a vector of length embedding-size\\
   (B, F, D) -> (B, F, E)"
    [dictionary-size embedding-size]
    (-> (EmbedBlock/builder)
        (.setEmbeddingSize embedding-size)
        (.setDictionarySize dictionary-size)
        (.build)))

(defn linear-embedding
  "Creates an embedding block that turns a one/multi-hot encoded vector of length dictionary-size
   into a vector of length embedding-size\\
   (B, F, D) -> (B, F, E)"
  [embedding-size]
  (-> (LinearEmbedding/builder)
      (.setUnits embedding-size)
      (.build)))

(defn mul-block
  "Builds and returns a multiplication block\\
   units: the extra leftmost dimension returned\\
   A mul-block with units 2 will turn a (1,2) shape into a (2,1,2) shape\\
   -> MultiplicationBlock"
  [& {:keys [units]
      :or {units 1}}]
  (.build
   (.setUnits
    (ai.djl.nn.core.Multiplication/builder)
    units)))

(defn squeeze-block
  "Builds and returns a block that squeezes an axis (default 0) out of an ndarray\\ 
   -> LambdaBlock"
  [& {:keys [axis]
      :or {axis 0}}]
  (LambdaBlock.
   (utils/make-function #(ndarray/ndlist (.squeeze (.singletonOrThrow %) axis)))))


(defn sequential-block
  "Creates a sequential block that applies each given block in order\\
   [Block ...] -> Block"
  [& blocks]
  (let [s (SequentialBlock.)]
    (.addAll s blocks)
    s))

(defn parallel-block
  "Given a Function<List<NDList>, NDList> fn and an optional List<Block> blocks, \\
   returns a parallel block that applies each block to the inputs and combines the outputs using the given function\\
   In contrast to separate-parallel-block, each block is applied to the same input NDList.\\
   Function, List<Blocks> -> Block"
  ([fn blocks]
   (assert (instance? Function fn))
   (ParallelBlock. fn blocks))
  ([fn] (ParallelBlock. fn)))

#_(defn add-input-block
    "Given an NDList of additional inputs and a Block, creates a block that takes an NDList of inputs,
   appends the additional inputs, and feeds it to the block.\\
   -> Block"
    [additional-inputs block]
    (-> (AddInput.)
        (.setAdditionalInputs additional-inputs)
        (.setCoreBlock block)))


(defn unembed-block
  "Uses an Embedding block to create an unembedding block with shared weights\\
   Unembedding multiplies input by the transpose of the embedding matrix\\
   Block -> Block"
  [embed-block]
  (.setEmbedding (UnembedBlock.) embed-block))

;;see parallel-embedding


(defn separate-parallel-block
  "Creates a parallel block that process separate inputs in parallel and then combines them to 
   form a final NDArray output.\\
   function: List<NDList> -> NDList function to combine parallel outputs at the end\\
   numInputs: list of the number of NDArray inputs to each parallel block\\
   blocks: The parallel blocks to apply separately to each set of input NDArrays. 
   Each block should correspond to an element of numInputs. \\
   -> Block"
  [function numInputs & blocks]
  (let [function (if (instance? Function function)
                   function
                   (utils/make-function function))
        p (SeparateParallelBlock. function)]
    (.setNumInputs p (map int numInputs))
    (when (seq blocks) (.addAll p blocks))
    p))

#_(with-open [manager (nd/new-base-manager)]
    (let [model1 (SeparateParallelBlock. (utils/make-function ndarray/add-NDArrays))
          model2 (SeparateParallelBlock. (utils/make-function ndarray/add-NDArrays))]
      (.setNumInputs model1 (map int [1 1]))
      (.addAll model1 [(linear 2) (linear 2)])
      (.setNumInputs model2 (map int [2 1]))
      (.addAll model2 [model1 (linear 2)])
      (initialize-model model2 manager "float" (into-array Shape [(nd/new-shape [1 2])
                                                                  (nd/new-shape [1 3])
                                                                  (nd/new-shape [1 1])]))
      #_(println (ndarray/get-parameters model2))
      (println (count (.getManagedArrays manager)))
      (forward model2 (ndarray/ndlist manager [[1 0]] [[1 0 0]] [[1]]))
      (println (count (.getManagedArrays manager)))
      #_(println (.singletonOrThrow (forward model2 (ndarray/ndlist manager [[1 0]] [[1 0 0]] [[1]]))))))

(defn create-mlp
  "Given an output size and any number of hidden-sizes, creates but does not initialize a multilayer perceptron\\
   Structure: sequential Nx(Linear -> Activation) -> Linear\\
   hidden-sizes: list of the number of units in each hidden layer\\
   activation-function: activation function to be applied after each hidden layer\\
   bias: Whether to include bias for all linear blocks\\
   -> Block"
  [output-size & {:keys [hidden-sizes activation-function bias?]
                  :or {hidden-sizes []
                       bias? true
                       activation-function ndarray/relu-function}}]
  (let [s (SequentialBlock.)]
    (run! #(.add s %)
          (reduce #(conj %1
                         (linear %2 :bias? bias?)
                         (ndarray/process-activation activation-function))
                  []
                  hidden-sizes))
    (.add s (linear output-size :bias?  bias?))
    s))

#_(with-open [m (nd/new-base-manager)]
    (let [mlp (create-mlp 2 :hidden-sizes [4] :bias? true)]
      (initialize-model mlp m "float" [2 2] :childblocks? true)
      (println (get-parameters mlp))
      (initialize-model mlp m "float" [2 2] :childblocks? true)
      (println (get-parameters mlp))))


(defn parallel-embedding
  "Creates a parallel embedding block given the embedding blocks\\
   axis: axis along which to interleave output NDArrays\\
   embeddings: embeddings to be applied in parallel to the input NDArrays\\
   -> Block"
  [axis & embeddings]
  (let [pe (ParallelEmbedding.)]
    (.setAxis pe axis)
    (.setEmbeddings pe embeddings)
    pe))

#_(with-open [m (nd/new-base-manager)]
    (let [e1 (linear-embedding 2)
          e2 (linear-embedding 2)
          pe (parallel-embedding 1 e1 e2)
          reverse-pe (unembed-block pe)]
      (initialize-model pe m "float" (into-array Shape [(nd/shape [1 3 2])
                                                        (nd/shape [1 2 1])]))
      (println (get-parameters pe))
      (let [f (forward pe (ndlist m [[[1 1] [1 0] [0 -1] [-1 -1]]]
                                  [[[1] [-1] [2] [3]]]))]
        (println (.head f))
        (println (.head (forward reverse-pe f)))
        (println (.get (forward reverse-pe f) 1)))))


(defn scaled-dot-product-attention-block
  "Creates a scaled dot product attention block\\
   -> Block"
  [embedding-size & {:keys [dropout-prob head-count]
                     :or {dropout-prob 0.1
                          head-count 1}}]
  (-> (ScaledDotProductAttentionBlock/builder)
      (.optAttentionProbsDropoutProb dropout-prob)
      (.setEmbeddingSize embedding-size)
      (.setHeadCount head-count)
      (.build)))

#_(with-open [manager (ndarray/new-base-manager)]
    (let [m (scaled-dot-product-attention-block 4)]
      (initialize-model m manager "float" (into-array Shape [(ndarray/shape [1 2 4])
                                                             (ndarray/shape [1 2 2])]))
      #_(set-all-parameters m 1 float-array)
      #_(println (get-parameters m))
      (println (into [] (forward m (ndarray/ndlist manager
                                                   [[[1 2 3 4] [3 4 5 6]]]
                                                   [[[1 0] [1 1]]]))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Manipulating Blocks    ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;




(defn set-parameter!
  "Given a block, the name of a parameter, and an array of the values in the parameter,
   sets the values of that parameter to the values in the array.\\
   Block, String, java array -> Block"
  [block pname pvalues]
  (.set (.getArray (.get (.getParameters block) pname)) pvalues)
  block)

(defn get-pnames
  "Given a block, returns a vector of the names of the parameters in the block\\
   -> [names ...]"
  [block]
  (into [] (.keys (.getParameters block))))

#_(with-open [m (ndarray/new-base-manager)]
    (clojure.pprint/pprint
     (get-pnames (causal-mask-block m [1 2 3]))))

(defn get-parameters
  "Given a block, returns a map from the name of a parameter to the value, either as an NDArray or as a nested vector, of that parameter\\
   -> {pname pvalue}"
  [block & {:keys [as-array?]
            :or {as-array? true}}]
  (let [p (.getParameters block)
        k (.keys p)]
    (zipmap k
            (for [n k]
              ((if as-array?
                 #(.getArray %)
                 identity)
               (.get p n))))))

#_(with-open [m (ndarray/new-base-manager)]
    (clojure.pprint/pprint
     (get-parameters (causal-mask-block m [1 2 3]) :as-array? true)))

(defn get-pcount
  "Given a block, gets the number of learnable parameters"
  [block]
  (reduce #(+ %1 (.size (second %2))) 0 (get-parameters block)))

#_(with-open [m (ndarray/new-base-manager)]
    (clojure.pprint/pprint
     (get-pcount (causal-mask-block m [1 2 3]))))


(defn set-all-parameters
  "Set all parameters in a model to a value"
  [model value arrfn]
  (let [names (get-pnames model)
        p (get-parameters model)]
    (run! (fn [n]
            (set-parameter! model n (arrfn (repeat (.size (p n)) value))))
          names)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;       Individuals       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#_(defn mutate
    "Given an individual whose :parameter-map is {:pname :pvalues}, perturbs each
   of the pvalues with gaussian noise."
    [individual]
    (update individual
            :parameter-map
            #(into {}
                   (map (fn [[key value]]
                          [key (mapv +
                                     value
                                     (utils/rand-normal-length 0 1 (count value)))]))
                   %)))

(defn set-parameters!
  "Given a neural net and a map of {parameter-name float-array}, where the float-array or vector
   has the same number of elements as the corresponding parameter, sets the data of the neural
   net to the data in the float-arrays\\
   -> nnet"
  [nnet parameter-map]
  (let [params (get-parameters nnet :as-array? true)]
    (run! #(.set (params %)
                 (let [arr (parameter-map %)]
                   (if (instance? (Class/forName "[F") arr)
                     arr
                     (float-array arr))))
          (map first params))
    nnet))

(defn add-parameters!
  "Given a neural net and a map of {parameter-name float-array}, where the float-array or vector
   has the same number of elements as the corresponding parameter, sets the data of the neural
   net to the data in the float-arrays\\
   -> nnet"
  [nnet parameter-map]
  (let [params (get-parameters nnet :as-array? true)]
    (run! #(.set ^NDArray (params %)
                 (float-array (map +
                                   (parameter-map %)
                                   (.toArray ^NDArray (params %)))))
          (map first params))
    nnet))


#_(with-open [m (nd/new-base-manager)]
    (let [l (linear 4)]
      (initialize-model l m "float" [1 2])
      (add-parameters! l {"weight" [10 10 10 10 10 10 10 10]
                          "bias" [-10 -10 -10 -10]})
      (println (ndarray/get-parameters l))))

#_(defn add-gaussian-model!
    "Adds gaussian noise to an individual\\
   Either uses a common mean and stdev or a mean of 0 and an ndarray of the same size as the stdev\\
   -> ndarray"
    ([nnet mean stdev arrfn]
     (run! #(ndarray/add-gaussian-noise! % mean stdev arrfn)
           (map second (ndarray/get-parameters nnet :as-array? true)))
     nnet)
    ([model stdev arrfn]
     (let [p (ndarray/get-parameters model :as-array? true)
           pnames (map first p)]
       (run! #(ndarray/add-gaussian-noise! (get p %) (get stdev %) arrfn)
             pnames)
       model)))

#_(with-open [m (nd/new-base-manager)]
    (let [model (linear 2 :bias? false)]
      (initialize-model model m "float" [1 2])
      (println (get-parameters model))
      (add-gaussian-model!
       model
       {"weight" (ndarray m float-array [[0 0.001] [0.01 0.1]])}
       float-array)
      (println (get-parameters model))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;   Initialized Blocks    ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn mask-block
  "Creates and initializes a mul-squeeze block for masking\\
   input-shape: vector [B F1 F2] or higher dimension\\
   creates a [1 1 F1 F2] lower triangular masking block\\
   -> Block"
  [manager input-shape & {:keys [datatype]
                          :or {datatype DataType/FLOAT32}}]
  (let [s (SequentialBlock.)]
    (.add s (mul-block))
    (.add s (squeeze-block))
    (initialize-model s manager datatype input-shape :childblocks? true)
    (.freezeParameters s true)
    s))

#_(with-open [m (nd/new-base-manager)]
    (clojure.pprint/pprint (get-parameters (mask-block m [2 3 3]))))

(defn causal-mask-block
  "Creates and initializes a masking block\\
   input-shape: [B F1 F2] or higher dimension\\
   creates a [1 1 F1 F2] lower triangular masking block\\
   -> Block"
  [manager input-shape & {:keys [datatype axis arrfn]
                          :or {datatype DataType/FLOAT32
                               axis -2
                               arrfn float-array}}]
  (assert (>= (count input-shape) 3) (str "Shape size too small. Must be at least [B F1 F2]. Shape: " input-shape))
  (let [m (mask-block manager input-shape :datatype datatype)]
    (set-parameter! m
                    "01Multiplication_weight"
                    (arrfn
                     (flatten
                      (ndarray/causal-mask (drop 1 input-shape)
                                           axis))))
    (.freezeParameters m true)
    m))

#_(with-open [m (ndarray/new-base-manager)]
    (clojure.pprint/pprint (get-parameters (causal-mask-block m [2 3 3]))))

(defn n-fold-mask-block
  "Creates a parallel mask to zero out every nth item starting from an i<n\\
   axis: axis to apply parallel mask along\\
   arrfn: function to convert clojure collection into java array\\
   input-shape: vector [B F] or higher dimension\\
   creates a [1 1 F] or more parallel masking block\\
   -> Block"
  [manager input-shape & {:keys [datatype axis arrfn n i]
                          :or {datatype DataType/FLOAT32
                               axis -2
                               n 3
                               i 0
                               arrfn float-array}}]
  (let [m (mask-block manager input-shape :datatype datatype)]
    (assert (>= (count input-shape) 2) (str "Shape size too small. Must be at least [B F]. Shape: " input-shape))
    (set-parameter! m
                    "01Multiplication_weight"
                    (arrfn
                     (flatten
                      (ndarray/n-fold-mask (drop 1 input-shape) axis n i))))
    (.freezeParameters m true)
    m))

#_(with-open [m (ndarray/new-base-manager)]
    (println (get-parameters (n-fold-mask-block m [2 6 3]))))




#_(defn n-fold-embedding-blocks
    "A block that applies a multilayer perceptron to every nth item along the given axis\\
   input -> parallel nx(linear -> mask) -> add -> (log-softmax) -> output

   input-shape - vector of input dimensions: (B, F)\\
   datatype - data type of NDArrays\\
   axis - axis to apply parallel embedding to\\
   n - number of parallel embedding blocks\\
   arrfn - function to convert clojure collections to java arrays\\

   embedding: (B, F, D) -> (B, F, E), axis = -2 (F)\\
   unembedding: (B, F, E) -> (B, F, D), axis = -2 (F)\\
   -> Block"
    [manager input-shape dictionary-sizes embedding-size & {:keys [datatype axis arrfn n]
                                                            :or {datatype DataType/FLOAT32
                                                                 axis -2
                                                                 n 3
                                                                 arrfn float-array}}]
    (let [dictionary-sizes (if (number? dictionary-sizes)
                             (into [] (repeat n dictionary-sizes))
                             dictionary-sizes)
          embeddings (into []
                           (map #(embed-block (dictionary-sizes %) embedding-size)
                                (range n)))
          unembeddings (into [] (map unembed-block embeddings))
          embed-maskings (into [] (map (fn [i]
                                         (n-fold-mask-block manager
                                                            (conj input-shape embedding-size)
                                                            :datatype datatype
                                                            :axis axis
                                                            :arrfn arrfn
                                                            :n n
                                                            :i i))
                                       (range n)))
          #_unembed-maskings #_(into [] (map (fn [i]
                                               (n-fold-mask-block manager
                                                                  (conj input-shape (dictionary-sizes i))
                                                                  :datatype datatype
                                                                  :axis axis
                                                                  :arrfn arrfn
                                                                  :n n
                                                                  :i i))
                                             (range n)))
          parallel-embed (parallel-block (utils/make-function add-NDArrays)
                                         (map sequential-block
                                              embeddings
                                              embed-maskings))
          parallel-unembed (parallel-block (utils/make-function add-NDArrays)
                                           (map sequential-block
                                                embed-maskings
                                                unembeddings))]
      (initialize-model parallel-embed manager datatype input-shape)
      {:embedding parallel-embed
       :unembedding parallel-unembed}))

#_(with-open [m (nd/new-base-manager)]
    (let [{e :embedding
           u :unembedding} (n-fold-embedding-blocks m [1 6 3] 3 2)
          input [[[1 0 0] [1 0 0] [1 0 0] [0 0 0] [0 1 0] [0 1 0]]]]
      (set-parameter! e "01SequentialBlock_01EmbedBlock_embedding"
                      (float-array [1 0 1 1 1 0]))
      (set-parameter! e "02SequentialBlock_01EmbedBlock_embedding"
                      (float-array [1 1 0 0 1 1]))
      (set-parameter! e "03SequentialBlock_01EmbedBlock_embedding"
                      (float-array [-1 0 0 0 0 -1]))
      (let [emb (forward e (ndlist m float-array input))
            unemb (forward u emb)]
        (println "parameters" (get-parameters e))
        (println "input" input)
        (println "embedded input" (.singletonOrThrow emb))
        (println "unembedded embedded input" (.singletonOrThrow unemb)))))

(defn single-position-encoding
  "Given an array where arr[i] is the position encoding of the ith position,
   creates a block to create position encoding from integer positions\\
   -> Block (... F) -> (... F, E)"
  ([arr]
   (SinglePositionEncoding. arr))
  ([manager arrfn arr]
   (SinglePositionEncoding. (ndarray/ndarray manager arrfn arr))))

#_(let [em (SinglePositionEncoding. (ndarray m [[1 2 3] [2 3 4] [4 5 6] [5 6 7]]))
        input (ndlist m [[1 0] [1 1] [3 2]])]
    (println (forward em input :NDArray? true)))

(defn positional-encoding
  "Given a list of embedding sizes [E0 ...n] and a list of embeddings [Block0 ...n],
   creates a hierarchical positional encoding block that separately embeds each position using each block,
   and then concatenates them together to obtain a final positional embedding of size ΣEi = E\\
   -> Block: (... F, n) -> (... F, E)"
  [embedding-sizes embeddings]
  (let [p (PositionalEncoding.)]
    (.setEmbeddingSizes p (map int embedding-sizes))
    (.setEmbeddings p embeddings)
    p))

#_(let [em1 (SinglePositionEncoding. (ndarray m [[1 2 3] [2 3 4] [4 5 6] [5 6 7]]))
        em2 (SinglePositionEncoding. (ndarray m [[0.1 0.2] [0.3 0.4] [0.5 0.6]]))
        input (ndlist m [[1 0] [1 1] [3 2]])
        pe (PositionalEncoding.)]
    (.setEmbeddingSizes pe [3 2])
    (.setEmbeddings pe [em1 em2])
    (println (forward pe input :NDArray? true)))

(def default-xavier-initializer
  (XavierInitializer.
   ai.djl.training.initializer.XavierInitializer$RandomType/GAUSSIAN
   ai.djl.training.initializer.XavierInitializer$FactorType/IN
   2))

(defn transformer
  "Creates a transformer model\\
   manager: NDManager controls lifecycle of models and NDArrays\\
   input-shapes: The shapes of the inputs. \\
   Default <state: [B F1 D1], action: [B F2 D2] position: [B F 3] mask: [B F F]>\\
   The three hierarchical positions are: [game-num, round-num, action-num]\\
   d-model: The model / embedding dimension E. The transformer layer takes an input of 
   shape [B F d_model] = [B F E] and returns and input of shape [B F d_model] = [B F E]\\
   d-ff: The number of hidden units in the pointwise feed-forward layers. Traditionally 4xd_model\\
   num-layers: The number of transformer layers\\
   d-pe: The dimensions of each positional encoding (game, round, action). Should be a vector of 3 integers, each around d_model/3\\
   max-seq-length: The maximum context length of the model. Runtime scales as seq-length squared\\
   activation-function: the activation function of the feedforward layers\\
   initializer: The initializer to use for the weights\\
   component-map? Whether to return the final model, or a map of {component-name component-block} of each part of the model\\
   -> Block: input-shapes -> (take 2 input-shapes)"
  [manager input-shapes & {:keys [d-model d-ff num-layers num-heads d-pe max-seq-length activation-function dropout-probability initializer component-map?]
                           :or {activation-function (utils/make-function #(Activation/relu %))
                                dropout-probability 0.1
                                initializer default-xavier-initializer}
                           :as argmap}]
  (assert (and d-model d-ff num-layers d-pe max-seq-length) (str "Must specify all parameters. unspecified: " (filterv #(not (contains? argmap %)) [:d-model :d-ff :num-layers :d-pe :max-seq-length])))
  (assert (= d-model (reduce + d-pe))
          (str "The sum of positional encoding dimensions must equal the model dimension. 
                d-model: " d-model ", d-pe: " d-pe))
  (assert (= 0 (mod d-model num-heads))
          (str "The number of attention heads must evenly divide the model dimension. 
                d-model: " d-model ", num-heads: " num-heads))
  (let [input-shapes (ndarray/process-shape input-shapes :array? true)
        activation-function (ndarray/process-activation activation-function)
        embedding (apply parallel-embedding
                         1;;interleaving axis = F
                         (repeatedly 2 #(linear-embedding d-model)))
        pos-encoding (positional-encoding d-pe
                                          (map #(single-position-encoding
                                                 (ndarray/ndarray manager
                                                                  (utils/positional-encoding
                                                                   %1
                                                                   :num-positions max-seq-length)))
                                               d-pe))
        input-layer (separate-parallel-block ndarray/concat-NDArrays
                                             [3 1];;3 inputs and 1 mask
                                             (separate-parallel-block ndarray/add-NDArrays
                                                                      [2 1];;2 embeddings and 1 position
                                                                      embedding pos-encoding)
                                             (lambda-block identity))
        core-layer (apply sequential-block
                          (repeatedly num-layers #(transformer-decoder-block
                                                   d-model
                                                   num-heads
                                                   d-ff
                                                   :activation-function activation-function
                                                   :dropout-probability dropout-probability)))
        output-layer (separate-parallel-block
                      #(.get % 0);;get the only NDList in the List<NDList>
                      [1];;only take the first input, ignore the mask.
                      (unembed-block embedding))
        model (sequential-block input-layer core-layer output-layer)]
    (when initializer (.setInitializer model initializer ai.djl.nn.Parameter$Type/WEIGHT))
    (initialize-model model manager "float" input-shapes)
    (if component-map?
      {:embedding embedding :pos-encoding pos-encoding :input-layer input-layer :core-layer core-layer :output-layer output-layer :model model}
      model)))



#_(with-open [manager (ndarray/new-base-manager)]
    (let [input-shapes (into-array Shape (map ndarray/shape [[1 4 4]
                                                             [1 3 2]
                                                             [7 3]
                                                             [1 7 7]]))
          {model :model
           input-layer :input-layer
           embedding :embedding} (transformer manager input-shapes
                             :d-model 10
                             :d-ff 20
                             :num-layers 1
                             :num-heads 2
                             :d-pe [4 3 3]
                             :max-seq-length 10
                             :component-map? true)]
      #_(println (get-parameters model))
      (let [first-size (transduce (map #(.size %)) + (.getManagedArrays manager))]
        (into [] (forward embedding (ndarray/ndlist manager
                                              [[[-1 -2 -3 -4] [-5 -6 -7 -8] [-1 -3 -5 -7] [1 1 1 1]]]
                                              [[[0 1] [1 0] [1 1]]]
                                              #_[[1 2 3] [2 3 4] [3 4 5] [4 5 6] [5 6 7] [6 7 8] [1 3 4]]
                                              #_(ndarray/causal-mask [1 7 7] 0))))
      (println (.getManagedArrays manager)
               (- (transduce (map #(.size %)) + (.getManagedArrays manager))
                  first-size)))))



#_(defn initialize-stdevs
    "For each parameter NDArray of nnet, creates an array of ones with the same shape 
   representing an adaptive standard deviation for each entry in the array\\
   -> {name NDArray}"
    [nnet manager]
    (let [p (.getParameters nnet)
          k (.keys p)]
      (zipmap k
              (map #(nd/ones manager
                             (-> p
                                 (.get %)
                                 (.getArray)
                                 (.getShape)))
                   k))))

#_(with-open [m (nd/new-base-manager)]
    (let [model (linear 2)]
      (initialize-model model m "float" [1 2])
      (println (initialize-stdevs model m))))

(def transformer-parameters
  (volatile! {:d-model 64;;
              :d-ff 256;;
              :num-layers 6;;
              :num-heads 8
              :d-pe [16 16 16 16];;
              :max-seq-length 512}))

(defn current-transformer
  "The current transformer model being evolved. Subject to change based on 
   computing constraints, meta-evolution, and ablation studies.\\
   -> Block"
  [manager]
  (apply transformer manager
         (into-array Shape
                     (map ndarray/shape
                          [[1 256 onehot/state-length];;state
                           [1 256 onehot/action-length];;action
                           [1 512 4];;position
                           [1 512 512]]));;mask
         (mapcat identity (into [] @transformer-parameters))))

#_(vreset! transformer-parameters
           {:d-model 128;;
            :d-ff 512;;
            :num-layers 12;;
            :num-heads 8
            :d-pe [32 32 32 32];;
            :max-seq-length 512})

#_(with-open [m (ndarray/new-base-manager)]
    (get-pcount (current-transformer m)))



(defn parameter-map
  "The initial parameter map based on the current transformer"
  []
  (with-open [m (ndarray/new-base-manager)]
    (let [t (current-transformer m)]
      (transduce (map (fn [[k v]] [k (float-array (.size v))]))
                 conj
                 {}
                 (get-parameters t)))))

(def initial-parameter-map
  (volatile! (parameter-map)))

(defn set-parameters
  [pmap]
  (vreset! transformer-parameters pmap)
  (vreset! initial-parameter-map (parameter-map))
  (System/gc))

#_(with-open [manager (ndarray/new-base-manager)
              model (Model/newInstance "transformer")]
    (let [s (into-array Shape [(ndarray/shape [1 2 3])])
          arr [(ndarray/ndarray manager [[[1 2 3] [4 5 6]]])]
          l (linear 10)]
      (initialize-model l manager "float" s)
      (.setBlock model l)
      (println (.getManagedArrays manager))
      (with-open [p (.newPredictor model (TransformerTranslator. manager))]
        (println p)
        (println "prediction: " (map vec (into [] (.predict p arr)))))
      (println "________NEWLINE________")
      (println (.getManagedArrays manager))))


(defn close-individual
  "Closes the manager of the individual\\
   -> individual"
  [individual]
  (.close (:manager individual))
  (.close (:model individual))
  (dissoc individual :manager :model :mask))

(defn initialize-individual
  "Creates an individual of the form {model mask manager max-seq-length id (stdev)} 
   with the neural net and possibly also the standard deviations for 
   mutating each of the nnet parameters\\
   initial: option to preset parameters with an argmap\\
   nn-factory: function that will return the neural net after calling (nn-factory manager)
   with a supplied manager\\
   parameter-map: the weights in each of the parameters of the neural net\\
   max-seq-length: the maximum sequence length of the individual\\
   id: the individual's id\\
   stdev-map: an optional adaptive standard deviation for the mutations of each learnable weight\\
   -> individual: {parameter-map nn-factory max-seq-length id (stdev-map)}"
  [& {:keys [initial nn-factory parameter-seeds max-seq-length id stdev-map]
      :or {initial {}}
      :as argmap}]
  (assert (every? #(or (% argmap) (% initial)) [:id :nn-factory :parameter-seeds :max-seq-length])
          (str "Incomplete information provided. Unset variables: "
               (filterv #(not (or (% argmap) (% initial))) [:id :nn-factory :parameter-seeds :max-seq-length])))
  (let [[nn-factory parameter-seeds max-seq-length id]
        (mapv #(or (% argmap) (% initial))
              [:nn-factory :parameter-seeds :max-seq-length :id])]
    (merge initial
           {:parameter-seeds parameter-seeds
            :max-seq-length max-seq-length
            :nn-factory nn-factory
            :id id
            :error {}}
           (when stdev-map
             {:stdev stdev-map}))))


#_(defn index-into-block
    "Helper for indexing into a block of random noise. Returns updated indices\\
   indices: vector of indices into the block\\
   block: float array of random noise\\
   n: number of samples to take\\
   processing: postprocessing after summing all of the indexed numbers\\
   -> {:indices :result}"
    [i-start indices ^floats block n & {:keys [processing]
                                        :or {processing identity}}]
    (let [N (alength block)
          compute #(mod (* %1 %2) N)]
      (loop [arr (transient [])
             i 0
             i-start i-start]
        (if (= i n)
          {:i-start i-start :result (persistent! arr)}
          (recur (conj! arr
                        (processing
                         (transduce (map (comp (partial aget block)
                                               (partial compute i-start)))
                                    +
                                    indices)))
                 (inc i)
                 (inc i-start))))))


#_(defn index-into-block2
    "Helper for indexing into a block of random noise. Returns updated indices\\
   indices: vector of indices into the block\\
   block: float array of random noise\\
   n: number of samples to take\\
   processing: postprocessing after summing all of the indexed numbers\\
   -> {:indices :result}"
    [i-start ^ints indices ^floats block n & {:keys [processing]
                                              :or {processing identity}}]
    (let [N (alength block)
          compute #(mod (* %1 %2) N)
          arr (float-array n)]
      (loop [i 0
             i-start i-start]
        (if (= i n)
          {:i-start i-start :result arr}
          (do
            (aset arr i
                  (float (processing
                          (utils/indexedsum indices block
                                            :processing (partial compute i-start)))))
            (recur (inc i)
                   (inc i-start)))))))

;;; I ended up having to make this code java, because clojure just wasn't cutting it,
;;; even with the two optimized examples above. I thought I was writing very optimized code,
;;; but for some reason pure java is an entire order of magnitude faster
(defn index-into-block
  "Helper for indexing into a block of random noise. Returns updated indices\\
   indices: vector of indices into the block\\
   block: float array of random noise\\
   n: number of samples to take\\
   stdev: number to multiply each sampled number by\\
   -> {:indices :result}"
  [i-start indices block n & {:keys [stdev]
                              :or {stdev 0.005}}]
  {:i-start (+ i-start n)
   :result (Indexing/indexIntoBlock (int i-start)
                                    (int-array indices)
                                    block
                                    (int n)
                                    (float stdev))})

#_(utils/initialize-random-block 1000000 1)

#_(utils/initialize-random-block 2.5e8 1)

#_(index-into-block 1 [-10 10] @utils/random-block 100)

#_(time (do (dotimes [_ 10] (index-into-block 1  (range 100) @utils/random-block 100000)) nil))


(defn expand-via-indexing
  "Expand seeds into parameter block by indexing into preinstantiated random
   noise block"
  [individual random stdev]
  (assoc individual
         :parameter-map
         (let [indices (:parameter-seeds individual)]
           (loop [to-return (transient {})
                  p @initial-parameter-map
                  i 0]
             (if (empty? p)
               (persistent! to-return)
               (let [[k v] (first p)
                     {i :i-start
                      res :result} (index-into-block i indices random (count v)
                                                     :stdev stdev)]
                 (recur (assoc! to-return k res)
                        (dissoc p k)
                        i)))))))

(defn expand-via-sampling
  "Expand seeds into parameter block by sampling noise from random number generators"
  [individual random stdev]
  (assoc individual
         :parameter-map
         (into {}
               (map (fn [[key value]]
                      [key (utils/make-vector (fn []
                                                (transduce (map #(* stdev (.nextGaussian ^Random %)))
                                                           +
                                                           random))
                                              (count value))]))
               @initial-parameter-map)))

(defn expand-param-seeds
  "Given an individual, expands its parameter seeds into a set of parameter weights
   for the neural net\\
   -> {parameter-seeds parameter-map}"
  [individual & {:keys [stdev from-block?]
                 :or {stdev 1}}]
  (let [stdev (or (:stdev individual) stdev)]
    (if-let [random (and from-block? @utils/random-block)]
      (expand-via-indexing individual random stdev)
      (expand-via-sampling individual (map utils/random (:parameter-seeds individual)) stdev))))

#_(loop [to-return (transient {})
         p initial-parameter-map
         i (:parameter-seeds individual)]
    (if (empty? p)
      (persistent! to-return)
      (let [[k v] (first p)
            {i :indices res :result} (index-into-block i random (count v))]
        (recur (assoc to-return k res)
               (dissoc p k)
               i))))
#_(let [stdev (or (:stdev individual) stdev)
        randoms (map utils/random (:parameter-seeds individual))]
    (assoc individual
           :parameter-map
           (into {}
                 (map (fn [[key value]]
                        [key (utils/make-vector (fn []
                                                  (transduce (map #(* stdev (.nextGaussian %)))
                                                             +
                                                             randoms))
                                                (count value))]))
                 initial-parameter-map)))



(def max-seq-length
  "The maximum sequence length for transformer models.\\
   Currently set at 100 for the fastest possible models with still decent-ish sequence
   length"
  100)

(defn make-model
  "Given an individual, creates a transformer model from its parameter-map\\
   Also removes the parameter map\\
   -> individual with {model manager mask}"
  [individual manager mask]
  (let [model (Model/newInstance (str "transformer " (:id individual)))
        m (.newSubManager manager)]
    (.setBlock model ((:nn-factory individual) m))
    (add-parameters! (.getBlock model) (:parameter-map individual))
    (assoc (dissoc individual :parameter-map)
           :model model
           :manager m
           :mask mask)))

#_(with-open [m (nd/new-base-manager)]
    (initialize-individual :manager m :nn-factory #(current-transformer m)
                           :mask (ndarray/ndarray m (ndarray/causal-mask [1 7 7] -2))
                           :id :p0
                           :max-seq-length 7))

(defn model-from-seeds
  "Given a map of seeds and ids, returns an individual with the current
   default settings"
  [individual max-seq-length manager mask & {:keys [stdev from-block?]
                                             :or {stdev 1}}]
  (let [{seeds :seeds id :id std :stdev} individual]
    (.setRandomSeed (Engine/getInstance) (first seeds))
    (-> (initialize-individual
         :nn-factory current-transformer
         :parameter-seeds (rest seeds)
         :id id
         :max-seq-length max-seq-length)
        (expand-param-seeds :stdev (or std stdev) :from-block? from-block?)
        (make-model manager mask))))


#_(with-open [m (ndarray/new-base-manager)]
    #_(println (-> (initialize-individual
                    :nn-factory current-transformer
                    :parameter-seeds []
                    :id :p0
                    :max-seq-length max-seq-length)
                   (expand-param-seeds :stdev 1)))
    (println (get-parameters (.getBlock (:model (model-from-seeds {:seeds [1]
                                                                   :id :p0}
                                                                  10
                                                                  m
                                                                  1
                                                                  :from-block? true))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;    Gameplay Interface   ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn parse-action-encoding-type
  "Helper method for parse-action-encoding\\
   Dynamically interprets Bet as either Bet or Raise as appropriate\\
   -> string ∈ [Bet Raise Call Fold All-In Check]"
  [encoding preprocessing sampler legal-actions]
  (let [legal-types (set (map first legal-actions))
        type-mask (onehot/multi-hot (mapv onehot/action-type-idx legal-types)
                                    (count onehot/action-types))
        chosen-type (sampler (mapv vector
                                   onehot/action-types
                                   (preprocessing (take 5 encoding) type-mask)))]
    (if (and (= chosen-type "Bet")
             (contains? legal-types "Raise")
             (not (contains? legal-types "Bet")))
      "Raise"
      chosen-type)))

#_(parse-action-encoding-type (onehot/encode-action-type "Bet")
                              #(mapv * (into [] %1) %2)
                              #(first (apply max-key second %))
                              [["All-In" 200.0 200.0] ["Check" 0.0 0.0] ["Raise" 1.0 199.0] ["Fold" 0.0 0.0]])

(defn parse-action-encoding-amount
  "Helper method for parse-action-encoding\\
   -> float ∈ (1,200) or (1,400)"
  [encoding preprocessing sampler legal-actions buckets]
  (assert (every? identity buckets) "Cannot have a nil bucket")
  (assert (every? #(every? identity %) legal-actions)
          (str "Cannot have nils in legal-actions " legal-actions))
  (let [[min-amt max-amt] (first (filter #(< (first %) (second %))
                                         (map rest legal-actions)))
        money-mask (mapv #(if (utils/in-range % min-amt max-amt) 1 0) buckets)]
    (sampler (mapv vector buckets (preprocessing (drop 5 encoding) money-mask)))))

;;Mean squared error for information loss upon encoding then decoding a range of monetary amounts 
#_(reduce +
          (map utils/square
               (for [i (utils/log-scale 1 200 :num-buckets 20)]
                 (let [amount (parse-action-encoding-amount (concat [1 1 1 1 1] (onehot/encode-money i 10 200 onehot/default-action-buckets :multi-hot? false :logscale? false))
                                                            #(mapv * (into [] %1) %2)
                                                            #(first (apply max-key second %))
                                                            [["All-In" 200.0 200.0] ["Check" 0.0 0.0] ["Bet" 1.0 199.0] ["Fold" 0.0 0.0]]
                                                            (onehot/buckets-to-money onehot/default-action-buckets 10 200))]
                   (/ (abs (- i amount))
                      i)))))

(defn parse-action-encoding
  "Parse an action encoding into a sampled action type and a sampled action amount.\\
   -> [type amount]"
  [encoding game-state & {:keys [buckets sample? need-softmax? need-exp?]
                          :or {buckets onehot/default-action-buckets
                               sample? true
                               need-softmax? true}}]
  (assert (not (and need-exp? need-softmax?)) "Encoding cannot simultaneously represent log-softmax and unsoftmaxed weights")
  (assert (every? identity encoding) "Cannot have nil in the encoding")
  (let [legal-actions (utils/legal-actions game-state)
        converter (if (or need-exp? need-softmax?) #(if (zero? %) ##-Inf %) identity)
        preprocessing #((comp (if need-softmax? utils/softmax identity)
                              (if need-exp? (partial mapv (fn [x] (Math/exp x))) identity))
                        (mapv (if (or need-softmax? need-exp?) + *)
                              %1
                              (map converter %2)))
        sampler #(first ((if sample?
                           utils/random-weighted
                           (partial apply max-key))
                         second
                         %))
        chosen-type (parse-action-encoding-type encoding
                                                preprocessing
                                                sampler
                                                legal-actions)
        chosen-amount (if (contains? #{"Bet" "Raise"} chosen-type)
                        (parse-action-encoding-amount encoding
                                                      preprocessing
                                                      sampler
                                                      legal-actions
                                                      (onehot/buckets-to-money buckets game-state))
                        (utils/sfirst (filter #(= chosen-type (first %)) legal-actions)))]
    (assert (and chosen-type chosen-amount) (str "cannot have nil chosen type or amount " chosen-type chosen-amount))
    [chosen-type chosen-amount]))

#_(let [g (poker.headsup/init-game)]
    #_(onehot/encode-action ["Fold" 0.0] g)
    (parse-action-encoding (onehot/encode-action ["Bet" 100.0] g)
                           g
                           (utils/legal-actions g)))


(defn minus-baseline
  "Helper for processing inputs during slice-inputs.
   Edits game-number of positional encoding so that it starts at 0. This way
   it will not increase beyond the maximum sequence length of the transformer"
  [pos-encoding manager]
  (with-open [m (.newSubManager manager)]
    (.tempAttachAll m (into-array ai.djl.ndarray.NDResource [pos-encoding]))
    (.set pos-encoding
          (ndarray/ndindex "...,0")
          (utils/make-function #(do (.addi % (.muli (.min %) -1)) %)))
    pos-encoding))

(defn slice-inputs
  "Given a set of inputs for the decision transformer, slices them so that they
   do not exceed the maximum sequence length of the transformer and returns them in order 
   as a vector\\
   If the maximum sequence length n is odd, there will be at most ⌊n/2⌋ actions and ⌈n/2⌉ states\\
   -> [state actions positions mask]"
  [^NDArray state ^NDArray actions ^NDArray position ^NDArray mask max-seq-length]
  (let [[^Shape state-shape ^Shape action-shape ^Shape position-shape] (map ndarray/get-shape [state actions position])
        get-slice (fn [^NDArray arr idx]
                    (if (zero? (.size arr))
                      (.create (.getManager arr) (.getShape arr))
                      (.get arr ^NDIndex (ndarray/ndindex (str "...," (int idx) ":,:")))))
        ^NDArray position-slice (get-slice position
                                           (max 0 (- (ndarray/get-axis position-shape -2)
                                                     max-seq-length)))
        ^NDArray position-slice (if (= "PyTorch" (ai.djl.engine.Engine/getDefaultEngineName))
                                  (.toType position-slice DataType/INT32 false)
                                  position-slice)
        position-slice (minus-baseline position-slice (.getManager position-slice))
        mask-start (max 0 (- (ndarray/get-axis (ndarray/get-shape mask) -2)
                             (ndarray/get-axis (ndarray/get-shape position-slice) -2)))
        mask-slice (.get mask ^NDIndex (ndarray/ndindex (str "...," mask-start ":," mask-start ":")))]
    [(get-slice state (max 0 (- (ndarray/get-axis state-shape -2)
                                (Math/ceil (/ max-seq-length 2.0)))))
     (get-slice actions (max 0 (- (ndarray/get-axis action-shape -2)
                                  (Math/floor (/ max-seq-length 2.0)))))
     position-slice
     mask-slice]))



#_(with-open [manager (ndarray/new-base-manager)
            model (Model/newInstance "transformer")]
  (let [actions (ndarray/ndarray manager float-array (ndarray/identical-array [1 100 onehot/action-length] 1))
        state (ndarray/ndarray manager float-array (ndarray/identical-array [1 100 onehot/state-length] 1))
        position (ndarray/ndarray manager int-array (ndarray/identical-array [1 200 onehot/position-length] 1))
        mask (ndarray/ndarray manager float-array (ndarray/identical-array [1 200 200] 1))
        {embedding :embedding
         pos-encoding :pos-encoding
         input-layer :input-layer
         core-layer :core-layer
         output-layer :output-layer
         m :model} (transformer manager
                                (into-array Shape
                                            (map ndarray/shape
                                                 [[1 256 onehot/state-length];;state
                                                  [1 256 onehot/action-length];;action
                                                  [1 512 4];;position
                                                  [1 512 512]]));;mask
                                :d-model 64
                                :d-ff 256
                                :num-layers 1
                                :num-heads 8
                                :d-pe [16 16 16 16]
                                :max-seq-length 512
                                :component-map? true)]
    (.setBlock model m)
    (let [first-size (transduce (map #(.size %)) + (.getManagedArrays manager))]
      (println (forward (sequential-block input-layer core-layer) (ndarray/ndlist manager state actions position mask)))
      #_(with-open [p (.newPredictor model (TransformerTranslator. manager))]
          (println "prediction: " (map vec (into [] (.predict p [state actions position mask])))))
      #_(println (.getManagedArrays manager))
      (println "Additional size: " (- (transduce (map #(.size %)) + (.getManagedArrays manager))
                                      first-size)))))


#_(with-open [manager (nd/new-base-manager)
              model (Model/newInstance "transformer")]
    (let [m1 (linear-embedding 4)
          m2 (linear-embedding 4)
          m (parallel-embedding 1 m1 m2)
          arr1 (ndarray/ndarray manager [[1 2]])
          arr2 (ndarray/ndarray manager [[1 2 3]])
          arr3 (ndarray/ndarray manager [[[1 2 3 4]]])]
      (initialize-model m manager "float" (into-array Shape [(nd/shape [1 2])
                                                             (nd/shape [1 3])]))
      (.setBlock model (unembed-block m))
      #_(println (into [] (.getManagedArrays manager)))
      #_(println (forward (unembed-block m) (nd/ndlist arr3)))
      (let [first-arrays (.getManagedArrays manager)]
        (with-open [p (.newPredictor model (TransformerTranslator. manager))]
          (println "prediction: " (map vec (into [] (.predict p [arr3])))))
        (println (- (count (.getManagedArrays manager)) (count first-arrays)))
        #_(println (into [] (.getManagedArrays manager))))))

(defn as-agent
  "Given an individual, returns a function that uses the individual's
   neural net to make a decision based on a game-state and game-encoding\\
   -> IFn"
  [individual]
  (fn [game-state game-encoding]
    (let [{{state (:id individual)} :state
           actions :actions
           positions :position} game-encoding
          {mask :mask
           max-seq-length :max-seq-length
           ^Model model :model
           manager :manager} individual
          input (slice-inputs state
                              actions
                              positions
                              mask
                              max-seq-length)
          encoded-action (utils/sfirst
                          (with-open [p (.newPredictor
                                         model
                                         ^TransformerTranslator (TransformerTranslator. manager))]
                            (.batchPredict p [input])))]
      (parse-action-encoding encoded-action game-state))))

(defn as-player
  "Given an individual, returns a player with the individual's id\\
   -> player"
  [individual]
  (utils/init-player (as-agent individual) (:id individual)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;         Runtime         ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#_(with-open [m (nd/new-base-manager)]
    (time (do (model-from-seeds {:seeds [1761283695 1429008869 26273138]
                                 :id :p0}
                                20
                                m
                                1)
              nil)))

#_(with-open [manager (nd/new-base-manager)]
    (time (let [[B F E] [1 2 64];;[10 10 512] 
                [D1 D2] [64 183];;[1024 256]
                input (ndarray/ndlist manager
                                      (ndarray/identical-array [B (/ F 2) D1] 1);;state = [B F/2 D1]
                                      (ndarray/identical-array [B (/ F 2) D2] 1);;[B F/2 D2]
                                      (ndarray/identical-array [B F 4] 1);;B F 3
                                      (ndarray/causal-mask [B F F] 1))

                input-shapes #_(into-array Shape (map nd/new-shape [[1 4 4]
                                                                    [1 3 2]
                                                                    [7 3]
                                                                    [1 7 7]]))
                (into-array Shape (map nd/new-shape [[B (/ F 2) D1]
                                                     [B (/ F 2) D2]
                                                     [B F 4]
                                                     [B F F]]))

                model (transformer manager input-shapes
                                   :d-model 64;;512
                                   :d-ff 256;;2048
                                   :num-layers 6;;6
                                   :num-heads 8;;8
                                   :d-pe [16 16 16 16];;[256 128 128]
                                   :dropout-probability 0.1
                                   :max-seq-length 256;;1024
                                   :component-map? true)
                {embedding :embedding
                 pos-encoding :pos-encoding
                 input-layer :input-layer
                 core-layer :core-layer
                 output-layer :output-layer
                 model :model} model];;B F F
            #_(set-all-parameters model 1 float-array)
            #_(get-pnames model)
            (println manager)
            (time (forward model input))
            (println manager)
            #_(time (add-gaussian-model! model 0 1 float-array))
            #_(type model)
            #_(get-pcount model)
            #_(println (into [] (forward core-layer (ndlist manager input-output (causal-mask [1 7 7] 1)
                                                            #_(into [] (repeat 7 (into [] (repeat 7 1))))))))
            #_(println (into [] (forward input-layer (ndlist manager
                                                             [[[-1 -2 -3 -4] [-5 -6 -7 -8] [-1 -3 -5 -7] [1 1 1 1]]]
                                                             [[[0 1] [1 0] [1 1]]]
                                                             [[[1 2 3] [2 3 4] [3 4 5] [4 5 6] [5 6 7] [6 7 8] [1 3 4]]]
                                                             (causal-mask [1 7 7] 1))))))))

(defmacro from-seeds
  "Given an unexpanded individual {:seeds :id :stdev}, expands the individual using a new base manager,
   and wraps a call to the given function and arguments with optional arguments :model :manager for the expanded individual
   and manager in a with-open expression that closes the model."
  [individual from-block? f & args]
  `(with-open [manager# (ndarray/new-base-manager)]
     (let [max-seq-length# (:max-seq-length @transformer-parameters)
           mask# (ndarray/ndarray manager# (ndarray/causal-mask [1 max-seq-length# max-seq-length#] -2))
           model# (model-from-seeds ~individual
                                    max-seq-length#
                                    manager#
                                    mask#
                                    :stdev 0.005
                                    :from-block? ~from-block?)]
       (with-open [_model# (utils/make-closeable model# close-individual)]
         (~f ~@args :model model# :manager manager#)))))



