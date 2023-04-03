(ns poker.ERL
  (:require
   [clj-djl.ndarray :as nd]
   [clojure.core.matrix :as matrix]
   [clj-djl.model :as m]
   [clj-djl.nn :as nn]
   [clj-djl.training :as training]
   [clj-djl.training.dataset :as dataset]
   [clj-djl.training.loss :as loss]
   [clj-djl.training.optimizer :as optimizer]
   [clj-djl.training.tracker :as tracker]
   [clj-djl.training.listener :as listener]
   [clj-djl.nn.parameter :as param]
   [clj-djl.dataframe.column-filters :as cf]
   [clj-djl.dataframe.functional :as dfn]
   [poker.utils :as utils])
  (:import poker.TransformerDecoderBlock
           poker.Test
           poker.Utils
           ai.djl.ndarray.types.DataType
           ai.djl.ndarray.types.Shape
           ai.djl.ndarray.NDArray
           ai.djl.nn.Activation
           java.lang.Class))



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

;;Transformer preprocessing:
;;Either 1. split up inputs into an R, an A, and an S array, put it into a parallelblock, and interleave results
;;or 2. Apply embeddings for R, A, and S, then mask by positionwise multiplication and add together 
;;Transformer postprocessing:
;;Same options as preprocessing. Mask and add or split, apply, interleave.
;;Which of these will be differentiable?
;;Focus is on getting a mutatable, crossoverable transformer that can play poker.
;;Need to change game-history to be (RSA) or {:R :S :A}
;;I suppose we're not always going to have the same amount of R S and A, so we have to do mask and add.
;;Or elementwise application.
;;Parallel block to process one-hot encoded parts of state
;;Parallel? block with masking for (R S A) 


(defn get-djl-type 
  [type]
  (condp = (clojure.string/lower-case type)
         "float32" DataType/FLOAT32
         "float16" DataType/FLOAT16
         "float64" DataType/FLOAT64
         "float" DataType/FLOAT32
         "bool" DataType/BOOLEAN
         "boolean" DataType/BOOLEAN
         "complex" DataType/COMPLEX64
         "complex64" DataType/COMPLEX64
         "int8" DataType/INT8
         "int32" DataType/INT32
         "int64" DataType/INT64
         "int" DataType/INT32
         "string" DataType/STRING
         "uint8" DataType/UINT8
         DataType/UNKNOWN))




(defn transformer-decoder-block
  "Creates and initializes transformer decoder block.\\
   Manager: controls lifecycle of NDArrays\\
   embedding-size: length of embedded vectors\\
   head-count: number of attention heads\\
   hidden-size: number of hidden units in positional feed-forward network\\
   input-shape ([Shape/Shape/collection): Shape of input [B F E]\\
   activation-function (ifn/Function): activation function of positional feed-forward network\\
   datatype (string/DataType): datatype of transformer block\\
   -> TransformerDecoderBlock"
  [manager
   embedding-size
   head-count
   hidden-size
   input-shape
   & {:keys [activation-function dropout-probability datatype]
      :or {activation-function (utils/make-function #(Activation/relu %))
           dropout-probability 0.1
           datatype DataType/FLOAT32}}]
  (let [input-shape (cond (instance? (Class/forName "[Shape") input-shape) input-shape
                          (instance? Shape input-shape) (into-array Shape [input-shape])
                          :else (nd/new-shape (vec input-shape)))
        activation-function (if (ifn? activation-function)
                              (utils/make-function activation-function)
                              activation-function)
        datatype (if (string? datatype) (get-djl-type datatype) datatype)
        b (TransformerDecoderBlock. embedding-size
                                    head-count
                                    hidden-size
                                    dropout-probability
                                    activation-function)]
    (.initializeChildBlocks b
                            manager
                            datatype
                            input-shape)
    b))


(defn initialize-stdevs 
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

(defn initialize-individual
  "Creates an individual of the form {:nnet :stdev} 
   with the neural net and possibly also the standard deviations for 
   mutating each of the nnet parameters"
  [individual nnet & {:keys [stdev?]
                      :or {stdev? true}}]
  (if stdev?
    {:nnet nnet
   :stdev (initialize-stdevs nnet (:manager individual))}
    {:nnet nnet}))

(defn create-individual
  "Given a manager, assigns a new submanager of that manager to a new individual.\\
   Given a function (fn [manager] nnet), also initializes the individual\\
   -> {:manager :nnet? :stdev?}"
  ([manager & {:keys [nn-factory stdev?]
               :or {stdev? true}}]
   (if nn-factory
     (initialize-individual (create-individual manager)
                            (nn-factory manager)
                            stdev?)
     {:manager (.newSubManager manager)})))


(defn error-function [individual]
  (assoc individual :error 0))


(defn add-gaussian-noise!
  ([ndarray mean stdev]
   (let [v (nd/to-array ndarray)
         newv (map #(+ % (utils/rand-normal mean stdev))
                   v)]
     (.set ndarray (to-array newv))
     ndarray))
  ([value-array stdev-array]
   (let [v (nd/to-array value-array)
         s (nd/to-array stdev-array)
         new-s (map (partial utils/rand-normal 0) s)]
     (.set value-array (to-array (map + v new-s)))
     value-array)))

(defn close-individual [individual]
  (.close (:manager individual)))

(defn shape [arr]
  (nd/new-shape (utils/shape arr)))

(defn ndarray 
  "Given a manager, a function, and a vector array, creates an NDArray object\\
   Applies the function to the vector array\\
   NDManager, arr -> NDArray"
  [manager arrfn arr]
  (nd/create manager 
             (arrfn (flatten arr))
             (utils/shape arr)))

(defn ndlist 
  "Given a manager, a function, and any number of vector arrays, creates an NDList object
   holding all of the arrays in order.\\
   Applies arrfn to each vector array\\
   NDManager, arr1, arr2, ... -> NDList"
  [manager arrfn & arrs]
  (let [ndarrays (map (partial ndarray manager arrfn) arrs)]
    (apply nd/ndlist ndarrays)))

(defn ndarray-to-vector 
  [arr]
  (let [s (nd/to-array (.getShape arr))
        f (nd/to-array arr)]
    ))

(defn concat-ndlist
  "Given an list of singleton NDLists, concatenates the NDArray 
   at the head of each NDList by the given axis and returns the result 
   as another NDList\\
   todo: reframe as transduce\\
   java.lang.List<NDList> -> NDList"
  [ndlist-list axis]
  (let [flist (.singletonOrThrow (.get ndlist-list 0))
        len (.size ndlist-list)]
    (loop [flist flist
           i 1]
      (if (= i len)
        (nd/ndlist flist)
        (recur (.concat flist
                        (.singletonOrThrow (.get ndlist-list i))
                        axis)
               (inc i))))))


#_(class (concat-ndlist [(ndlist m float-array [[1 2] [1 2]] )
                (ndlist m float-array [[3 4] [4 3]])] 
               0))

(defn interleave-ndlist
  "Given a list of singleton NDLists where each NDArray has the same shape, 
   expands them to get a new axis along axis, concatenates them along the new axis,
   and reshapes to target-shape.\\
   axis 1 = interleave (B f E)xn -> (B nf E)\\
   [[3 2] [3 2]], 0 -> [[3 1 2] [3 1 2]] -> [3 2 2] -> [6 2] \"interleaving\" along axis 0\\
   [[1 2] [3 4] [5 6]] [[11 2] [31 4] [51 6]] -> [[1 2] [11 2] [3 4] [31 4] [5 6] [51 6]]\\
   todo: reframe as transduce\\
   java.lang.Iterable<NDList> -> NDList"
  [ndlist-list axis]
  (let [extra-axis (inc axis)
        flist (.expandDims (.singletonOrThrow (.get ndlist-list 0)) extra-axis)
        s (.getShape (.singletonOrThrow (.get ndlist-list 0)))
        s (Shape/update s axis (* (.size ndlist-list)
                                  (.get s axis)))
        len (.size ndlist-list)]
    (loop [flist flist
           i 1]
      (if (= i len)
        (nd/ndlist (nd/reshape flist s))
        (recur (.concat flist
                        (.expandDims (.singletonOrThrow (.get ndlist-list i)) extra-axis)
                        extra-axis)
               (inc i))))))

(defn set-parameter!
  "Given a block, the name of a parameter, and an array of the values in the parameter,
   sets the values of that parameter to the values in the array.\\
   -> block"
  [block pname pvalues]
  (.set (.getArray (.get (.getParameters block) pname)) pvalues)
  block)

(defn get-pnames
  "Given a block, the name of a parameter, and an array of the values in the parameter,
   sets the values of that parameter to the values in the array.\\
   -> block"
  [block]
  (into [] (.keys (.getParameters block))))

(defn get-parameters [block]
  (let [p (.getParameters block)
        k (.keys p)]
    (zipmap k
          (for [n k] 
            (.getArray (.get p n))))))


;;(2 2 2)x3 -> (2 6 2)
#_(println (.head (interleave-ndlist [(ndlist m [[[1 2] [3 4]] [[11 21] [31 41]]])
                                    (ndlist m [[[5 6] [7 8]] [[51 61] [71 81]]])
                                    (ndlist m [[[9 10] [11 12]] [[91 101] [111 121]]])]
                                   1)))

(let [a (ndarray m [[[1 2] [3 4]] [[5 6] [7 8]]])]
  (println a)
  (println (.flatten a 0 -1)))

(doall (map #(println %)  
            (java.util.ArrayList. [(ndlist m [[1 2]])
                                                  (ndlist m [[[1 2]]])
                                                  (ndlist m [1 2])])))

(class [])
(println (nd/ (nd/stack (ndarray m [[1 2] [3 4]])
                               (ndarray m [[5 6] [7 8]])
                               (ndarray m [[5 6] [7 8]])
                               1)
                     (nd/shape [5 2])))

(println (nd/reshape (nd/concat (.expandDims (ndarray m [[1 2] [3 4]]) -1)
                       (.expandDims (ndarray m [[5 6] [7 8]]) -1)
                       -1)
            (nd/shape [4 2])))



(println (.head (interleave-ndlist [(ndlist m [[1 2] [3 4] [5 6]])
                                    (ndlist m [[1 2] [3 4] [5 6]])]
                                   0
                                   (nd/shape [6 2]))))
;;B f1 E -> B F E
[1 2 3]
[1 3 3] -> [1 5 3]

[[[1 2 3] [1 2 3]]]
[[[4 5 6] [4 5 6] [4 5 6]]]





