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

(defn new-default-manager 
  "Defines m as a new base manager"
  [] 
  (def m (nd/new-base-manager)))

(defn close-default-manager 
  "Closes the manager m"
  [] 
  (when m (.close m)))

(defn shape
  "Turns a vector into a Shape object\\
   -> Shape"
  [arr]
  (nd/new-shape (utils/shape arr)))

#_(shape [[1 2] [3 4] [5 6]])

(defn ndarray
  "Given a manager, a function, and a vector array, creates an NDArray object\\
   Applies the function to the vector array\\
   NDManager, arr -> NDArray"
  [manager arrfn arr]
  (nd/create manager
             (arrfn (flatten arr))
             (utils/shape arr)))

#_(with-open [m (nd/new-base-manager)]
    (ndarray int-array [[1 2] [3 4]]))

(defn process-shape 
  "Turns a vector/Shape/[Shape into a Shape or [Shape\\
   -> Shape or [Shape"
  [s & {:keys [array?]
        :or {array? false}}]
  (if array?
    (if (instance? (Class/forName "[Lai.djl.ndarray.types.Shape;") s)
      s
      (into-array Shape [(process-shape s false)]))
    (if (instance? Shape s)
      s
      (nd/new-shape (vec s)))))

#_(process-shape [1 2] :array true)

(defn process-activation 
  "Turns an activation iFn/Function into a java Function\\
   -> Function"
  [a]
  (if (ifn? a)
    (utils/make-function a)
    a))

#_(process-activation #(Activation/relu %))

(defn process-datatype
  "Turns a string or DataType into a DataType\\
   -> DataType"
  [d]
  (if (string? d) 
    (get-djl-type d)
    d))

#_(process-datatype "float")

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
  (let [input-shape (process-shape input-shape :array? true)
        activation-function (process-activation activation-function)
        datatype (process-datatype datatype)
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


(defn error-function 
  "NOT FINISHED\\
   -> individual"
  [individual]
  (assoc individual :error 0))


(defn add-gaussian-noise!
  "Adds gaussian noise to an ndarray\\
   Either uses a common mean and stdev or a mean of 0 and an ndarray of the same size as the stdev\\
   -> ndarray"
  ([ndarray mean stdev arrfn]
   (let [v (nd/to-array ndarray)
         newv (map #(+ % (utils/rand-normal mean stdev))
                   v)]
     (.set ndarray (arrfn newv))
     ndarray))
  ([value-array stdev-array arrfn]
   (let [v (nd/to-array value-array)
         s (nd/to-array stdev-array)
         new-s (map (partial utils/rand-normal 0) s)]
     (println (into-array (map + v new-s)))
     (.set value-array (arrfn (map + v new-s)))
     value-array)))

#_(with-open [m (nd/new-base-manager)]
    (println (add-gaussian-noise!
              (ndarray m float-array [[1 2] [3 4]])
              (ndarray m float-array [[0 0.001] [0.01 0.1]])
              float-array)))

(defn close-individual 
  "Closes the manager of the individual\\
   -> individual"
  [individual]
  (.close (:manager individual))
  individual)



(defn ndlist 
  "Given a manager, a function, and any number of vector arrays, creates an NDList object
   holding all of the arrays in order.\\
   Applies arrfn to each vector array\\
   NDManager, arr1, arr2, ... -> NDList"
  [manager arrfn & arrs]
  (let [ndarrays (map (partial ndarray manager arrfn) arrs)]
    (apply nd/ndlist ndarrays)))

#_(with-open [m (nd/new-base-manager)]
    (ndlist int-array [[1 2] [3 4]] [[5 6] [7 8]]))

(defn ndarray-to-vector 
  "Given an nd-array, converts it into a clojure nested vector of the same dimensions."
  [ndarr]
  (matrix/reshape (nd/to-array ndarr)
                  (nd/to-array (nd/shape ndarr))))

#_(with-open [m (nd/new-base-manager)]
    (ndarray-to-vector (ndarray m int-array [[1 2] [3 4]])))


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


#_(with-open [m (nd/new-base-manager)]
    (concat-ndlist [(ndlist m float-array [[1 2] [3 4]])
                    (ndlist m float-array [[5 6] [7 8]])]
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

;;(2 2 2)x3 -> (2 6 2)
#_(with-open [m (nd/new-base-manager)]
    (println (.head (interleave-ndlist [(ndlist m int-array [[[1 2] [3 4]] [[11 21] [31 41]]])
                                        (ndlist m int-array [[[5 6] [7 8]] [[51 61] [71 81]]])
                                        (ndlist m int-array [[[9 10] [11 12]] [[91 101] [111 121]]])]
                                       1))))

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

#_(with-open [m (nd/new-base-manager)]
    (get-pnames (transformer-decoder-block m 2 1 3 [1 2 2])))

(defn get-parameters 
  "Given a block, returns a map from the name of a parameter to the value, either as an NDArray or as a nested vector, of that parameter\\
   -> {pname pvalue}"
  [block & {:keys [as-array?]
            :or {as-array? false}}]
  (let [p (.getParameters block)
        k (.keys p)]
    (zipmap k
          (for [n k] 
            ((if as-array?
               ()
               identity)
             (.getArray (.get p n)))))))

#_(with-open [m (nd/new-base-manager)]
    (clojure.pprint/pprint
     (get-parameters (transformer-decoder-block m 2 1 3 [1 2 2]))))

(defn identical-array
  "Creates a vector array with a shape populated by only the given value\\
  vector value -> vector"
  [shape value]
  ((apply comp 
          (map #(fn [v] (into [] (repeat % v))) 
               shape)) 
   value))

#_(identical-array [3 3] 2)

(defn mask-2d 
  "Given an mxn array and a length m vector, sets everything in the mxn array 
   that exceeds the sequence lengths in the seq-length vector to a given value\\
   [[... n]... m] [... m]-> [[... n]... m]"
  [array & {:keys [seq-length value]
            :or {seq-length nil
                 value 0}}]
  (let [m (count array)
        n (count (first array))
        seq-length (if seq-length
                     seq-length
                     (range 1 (inc m)))]
    (into [] 
          (map #(into [] 
                      (concat (take %2 %1) 
                              (repeat (- n %2) value))) 
               array 
               seq-length))))

#_(mask-2d [[1 2 3 4] [1 2 3 4] [1 2 3 4] [1 2 3 4]]
         :seq-length [4 2 3 2]
         :value 0)

(defn causal-mask
  "Given a shape and an axis, constructs a mask for that axis.\\
   axis: start dimension for mask - for a nd shape, (axis-1)d [2d mask] (n-axis-1)d\\
   vector, integer -> vector of 0s and 1s"
  [shape axis]
  (let [shape (vec shape)
        axis (if (< axis 0)
               (+ axis (count shape))
               axis)]
    (assert (<= 2 (count shape)) (str "Shape dimension is too small: shape = " shape))
    (assert (<= axis (- (count shape) 2)) (str "Axis is too large: axis = " axis))
    (let [first-identical (take axis shape)
          second-identical (drop (+ 2 axis) shape)
          m (shape axis)
          n (shape (inc axis))]

      (identical-array first-identical
                       (mask-2d (identical-array [m n] (identical-array second-identical 1))
                                :value (identical-array second-identical 0))))))

#_(causal-mask [3 3 3] 0)

(defn parallel-mask
  "Sets everything to 0 except every nth item of an array\\
   axis: axis to apply parallel mask to\\
   n: every nth item is nonzero\\
   i: every i (mod n) item is nonzero\\
   vector, int, int, int -> vector"
  [shape axis n i]
  (let [shape (vec shape)
        axis (if (< axis 0)
               (+ axis (count shape))
               axis)
        m (shape axis)
        i (mod i n)
        first-identical (take axis shape)
        second-identical (drop (inc axis) shape)
        get-value #(if (= i (mod % n))
                     (identical-array second-identical 1)
                     (identical-array second-identical 0))]
    (identical-array first-identical
                     (into [] (map get-value (range m))))))

#_(parallel-mask [3 6 3] 1 3 1)


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
  (ai.djl.nn.LambdaBlock.
   (utils/make-function #(nd/ndlist (.squeeze (.singletonOrThrow %) axis)))))


(defn mask-block
  "Creates and initializes a mul-squeeze block for masking\\
   input-shape: [B F1 F2] or higher dimension\\
   creates a [1 1 F1 F2] lower triangular masking block\\
   -> Block"
  [manager input-shape & {:keys [datatype]
                          :or {datatype DataType/FLOAT32}}]
  (assert (>= (count input-shape) 3) (str "Shape size too small. Must be at least [B F E]. Shape: " input-shape))
  (let [s (ai.djl.nn.SequentialBlock.)]
    (.add s (mul-block))
    (.add s (squeeze-block))
    (.initializeChildBlocks s
                            manager
                            (process-datatype datatype)
                            (process-shape input-shape :array? true))
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
  (assert (>= (count input-shape) 3) (str "Shape size too small. Must be at least [B F E]. Shape: " input-shape))
  (let [m (mask-block manager input-shape :datatype datatype)]
    (set-parameter! m
                    "01Multiplication_weight"
                    (arrfn
                     (flatten
                      (causal-mask (drop 1 input-shape)
                                 axis))))
    m))

#_(with-open [m (nd/new-base-manager)]
    (clojure.pprint/pprint (get-parameters (causal-mask-block m [2 3 3]))))

(defn parallel-mask-block
  "Creates a parallel mask to zero out every nth item starting from an i<n\\
   input-shape: [B F] or higher dimension\\
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
                      (parallel-mask (drop 1 input-shape) axis n i))))
    m))

#_(with-open [m (nd/new-base-manager)]
    (clojure.pprint/pprint (get-parameters (parallel-mask-block m [2 6 3]))))

(defn mlp-block
  [manager input-shape ])

(def mlp (ai.djl.basicmodelzoo.basic.Mlp. 2 2 (int-array [4])))

mlp

(.initializeChildBlocks mlp m
ai.djl.ndarray.types.DataType/FLOAT32
(into-array ai.djl.ndarray.types.Shape [(nd/new-shape [2 1 2])]))

(.forward
 mlp
 (ai.djl.training.ParameterStore.)
 (poker.ERL/ndlist m float-array [[[1 1]][[1 1]]])
 false
 nil)




