(ns poker.ndarray
  (:require
   [clj-djl.ndarray :as nd]
   [clojure.core.matrix :as matrix]
   [clj-djl.model :as m]
   [clj-djl.nn :as nn]
   [poker.utils :as utils])
(:import poker.Utils
         ai.djl.ndarray.types.DataType
         ai.djl.ndarray.types.Shape
         ai.djl.ndarray.NDArray
         ai.djl.ndarray.NDManager
         ai.djl.ndarray.NDList
         ai.djl.nn.Activation
         java.lang.Class))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Auxiliaries and Constants;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


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

(def relu-function
  "Returns a java Function that applies the relu activation function"
  (utils/make-function #(Activation/relu %)))

(defn new-default-manager
  "Defines m as a new base manager"
  []
  (def m (nd/new-base-manager)))

(defn close-default-manager
  "Closes the manager m"
  []
  (when (and m (instance? NDManager m)) (.close m)))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  NDArrays and NDLists   ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-axis
  "Gets the length of the shape along an axis. Supports any integer for axis.\\
   Integer indexing loops around the array of dimensions\\
   -> int"
  [shape axis]
  (nd/get shape (if (>= axis 0)
                  axis
                  (mod (+ axis (abs (* axis (.dimension shape))))
                       (.dimension shape)))))


(defn shape
  "Turns a vector into a Shape object\\
   -> Shape"
  [arr]
  (nd/new-shape (utils/shape arr)))

#_(shape [[1 2] [3 4] [5 6]])

(defn ndarray
  "Given a manager, a function, and a vector array, creates an NDArray object\\
   Applies the function to the vector array\\
   If passed an NDArray, returns the NDArray\\
   NDManager, arr -> NDArray"
  ([manager arrfn arr]
   (if (instance? NDArray arr)
     arr
     (nd/create manager
                (arrfn (flatten arr))
                (utils/shape arr))))
  ([manager arr] (ndarray manager float-array arr)))


#_(with-open [m (nd/new-base-manager)]
    (println (ndarray m int-array [[1 2] [3 4]])))

(defn add-to-ndarray
  "Concatenates an addition to the "
  [axis arr addition]
  (let [s (nd/get-shape arr)
        s (Shape/update s axis 1)
        addition (if (instance? NDArray addition)
                   addition
                   (ndarray (.getManager arr) addition))
        addition (.reshape addition s)]
    (.concat arr addition)))



(defn random-array
  "Returns a vector of the given shape with randomly initialized elements between 0 and 1\\
   -> nested vector"
  [shape]
  (let [fns (map (fn [len]
                   #(fn [] (into [] (repeatedly len %)))) shape)]
    (((apply comp fns) rand))))

(defn nd-get-index
  "Gets an index along an axis of an ndarray\\
   axis can be negative\\
   -> NDArray"
  [ndarray axis index]
  (let [axis (if (< axis 0) (+ axis (.dimension (nd/get-shape ndarray))) axis)]
    (nd/get ndarray
            (nd/index (str (apply str
                                  (repeat axis ":,"))
                           index
                           ",...")))))

#_(with-open [m (nd/new-base-manager)]
    (let [arr (ndarray m [[[1 2 3] [1 3 3]]])]
      (println (nd-get-index arr 1 -1))))


(defn ndlist
  "Given a manager, an optional function (default float-array), and any number of vector arrays, creates an NDList object
   holding all of the arrays in order.\\
   Applies optional function to each vector array\\
   NDManager, (arrfn), arr1, arr2, ... -> NDList"
  [manager & args]
  (let [[arrfn arrs] (if (or (coll? (first args)) (instance? NDArray (first args)))
                       [float-array args]
                       [(first args) (rest args)])
        ndarrays (map (partial ndarray manager arrfn) arrs)]
    (apply nd/ndlist ndarrays)))


#_(with-open [m (nd/new-base-manager)]
    (println (ndlist m (ndarray m [[1 2] [3 4]]) [[5 6] [7 8]])))

(defn ndarray-to-vector
  "Given an nd-array, converts it into a clojure nested vector of the same dimensions."
  [ndarr]
  (matrix/reshape (nd/to-array ndarr)
                  (nd/to-array (nd/shape ndarr))))

#_(with-open [m (nd/new-base-manager)]
    (ndarray-to-vector (ndarray m int-array [[1 2] [3 4]])))


(defn process-shape
  "Turns a vector/Shape/[Shape into a Shape or [Shape\\
   -> Shape or [Shape"
  [s & {:keys [array?]
        :or {array? false}}]
  (if array?
    (if (instance? (Class/forName "[Lai.djl.ndarray.types.Shape;") s)
      s
      (into-array Shape [(process-shape s)]))
    (if (instance? Shape s)
      s
      (nd/new-shape (vec s)))))

#_(process-shape [1 2] :array? true)

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
  (println (count (.getManagedArrays m)))
    (println (.head (interleave-ndlist [(ndlist m int-array [[[1 2] [3 4]] [[11 21] [31 41]]])
                                        (ndlist m int-array [[[5 6] [7 8]] [[51 61] [71 81]]])
                                        (ndlist m int-array [[[9 10] [11 12]] [[91 101] [111 121]]])]
                                       1)))
  (println (count (.getManagedArrays m))))


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

#_(with-open [m (nd/new-base-manager)]
    (clojure.pprint/pprint
     (get-parameters (causal-mask-block m [1 2 3]) :as-array? true)))

(defn get-pcount
  "Given a block, gets the number of learnable parameters"
  [block]
  (reduce #(+ %1 (.size (second %2))) 0 (get-parameters block)))

#_(with-open [m (nd/new-base-manager)]
    (clojure.pprint/pprint
     (get-pcount (causal-mask-block m [1 2 3]))))


(defn set-all-parameters
  "Set all parameters in a model to a value"
  [model value arrfn]
  (let [names (get-pnames model)
        p (get-parameters model)]
    (run! (fn [n]
            (set-parameter! model n (arrfn (repeat (nd/size (p n)) value))))
          names)))

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

#_(causal-mask [1 3 3] 1)

(defn n-fold-mask
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

#_(n-fold-mask [3 6 3] 1 3 1)


(defn add-NDArrays
  "Function that takes a list of singleton NDLists and adds all their NDArrays up elementwise\\
   -> IFn"
  [ndlist]
  (let [[array & rest] (map #(.singletonOrThrow %) ndlist)]
    (nd/ndlist (reduce #(.addi %1 %2) array rest))))

#_(with-open [m (nd/new-base-manager)]
    (let [l1 (ndlist m float-array [[1 2] [3 4]])
          l2 (ndlist m float-array [[0.1 0.2] [0.3 0.4]])
          l3 (ndlist m float-array [[10 20] [30 40]])]
      #_(println (.singletonOrThrow (add-NDArrays (java.util.ArrayList. [l1 l2 l3]))))))

(defn concat-NDArrays
  "Function that takes a list of NDLists and returns an NDList containing all of their NDArrays\\
   -> IFn"
  [ndlist]
  (reduce #(.addAll %1 %2) ndlist))


#_(with-open [m (nd/new-base-manager)]
    (let [l1 (ndlist m float-array [[1 2] [3 4]])
          l2 (ndlist m float-array [[0.1 0.2] [0.3 0.4]])
          l3 (ndlist m float-array [[10 20] [30 40]])]
      (println (into [] (concat-NDArrays (java.util.ArrayList. [l1 l2 l3]))))))


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
     (.set value-array (arrfn (map + v new-s)))
     value-array)))

#_(with-open [m (nd/new-base-manager)]
  (add-gaussian-noise!
   (ndarray m float-array [[1 2] [3 4]])
   (ndarray m float-array [[0 0.001] [0.01 0.1]])
   float-array)
    #_(println (add-gaussian-noise!
              (ndarray m float-array [[1 2] [3 4]])
              (ndarray m float-array [[0 0.001] [0.01 0.1]])
              float-array))
  (println (.getManagedArrays m)))


(defn add-lognormal-noise!
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


