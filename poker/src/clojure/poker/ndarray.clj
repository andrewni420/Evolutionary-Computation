(ns poker.ndarray
  (:require
   [clojure.core.matrix :as matrix]
   [poker.utils :as utils]
   [poker.ndarray :as ndarray]
   [clojure.string :as string])
(:import ai.djl.ndarray.types.DataType
         ai.djl.ndarray.types.Shape
         ai.djl.ndarray.NDArray
         ai.djl.ndarray.NDManager
         ai.djl.ndarray.index.NDIndex
         ai.djl.ndarray.NDList
         ai.djl.nn.Activation
         java.lang.Class))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Auxiliaries and Constants;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn get-djl-type
  [type]
  (condp = (string/lower-case type)
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

(defn new-base-manager
  []
  (NDManager/newBaseManager))

(defn new-default-manager
  "Defines m as a new base manager"
  []
  (def m (new-base-manager)))

(defn close-default-manager
  "Closes the manager m"
  []
  (when (and m (instance? NDManager m)) (.close m)))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  NDArrays and NDLists   ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-axis
  "Gets the length of the shape along an axis. Supports any integer for axis.\\
   Negative indexing loops around the array of dimensions\\
   -> int"
  [shape axis]
  (.get shape (if (>= axis 0)
                  axis
                  (mod (+ axis (abs (* axis (.dimension shape))))
                       (.dimension shape)))))

#_(get-axis (Shape. (long-array [1 2])) 0)

(defn shape
  "Converts a vector of dimensions to a shape object\\
   -> Shape"
  [s]
  (Shape. (long-array s)))

#_(shape [1 2 3])

(defn get-shape
  "Gets the shape of a nested vector\\
   -> Shape"
  [arr]
  (if (instance? NDArray arr)
    (.getShape arr)
    (shape (utils/shape arr))))

#_(get-shape [[1 2] [3 4] [5 6]])

(defn ndarray
  "Given a manager, a function, and a vector array, creates an NDArray object\\
   Applies the function to the vector array\\
   If passed an NDArray, returns the NDArray\\
   NDManager, arr -> NDArray"
  ([manager arrfn arr]
   (if (instance? NDArray arr)
     arr
     (.create manager
                (arrfn (flatten arr))
                (get-shape arr))))
  ([manager arr] (ndarray manager float-array arr)))

#_(with-open [m (nd/new-base-manager)]
    (println (ndarray m int-array [[1 2] [3 4]])))

(defn add-to-ndarray
  "Concatenates an addition to an NDArray along the axis given. For example, 
   adds a 2x3 array to a 2x4x3 matrix to make a 2x5x3 matrix"
  [arr addition axis]
  (let [s (get-shape arr)
        s (Shape/update s axis 1)
        addition (if (instance? NDArray addition)
                   addition
                   (ndarray (.getManager arr) addition))
        addition (.reshape addition s)]
    (.concat arr addition axis)))

#_(with-open [m (new-base-manager)]
  (println (add-to-ndarray (ndarray m [[[0 0 1] [0 0 2]]])
                  (ndarray m [[1 2 3]])
                  1)))

(defn ndindex 
  "Creates an ndindex object given a string and some numbers
   to fit into the {} in the string\\
   Params = string, object, object, ...\\
   -> NDIndex"
  [& params]
  (assert (> (count params) 0) "Must have at least one parameter, the string")
  (NDIndex. (first params) (to-array (rest params))))

#_(ndindex "...,{},:2" 1)

(defn random-array
  "Returns a vector of the given shape with randomly initialized elements between 0 and 1\\
   -> nested vector"
  [shape]
  (let [fns (map (fn [len]
                   #(fn [] (into [] (repeatedly len %)))) shape)]
    (((apply comp fns) rand))))

#_(random-array [2 3])

(defn nd-get-index
  "Gets an index along an axis of an ndarray\\
   axis can be negative\\
   -> NDArray"
  [ndarray axis index]
  (let [axis (if (< axis 0) (+ axis (.dimension (get-shape ndarray))) axis)]
    (.get ndarray
            (ndindex (str (apply str
                                  (repeat axis ":,"))
                           index
                           ",...")))))

#_(with-open [m (nd/new-base-manager)]
    (let [arr (ndarray m [[[1 2 3] [1 3 3]]])]
      (println (nd-get-index arr 1 -1))))


(defn ndlist
  "Given a list of NDArrays, creates an NDList\\
   Given a manager, an optional function (default float-array), and any number of vector arrays, creates an NDList object
   holding all of the arrays in order.\\
   Applies optional function to each vector array\\
   NDManager, (arrfn), arr1, arr2, ... -> NDList"
  [& args]
  (if (instance? NDManager (first args))
    (let [[manager & args] args
          [arrfn arrs] (if (or (coll? (first args)) (instance? NDArray (first args)))
                         [float-array args]
                         [(first args) (rest args)])
          ndarrays (map (partial ndarray manager arrfn) arrs)]
      (NDList. ndarrays))
    (do (assert (every? (partial instance? NDArray) args) "Can only pass NDArrays to manager-less constructor")
        (NDList. args))))


#_(with-open [m (nd/new-base-manager)]
    (println (ndlist m float-array [1 2 3]))
    (println (ndlist m (ndarray m [[1 2] [3 4]]) [[5 6] [7 8]])))

(defn ndarray-to-vector
  "Given an nd-array, converts it into a clojure nested vector of the same dimensions."
  [ndarr]
  (matrix/reshape (.toArray ndarr)
                  (.getShape (.getShape ndarr))))

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
      (shape (vec s)))))

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
        (NDList. [flist])
        (recur (.concat flist
                        (.singletonOrThrow (.get ndlist-list i))
                        axis)
               (inc i))))))

#_(with-open [m (nd/new-base-manager)]
    (println (concat-ndlist [(ndlist m float-array [[1 2] [3 4]])
                    (ndlist m float-array [[5 6] [7 8]])]
                   0)))

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
        (ndlist (.reshape flist s))
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
  [list-ndlist]
  (let [[array & rest] (map #(.singletonOrThrow %) list-ndlist)]
    (ndlist (reduce #(.addi %1 %2) array rest))))

#_(with-open [m (new-base-manager)]
    (let [l1 (ndlist m float-array [[1 2] [3 4]])
          l2 (ndlist m float-array [[0.1 0.2] [0.3 0.4]])
          l3 (ndlist m float-array [[10 20] [30 40]])]
      (println (add-NDArrays [l1 l2 l3]))))

(defn concat-NDArrays
  "Function that takes a list of NDLists and returns an NDList containing all of their NDArrays\\
   -> IFn"
  [ndlist]
  (reduce #(.addAll %1 %2) ndlist))


#_(with-open [m (new-base-manager)]
    (let [l1 (ndlist m float-array [[1 2] [3 4]])
          l2 (ndlist m float-array [[0.1 0.2] [0.3 0.4]])
          l3 (ndlist m float-array [[10 20] [30 40]])]
      (println (into [] (concat-NDArrays (java.util.ArrayList. [l1 l2 l3]))))))


(defn add-gaussian-noise!
  "Adds gaussian noise to an ndarray\\
   Either uses a common mean and stdev or a mean of 0 and an ndarray of the same size as the stdev\\
   -> ndarray"
  ([ndarray mean stdev arrfn]
   (let [v (.toArray ndarray)
         newv (map #(+ % (utils/rand-normal mean stdev))
                   v)]
     (.set ndarray (arrfn newv))
     ndarray))
  ([value-array stdev-array arrfn]
   (let [v (.toArray value-array)
         s (.toArray stdev-array)
         new-s (map (partial utils/rand-normal 0) s)]
     (.set value-array (arrfn (map + v new-s)))
     value-array)))

#_(with-open [m (new-base-manager)]
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
   (let [v (.toArray ndarray)
         newv (map #(+ % (utils/rand-normal mean stdev))
                   v)]
     (.set ndarray (arrfn newv))
     ndarray))
  ([value-array stdev-array arrfn]
   (let [v (.toArray value-array)
         s (.toArray stdev-array)
         new-s (map (partial utils/rand-normal 0) s)]
     (println (into-array (map + v new-s)))
     (.set value-array (arrfn (map + v new-s)))
     value-array)))

#_(with-open [m (nd/new-base-manager)]
    (println (add-gaussian-noise!
              (ndarray m float-array [[1 2] [3 4]])
              (ndarray m float-array [[0 0.001] [0.01 0.1]])
              float-array)))


