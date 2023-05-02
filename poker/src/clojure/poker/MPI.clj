(ns poker.MPI
  (:require [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :as py :refer [py. py.. py.-]]
            [poker.ERL :as ERL]
            [poker.headsup :as headsup]
            [poker.utils :as utils])
  (:import java.util.Random))

;;;;;;;;;
;;; Message Passing Interface (MPI) allows for distributed training across 
;;; multiple nodes. Allows for a lot of flexibility in compute architecture 
;;; such as beowulf clusters, and can take advantage of leftover compute power
;;; MPI is only officially available for fortran, C++, and python, so I decided to 
;;; use libpython and mpi4py instead of MPJ, the java MPI implementation
;;;;;;;;;

;;;;;;;;;
;;; The master-slave architecture is used, in which a central MPI thread
;;; controls the evolutionary loop, and delegates fitness function evaluations
;;; to worker threads, which await a signal from the master thread, evaluate the 
;;; fitness, send the result to the master thread, and await another task.
;;;
;;; In master-slave communication, a tag of 0 indicates a task. A tag of 1 indicates
;;; a termination message.
;;;;;;;;;

;;; MPI Import and Environment Variables ;;;
(require-python '[mpi4py :as mpi])
#_(println (py/get-attr mpi/rc "finalize"))
#_(println (py/get-attr mpi/rc "initialize"))
(py/set-attr! mpi/rc "initialize" false)
(py/set-attr! mpi/rc "finalize" false)
(require-python '[mpi4py.MPI :as MPI])
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;; MPI Setup and Close ;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn close-MPI
  "Finalizes the MPI environment"
  []
  (when (MPI/Is_initialized)
    (MPI/Finalize)))

(defn initialize-MPI
  "Initializes the MPI environment"
  []
  (when-not (MPI/Is_initialized)
    (MPI/Init)))


(defmacro with-MPI
  "Wraps body in an initialize MPI / finalize MPI call"
  [& body]
  `(try (initialize-MPI)
        ~@body
        (finally (close-MPI))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;              MPI code                ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn send-and-collect
  "Send off tasks to the threads given by their rank, and return a vector of 
   Requests awaiting the responses of those threads upon finishing the task\\
   -> [Request ...]"
  [comm threads matches]
  (mapv #(py. comm isend (py/->py-dict %2)
              :dest %1
              :tag 0)
        threads
        matches)
  (mapv #(vector %
                 (py. comm irecv :source % :tag 0))
        threads))

(defn collect-fitness
  "Does not check to see if communication has been received"
  [comm matches num-threads max-actions deck-seed & {:keys [symmetrical?]}]
  (let [process-result #(utils/recursive-copy (py/->jvm (py. (second %) wait)) :to-keyword? true)
        matches (mapv #(assoc {} :players %
                              :max-actions max-actions
                              :deck-seed deck-seed)
                      (if symmetrical?
                        (concat matches (map reverse matches))
                        matches))]
    (loop [matches (drop (dec num-threads) matches)
           requests (send-and-collect comm (range 1 num-threads) matches)
           results []]
      (if (empty? matches)
        (into [] (concat results (map process-result requests)))
        (let [{finished true
               unfinished false} (group-by #(py. (second %) Get_status) requests)]
          (recur (drop (count finished) matches)
                 (concat unfinished (send-and-collect comm (map first finished) matches))
                 (doall (concat results (mapv process-result finished)))))))))

(defn benchmark
  [comm pop bench & {:keys [deck-seed  max-actions symmetrical?]
                     :or {deck-seed 1
                          max-actions ##Inf}}]
  (let [matches (for [ind1 pop
                      ind2 bench
                      :when (not (= ind1 ind2))]
                  [ind1 ind2])]
    (ERL/process-results pop
                         bench
                         (collect-fitness comm
                                          matches
                                          (py. comm Get_size)
                                          max-actions
                                          deck-seed
                                          :symmetrical? symmetrical?))))


(defn terminate-slaves
  "Terminates the slave threads"
  [comm num-threads]
  (run! #(py. comm isend (py/->python 1) :dest % :tag 1) (range 1 num-threads)))


(defn master
  "Code executed by the main MPI thread with rank=0\\
   Controls GA loop and delegates the evaluation of fitnesses
   to worker MPI threads\n
   Accepts the following messages from the slave threads:\\
   Fitness evaluation results\n
   Sends the following messages to the slave threads:\\
   Evaluate fitness of two opponents\\
   Stop evaluation early and return result\\
   Notify slaves that ERL loop is over"
  [comm & {:keys [pop-size num-generations benchmark-count random-seed stdev]
           :or {pop-size 3
                num-generations 1
                benchmark-count 5
                random-seed 1
                stdev 0.005}
           :as argmap}]
  (assert (MPI/Is_initialized) "MPI must be initialized for master thread to run")
  (let [r (if (int? random-seed) (utils/random random-seed) random-seed)]
    (loop [generation 0
           pop (ERL/initialize-pop pop-size :r r :stdev stdev)
           hof []
           max-actions ##Inf]
      (if (= generation num-generations)
        (do (terminate-slaves comm (py. comm Get_size))
            {:last-pop pop
             :hall-of-fame (ERL/round-errors hof 3)})
        (let [{{p :pop
                b :benchmark
                a :action-counts} :result
               t :time} (utils/get-time
                         (benchmark comm
                                    pop
                                    (ERL/get-benchmark benchmark-count pop hof 0.5)
                                    :deck-seed (.nextInt r)
                                    :max-actions max-actions
                                    :symmetrical? true))
              [p h] (ERL/next-generation p r)]
          (ERL/report-generation pop generation
                                 :max-actions max-actions
                                 :time-ms t)
          (recur (inc generation)
                 p
                 (-> hof
                     (ERL/update-hof b)
                     (ERL/cull-hof)
                     (conj h))
                 (* 2 (utils/mean a))))))))

(defn process-message [message]
  (let [{[{id1 "id" seeds1 "seeds" std1 "stdev"}
          {id2 "id" seeds2 "seeds" std2 "stdev"}] "players"
         max-actions "max-actions"
         deck-seed "deck-seed"} message]
    {:players [{:id (keyword id1) :seeds seeds1 :std std1}
               {:id (keyword id2) :seeds seeds2 :std std2}]
     :max-actions max-actions
     :deck-seed deck-seed}))



(defn slave
  "Code executed by a worker MPI thread with rank>0\\
   Computes fitness functions and then queries the main thread
   for more work\n
   Accepts the following messages from the master thread:\\
   compute fitness of two opponents\\
   Early termination of fitness computation\\
   Stop waiting for tasks and exit loop\n
   Sends the following messages to the master thread:\\
   Fitness evaluation results"
  [comm max-seq-length num-games stdev from-block?]
  (assert (and comm max-seq-length num-games stdev) "Cannot be passed nil parameters")
  (assert (MPI/Is_initialized) "MPI must be initialized for slave thread to run")
  (loop [task (py. comm irecv :source 0 :tag 0)
         terminate (py. comm irecv :source 0 :tag 1)]
    (cond (py. terminate Get_status) (do (py. task cancel)
                                         (py. terminate cancel)
                                         nil)
          (py. task Get_status) (do (let [message (py. task wait)
                                          {players :players
                                           max-actions :max-actions
                                           deck-seed :deck-seed} (process-message (py/->jvm message))
                                          res (ERL/versus (first players)
                                                          (second players)
                                                          max-seq-length
                                                          num-games
                                                          :net-gain? true
                                                          :stdev stdev
                                                          :decks deck-seed
                                                          :max-actions max-actions
                                                          :action-count? true
                                                          :from-block? from-block?)]
                                      (py. comm isend (py/->py-dict res) :dest 0 :tag 0))
                                    (recur (py. comm irecv :source 0 :tag 0)
                                           terminate))
          :else (recur task terminate))))


(defn ERL
  "Main function assigning the master-slave roles to MPI threads
   based on rank"
  [& {:keys [pop-size num-generations benchmark-count random-seed num-games max-seq-length stdev from-block? block-size]
      :or {pop-size 3
           num-generations 1
           benchmark-count 5
           random-seed 1
           num-games 10
           max-seq-length 10
           block-size 1e8
           stdev 0.005}
      :as argmap}]
  (let [r (utils/random random-seed)]
    (when from-block? (utils/initialize-random-block (int block-size) r))
    (with-MPI
      (let [comm mpi4py.MPI/COMM_WORLD
            rank (py. comm Get_rank)]
        (if (= 0 rank)
          (println (master comm :pop-size pop-size
                           :num-generations num-generations
                           :benchmark-count benchmark-count
                           :random-seed r
                           :stdev stdev))
          (slave comm
                 max-seq-length
                 num-games
                 stdev
                 from-block?))))))

;;Get the average time of every fitness evaluation that's finished when the buffer
;;finally empties. Then wait 2x that amount of time

(defn test-mpi []
  (MPI/Init)
  (let [comm mpi4py.MPI/COMM_WORLD
        rank (py/call-attr comm "Get_rank")]
    "if rank == 0:
    data = {'a': 7, 'b': 3.14}
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)"
    (cond (= 0 rank) (do (println "sending data " rank)
                         (py/py. comm isend (py/->py-dict {:players [{:seeds [-1155869325] :id :p0 :stdev 0.005}, {:seeds [1761283695] :id :p2 :stdev 0.005}], :max-actions ##Inf, :deck-seed 7515937759503895804}) :dest 1 :tag 11)
                         (println "data sent"))
          (= 1 rank) (do (println "receiving data " rank)
                         (let [d (py/py. comm irecv :source 0 :tag 11)]
                           (py. d Test)
                           (println "data received " (py. d wait))))))
  (MPI/Finalize))
