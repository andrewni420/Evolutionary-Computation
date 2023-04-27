(ns poker.MPI
  (:require [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :as py :refer [py. py.. py.-]]
            [poker.ERL :as ERL]
            [poker.headsup :as headsup]))

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
(req/require-python '[mpi4py.MPI :as MPI])
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
        (~@body)
        (finally (close-MPI))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;              MPI code                ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn master 
  "Code executed by the main MPI thread with rank=0\\
   Controls GA loop and delegates the evaluation of fitnesses
   to worker MPI threads\n
   Accepts the following messages from the slave threads:\\
   Fitness evaluation results\n
   Sends the following messages to the slave threads:\\
   Evaluate fitness of two opponents\\
   Stop evaluation early and return result\\
   Terminate thread"
  [comm]
  (assert (MPI/Is_initialized) "MPI must be initialized for master thread to run")
  ())

(defn slave
  "Code executed by a worker MPI thread with rank>0\\
   Computes fitness functions and then queries the main thread
   for more work\n
   Accepts the following messages from the master thread:\\
   compute fitness of two opponents\\
   Early termination of fitness computation\\
   Terminate thread\n
   Sends the following messages to the master thread:\\
   Fitness evaluation results"
  [comm]
  (assert (MPI/Is_initialized) "MPI must be initialized for slave thread to run")
  (loop [task (py. comm irecv :source 0 :tag 0)
         terminate (py. comm irecv :source 0 :tag 1)]
    (cond (py. terminate test) (do (py. test cancel)
                                   (py. terminate cancel)
                                   nil)
          (py. task test) (do ()
                              (recur (py. comm irecv :source 0 :tag 0)
                                     terminate))
          :else (recur task terminate))));;irecv and test for completion

(defn main
  "Main function assigning the master-slave roles to MPI threads
   based on rank"
  []
  (with-MPI
    (let [comm mpi4py.MPI/COMM_WORLD
          rank (py. comm Get_rank)]
      (if (= 0 rank)
        (master comm)
        (slave comm)))))

;;Get the average time of every fitness evaluation that's finished when the buffer
;;finally empties. Then wait 2x that amount of time

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
