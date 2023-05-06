(ns poker.MPI
  (:require [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :as py :refer [py. py.. py.-]]
            [poker.ERL :as ERL]
            [poker.headsup :as headsup]
            [poker.concurrent :as concurrent]
            [poker.utils :as utils]
            [poker.transformer :as transformer]
            [clojure.test :as t])
  (:import java.util.Random
           java.lang.Runtime))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;   Using the Message Passing Interface (MPI) to distribute the   ;;;
;;;            Evolutionary Algorithm over many nodes               ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;
;;; Overview:
;;; Making the leap from parallel to distributed computing is a daunting but
;;; highly rewarding task. On the one hand, the lack of shared memory results in
;;; a large amount of time being spent in I/O, communicating between threads. 
;;; However, a distributed program is able to be scaled to orders of magnitude more 
;;; processors than a simply parallel program, potentially resulting in much faster wall-clock times
;;; 
;;; As Evolutionary Algorithms are embarrassingly parallel it makes sense that EAs
;;; would be a promising application of distributed computing, and many distributed 
;;; architectures such as master-slave or island models using environments such as
;;; MapReduce or Hadoop have been proposed to massively scale up evolutionary algorithms.
;;;
;;; In the case of deep neuroevolution, the master-slave architecture that most closely
;;; mimics conventional serial GAs becomes impossible due to the prohibitively large cost of
;;; passing the millions of neural network parameters that compose an individual to a slave thread
;;; for fitness evaluation. To combat this, first OpenAI ES https://arxiv.org/abs/1703.03864, 
;;; then Such et al. https://arxiv.org/abs/1712.06567 and Klijn et al. https://arxiv.org/pdf/2104.05610.pdf
;;; adopt the practice of representing evolved neural nets as the series of mutations that produced
;;; the individual from the first initialized population in the EA, where each mutation is represented by a
;;; distinct integer. To put it another way, instead of representing an individual as a node in the
;;; phylogenic tree, they represent it as the series of branches leading to that node from
;;; the initial population. Since EAs typically only last for hundreds or thousands of generations, this encoding
;;; amounts to a compression on the order of 1000x to 10000x.  
;;;;;;;;;

;;;;;;;;;
;;; Implementation Details:
;;;
;;; I chose to use the Message Passing Interface (MPI) as the environment
;;; in which to perform distributed computing. MPI is only officially 
;;; available for fortran, C++, and python. Although there is an ongoing effort
;;; to provide a java MPI implementation in MPJ (and many other libraries), I decided
;;; to use the official python library using libpython.clj. 
;;; 
;;; This implementation uses the master-slave architecture, in which a central MPI process
;;; controls the evolutionary loop, and delegates fitness function evaluations
;;; to worker threads, which await a signal from the master thread, evaluate the 
;;; fitness, send the result to the master thread, and await another task.
;;;
;;; In master-slave communication, a tag of 0 indicates a task sent to the slave
;;; or the result of a fitness evaluation sent to the master. A tag of 1 indicates
;;; a termination message sent to the slave.
;;;
;;; I kept running into out of memory issues when using each hardware thread as an MPI
;;; thread. I assume there's some kind of overhead or lack of synchronization that MPI 
;;; has but multithreading doesn't. In any case, for that reason, my implementation is actually
;;; a hybrid MPI-multithreading implementation, in which each compute node has its own MPI process,
;;; and uses its cores to compute multiple tasks in parallel. One MPI process is designated to 
;;; be the master thread, and assigns tasks to slave MPI processes as well as its own threads.
;;; 
;;; Every MPI process only processes as many fitness evaluations in parallel as the number of 
;;; cpus it possesses, and when some fitness evaluations are finished, it sends the results to the
;;; master process and resumes waiting for the same number of tasks as the number of
;;; tasks it just finished. This minimizes the downtime of processes and threads waiting for other
;;; processes and threads to finish. 
;;;;;;;;;


;;;;;;; Example Usage ;;;;;;;
;; This example runs evolution on 10 MPI processes with 10 threads each
;; Need to install MPI
;;pip install mpi4py
;; How to call evolutionary loop with default arguments:
#_(ERL)
;; Use either of these two commands in a shell:
;;mpiexec -n 10 -map-by socket:PE=5 lein run poker.MPI ERL
;;srun --mpi pmix -n 10 --cpus-per-task 10 lein run poker.MPI ERL
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;; MPI Import and Environment Variables ;;;
(require-python '[mpi4py :as mpi])
(py/set-attr! mpi/rc "initialize" false)
(py/set-attr! mpi/rc "finalize" false)
(require-python '[mpi4py.MPI :as MPI])
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;; MPI Setup and Closing ;;;;;;
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Distributed Fitness Evaluation ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn evaluate-task
  "Computes a fitness evaluation given two individuals and hyperparameters governing 
   how the fitness evaluation is to be carried out. \\
   Arguments can be specified in two ways:\\
   default arguments are specified in the keys. Calling (evaluate-task :players players) will call 
   (ERL/versus players ...) by default\\
   Overriding argmaps passed down from MPI/ERL and ultimately from core/hotstart are supplied in the 
   :argmaps optional argument. These are merged together in order, with later argmaps overriding earlier
   argmaps, and passed to ERL/versus\\
   -> {(:ind1 :ind2) (:net-gain) (:winner) (:action-count)} \\
   cf. ERL/versus"
  [& {:keys [players max-actions deck-seed max-seq-length num-games stdev from-block? argmaps]}]
  (assert (or (and players max-actions deck-seed max-seq-length num-games)
              argmaps)
          (str "Must have all required parameters " players max-actions deck-seed max-seq-length num-games))
  #_(println "rank " (py. mpi4py.MPI/COMM_WORLD Get_rank) " evaluating task")
  (time (apply ERL/versus
         (first players)
         (second players)
         max-seq-length
         num-games
         (mapcat identity
                 (into [] (merge
                           {:net-gain? true
                            :stdev stdev
                            :decks deck-seed
                            :max-actions max-actions
                            :action-count? true
                            :from-block? from-block?
                            :gc? true}
                           argmaps))))))

(defn process-result
  "Processes the result received either from MPI or from derefing a thread. 
   :rank - the rank of the process that's supposed to send the result\\
   If the rank is 0, that means it's the master MPI process's own threads sending
   the result, and the result is therefore a future, not an MPI Request\\
   :result - the result to receive. Either a future or an MPI Request object\\
   -> {:net-gain :action-count}"
  [& {:keys [rank result]}]
  #_(println "process result from rank " rank)
  (assert (and rank result) (str "Must provide rank and result" rank result))
  (if (zero? rank)
    (deref result)
    (-> (py. result wait)
        (py/->jvm)
        (utils/recursive-copy :to-keyword? true))))

(defn test-result
  "Tests whether the future or Request result has been realized or sent\\
   -> boolean\\
   See process-result"
  [& {:keys [rank result]}]
  (assert (and rank result) (str "Must provide rank and result" rank result))
  (if (zero? rank)
    (.isDone result)
    (py. result Get_status)))

(defn send-task
  "Sends off a task to be executed in a thread of a process. Returns information
   about the rank and thread, and a future to receive the results.\\
   :comm - world communicator for MPI\\
   :task - fitness evaluation task to be sent to slave thread\\
   :rank - the rank of the process in which the slave thread resides\\
   :thread - the index of the target thread in the target MPI process\\
   Doesn't correspond to an actual hardware thread. Used for keeping track 
   of how many tasks have been sent to each MPI process\\
   :args - arguments to be passed to the fitness evaluation task\\
   see ERL/versus, evaluate-task\\
   -> {:rank :thread :result}"
  [& {:keys [comm task rank thread args]
      :as argmap}]
  (assert (and comm task rank thread args) (str "Must provide task, rank, and thread" comm task rank thread args))
  #_(println "sending task to rank " rank " thread " thread)
  (assert (>= thread 0) "Thread cannot be negative")
  (when-not (zero? rank) (py. comm isend (py/->py-dict task)
                              :dest rank
                              :tag thread))
  {:rank rank
   :thread thread
   :result (if (zero? rank)
             (concurrent/msubmit
              (utils/apply-map evaluate-task
                               task
                               args
                               {:argmaps [args task]}))
             (py. comm irecv
                  :source rank
                  :tag thread))})

#_(defn send-and-collect
    "Send off tasks to the threads given by their rank, and return a vector of 
   Requests awaiting the responses of those threads upon finishing the task\\
   -> [Request ...]"
    [comm threads matches]
    (mapv #(do (py. comm isend (py/->py-dict %2)
                    :dest %1
                    :tag 0))
          threads
          matches)
    (mapv (fn [t _]
            (vector t
                    (py. comm irecv :source t :tag 0)))
          threads
          matches))

(defn resend-tasks
  "Send tasks out to all of the threads in processes given by a list of 
   {:rank :thread} or a range of ranks and threads to send out to\\
   :comm - MPI world communicator\\
   :args - arguments passed to fitness evaluation function\\
   :rank-threads - list of {:rank :thread} designating slave threads to which
   these tasks are being sent\\
   :ranks, :threads - alternatively specify a list of ranks and a list of threads,
   and send tasks to every combination of rank and thread\\
   :tasks - total tasks to be sent. Only actually sends as many tasks as the number
   of slave threads specified by :rank-threads or :ranks and :threads\\
   -> [{:rank :thread :result} ...]\\
   cf. send-task"
  ([comm args rank-threads tasks]
   #_(when (seq rank-threads) (println "resend-tasks to " (map (fn [{rank :rank thread :thread}] 
                                      {:rank rank :thread thread}) 
                                    (take (count tasks) rank-threads))))
   (mapv #(utils/apply-map send-task
                           {:comm comm :args args}
                           %1
                           {:task %2})
         rank-threads
         tasks))
  ([comm args ranks threads tasks]
   (resend-tasks comm
                 args
                 (for [r ranks
                       t threads]
                   {:rank r :thread t})
                 tasks)))

(defn collect-fitness
  "Send off and collect the results of fitness evaluations from all slave threads,
   including threads on the master MPI process. \\
   :comm - MPI world communicator\\\
   :matches - vector [[individual1 individual2] ...] of matchups to be played\\
   :num-ranks - number of MPI processes in total\\
   :max-actions - adaptive cap on the number of actions allowed per matchup for 
   for higher CPU utilization\\
   :deck-seed - random seed to synchronize decks played between different matchups\\
   :symmetrical? - whether each matchup happens once normally and once with switched positions\\
   :args - overriding arguments to be passed to the fitness evaluation\\
   -> [{:net-gain :action-count} ...]"
  [comm matches num-ranks max-actions deck-seed & {:keys [symmetrical? args]}]
  (let [matches (mapv #(assoc {} :players %
                              :max-actions max-actions
                              :deck-seed deck-seed)
                      (if symmetrical?
                        (concat matches (map reverse matches))
                        matches))]
    #_(println "Collecting fitness. Ranks: " num-ranks " threads: " (utils/num-processors) "match count: " (count matches))
    (loop [requests (resend-tasks comm args (range num-ranks) (range (utils/num-processors)) matches)
           matches (drop (count requests) matches)
           results []]
      (if (empty? matches)
        (do #_(println "matches empty. Waiting on: " (map #(dissoc % :result) (get (group-by (partial utils/apply-map test-result) requests) false)))
         (into [] (concat results (map (partial utils/apply-map process-result) requests))))
        (let [{finished true
               unfinished false} (group-by (partial utils/apply-map test-result) requests)]
          (recur (concat unfinished (resend-tasks comm args finished matches))
                 (drop (count finished) matches)
                 (doall (concat results (mapv (partial utils/apply-map process-result) finished)))))))))

(defn benchmark
  "Given a population and a set of benchmarking individuals, matches each population
   individual with each benchmark individual, collects the results of the matches, 
   and processes them to get the updated population and benchmark individuals\\
   -> {:pop :benchmark :results}\\
   cf. ERL/process-results, ERL/benchmark"
  [comm pop bench & {:keys [deck-seed  max-actions symmetrical? args]
                     :or {deck-seed 1
                          max-actions ##Inf}}]
  (let [matches (for [ind1 pop
                      ind2 bench
                      :when (not (= ind1 ind2))]
                  [(dissoc ind1 :error) 
                   (dissoc ind2 :error)])]
    (ERL/process-results pop
                         bench
                         (collect-fitness comm
                                          matches
                                          (py. comm Get_size)
                                          max-actions
                                          deck-seed
                                          :symmetrical? symmetrical?
                                          :args args))))


(defn terminate-slaves
  "Terminates the slave MPI processes when the evolutionary loop is over."
  [comm num-ranks]
  (run! #(do #_(println "sending terminate message to rank " %)
          (py. comm isend (py/->python 1) :dest % :tag (utils/num-processors)))
        (range 1 num-ranks)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;        Master MPI Process      ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;   Controls central ERL loop    ;;
;; Assigns fitness evaluations to ;;
;;         slave threads          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  "Anytime algorithm" quality   ;;
;; via hot-starts and info caching;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn start-gen
  "Hot-start using the cached state of a previous ERL run. Reads in the previous
   generation number, population, max-actions, and time taken"
  [hot-start gen-input pop-size r stdev]
  (if hot-start
    (last hot-start)
    (or (try (read-string (slurp gen-input))
             (catch Exception _ {:generation 0
                                 :pop (ERL/initialize-pop pop-size :r r :stdev stdev)
                                 :max-actions ##Inf
                                 :time 0}))
        {:generation 0
         :pop (ERL/initialize-pop pop-size :r r :stdev stdev)
         :max-actions ##Inf
         :time 0})))

(defn start-hof
  "Hot-start using the cached state of a previous ERL run. Reads in the previous
   hall of fame"
  [hof-input]
  (cond hof-input (if (string? hof-input)
                    (or (try (read-string (slurp hof-input))
                             (catch Exception _ [])) 
                        [])
                    hof-input)
        :else []))


(defn master
  "Code executed by the main MPI process with rank=0\\
   Controls EA loop and delegates the evaluation of fitnesses
   to worker MPI processes\n
   Accepts the following messages from the slave threads:\\
   Fitness evaluation results\n
   Sends the following messages to the slave threads:\\
   Evaluate fitness of two opponents\\
   Stop evaluation early and return result\\
   Notify slaves that ERL loop is over"
  [comm & {:keys [pop-size num-generations benchmark-count random-seed stdev hot-start hof-output hof-input gen-output gen-input bench-method next-gen-method prop-hof]
           :or {pop-size 3
                num-generations 1
                benchmark-count 5
                random-seed 1
                bench-method :exp
                next-gen-method :parents
                stdev 0.005
                prop-hof 0.5}
           :as argmap}]
  #_(println "Master argmap: " argmap)
  (assert (MPI/Is_initialized) "MPI must be initialized for master thread to run")
  (let [r (if (int? random-seed) (utils/random random-seed) random-seed)
        gen (start-gen hot-start gen-input pop-size r stdev)]
    (loop [generation (:generation gen)
           pop (:pop gen)
           hof (start-hof hof-input)
           max-actions (:max-actions gen)
           t (:time-ms gen)]
      ;; Report on the status of each generation
      (ERL/report-generation pop generation
                             :max-actions max-actions
                             :time-ms t
                             :gen-output gen-output
                             :hof-output hof-output
                             :hof hof)
      (System/gc)
      (if (= generation num-generations)
        ;;Terminate slave MPI processes and return final result
        (do (terminate-slaves comm (py. comm Get_size))
            {:last-pop pop
             :hall-of-fame (ERL/round-errors hof 3)})
        ;;Fitness evaluation using benchmarking individuals
        (let [{{p :pop
                b :benchmark
                a :action-counts} :result
               t :time} (utils/get-time
                         (benchmark comm
                                    pop
                                    (ERL/get-benchmark benchmark-count pop hof prop-hof :method bench-method)
                                    :deck-seed (.nextInt r)
                                    :max-actions max-actions
                                    :symmetrical? true
                                    :args argmap))
              ;; Selection, mutation, and updating errors of individuals in the hall of fame
              [p h] (ERL/next-generation p r :method next-gen-method)]
          ;; Recur with updated population, hall of fame, max-actions, and time-taken
          (recur (inc generation)
                 p
                 (-> hof
                     (ERL/update-hof b)
                     (ERL/cull-hof)
                     (conj h))
                 (* 2 (utils/mean a))
                 t))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;        Slave MPI Process       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;   Accepts fitness evaluation   ;;
;;tasks from the master thread and;;
;;  returns evaluation results    ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;   The overall lifecycle of a   ;;
;;slave MPI process is as follows:;;
;;                                ;;
;;For each available processor:   ;;
;;   await messages from master   ;;
;;        receive message         ;;
;; sendoff evaluation to a thread ;;
;;  deref result of evaluation    ;;
;;    send result back to master  ;;
;;      await next message        ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn process-message
  "Converts a python dict message containing the
   details of the fitness evaluation task received from
   the main MPI process into a clojure object"
  [message]
  (let [{[{id1 "id" seeds1 "seeds" std1 "stdev"}
          {id2 "id" seeds2 "seeds" std2 "stdev"}] "players"
         max-actions "max-actions"
         deck-seed "deck-seed"} message]
    #_(println "rank " (py. mpi4py.MPI/COMM_WORLD Get_rank) "processing message " id1 id2)
    {:players [{:id (keyword id1) :seeds seeds1 :std std1}
               {:id (keyword id2) :seeds seeds2 :std std2}]
     :max-actions max-actions
     :deck-seed deck-seed}))



(defn thread-to-messages
  "Sends results for each thread in results, and returns
   Ireceives awaiting future tasks for each thread in results\\ 
   [{:thread :result}] -> [{:thread :message} ...]"
  [comm results]
  #_(println "rank " (py. mpi4py.MPI/COMM_WORLD Get_rank) "sent results of threads " (mapv :thread results) "to master")
  (assert (seqable? results) (str "not seq results "results))
  ;; Send evaluation results back to main MPI process
  (mapv #(do (assert (>= (:thread %) 0) "Thread cannot be negative")
             (py. comm isend (py/->py-dict (:result %))
              :dest 0
              :tag (:thread %)))
        results)
  ;; Await new tasks from main MPI process
  (mapv (fn [{t :thread}]
          (assert (>= t 0) "Thread cannot be negative")
          #_(println "rank " (py. mpi4py.MPI/COMM_WORLD Get_rank) " awaiting message on thread " t)
          {:thread t
           :message (py. comm irecv
                         :source 0
                         :tag t)})
        results))

(defn message-to-thread
  "Submits received fitness evaluation tasks to the ExecutorService and
   returns futures awaiting the evaluation results\\
   [{:thread :message} ...] -> [{:thread :result} ...]"
  [messages & {:keys [args]}]
  (assert (and args (map? args)) "Must supply argmap")
  ;; Get message, convert to a jvm object, parse and convert to 
  ;; clojure format with keyword keys, combine argument maps, and 
  ;; submit for evaluation
  (let [submit #(concurrent/msubmit
                 (->> (py. % wait)
                      (py/->jvm)
                      (process-message)
                      (utils/apply-map
                               evaluate-task
                               {:net-gain? true
                                :action-count? true
                                :gc? true}
                               args)))]
    #_(println "rank " (py. mpi4py.MPI/COMM_WORLD Get_rank) 
             " sent futures to threads given the messages " messages)
    (let [m (mapv (fn [{t :thread
                m :message}]
            {:thread t
             :result (submit m)})
          messages)]
      #_(println "MESSAGE TO THREAD " m)
      m)))



(defn slave
  "Code executed by a worker MPI process with rank>0\\
   Computes fitness functions and then queries the main thread
   for more work\n
   Accepts the following messages from the master thread:\\
   tag 0: compute fitness of two opponents\\
   tag 1: Stop waiting for tasks and exit loop\n
   Sends the following messages to the master thread:\\
   tag 0: Fitness evaluation results"
  [comm & {:keys [args]}]
  (assert (and comm args) "Cannot be passed nil parameters")
  (assert (MPI/Is_initialized) "MPI must be initialized for slave thread to run")
  (System/gc)
  ;;Wait for a signal (fitness evaluation task / termination signal) from the main MPI process
  (let [get-status #(py. (:message %) Get_status)
        rank (py. comm Get_rank)]
    (loop [messages (mapv (fn [t]
                            {:thread t
                             :message (py. comm irecv :source 0 :tag t)})
                          (range (utils/num-processors)))
           threads []
           terminate (py. comm irecv :source 0 :tag (utils/num-processors))]
            ;;Terminate waiting loop and proceed to MPI finalization
      (cond (py. terminate Get_status) (do #_(println "rank " (py. comm Get_rank) " terminating with message " (py. terminate wait))
                                         (mapv #(py. (:message %) cancel) messages)
                                           (py. terminate cancel)
                                           nil)
            ;;Parse received message and send off fitness evaluation into a thread
            (some get-status messages) (do (let [{received true
                                                  unreceived false} (group-by get-status messages)]
                                             #_(println "rank " rank " received messages from master " received)
                                             (recur unreceived
                                                    (into threads (message-to-thread received :args args))
                                                    terminate)))
            ;;Send derefed result of thread evaluation to master and await more messages
            (some #(.isDone (:result %)) threads) (do #_(println "process " rank "realized futures on threads " (mapv :thread (filter #(.isDone (:result %)) threads)))
                                                        (let [{realized true
                                                               unrealized false} (group-by #(.isDone (:result %)) threads)
                                                              results (mapv #(update % :result deref) realized)]
                                                          #_(println "rank " rank " realized results from futures: " results)
                                                          (recur (into messages (thread-to-messages comm results))
                                                                 unrealized
                                                                 terminate)))
            ;;Received nothing, so do nothing
            :else (recur messages threads terminate)))))


(defn ERL
  "Main function assigning the master-slave roles to MPI processes
   based on rank\\
   pop-size: size of population\\
   num-generations: number of generations to run for\\
   benchmark-count: number of individuals to choose for the benchmark\\
   random-seed: seed for the random number generator that produces the integers representing mutations in individuals
   If from-block? is true, also seeds the pre-instantiated block of gaussian noise\\
   num-games: number of games to play between individuals for fitness evaluations\\
   max-seq-length: maximum context length of the individuals' transformer models\\
   stdev: The standard deviation of the gaussian mutation applied to individuals\\
   from-block?: whether to pre-instantiate a large block of gaussian noise or to generate the mutations dynamically
   using a random number generator. WARNING: the same integer will map to different mutations depending on
   the value of from-block? and the random seed\\
   block-size: Number of floats in the pre-instantiated block of random noise. The larger the size of the block,
   the more independent mutations indexed by different integers will be\\
   hof-output/hof-input: output/input files for caching hall of fame members\\
   gen-output/gen-input: output/input files for caching information about the current generation\\
   param-output/param-input: output/input files for caching hyperparameter information\\
   bench-method: Method to use for selecting benchmark individuals from the hall of fame. See ERL/get-benchmark\\
   next-gen-method: Method to use for selecting individuals to enter the hall of fame. See ERL/next-generation\\
   prop-hof: The proportion of benchmark individuals that will come from the hall of fame. See ERL/get-benchmark
   as opposed to the current population\\
   transformer-parameters: optional parameters to specify the transformer model architecture\\
   -> Reports out each generation\\
   -> caches information in files for resuming evolution\\
   -> returns the final population and hall-of-fame."
  [& {:keys [pop-size num-generations benchmark-count random-seed num-games max-seq-length stdev from-block? block-size hot-start hof-output hof-input gen-output gen-input param-output param-input bench-method next-gen-method prop-hof transformer-parameters]
      :or {pop-size 3
           num-generations 1
           benchmark-count 5
           random-seed 1
           num-games 10
           max-seq-length 10
           block-size 1e8
           stdev 0.005
           bench-method :exp
           next-gen-method :parents
           prop-hof 0.5
           hof-output "src/clojure/poker/Andrew/results/_hof.out"
           hof-input "src/clojure/poker/Andrew/results/_hof.out"
           gen-output "src/clojure/poker/Andrew/results/_gen.out"
           gen-input "src/clojure/poker/Andrew/results/_gen.out"
           param-output "src/clojure/poker/Andrew/results/_param.out"
           param-input "src/clojure/poker/Andrew/results/_param.out"}
      :as argmap}]
  (let [r (utils/random random-seed)]
    ;;Pre-instantiate large block of gaussian noise
    (when from-block? (utils/initialize-random-block (int block-size) r))
    (when transformer-parameters (transformer/set-parameters transformer-parameters))
    ;Open and close MPI environment
    (with-MPI
      (let [comm mpi4py.MPI/COMM_WORLD
            rank (py. comm Get_rank)]
        ;Rank 0 process is master thread
        (if (= 0 rank)
          (do (println (dissoc argmap :hof-start))
              (when param-output (spit param-output (with-out-str (println argmap))))
              #_(when hot-start (run! println hot-start))
              ;pass on parameters to master process
              (println (apply master comm (mapcat identity (into [] argmap)))))
          ;All other processes are slave threads
          (slave comm :args argmap))
        #_(println "Finalizing rank " rank)))))

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
