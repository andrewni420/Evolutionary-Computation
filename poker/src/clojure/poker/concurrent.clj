(ns poker.concurrent
  (:import java.util.concurrent.ExecutorService
           java.util.concurrent.Executors))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Overview:
;;; A collection of wrappers around java's ExecutorService
;;; providing control over the thread pool
;;;
;;; Each Clojure future executes in its own thread. 
;;; With 100 individuals playing 2 sets of games against each of 
;;; 10 benchmark individuals, that amounts to 2000 threads.
;;; Since the number of processors is capped at 116, this requires
;;; manual control of thread creation via a thread pool to avoid
;;; unnecessary overhead
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Structure:
;;; Global work-stealing thread pool with size equal to 
;;; the number of processors handles all asynchronous tasks
;;;
;;; submit and msubmit (macro version) submit a runnable, which is implemented by an ifn with 
;;; no arguments, and returns a future that can be derefed to obtain the result. 
;;; The submitted runnable is added to the global ExecutorService's task queue
;;; and is picked up by threads in the pool when available
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(def ^ExecutorService service 
  "Work-stealing pool with target parallelism equal to the number of 
   cores detected. Manages concurrency of the entire project"
  (Executors/newWorkStealingPool))

(defn submit 
  "Submit a job (function with no arguments) to the global 
   work-stealing pool. Returns a deref-able future.\\
   -> Future"
  [f]
  (.submit service
           ^Callable f))

(defmacro msubmit
  "Macro version of submit that executes the body as a no argument
   function\\
   -> Future"
  [f]
  `(submit (fn [] ~f)))


(defn parse-time-unit
  "Parse a string as a time unit. Possible values:\\
   DAYS, HOURS, MINUTES, SECONDS, MICROSECONDS, MILLISECONDS, NANOSECONDS\\
   -> java.util.concurrent.TimeUnit"
  [unit]
  (condp = unit
    "DAYS" java.util.concurrent.TimeUnit/DAYS
    "HOURS" java.util.concurrent.TimeUnit/HOURS
    "MICROSECONDS" java.util.concurrent.TimeUnit/MICROSECONDS
    "MILLISECONDS" java.util.concurrent.TimeUnit/MILLISECONDS
    "MINUTES" java.util.concurrent.TimeUnit/MINUTES
    "NANOSECONDS" java.util.concurrent.TimeUnit/NANOSECONDS
    "SECONDS" java.util.concurrent.TimeUnit/SECONDS))


(defn submit-all
  "Submits all of the given no-argument functions to the global
   ExecutorService.\\
   -> List<Future>"
  ([functions timeout unit]
  (.invokeAll ^java.util.Collection functions
              timeout
              (if (string? unit) 
                (parse-time-unit unit)
                unit)))
  ([functions]
   (.invokeAll ^java.util.Collection functions)))

(defn shutdown
  "Shuts down the service"
  []
  (.shutdown service))

