(ns poker.concurrent
  (:import java.util.concurrent.ExecutorService
           java.util.concurrent.Executors))

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

