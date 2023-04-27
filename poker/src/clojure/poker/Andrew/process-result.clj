(ns poker.Andrew.process-result
  (:require [poker.ERL :as ERL]
            [poker.utils :as utils]
            [poker.concurrent :as concurrent]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;    Utilities for parsing and     ;;;
;;;     comparing the outputs of     ;;;
;;;        evolutionary runs         ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; ERL output is structured like so:
;;;
;;; [{:generation 0 :pop pop0}
;;; "Elapsed time: time0"
;;; {:generation 1 :pop pop1}
;;; ...
;;; {:generation last-generation :pop last-pop}
;;; {:last-pop last-pop :hall-of-fame hof}]
;;;
;;; Sometimes multiple of these will be present in a file if there were multiple
;;; ERL runs in that batch
;;;
;;; Uses for parsed ERL outputs include:
;;; - Analyzing progress of evolutionary run by 
;;; extracting a best individual from each generation and comparing them
;;; - Comparing different hyperparameter settings by comparing 
;;; best individuals from the last population of the runs for each hyperparameter
;;; - Selecting best individual to play against Slumbot
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn read-file 
  "Reads in a clojure data structure from a file. Automatically prepends the path
   src/clojure/poker/Andrew/results/ \\
   Use make-vector to read all data structures in the file, not just the first, into a 
   vector."
  [name & {:keys [make-vector?]}]
  (read-string
   (if make-vector?
     (str "[" (slurp (str "src/clojure/poker/Andrew/results/" name)) "]")
     (slurp (str "src/clojure/poker/Andrew/results/" name)))))

(def output-structure
  "ERL output is structured like so:\\
   [{:generation 0 :pop pop0}\\
   \"Elapsed time: time0\"\\
   {:generation 1 :pop pop1}\\
   ...\\
   {:generation last-generation :pop last-pop}\\
   {:last-pop last-pop :hall-of-fame hof}]\\
   Sometimes multiple of these will be present in a file if there were multiple
   ERL runs in that batch"
  nil)

(defn process-results
  "If the results are given as a file path, reads it in, possibly adding 
   brackets to turn multiple printed outputs into a vector, and returning
   the resulting data structure"
  [results make-vector?]
  (if (string? results)
    (read-string (if make-vector?
                   (str "[" (slurp results) "]")
                   (slurp results)))
    results))

(defn get-generations
  "Given the output of a single run, parses it to get the {:generation :pop} report
   on the population at each generation\\
   -> [{:generation :pop} ...]"
  [results & {:keys [make-vector?]}]
  (keep-indexed (fn [idx item]
                  (when (odd? idx) item))
                (process-results results make-vector?)))

(defn get-hof
  "Get the hall-of-fame from ERL results. It's taken from the last output, which was
   a {:last-pop :hof} map"
  [results & {:keys [make-vector?]}]
  (:hall-of-fame (last (process-results results make-vector?))))

(defn get-last-pop
  "Get the last population from ERL results. It's taken from the last output, which was
   a {:last-pop :hof} map"
  [results & {:keys [make-vector?]}]
  (:last-pop (last (process-results results make-vector?))))





#_(mapv (partial nth (first pretest)) (range 2 (dec (count (first pretest))) 2))

#_(nth (first pretest) 2)

#_(let [lg (:pop (first (take-last 2 pretest)))
      fg (:pop (nth pretest 1))]
  (time (poker.ERL/versus (first lg) (second lg) 100 500 :stdev 0.005 :net-gain? true))
  (time (poker.ERL/versus (first fg) (second fg) 100 500 :stdev 0.005 :net-gain? true)))

(defn extract-winners 
  "Extracts the best individual from each generation as determined by a single elimination
   tournament"
  [generations num-tournaments max-seq-length num-games & {:keys [reset-id? symmetrical? decks stdev]
                                                           :or {symmetrical? true
                                                                stdev 0.05}}]
  ((if reset-id?
     (partial mapv #(assoc %2 :id %1) (range (count generations)))
     identity)
    (mapv #(rand-nth (ERL/single-elim % num-tournaments max-seq-length num-games
                                      :symmetrical? symmetrical?
                                      :stdev stdev
                                      :decks decks))
          generations)))


(defn benchmark-other
  "Benchmarks the population against a set of non-transformer individuals"
  [pop benchmark max-seq-length num-games & {:keys [manager decks symmetrical?]}]
  (let [decks (if symmetrical?
                (or decks
                    (repeatedly num-games
                                #(shuffle utils/deck)))
                decks)
        vs (if manager
             #(ERL/versus-other %1 %2 max-seq-length num-games :reverse? %3 :decks decks :manager manager)
             #(ERL/versus-other %1 %2 max-seq-length num-games :reverse? %3 :decks decks))
        res1 (doall
              (for [ind1 pop
                    ind2 benchmark :when (not (= ind1 ind2))]
                (concurrent/msubmit (vs ind1 ind2 false))))
        res2 (when symmetrical?
               (doall
                (for [ind1 pop
                      ind2 benchmark :when (not (= ind1 ind2))]
                  (concurrent/msubmit (vs ind1 ind2 true)))))]
    (reduce (fn [p res]
              (map #(ERL/update-individual % res :merge-fn conj) p))
            pop
            (map deref (concat res1 res2)))))