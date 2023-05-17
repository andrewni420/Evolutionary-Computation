(ns poker.Andrew.processresult
  (:require [poker.ERL :as ERL]
            [poker.utils :as utils]
            [poker.ndarray :as ndarray]
            [incanter.core]
            [incanter.charts]
            [incanter.stats]
            [poker.concurrent :as concurrent]
            [poker.transformer :as transformer]))

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
  [results & {:keys [make-vector? keep-all?]}]
  (keep-indexed (fn [idx item]
                  (when (if keep-all? (:generation item) (odd? idx)) item))
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
   (mapv #(concurrent/msubmit
           (rand-nth
            (ERL/single-elim % num-tournaments max-seq-length num-games
                             :symmetrical? symmetrical?
                             :stdev stdev
                             :decks decks)))
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

(defn extract-a-winner
  "Extracts a best-of-generation from each generation using a single elimination
   tournament"
  [pop & {:keys [stdev from-block?]
          :or {stdev 0.005}}]
  (ERL/single-elim pop 10 100 1000
                   :symmetrical? true
                   :stdev stdev
                   :from-block? from-block?))


(defn process-multirun
  "Helper for comparison of different runs in a single file"
  [filename & {:keys [keep-all?]}]
  (let [output (read-file filename)]
    (map #(assoc {} :parameters %1 :best-per-gen %2)
         (map first output)
         (map (fn [run] (map #(update % :pop extract-a-winner)
                             (take-last 3 run)))
              (map #(get-generations % :keep-all? keep-all?) output)))))

(defn process-multirun-first
  "Helper for studying the fitness improvement as generation increases"
  [filename]
  (let [output (first (read-file filename))]
    {:parameters (first output)
     :best-per-gen (map #(update % :pop extract-a-winner)
                        (get-generations output))}))

(defn take-generation
  "Helper for studying the fitness improvement as generation increases"
  [{best-per-gen :best-per-gen}]
  (mapv #(let [{{seeds :seeds} :pop
                gen :generation} %]
           (-> %
               (assoc :seeds seeds
                      :id (keyword (str "p" gen)))
               (dissoc :pop :generation)))
        best-per-gen))

(defn take-3
  "Helper for comparing multiple runs in one file.
   Take the best-of-generation of the last 3 generations and rename them"
  [params {parameters :parameters best-per-gen :best-per-gen} & {:keys [n]
                                                                 :or {n 3}}]
  (let [last3 (take-last n best-per-gen)]
    (mapv #(let [{{seeds :seeds} :pop
                  gen :generation} %]
             (-> %
                 (assoc :parameters parameters
                        :seeds seeds
                        :id (keyword (str "p-" (.indexOf params parameters) "-" gen)))
                 (dissoc :pop :generation)))
          last3)))

(defn round-robin
  [pop & {:keys [num-games seq-length]
        :or {num-games 5000
             seq-length 100}}]
  (let [futs (doall (for [ind1 pop
                          ind2 pop :when (not (= ind1 ind2))]
                      (concurrent/msubmit
                       (ERL/versus ind1 ind2 seq-length num-games
                                   :net-gain? true
                                   :as-list? true
                                   :from-block? true
                                   :gc? true))))]
    (map #(:net-gain (deref %)) futs)))

(defn round-round-robin
  [pop & {:keys [num-games seq-length]
        :or {num-games 5000
             seq-length 100}}]
  (let [futs (doall (for [pop1 pop
                          pop2 pop :while (not (= pop1 pop2))
                          ind1 pop1
                          ind2 pop2 :when (not (= ind1 ind2))]
                      (concurrent/msubmit
                       (ERL/versus ind1 ind2 seq-length num-games
                                   :net-gain? true
                                   :as-list? true
                                   :from-block? true
                                   :gc? true))))]
    (map #(:net-gain (deref %)) futs)))

(defn generation-versus
  "Helper for studying the fitness improvement as generation increases
   Extracts winners, renames them, pits them against each other for 5000 games symmetrically,
   and prints the match results"
  [filename]
  (println filename)
  (let [[params & pop] (drop-last 2 (read-file filename))
        processed {:parameters params
                   :best-per-gen (map #(update % :pop extract-a-winner)
                                      (get-generations pop))}
        last-4s (take-generation processed)]
    (println (:parameters processed))
    (println last-4s)
    (let [futs []#_(doall (for [ind1 last-4s
                            ind2 last-4s :when (not (= ind1 ind2))]
                        (concurrent/msubmit
                         (ERL/versus ind1 ind2 100 5000
                                     :net-gain? true
                                     :as-list? true))))
          futs2 (doall (for [ind1 last-4s]
                         (concurrent/msubmit
                          (ERL/versus-other ind1 
                                            utils/random-agent
                                            100
                                            5000
                                            :net-gain? true
                                            :as-list? true))))]
      #_(println (map #(:net-gain (deref %)) futs))
      (println (map deref futs2)))))


(defn multirun-versus
  "Helper for comparison of multiple runs in one file. 
   Extracts winners, renames them, pits them against each other for 5000 games symmetrically,
   and prints the match results"
  [filenames & {:keys [keep-all?]}]
  (println filenames)
  (let [filenames (if (coll? filenames) filenames [filenames])
        processed (mapcat #(process-multirun % :keep-all? keep-all?) filenames)
        params (mapcat #(mapv first (read-file %)) filenames)
        last-4s (map (partial take-3 params) processed)]
    (println params)
    (let [futs (doall (for [p1 last-4s
                            p2 last-4s :when (not (= p1 p2))
                            ind1 p1
                            ind2 p2]
                        (concurrent/msubmit
                         (ERL/versus ind1 ind2 100 5000
                                     :net-gain? true
                                     :as-list? true))))]
      (println (map #(:net-gain (deref %)) futs)))))

(defn process-singlerun
  "Helper for processing a single run in a file"
  [filename]
  (let [output (read-file filename :make-vector? true)]
    {:parameters (first output)
     :best-per-gen (map #(update % :pop extract-a-winner)
                        (take-last 3 (get-generations output)))}))

(defn singlerun-versus
  "Helper for comparison of multiple files of one run per file"
  [filenames]
  (let [processed (map process-singlerun filenames)
        params (map #(first (read-file % :make-vector? true)) filenames)
        last-4s (map (partial take-3 params) processed)]
    (println params)
    (let [futs (doall (for [p1 last-4s
                            p2 last-4s :when (not (= p1 p2))
                            ind1 p1
                            ind2 p2]
                        (concurrent/msubmit
                         (ERL/versus ind1 ind2 100 5000
                                     :net-gain? true
                                     :as-list? true))))]
      (println (map #(:net-gain (deref %)) futs)))))

(defn results-vs
  [filename]
  (let [file (read-file filename :make-vector? true)
        params (first file)]
    (ndarray/initialize-random-block (int (:block-size params))
                                   (:random-seed params)
                                     :ndarray? true)
    (transformer/set-parameters (:transformer-parameters params))
    (let [individuals (->> file
                           (filter :results)
                           (first)
                           (:results)
                           (first)
                           (:hall-of-fame)
                           (map first)
                           (map #(dissoc % :error))
                           (map-indexed #(assoc %2 :id (keyword (str "p" %1)))))
          results (doall (for [ind1 individuals
                               ind2 individuals :when (not (= ind1 ind2))]
                           (concurrent/msubmit (ERL/versus ind1 ind2 100 5000
                                                           :net-gain? true
                                                           :as-list? true
                                                           :from-block? true
                                                           :stdev (:stdev params)
                                                           :gc? true))))]
      (run! #(println (deref %)) results))))



(defn moving-average
  [coll n]
  (assert (odd? n) "Even results in too many moving average elements")
  (into []
        (eduction 
         (map (partial filter identity))
                  (map utils/mean)
                  (partition n 1 (concat (repeat (/ n 2) nil)
                                         coll
                                         (repeat (/ n 2) nil))))))

(defn moving-average-simple
  [coll n]
  (map utils/mean (partition n 1 coll)))


(defn multiplot
  "Plots multiple y-series against x-series
   ys: [y-values label]"
  [x & ys]
  (incanter.core/view
   (reduce
    #(incanter.charts/add-lines %1 x (first %2) :series-label (second %2))
    (incanter.charts/xy-plot x (first (first ys)) :series-label (second (first ys)) :legend true :y-label "y")
    (rest ys))))


(defn plot-best-fit
  [y & {:keys [label window]
        :or {window 11}}]
  (let [x (range (count y))
        coefs (:coefs (incanter.stats/linear-model y (range (count y))))
        best-fit (map #(+ (* % (second coefs)) (first coefs)) x)]
    (println label coefs)
    (multiplot x 
               [y (or label "line")] 
               [best-fit "best-fit"]
               [(moving-average y window) (str "Moving Average, n=" window)])))


#_(def file (read-file "0.0005-big-8b-125-2e-best-hard-0.75h-1200-74193.out" :make-vector? true))


(defn process-inter
  [filename]
  (let [f (read-file filename :make-vector? true)
        r (->> f
               (filter :inter-run)
               (first)
               (:inter-run))
        k (->> r
               (mapcat keys)
               (into #{}))]
    (zipmap k (map #(->> r
                         (map %)
                         (map :mean)
                         (filter identity)
                         (reduce +))
                   k))))

#_(process-inter "0.0005-small-8b-50-2e-best-hard-0.75h-74289.out")

#_(plot-best-fit
   (drop 0 (map second
        (sort-by
         #(read-string (apply str (drop 2 (str (first %)))))
         (let [f (read-file "0.005s-8b-0.75h-parents-hard-74070.out" :make-vector? true)]
           (apply merge-with
                  +
                  (map (fn [gain]
                         (into {}
                               (map #(vector (first %)
                                             (:mean (second %))
                                             #_(* (Math/sqrt 5000)
                                                  (/ (:mean (second %))
                                                     (:stdev (second %)))))
                                    gain)))
                       #_(map :net-gain f) (->> f
                                                (filter :intra-run)
                                                (first )
                                                (:intra-run))))))))
   :label "0.005s-8b-0.75h-parents-hard-74070.out")



#_[;;; small transformer models
   ;;prop-hof
   "0.5-prop-hof-8-small-results-73816.out"
   ;; k-best vs hardexp, 8 vs 16 benchmark, how low can num-games go?
   "0.005s-8b-0.75h-best-best-250-results-74053.out"
   "0.005s-8b-0.75h-best-hard-250-results-74055.out"
   "0.005s-8b-0.75h-best-hard-results-74059.out"
   "0.005s-16b-1h-best-hard-125-results-74058.out"
   "0.0005-small-8b-50-2e-best-hard-0.75h-74289.out"
   ;; 8 vs 12 benchmark on 1.5 exp
   "0.005s-12b-0.75h-parents-hard-1.5e-74071.out"
   "0.005s-8b-0.75h-parents-hard-74070.out"
   ;; stdev for small models
   "0.0005s-8b-0.75h-best-hard-74067.out"
   "0.001s-8b-0.75h-best-hard-74068.out"
   "0.025s-8b-0.75h-best-hard-74069.out"
   ;; sparse vs not
   "0.0005-small-8b-125-2e-best-hard-0.75h-sparse3.vs.not-1K.intra-10K.inter-74281.out"
   
   ;;;big transformer models
   ;;stdev ablation
   "0.0005-big-8b-125-2e-parents-hard-0.75h-74187.out";;good improvement
   "0.0001-big-8b-125-2e-best-hard-0.75h-74280.out";;actually k-best
   "0.00005-big-8b-125-2e-parents-hard-0.75h-74191.out";;no improvement
   ;;pop-size vs gen-number comparison
   "0.0005-big-8b-125-2e-best-hard-0.75h-1200-74193.out"]


#_(plot-best-fit
   (map second (sort-by
                #(read-string (apply str (drop 2 (str (first %)))))
                (:error
                 (first
                  (let [r (->> file (filter :intra-run) (first) (:intra-run) (map :net-gain))]
                    (reduce (fn [res incr]
                              (map #(ERL/update-individual % incr)
                                   res))
                            [{:id :p99 :error {}}]
                            (filter :p99 r)))
                  #_(let [f (read-file "single-experiment-results-74168.out" :make-vector? true)]
                    (reduce (fn [res incr]
                              (map #(ERL/update-individual % incr)
                                   res)) [{:id :p49 :error {}}]
                            (map (fn [gain]
                                   (into {} (map #(vector (first %)
                                                          (second %)
                                                          #_(:mean (second %))
                                                          #_(* (Math/sqrt 5000)
                                                               (/ (:mean (second %))
                                                                  (:stdev (second %)))))
                                                 gain)))
                                 #_(map :net-gain f) (->> f
                                                          (filter :intra-run)
                                                          (first)
                                                          (:intra-run)
                                                          (map :net-gain)
                                                          (take 1000)))))))))
   :label "single-experiment-results-74168.out")


(defn get-best
  [filename & {:keys [multiple?]}]
  (let [[params & pop] (read-file filename)]
    (when (:from-block? params) (ndarray/initialize-random-block (or (:block-size params) (int 1e8))
                                                               (:random-seed params)
                                                               :ndarray? true))
    (let [pop (drop-last 2 pop)
          erl #(ERL/single-elim (:pop %)
                                20
                                100
                                2000
                                :symmetrical? true
                                :from-block? true
                                :stdev (:stdev params))
          res (if multiple? 
                [(concurrent/msubmit (erl (last pop)))]
                  (mapv #(concurrent/msubmit (erl %)) pop))]
      (run! #(println (deref %)) res))))

#_(let [f (read-file "ERL-250x100-67545.out")]
    (println (mapv :time-ms (drop-last 2 f))))


(def aggregated-results
  "Aggregated results of ablation experiments"
  [{:experiment "num-games: 125, 250, 500, 1000",
    :metric "sum";;doesn't seem to be a whole lot of difference. Might just stick to 500 or 250. Might switch to 125
    :0 3.389845103885045,
    :1 -1.9500472185276099,
    :2 -0.12553557325473363,
    :3 -1.314262312102703}
   {:experiment "num-games: 125, 250, 500, 1000",
    :metric "sum in terms of stdev";;what about runtime?
    :0 6.149818856880755,
    :1 -6.666554619972082,
    :2 -4.0308269786619215,
    :3 4.547562741753248}
   {:experiment "pop ablation: 25p25g, 20p30g, 15p40g",
    :metric "sum";;Inconclusive. Need to compare to other pop/gen studies.
    :0 0.9079489889353969,
    :1 -2.110210027115829,
    :2 1.2022610381804304}
   {:experiment "pop ablation: 25p25g, 20p30g, 15p40g",
    :metric "sum in terms of stdev"
    :0 0.17146744541586711,
    :1 2.539536814345111,
    :2 -2.71100425976098}
   {:experiment "benchmark comparison: 4, 6, 8, 16"
    :metric "sum";;need to compare to different benchmark selection strategies. Also compare with runtime
    :0 -5.2519197636431825,;;8 seems to be the best, but what about runtime?
    :1 -6.122138812336342,
    :2 13.637682858495676,
    :3 -2.2636242825161563}
   {:experiment "benchmark comparison: 4, 6, 8, 16",
    :metric "sum in terms of stdev"
    :0 -4.8572393702023895,
    :1 -3.6223334749501865,
    :2 5.250030437369037,
    :3 3.229542407783533}
   {:experiment "generation ablation: 30p20g, 40p15g",
    :metric "sum";;have to compare to pop ablation and scaling studies like the 1200 study
    :0 -1.7933880949143384,;;40p15 is better
    :1 1.7933880949143362}
   {:experiment "generation ablation: 30p20g, 40p15g",
    :metric "sum in terms of stdev"
    :0 -2.184460415459356,
    :1 2.184460415459356}
   {:experiment "positional encoding: [32 16 8 8], [36 20 4 4], [48 8 4 4], [52 4 4 4]",
    :metric "sum",;;[32 16 8 8] is best, but still have to compare vs [16 16 16 16]
    :0 28.353950232764937,
    :1 -17.482229163280124,
    :2 2.875050660889375,
    :3 -13.746771730374194}
   {:experiment "positional encoding: [32 16 8 8], [36 20 4 4], [48 8 4 4], [52 4 4 4]",
    :metric "sum in terms of stdev",
    :0 19.77345304205734,
    :1 -8.835545495991669,
    :2 -3.450356152728985,
    :3 -7.487551393336691}
   {:experiment "num-heads: 2, 4, 8, 16",
    :metric "sum",;;8 heads seems to be the best
    :0 3.0141126828954974,
    :1 -18.11236139100984,
    :2 16.753031427915953,
    :3 -1.6547827198016103}
   {:experiment "num-heads: 2, 4, 8, 16",
    :metric "sum in terms of stdev",
    :0 7.044787348876627,
    :1 -6.405753574635852,
    :2 7.008001165681474,
    :3 -7.647034939922247}
   {:experiment "seq-length: 25, 50, 100, 200",
    :metric "sum",;;sequence length seems to not matter too much so I'm going to use 100
    :0 11.430990281119676,
    :1 -27.134550742299318,
    :2 10.616033957470478,
    :3 5.087526503709162}
   {:experiment "seq-length: 25, 50, 100, 200",
    :metric "sum in terms of stdev",
    :0 3.345839020846146,
    :1 -8.693347444189376,
    :2 5.282064185267595,
    :3 0.06544423807562172}
   {:experiment "stdev: 0.5, 0.05, 0.005",
    :metric "sum",;;0.005 is the best, makes sense. May have to experiment more around 0.005
    :0 -7.65426523778135,
    :1 -7.097659643258988,
    :2 14.751924881040337}
   {:experiment "stdev: 0.5, 0.05, 0.005",
    :metric "sum in terms of stdev",
    :0 -7.97755955737327,
    :1 -1.6530528654843164,
    :2 9.630612422857585}
   {:experiment "1200 pop*gen: 15p80g, 80p15g, 25p50g, 50p25g, 34p34g, 57p21g",
    :metric "sum",;;seems like number 57p21g is the best, followed by 50p25g
    :0 11.169751801957542,
    :1 -4.985691783262645,
    :2 -16.477825971966297,
    :3 3.518693546835202,
    :4 -14.858707350941447,
    :5 21.633779757377624}
   {:experiment "1200 pop*gen: 15p80g, 80p15g, 25p50g, 50p25g, 34p34g, 57p21g",
    :metric "sum in terms of stdev",
    :0 10.699136927854514,
    :1 10.601250242621642,
    :2 0.6806537835231659,
    :3 26.73996305363102,
    :4 -99.43855825408842,
    :5 50.717554246458064}
   {:experiment "300 pop*gen: 25p12g, 12p25g, 17p17g, 15p20g, 20p15g",
    :metric "sum",;;seems like 20p15g is the best, followed by 17p17g. Others are probably just too little 
    :0 -24.693096739615907,
    :1 -22.834150277200692,
    :2 14.491053831479014,
    :3 2.1367267723934043,
    :4 30.899466412944165}
   {:experiment "300 pop*gen: 25p12g, 12p25g, 17p17g, 15p20g, 20p15g",
    :metric "sum in terms of stdec",
    :0 -15.548416804392632,
    :1 -15.88522460725057,
    :2 11.364300528505492,
    :3 -0.764613161862278,
    :4 20.833954044999988}])



#_{:slumbot-results {:mean -1.1848, :stdev 12.66235558496115, :CI-95% [-1.5429454999968553 -0.8266545000031449]}}

#_(sort-by
#(read-string (apply str (drop 2 (str (first %)))))
 {:p11 -0.03702653464756178,
  :p2 1.6385235421632935,
  :p4 -1.8559997958559449,
  :p3 -0.17911542014239362,
  :p35 -1.993864468087909,
  :p33 2.2886050274045893,
  :p21 0.06914137130681175,
  :p46 1.5301674679679007,
  :p28 -0.7540962325062,
  :p36 -1.7429776906711776,
  :p48 -1.8219191571933666,
  :p5 3.090059120475086,
  :p8 4.737941163951412,
  :p47 -1.8907586101924203,
  :p43 -1.7739529644468206,
  :p9 2.1575406098591614,
  :p45 -4.52806140192151,
  :p22 -0.8177429214810317,
  :p25 -2.100719504952203,
  :p24 -0.5139993567750836,
  :p12 -6.543237331302951,
  :p41 0.13122008985999645,
  :p49 2.007870608905415,
  :p0 3.622821538607565,
  :p32 -5.062701202135638,
  :p26 2.2265419427205275,
  :p39 0.279608395743453,
  :p38 -0.3666958272448248,
  :p7 -0.8959226935856377,
  :p19 0.8233624976263598,
  :p30 0.8910468763515909,
  :p14 -0.2961568194204476,
  :p15 2.5149466124094912,
  :p13 2.5518138079002437,
  :p17 -2.940865661287308,
  :p10 2.8346772776940847,
  :p18 2.8946625983966014,
  :p16 4.400494013372673,
  :p44 -0.16067135578242409,
  :p40 7.3405313633939855,
  :p20 -0.4937134700706811,
  :p37 -2.134318125665856,
  :p31 -4.0772920983975345,
  :p27 -3.5538817623517107,
  :p34 -3.9267878274205925,
  :p42 -0.8350671383951784,
  :p6 0.27571770947597685,
  :p23 0.5286252501963601,
  :p29 5.339510424053925})


#_(run! #(println % ",") (map second 
                              [[:p0 3.622821538607565]
[:p2 1.6385235421632935]
[:p3 -0.17911542014239362]
[:p4 -1.8559997958559449]
[:p5 3.090059120475086]
[:p6 0.27571770947597685]
[:p7 -0.8959226935856377]
[:p8 4.737941163951412]
[:p9 2.1575406098591614]
[:p10 2.8346772776940847]
[:p11 -0.03702653464756178]
[:p12 -6.543237331302951]
[:p13 2.5518138079002437]
[:p14 -0.2961568194204476]
[:p15 2.5149466124094912]
[:p16 4.400494013372673]
[:p17 -2.940865661287308]
[:p18 2.8946625983966014]
[:p19 0.8233624976263598]
[:p20 -0.4937134700706811]
[:p21 0.06914137130681175]
[:p22 -0.8177429214810317]
[:p23 0.5286252501963601]
[:p24 -0.5139993567750836]
[:p25 -2.100719504952203]
[:p26 2.2265419427205275]
[:p27 -3.5538817623517107]
[:p28 -0.7540962325062]
[:p29 5.339510424053925]
[:p30 0.8910468763515909]
[:p31 -4.0772920983975345]
[:p32 -5.062701202135638]
[:p33 2.2886050274045893]
[:p34 -3.9267878274205925]
[:p35 -1.993864468087909]
[:p36 -1.7429776906711776]
[:p37 -2.134318125665856]
[:p38 -0.3666958272448248]
[:p39 0.279608395743453]
[:p40 7.3405313633939855]
[:p41 0.13122008985999645]
[:p42 -0.8350671383951784]
[:p43 -1.7739529644468206]
[:p44 -0.16067135578242409]
[:p45 -4.52806140192151]
[:p46 1.5301674679679007]
[:p47 -1.8907586101924203]
[:p48 -1.8219191571933666]
[:p49 2.007870608905415]]))


