(ns poker.Andrew.processresult
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
                                   :as-list? true))))]
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
                                   :as-list? true))))]
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



#_(sort-by
   #(read-string (apply str (drop 2 (str (first %)))))
   (let [f (read-file "generation-versus-69994.out" :make-vector? true)]
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
                 (nth f 3)))))

(defn get-best
  [filename & {:keys [multiple?]}]
  (let [[params & pop] (read-file filename)]
    (when (:from-block? params) (utils/initialize-random-block (or (:block-size params) (int 1e8))
                                                               (:random-seed params)))
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

#_(* 0.0024249158681258765 (Math/sqrt 5000))



#_(let [f (read-file "generation-versus-69994.out" :make-vector? true)]
    (reduce (fn [res incr]
              (map #(ERL/update-individual % incr)
                   res)) [{:id :p1 :error {}}]
            (map (fn [gain]
                   (into {} (map #(vector (first %)
                                          (:mean (second %))
                                          #_(* (Math/sqrt 5000)
                                               (/ (:mean (second %))
                                                  (:stdev (second %)))))
                                 gain)))
                 (nth f 3)))
    #_(spit "src/clojure/poker/Andrew/results/temp.txt"
            (with-out-str (clojure.pprint/pprint
                           (filter #(some (partial contains? (into #{} (map first %)))
                                          [:p-0-23 :p-0-24 :p-0-25]) (nth f 2))))))

#_(sort-by
   #(read-string (apply str (drop 2 (str (first %)))))
   {:p97 4.489108259836087,
    :p51 3.4304706497962947,
    :p11 3.0522443909213752,
    :p73 1.6564058522956815,
    :p87 -0.49232237160502246,
    :p3 -3.8512355318557687,
    :p35 -1.9053243740977936,
    :p33 -4.3268724140182675,
    :p75 10.64001517812119,
    :p91 1.0680747914768856,
    :p21 2.895022595208129,
    :p83 1.6397962612952106,
    :p99 5.743057437031725,
    :p5 3.0774176404618725,
    :p47 5.3306781491966735,
    :p43 -1.1909240320905414,
    :p9 0.6442596035585151,
    :p93 -0.014316886306870114,
    :p57 -2.170552598658751,
    :p45 0.059079355582477566,
    :p25 -1.395910966403542,
    :p61 2.345773837052664,
    :p41 0.02798426299927015,
    :p49 3.7213417210861977,
    :p39 1.9426201027739871,
    :p53 0.7156910789549791,
    :p7 -0.41942123286314015,
    :p65 5.833088180749535,
    :p19 -3.237639680846728,
    :p63 -0.9063584762739895,
    :p85 3.874275442033791,
    :p15 1.0747868384557135,
    :p13 -0.1684478397314404,
    :p79 2.122556028310089,
    :p17 1.7467111952669983,
    :p67 4.014855387391798,
    :p71 6.522252652248985,
    :p77 -0.38275973617459114,
    :p89 -0.7167953474374714,
    :p37 2.8565253256152103,
    :p31 6.99193142581964,
    :p81 3.268431534863338,
    :p27 -1.1149478410157057,
    :p69 1.8394766899496648,
    :p95 1.6764735729013005,
    :p59 -1.3684901948201427,
    :p55 6.442325449357152,
    :p23 3.199498184183935,
    :p29 1.1359888187914382})


#_(run! #(println % ",") (map second [[:p3 -3.8512355318557687]
[:p5 3.0774176404618725]
[:p7 -0.41942123286314015]
[:p9 0.6442596035585151]
[:p11 3.0522443909213752]
[:p13 -0.1684478397314404]
[:p15 1.0747868384557135]
[:p17 1.7467111952669983]
[:p19 -3.237639680846728]
[:p21 2.895022595208129]
[:p23 3.199498184183935]
[:p25 -1.395910966403542]
[:p27 -1.1149478410157057]
[:p29 1.1359888187914382]
[:p31 6.99193142581964]
[:p33 -4.3268724140182675]
[:p35 -1.9053243740977936]
[:p37 2.8565253256152103]
[:p39 1.9426201027739871]
[:p41 0.02798426299927015]
[:p43 -1.1909240320905414]
[:p45 0.059079355582477566]
[:p47 5.3306781491966735]
[:p49 3.7213417210861977]
[:p51 3.4304706497962947]
[:p53 0.7156910789549791]
[:p55 6.442325449357152]
[:p57 -2.170552598658751]
[:p59 -1.3684901948201427]
[:p61 2.345773837052664]
[:p63 -0.9063584762739895]
[:p65 5.833088180749535]
[:p67 4.014855387391798]
[:p69 1.8394766899496648]
[:p71 6.522252652248985]
[:p73 1.6564058522956815]
[:p75 10.64001517812119]
[:p77 -0.38275973617459114]
[:p79 2.122556028310089]
[:p81 3.268431534863338]
[:p83 1.6397962612952106]
[:p85 3.874275442033791]
[:p87 -0.49232237160502246]
[:p89 -0.7167953474374714]
[:p91 1.0680747914768856]
[:p93 -0.014316886306870114]
[:p95 1.6764735729013005]
[:p97 4.489108259836087]
[:p99 5.743057437031725]]))
