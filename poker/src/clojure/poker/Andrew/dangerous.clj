(ns poker.Andrew.dangerous
  (:require [poker.utils :as utils]
            [poker.headsup :as headsup]
            [poker.slumbot :as slumbot]))


;;Slumbot rollout - edits file
#_(loop [i 0]
  (if (= i 10)
    nil
    (do
      (time (slumbot/slumbot-rollout utils/random-agent
                                     "slumbot-history-random.txt"
                                     10
                                     240))
      (utils/combine-vectors "slumbot-history-random.txt")
      (flush)
      (recur (inc i)))))