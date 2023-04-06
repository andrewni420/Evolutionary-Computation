(ns poker.pyutils
  (:require [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :as py :refer [py. py.. py.-]]
            [poker.utils :as utils]))

(require-python '[requests :as requests])


;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; External Interface ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;

(def slumbot-bb
  "big blind bet in a game against slumbot"
  100.0)

(def slumbot-start-bb
  "Number of big blinds a player starts with when playing against slumbot"
  200.0)

(def slumbot-username "AndrewNi")

(def slumbot-password "HjHCLV@Q9dTqeka")

(defn handle-post-error [r]
  (if (not= 200 (py/get-attr r "status_code"))
    (clojure.pprint/pprint {:r r
                            :error "ERROR"
                            :json (try ((r "json"))
                                       (catch Exception e e))}) nil))

(defn get-json [r]
  (try ((py/get-attr r "json"))
       (catch Exception e (println "Can't get json" e))))

(defn handle-r-error [r input]
  (when (r "error_msg") (clojure.pprint/pprint {:error "ERROR"
                                                :r r
                                                :input input})))


(defn slumbot-login
  "Logs into the slumbot api using username and password\\
   -> token"
  ([username password]
   (let [request (requests/post "https://slumbot.com/api/login"
                                :json {"username" username
                                       "password" password})
         _ (handle-post-error request)
         r (get-json request)]
     (handle-r-error r {:username username :password password})
     (r "token")))
  ([] (slumbot-login slumbot-username slumbot-password)))

(defn slumbot-new-hand
  "Requests a new hand from the slumbot api using the login token\\
   -> [token response]"
  ([token]
   (let [request (requests/post "https://slumbot.com/api/new_hand"
                                :json {"token" token})
         _ (handle-post-error request)
         r (get-json request)]
     (handle-r-error r {:token token})
     [token r]))
  ([] (slumbot-new-hand (slumbot-login))))

(defn slumbot-send-action
  "Send an incremental action to the slumbot api using the login token\\
   -> response"
  [token action]
  (let [request (requests/post "https://slumbot.com/api/act"
                               :json {"token" token
                                      "incr" action})
        _ (handle-post-error request)
        r (get-json request)]
    (handle-r-error r {:token token :action action})
    r))

(def move-map
  "Map from the full name of a move to its one-letter abbreviation as used by slumbot\\
   {string string}"
  {"Call" "c",
   "Check" "k",
   "Fold" "f",
   "Bet" "b",
   "All-In" "b",
   "Raise" "b"})

(defn encode-action
  "Encode an action for processing by slumbot\\
   -> string"
  ([action bet-values cur-player]
   (let [new-bets (update bet-values
                          cur-player
                          (partial + (second action)))
         m (move-map (first action))]
     (cond
       (and (= (first action)
               "All-In")
            (= (first new-bets)
               (second new-bets))) "c"
       (= m "b") (str m (int (* slumbot-bb (new-bets cur-player))))
       :else (str m))))
  ([action game-state]
   (let [{bet-values :bet-values
          current-player :current-player} game-state]
     (encode-action action bet-values current-player))))

(defn encode-round
  "Encodes a round of betting for interface with Slumbot\\
   -> string"
  [round-history bet-values]
  (let [round-history (map second round-history)]
    (loop [s ""
           bet-values bet-values
           cur-player 0
           round-history round-history]
      (if (empty? round-history)
        s
        (let [new-bets (update bet-values cur-player (partial + (second (first round-history))))
              m (move-map (ffirst round-history))]
          (recur (if (= m "b")
                   (str s m (int (* slumbot-bb (new-bets cur-player))))
                   (str s m))
                 new-bets
                 (- 1 cur-player)
                 (rest round-history)))))))

(defn encode-action-history
  "Encode the actions in a game for interface with SlumBot\\
   -> string"
  [action-history]
  (clojure.string/join "/" (map-indexed #(encode-round %2 (if (= 0 %1) [0.5 1.0] [0.0 0.0])) action-history)))


(defn decode-action
  "Decode an incremental action from SlumBot\\
   -> action"
  ([action bet-values current-player current-bet money]
   (condp = (first action)
     \c ["Call" (- current-bet (bet-values current-player))]
     \k ["Check" 0.0]
     \b (let [value (/ (read-string (clojure.string/join (rest action))) slumbot-bb)]
          (cond
            (zero? (- (+ money (bet-values current-player)) value))
            ["All-In" money]
            (zero? current-bet) ["Bet" value]
            (> value current-bet)
            ["Raise" (- value (bet-values current-player))]
            :else ["Bet" value]))
     \f ["Fold" 0.0]))
  ([action game-state] (let [{current-bet :current-bet
                              bet-values :bet-values
                              current-player :current-player
                              players :players} game-state
                             money (:money (players current-player))]
                         (decode-action action bet-values current-player current-bet money))))

#_(encode-action ["Bet" 1.5] [0 0] 0)

(defn get-first-action
  "Get the string corresponding to the first action in a round of actions received from slumbot\\
   -> [round-actions first-action rest]"
  [round-actions]
  (re-find #"([a-zA-Z]\d*)(.*)" round-actions))

(defn get-action+roundover
  "get the string corresponding to the first action or a round-over marker 
   from a string of actions received from slumbot\\
   -> [round-actions first-action/roundover rest]"
  [round-actions]
  (re-find #"([a-zA-Z/]\d*)(.*)" round-actions))

(defn decode-round
  "Decodes a round of betting, given the money each player has, the player starting off the round,
   and the string of actions in the round. \\
   -> [decoded-actions money] the decoded actions and the updated money vector"
  [round-actions current-player bet-values money]
  (loop [current-player current-player
         bet-values bet-values
         current-bet (apply max bet-values)
         round-actions round-actions
         returned-actions []]
    (if (empty? round-actions)
      [returned-actions money]
      (let [[_ m r] (get-first-action round-actions)
            decoded (decode-action m bet-values current-player current-bet (money current-player))]
        (recur (- 1 current-player)
               (update bet-values
                       current-player
                       (partial + (second decoded)))
               (max current-bet (+ (bet-values current-player) (second decoded)))
               r
               (conj returned-actions decoded))))))

(defn decode-action-history
  "Decodes a string of actions from a slumbot game\\
   -> [round = [action = [action-type spent-money] ...] ...]"
  [action-history]
  (let [round-actions (clojure.string/split action-history #"/")]
    (loop [i 0
           decoded-actions []
           money [slumbot-start-bb slumbot-start-bb]]
      (if (= i (count round-actions))
        decoded-actions
        (let [[action money] (decode-round (round-actions i)
                                           (if (= 0 i) 0 1)
                                           (if (= 0 i) [0.5 1.0] [0.0 0.0])
                                           money)]
          (recur (inc i)
                 (conj decoded-actions action)
                 money))))))

(defn last-action
  "Returns the last action taken from a round of actions received from slumbot\\
   -> string"
  [action]
  (let [last-round (last (clojure.string/split action #"/"))]
    (loop [[_ a r] [nil nil last-round]]
      (if (empty? r)
        a
        (recur (get-first-action r))))))

(defn parse-card
  "Parses a card of the form \"Ah\" into the form [14 \"Hearts\"]
   -> card = [value suit]"
  [card]
  [(utils/value-from-facecard (str (first card)))
   (utils/suit-from-abbr (str (second card)))])


#_(decode-action-history "b200b300c/b100c/b100c/b100c")

#_(encode-action-history [[[:p0 ["Call" 0.5]] [:p1 ["Check" 0.0]]]
                          [[:p1 ["Check" 0.0]] [:p0 ["Check" 0.0]]]
                          [[:p1 ["Check" 0.0]] [:p0 ["Check" 0.0]]]
                          [[:p1 ["Check" 0.0]] [:p0 ["Bet" 1.0]] [:p1 ["All-In" 95.0]] [:p0 ["Call" 94.0]]]])


