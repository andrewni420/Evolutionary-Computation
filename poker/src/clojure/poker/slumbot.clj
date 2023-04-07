(ns poker.slumbot
  (:require [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :as py :refer [py. py.. py.-]]
            [clojure.pprint :as pprint]
            [clojure.string :as string]
            [poker.utils :as utils]
            [poker.headsup :as headsup]))

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
    (pprint/pprint {:r r
                            :error "ERROR"
                            :json (try ((r "json"))
                                       (catch Exception e e))}) nil))

(defn get-json [r]
  (try ((py/get-attr r "json"))
       (catch Exception e (println "Can't get json" e))))

(defn handle-r-error [r input]
  (when (r "error_msg") (pprint/pprint {:error "ERROR"
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
  (string/join "/" (map-indexed #(encode-round %2 (if (= 0 %1) [0.5 1.0] [0.0 0.0])) action-history)))


(defn decode-action
  "Decode an incremental action from SlumBot\\
   -> action"
  ([action bet-values current-player current-bet money]
   (condp = (first action)
     \c ["Call" (- current-bet (bet-values current-player))]
     \k ["Check" 0.0]
     \b (let [value (/ (read-string (string/join (rest action))) slumbot-bb)]
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
  (let [round-actions (string/split action-history #"/")]
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
  (let [last-round (last (string/split action #"/"))]
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



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;          Slumbot HUNL interface           ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn parse-winnings
  "Takes a response from a finished game against slumbot, and performs final effects
   such as updating the money of each player and revealing hands\\
   w: how much money (in 0.01bb) the client has won\\
   r: response from Slumbot\\
   game-state: current game state\\
   -> game-state"
  [w r game-state]
  (let [{hands :hands
         players :players
         action-history :action-history} game-state
        client-pos (r "client_pos")
        game-state (assoc game-state
                          :hands (assoc hands
                                        client-pos (map parse-card
                                                        (r "bot_hole_cards"))))
        game-state (if (not= "Fold" (utils/fsecond (last (last action-history))))
                     (headsup/showdown game-state)
                     game-state)]
    (assoc game-state
           :players (assoc players
                           client-pos (utils/set-money (players client-pos)
                                                       (- slumbot-start-bb
                                                          (/ w slumbot-bb)))
                           (- 1 client-pos) (utils/set-money (players (- 1 client-pos))
                                                             (+ slumbot-start-bb
                                                                (/ w slumbot-bb))))
           :game-over true)))


(defn parse-incr-action
  "Takes a response from slumbot, converts it into the incremental actions occurring
   since the last update, parses these incremental actions, and applies them to the game-state.\\
   If the game has ended, calls parse-winnings to return the final game-state.\\
   r: response from Slumbot\\
   game-state: current game state\\
   -> game-state"
  [r game-state]
  (let [_ (if (r "error_msg") (pprint/pprint {:error "ERROR"
                                                      :r r
                                                      :game-state game-state}) nil)
        old-action (r "old_action")
        action (r "action")
        incr (try (nth (re-find (re-pattern (str "(" old-action ")(.*)"))
                                action)
                       2)
                  (catch Exception e (println (.toString e) r game-state)))
        board (map parse-card (r "board"))
        client-hand (map parse-card (r "hole_cards"))
        client-pos (r "client_pos")]
    (loop [[_ a rem] (get-action+roundover incr)
           game-state (assoc game-state
                             :community board
                             :visible board
                             :hands (assoc [nil nil] (- 1 client-pos) client-hand)
                             :r r)]
      (if a
        (recur (get-action+roundover rem)
               (if (= a "/")
                 (headsup/next-round game-state)
                 (headsup/parse-action (decode-action a game-state)
                               game-state)))
        (if-let [w (r "winnings")]
          (parse-winnings w r game-state)
          game-state)))))


(defn init-game-slumbot
  "Initializes and plays a game of poker, returning the updated players
   and updated game-history\\
   token: Login token\\
   agent: agent to play against slumbot\\
   game-history: current history of games played against slumbot\\
   -> [token game-state game-history]"
  [token agent game-history]
  (let [[token r] (slumbot-new-hand token)
        players [(utils/init-player agent :client)
                 (utils/init-player :slumbot :bot)]
        client-pos (- 1 (r "client_pos"))
        game-state (headsup/pay-blinds (headsup/init-game (if (= 0 client-pos)
                                            players
                                            (into [] (reverse players)))))
        game-state (assoc game-state
                          :hands (assoc [nil nil]
                                        client-pos
                                        (map parse-card
                                             (r "hole_cards")))
                          :community (map parse-card (r "board"))
                          :visible (map parse-card (r "board"))
                          :r r)
        game-state (parse-incr-action r game-state)
        #_(if (= 0 client-pos)
            game-state
            (parse-action (decode-action (r "action") game-state)
                          game-state))]
    [token game-state game-history]))



#_(parse-incr-action {"old_action" "",
                      "action" "",
                      "client_pos" 0,
                      "hole_cards" ["Ks" "4h"],
                      "board" ["Ad", "7d", "5s", "9h", "Tc"],
                      "bot_hole_cards" ["Qh", "Qd"],
                      "winnings" -600}
                     (pay-blinds (init-game [{:money 200 :id :p0} {:money 200 :id :p1}])))

(defn remove-unneeded-r [r]
  (remove #(contains? #{"session_total"
                        "session_baseline_total"
                        "old_action"
                        "session_num_hands"
                        "won_pot"} (first %)) r))

(defn play-game-slumbot
  "Plays a game of poker from start to finish against Slumbot. When finished, summarizes the important features
   of the game and appends it to game-history.\\
   token: login token\\
   agent: agent to play against slumbot\\
   game-history: current history of agent vs slumbot games\\
   -> [token agent game-history]"
  [token agent game-history]
  (let [[token game-state game-history] (init-game-slumbot token agent game-history)]
    (loop [game-state game-state]
      (if (:game-over game-state)
        [token
         agent
         (conj game-history
               (assoc (headsup/state-to-history {:players (map #(utils/set-money % slumbot-start-bb)
                                                       (:players game-state))}
                                        game-state)
                      :r (into {}
                               (remove-unneeded-r (utils/recursive-copy
                                                   (:r game-state))))))]
        (let [action (headsup/make-move game-state game-history)
              r (slumbot-send-action token
                                           (encode-action action
                                                                game-state))]
          (recur (parse-incr-action r game-state)))))))

(defn iterate-games-slumbot
  "Plays multiple games against Slumbot and appends the results of those games to game-history.\\
   token: login token\\
   agent: agent to play against slumbot\\
   num-games: number of games to play\\
   game-history: current history of games played against slumbot\\
   -> [token game-history]"
  ([token agent num-games game-history]
   (loop [i num-games
          token token
          game-history game-history]
     (if (zero? i)
       [token game-history]
       (let [[token _agent game-history] (play-game-slumbot token
                                                            agent
                                                            game-history)]
         (recur (dec i)
                token
                game-history)))))
  ([token agent num-games] (iterate-games-slumbot token agent num-games []))
  ([agent num-games] (iterate-games-slumbot (slumbot-login)
                                            agent
                                            num-games)))

#_(iterate-games-slumbot utils/random-agent 5)

(defn slumbot-rollout
  "Pits agent against slumbot num-samples*num-iter times and writes their match history to the file txt.\\
   agent: agent to play against slumbot\\
   txt: text file to output history to\\
   num-samples: number of times to sample and print to file\\
   num-iter: number of games played per sample\\
   -> game-history"
  [agent txt num-samples num-iter]
  (loop [i 0
         token (slumbot-login)]
    (if (= i num-samples)
      nil
      (let [[token history] (iterate-games-slumbot token agent num-iter)]
        (spit txt
              (with-out-str (pprint/pprint history))
              :append true)
        (println i)
        (recur (inc i) token)))))




;;computing slumbot statistics
#_(let [l (read-string (slurp "slumbot-history-random.txt"))
        _ (println (count l))
        bot-hands (map #(:bot (into {} (:hands %))) l)
        bot-str (map #(let [{win :win total :total}
                            (utils/rollout (into #{} %))]
                        (try (float (/ win total))
                             (catch Exception e (println %))))
                     bot-hands)
        net-gain (map #(:bot (into {} (:net-gain %))) l)
        #_preflop-pot #_(map (fn [g]
                               (+ 1.5 (transduce (map utils/ssecond)
                                                 +
                                                 (first (:action-history g)))))
                             l)]
    (utils/corr bot-str net-gain))

;;calculating how representative my current slumbot history is
#_(let [l (group-by #(into #{} (first %))
                    (map #(vector
                           ((:r %) "hole_cards")
                           (/ ((:r %) "winnings") utils/slumbot-bb))
                         (read-string (slurp "slumbot-history-random.txt"))))]
    {:mean-games-per-hand (utils/mean (map #(count (second %)) l))})

;;Slumbot rollout
#_(loop [i 0]
    (if (= i 10)
      nil
      (do
        (time (slumbot-rollout random-agent
                               "slumbot-history-random.txt"
                               10
                               240))
        (utils/combine-vectors "slumbot-history-random.txt")
        (flush)
        (recur (inc i)))))


