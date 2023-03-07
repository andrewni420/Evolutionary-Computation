(ns poker.gp 
  (:require [propeller.push.state :as state]
            [poker.headsup :as headsup]))

(instructions/def-instruction
  :nort
  ^{:stacks #{:nort}}
  (fn [state]
    (instructions/make-instruction state #(and %1 %2) [:nort :nort] :nort)))

(instructions/get-stack-instructions #{:nort})
state/empty-state
(interpreter/interpret-program (list :nort) (assoc state/empty-state :nort [true false]) 100)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                Functions List             ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Constants
(def empty-state 
  (assoc state/empty-state
         :keyword (list)
         :game-state (list)
         :game-history (list)
         :action (list)
         ))

(def game-state-kws
  (keys (headsup/init-game [])))

(def game-history-kws
  [])

()



;;; Game State Functions

(instructions/def-instruction???
  :get-key-state
  ^{:stacks #{:game-state :string}}
  (fn [state]
    (instructions/make-instruction state #(%1 %2) [:keyword :game-state] :string)))


;;; Game History Functions
;;; Filter by conditions
;;; Aggregate probabilities