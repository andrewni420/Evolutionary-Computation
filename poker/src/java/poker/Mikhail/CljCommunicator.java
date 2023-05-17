package poker.Mikhail;

import java.util.*;
import clojure.java.api.Clojure;
import clojure.lang.IFn;
import clojure.lang.APersistentMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

// This is the exmaple of the function that will take the clojure map and 
// then translate it into Java variables that we can use for the UI

public class CljCommunicator 
{
    // This is the static maping that we will use the get the state of the game
    public APersistentMap maping;

    // List of the bets that players will make
    public List<Float> betValues;

    // Integer of the game number
    public Long gameNum;

    // float of the current bet on the table
    public double currentBet;

    // List of the Cards that are in players hands
    public List<List<String>> playerHands = new ArrayList<>();

    // String name of the Betting Round
    public String bettingRound;

    // float of the minimum bet
    public double minimumBet;
    
    // List of the Players Money
    public List<Double> playersMoney;

    // Is the game over
    public boolean isGameOver;

    // List of all the visible hands that we have
    public List<String> visibleCards;

    // Minimal Raise
    public double minimumRaise;

    // Value of the pot
    public double pot;

    // g and gameState
    public APersistentMap gameState;
    public APersistentMap g;

    // netGain
    public float netGain;

    // Step game function
    public IFn stepGame;

    public CljCommunicator(){
        gameState = null;
        g = null;
        netGain = 0;
        stepGame = Clojure.var("poker.headsup", "apply-step-game");
        gameNum = (long) 0;
    }

    public void updateMap(){
        // We get the maping to be the given map
        maping = (APersistentMap)g.get(Clojure.read(":game-state"));

        System.out.println("maping keys: " + maping.keySet());

        // We get the Bet Values
        betValues = (List<Float>)maping.get(Clojure.read(":bet-values"));
        System.out.println("betValues:" + betValues);

        // We get the game number
        gameNum = (Long)maping.get(Clojure.read(":game-num"));
        System.out.println("gameNum: " + gameNum);

        // We get the current bet
        currentBet = (double)maping.get(Clojure.read(":current-bet"));
        System.out.println("currentBet: " + currentBet);

        // We get the hands and then turn them into strings
        List<List<List<Object>>> placeHolder = (List<List<List<Object>>>)maping.get(Clojure.read(":hands"));
        playerHands.clear();

        for(List<List<Object>> players : placeHolder)
        {
            List<String> playersHandTemporary = new ArrayList<>();

            // We loop through all the players
            for(List<Object> cards : players)
            {
                // We loop through all the cards
                String card = cards.get(0).toString() + "_"+ cards.get(1).toString();
                playersHandTemporary.add(card);
            }
            List<String> tempCopy;
            playerHands.add(playersHandTemporary);
        }

        System.out.println("playerHands: " + playerHands);

        // We now get the betting round
        bettingRound = (String)maping.get(Clojure.read(":betting-round"));
        System.out.println("bettingRound: " + bettingRound);

        // Get the minimum bet
        minimumBet = (Double)maping.get(Clojure.read(":min-bet"));
        System.out.println("minimumBet: " + minimumBet);

        // We wnat to get players money
        List<APersistentMap> players = (List<APersistentMap>)maping.get(Clojure.read(":players"));

        APersistentMap playerMap1 = players.get(0);
        APersistentMap playerMap2 = players.get(1);

        double player1Money = (double)playerMap1.get(Clojure.read(":money"));
        double player2Money = (double)playerMap2.get(Clojure.read(":money"));

        playersMoney = new ArrayList<>();

        playersMoney.add(player1Money);
        playersMoney.add(player2Money);

        System.out.println("playersMoney: " + playersMoney);

        // Get the is Game Over Variable
        isGameOver = (boolean)maping.get(Clojure.read(":game-over"));

        System.out.println("isGameOver: " + isGameOver);

        // Get the visible hands
        visibleCards = new ArrayList<>();
        List<List<Object>> tempVisible = (List<List<Object>>)maping.get(Clojure.read(":visible"));
        for(List<Object> card : tempVisible)
        {
            visibleCards.add(card.get(0).toString() + "_" + card.get(1).toString());
        }

        System.out.println("visibleCards: " + visibleCards);

        // Get the Minimum Raise
        minimumRaise = (double)maping.get(Clojure.read(":min-raise"));

        System.out.println("minimumRaise: " + minimumRaise);

        // Get the pot
        pot = (double)maping.get(Clojure.read(":pot"));

        System.out.println("pot: " + pot);
    }

    public void update(float actionAmount, String actionType){
        try {
            System.out.println("stepgame is about to run");
            g = (APersistentMap) stepGame.invoke(g, Clojure.read(":action"), Clojure.read("[\"" + actionType + "\"" + Float.toString(actionAmount) + "]"));
            System.out.println("stepgame ran");
            gameState = (APersistentMap) g.get(Clojure.read(":game-state"));
            System.out.println("gamestate ran");
            System.out.println("g: " + g.get(Clojure.read(":net-gain")));
            // netGain = ((Number) g.get(Clojure.read(":net-gain"))).floatValue();
            // System.out.println("netgain: " + netGain);
            //update interface
        } catch (Exception e) {
            //not a legal action. Throws an AssertionError
            System.out.println("Not a legal action (Exception: " + e + ")");
            //display warning on interface
        }
    }

    public boolean testLegality(float actionAmount, String actionType){
        try {
            IFn isLegal = Clojure.var("poker.utils", "is-legal?");
            return (Boolean) isLegal.invoke(Clojure.read("[\"" + actionType + "\"" + Float.toString(actionAmount) + "]"), maping);
            // // System.out.println("stepgame is about to run");
            // APersistentMap g_tester = (APersistentMap) stepGame.invoke(g, Clojure.read(":action"), Clojure.read("[\"" + actionType + "\"" + Float.toString(actionAmount) + "]"));
            // APersistentMap gameState_tester = (APersistentMap) g_tester.get(Clojure.read(":game-state"));
            // // System.out.println("gamestate ran");
            // // System.out.println("g: " + g.get(Clojure.read(":net-gain")));
            // float netGain_tester = ((Number) g_tester.get(Clojure.read(":net-gain"))).floatValue();
            // System.out.println("netgain: " + netGain);
            // return true;
            //update interface
        } catch (java.lang.AssertionError e) {
            //not a legal action. Throws an AssertionError
            System.out.println("Not a legal action (Exception: " + e + ")");
            //display warning on interface
            return false;
        } catch (java.lang.NullPointerException ex) {
            System.out.println("Not a legal action (Exception: " + ex + ")");
            return false;
        }
    }
    public void init(){
        update((float) 0, "Fold");
    }
}
