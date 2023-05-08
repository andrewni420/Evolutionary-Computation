package poker.Mikhail;

import java.util.*;
import clojure.lang.*;
import clojure.java.api.Clojure;
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
    public static APersistentMap maping;

    // List of the bets that players will make
    static List<Float> betValues;

    // Integer of the game number
    static Long gameNum;

    // float of the current bet on the table
    static double currentBet;

    // List of the Cards that are in players hands
    static List<List<String>> playerHands = new ArrayList<>();

    // String name of the Betting Round
    static String bettingRound;

    // float of the minimum bet
    static double minimumBet;

    // List of the Players Money
    static List<Double> playersMoney;

    // Is the game over
    static boolean isGameOver;

    // List of all the visible hands that we have
    static List<String> visibleCards;

    // Minimal Raise
    static double minimumRaise;

    // Value of the pot
    static double pot;

    public static void main(APersistentMap givenMap)
    {
        // We get the maping to be the given map
        maping = givenMap;

        // We get the Bet Values
        betValues = (List<Float>)maping.get(Clojure.read(":bet-values"));
        System.out.println(betValues);

        // We get the game number
        gameNum = (Long)maping.get(Clojure.read(":game-num"));
        System.out.println(gameNum);

        // We get the current bet
        currentBet = (double)maping.get(Clojure.read(":current-bet"));
        System.out.println(currentBet);

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

        System.out.println(playerHands);

        // We now get the betting round
        bettingRound = (String)maping.get(Clojure.read(":betting-round"));
        System.out.println(bettingRound);

        // Get the minimum bet
        minimumBet = (Double)maping.get(Clojure.read(":min-bet"));
        System.out.println(minimumBet);

        // We wnat to get players money
        List<APersistentMap> players = (List<APersistentMap>)maping.get(Clojure.read(":players"));

        APersistentMap playerMap1 = players.get(0);
        APersistentMap playerMap2 = players.get(1);

        double player1Money = (double)playerMap1.get(Clojure.read(":money"));
        double player2Money = (double)playerMap2.get(Clojure.read(":money"));

        playersMoney = new ArrayList<>();

        playersMoney.add(player1Money);
        playersMoney.add(player2Money);

        System.out.println(playersMoney);

        // Get the is Game Over Variable
        isGameOver = (boolean)maping.get(Clojure.read(":game-over"));

        System.out.println(isGameOver);

        // Get the visible hands
        visibleCards = new ArrayList<>();
        List<List<Object>> tempVisible = (List<List<Object>>)maping.get(Clojure.read(":visible"));
        for(List<Object> card : tempVisible)
        {
            visibleCards.add(card.get(0).toString() + "_" + card.get(1).toString());
        }

        System.out.println(visibleCards);

        // Get the Minimum Raise
        minimumRaise = (double)maping.get(Clojure.read(":min-raise"));

        System.out.println(minimumRaise);

        // Get the pot
        pot = (double)maping.get(Clojure.read(":pot"));

        System.out.println(pot);
    }
}
