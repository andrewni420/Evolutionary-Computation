package poker.Carson;

// import javax.swing.*;
// import java.awt.*;
import javax.swing.*;
import java.awt.Image;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class GUI extends JPanel implements ActionListener {

    // Declare buttons
    protected JButton nextHandButton, foldButton, checkButton, callButton, minBetButton, betHalfPotButton, betPotButton, allInButton, peekButton, betButton;

    // Create a button panel
    JPanel buttonPanel = new JPanel();

    //  Create a board panel
    JPanel boardPanel = new JPanel();

    // Create a panel for the ai's cards
    JPanel aiCardsPanel = new JPanel();

    // Create a panel for the player's cards
    JPanel playerCardsPanel = new JPanel();

    // Create a panel for the community cards
    JPanel communityCardsPanel = new JPanel();

    // Create a panel for messages
    JPanel messagePanel = new JPanel();

    // Variables for pot, current bet, player hand, ai hand, player stack, ai stack, community cards, round, and if game is active
    int pot = 100;
    int current_bet = 50;
    String[] player_hand = new String[2];
    int player_stack = 100;
    String[] ai_hand = new String[2];
    int ai_stack = 200;
    String[] community_cards = new String[5];
    String round = "Pre-Flop";
    boolean game_active = false;

    JSpinner spinner;

    /**
     * TO DO: 
     * Use legal move logic to determine which buttons to display -> button.setEnabled(false)
     * Hook up java to clojure to get game state
     */


    // Constructor
    public GUI() {
        player_hand[0] = "ace spades";
        player_hand[1] = "9 diamonds";

        ai_hand[0] = "ace clubs";
        ai_hand[1] = "ace diamonds";

        community_cards[0] = "10 diamonds";
        community_cards[1] = "9 hearts";
        community_cards[2] = "2 clubs";
        community_cards[3] = "jack diamonds";
        community_cards[4] = "6 spades";


        // Set layout manager
        setLayout(new BorderLayout());

        // Create a panel for the game board
        boardPanel = new JPanel() {
            public Dimension getPreferredSize() {
                return new Dimension(800, 450);
            }

            public void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.setColor(Color.GREEN);
                g.fillRoundRect(100, 50, 800, 350, 350, 350);

                // Draw pot
                g.setColor(Color.RED);
                g.fillOval(680,255, 20, 20);
                g.fillOval(700,255, 20, 20);
                g.fillOval(720,255, 20, 20);
                g.setColor(Color.BLACK);
                g.drawString(Integer.toString(pot), 700, 255);

                // Draw player stack
                g.setColor(Color.RED);
                g.fillOval(580,360, 20, 20);
                g.fillOval(600,360, 20, 20);
                g.fillOval(590,345, 20, 20);
                g.setColor(Color.BLACK);
                g.drawString(Integer.toString(player_stack), 590, 395);

                // Draw AI stack
                g.setColor(Color.RED);
                g.fillOval(580,65, 20, 20);
                g.fillOval(600,65, 20, 20);
                g.fillOval(590,80, 20, 20);
                g.setColor(Color.BLACK);
                g.drawString(Integer.toString(ai_stack), 590, 60);

                // Draw current bet
                g.setColor(Color.RED);
                g.fillOval(580,65, 20, 20);
                g.fillOval(600,65, 20, 20);
                g.fillOval(590,80, 20, 20);
                g.setColor(Color.BLACK);
                g.drawString(Integer.toString(ai_stack), 590, 60);
            }
        };


        boardPanel.setLayout(new BoxLayout(boardPanel, BoxLayout.Y_AXIS));

        // Next hand button
        nextHandButton = new JButton("Next Hand");
        nextHandButton.setActionCommand("next_hand");

        // Fold button
        foldButton = new JButton("Fold");
        foldButton.setActionCommand("fold");

        // Check button
        checkButton = new JButton("Check");
        checkButton.setActionCommand("check");

        // Call button
        callButton = new JButton("Call");
        callButton.setActionCommand("call");

        // Min Bet button
        minBetButton = new JButton("Min Bet");
        minBetButton.setActionCommand("min_bet");

        // Bet Half Pot button 
        betHalfPotButton = new JButton("Bet Half Pot");
        betHalfPotButton.setActionCommand("bet_half_pot");

        // Bet Pot button
        betPotButton = new JButton("Bet Pot");
        betPotButton.setActionCommand("bet_pot");

        // All In button
        allInButton = new JButton("All In");
        allInButton.setActionCommand("all_in");

        // Peek button
        peekButton = new JButton("Peek");
        peekButton.setActionCommand("peek");

        // Bet button
        betButton = new JButton("Bet");
        betButton.setActionCommand("bet");

        // Spinner for bet amount
        SpinnerModel model = new SpinnerNumberModel(50, 10, player_stack, 10);     
        spinner = new JSpinner(model);

        //Listen for actions on buttons 1 and 3.
        nextHandButton.addActionListener(this);
        foldButton.addActionListener(this);
        checkButton.addActionListener(this);
        callButton.addActionListener(this);
        minBetButton.addActionListener(this);
        betHalfPotButton.addActionListener(this);
        betPotButton.addActionListener(this);
        allInButton.addActionListener(this);
        peekButton.addActionListener(this);
        betButton.addActionListener(this);

        // Tooltips (hover text)
        nextHandButton.setToolTipText("Click this button to get the next hand.");
        foldButton.setToolTipText("Click this button to fold the current hand.");
        checkButton.setToolTipText("Click this button to check.");
        callButton.setToolTipText("Click this button to call.");
        minBetButton.setToolTipText("Click this button to bet the minimum.");
        betHalfPotButton.setToolTipText("Click this button to bet half the pot.");
        betPotButton.setToolTipText("Click this button to bet the pot.");
        allInButton.setToolTipText("Click this button to go all in.");
        peekButton.setToolTipText("Click this button to peek at your cards.");
        betButton.setToolTipText("Click this button to bet a custom amount.");

        // Set buttons to false except for next hand
        nextHandButton.setEnabled(true);
        foldButton.setEnabled(false);
        checkButton.setEnabled(false);
        callButton.setEnabled(false);
        minBetButton.setEnabled(false);
        betHalfPotButton.setEnabled(false);
        betPotButton.setEnabled(false);
        allInButton.setEnabled(false);
        peekButton.setEnabled(false);
        betButton.setEnabled(false);

        // Add buttons to the button panel
        buttonPanel.add(nextHandButton);
        buttonPanel.add(foldButton);
        buttonPanel.add(checkButton);
        buttonPanel.add(callButton);
        buttonPanel.add(minBetButton);
        buttonPanel.add(betHalfPotButton);
        buttonPanel.add(betPotButton);
        buttonPanel.add(allInButton);
        buttonPanel.add(peekButton);
        buttonPanel.add(betButton);
        buttonPanel.add(spinner);

        // Set the layout manager for the card panels
        playerCardsPanel.setLayout(new BoxLayout(playerCardsPanel, BoxLayout.X_AXIS));
        communityCardsPanel.setLayout(new BoxLayout(communityCardsPanel, BoxLayout.X_AXIS));
        aiCardsPanel.setLayout(new BoxLayout(aiCardsPanel, BoxLayout.X_AXIS));
        // messagePanel.setLayout(new BoxLayout(messagePanel, BoxLayout.X_AXIS));

        // Add the panels to the main panel
        add(boardPanel, BorderLayout.NORTH);
        add(messagePanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);
    }

    // Function to draw cards based on round name
    public void drawCommunityCards(String round_name) {
        if (round_name == "Pre-Flop") {
            communityCardsPanel.removeAll();
            communityCardsPanel.add(Box.createRigidArea(new Dimension(0, 90))); // Add vertical space
            refreshElements();
        } else if (round_name == "Flop") {
            communityCardsPanel.removeAll();
            // Draw 3 community cards
            for (int i=0;i<3;i++){
                drawCard(communityCardsPanel, community_cards[i].substring(community_cards[i].indexOf(' ') + 1), community_cards[i].substring(0, community_cards[i].indexOf(' ')));
            }
            refreshElements();
        } else if (round_name == "Turn") {
            communityCardsPanel.removeAll();
            // Draw 4 community cards
            for (int i=0;i<4;i++){
                drawCard(communityCardsPanel, community_cards[i].substring(community_cards[i].indexOf(' ') + 1), community_cards[i].substring(0, community_cards[i].indexOf(' ')));
            }
            refreshElements();
        } else if (round_name == "River"){
            communityCardsPanel.removeAll();
            // Draw 5 community cards
            for (int i=0;i<5;i++){
                drawCard(communityCardsPanel, community_cards[i].substring(community_cards[i].indexOf(' ') + 1), community_cards[i].substring(0, community_cards[i].indexOf(' ')));
            }
            refreshElements();
        }
    }

    public void actionPerformed(ActionEvent e) {
        if ("next_hand".equals(e.getActionCommand())) {
            // Set game to active
            game_active = true;
            
            // Remove previous elements
            aiCardsPanel.removeAll();
            communityCardsPanel.removeAll();
            playerCardsPanel.removeAll();
            messagePanel.removeAll();
            boardPanel.removeAll();
            
            // Draw ai cards
            boardPanel.add(Box.createVerticalStrut(50)); // Add vertical space
            drawBackCard(aiCardsPanel);
            drawBackCard(aiCardsPanel);
            boardPanel.add(aiCardsPanel);

            boardPanel.add(Box.createVerticalStrut(40)); // Add vertical space

            drawCommunityCards("River");

            boardPanel.add(communityCardsPanel);

            boardPanel.add(Box.createVerticalStrut(40)); // Add vertical space

            // Draw player cards
            for (int i=0;i<player_hand.length;i++){
                drawCard(playerCardsPanel, player_hand[i].substring(player_hand[i].indexOf(' ') + 1), player_hand[i].substring(0, player_hand[i].indexOf(' ')));
            }
            boardPanel.add(playerCardsPanel);

            // Grey out the Next Hand, check, and peek buttons
            nextHandButton.setEnabled(false);
            foldButton.setEnabled(true);
            checkButton.setEnabled(false);
            callButton.setEnabled(true);
            minBetButton.setEnabled(true);
            betHalfPotButton.setEnabled(true);
            betPotButton.setEnabled(true);
            allInButton.setEnabled(true);
            peekButton.setEnabled(false);
            betButton.setEnabled(true);

            // Refresh the board
            refreshElements();
        } else if ("fold".equals(e.getActionCommand())){
            // Set game to inactive
            game_active = false;
            
            // Remove previous elements
            aiCardsPanel.removeAll();
            communityCardsPanel.removeAll();
            playerCardsPanel.removeAll();
            boardPanel.removeAll();

            // Print fold message
            displayMessage("You folded. AI wins the pot of $" + pot);

            // Enable the Next Hand button and disable the rest
            nextHandButton.setEnabled(true);
            foldButton.setEnabled(false);
            checkButton.setEnabled(false);
            callButton.setEnabled(false);
            minBetButton.setEnabled(false);
            betHalfPotButton.setEnabled(false);
            betPotButton.setEnabled(false);
            allInButton.setEnabled(false);
            peekButton.setEnabled(false);
            betButton.setEnabled(false);

            // Refresh the board
            refreshElements();

            // Check if game over
            checkIfGameOver();
        } else if ("check".equals(e.getActionCommand())){
            // Check logic here
        } else if ("call".equals(e.getActionCommand())){
            // Call logic here
            if (player_stack >= current_bet){
                player_stack = player_stack - current_bet;
                pot = pot + current_bet;
                refreshElements();
            } else {
                displayMessage("You don't have enough money to call. Going all in instead.");
                pot = pot + player_stack;
                player_stack = 0;
                refreshElements();
            }
        } else if ("min_bet".equals(e.getActionCommand())){
            // Min bet logic here
            if (player_stack >= current_bet){
                player_stack = player_stack - current_bet;
                pot = pot + current_bet;
                refreshElements();
            } else {
                displayMessage("You don't have enough money to bet the minimum. Going all in instead.");
                pot = pot + player_stack;
                player_stack = 0;
                refreshElements();
            }
        } else if ("bet_half_pot".equals(e.getActionCommand())){
            // Bet half pot logic here
            if (player_stack >= (pot / 2)){
                player_stack = player_stack - (pot / 2);
                current_bet = pot / 2;
                pot = pot + current_bet;
                refreshElements();
            } else {
                displayMessage("You don't have enough money to bet half the pot. Going all in instead.");
                pot = pot + player_stack;
                player_stack = 0;
                refreshElements();
            }
        } else if ("bet_pot".equals(e.getActionCommand())){
            // bet pot logic here
            if (player_stack >= pot){
                player_stack = player_stack - pot;
                pot = pot + pot;
                refreshElements();
            } else {
                displayMessage("You don't have enough money to bet half the pot. Going all in instead.");
                pot = pot + player_stack;
                player_stack = 0;
                refreshElements();
            }
        } else if ("all_in".equals(e.getActionCommand())){
            // all in logic here
            current_bet = player_stack;
            player_stack = 0;
            pot = pot + current_bet;
            refreshElements();
        } else if ("peek".equals(e.getActionCommand())){
            // peek logic here
            // Peek the AI cards for 3 seconds
            aiCardsPanel.removeAll();
            drawCard(aiCardsPanel, ai_hand[0].substring(ai_hand[0].indexOf(' ') + 1), ai_hand[0].substring(0, ai_hand[0].indexOf(' ')));
            drawCard(aiCardsPanel, ai_hand[1].substring(ai_hand[1].indexOf(' ') + 1), ai_hand[1].substring(0, ai_hand[1].indexOf(' ')));
            aiCardsPanel.revalidate();
            aiCardsPanel.repaint();
            refreshElements();
            displayMessage("Peeking AI cards for 3 seconds...");
            Timer timer = new Timer(3000, new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    aiCardsPanel.removeAll();
                    drawBackCard(aiCardsPanel);
                    drawBackCard(aiCardsPanel);
                    aiCardsPanel.revalidate();
                    aiCardsPanel.repaint();
                    refreshElements();
                }
            });
            timer.setRepeats(false);
            timer.start();
        } else if ("bet".equals(e.getActionCommand())){
            // Get bet amount
            int bet_amount = (int) spinner.getValue();

            // Check if valid bet amount
            if ((bet_amount <= player_stack) && (bet_amount >= 1)){
                // Set current bet
                current_bet = (int) spinner.getValue();

                // Remove bet amount from player stack
                player_stack = player_stack - current_bet;

                // Add player bet to main pot
                pot = pot + current_bet;
                
                // Refresh the board
                refreshElements();

            } else {
                // Display error message
                displayMessage("Invalid bet amount. Please try again.");
            }
        }
    }

    public void refreshElements(){
        // Check for button logic in here (Mikhail's legal actions)

        if (player_stack > 0){
            // Adjust the bet spinner
            try {
                SpinnerModel model = new SpinnerNumberModel(player_stack / 2, 1, player_stack, 1);
                spinner.setModel(model);
            } catch (IllegalArgumentException e){
                displayMessage("Invalid bet amount. Please try again.");
            }
        } else {
            try {
                SpinnerModel model = new SpinnerNumberModel(player_stack / 2, 0, player_stack, 1);
                spinner.setModel(model);
                spinner.setEnabled(false);
                betButton.setEnabled(false);
            } catch (IllegalArgumentException e){
                displayMessage("Invalid bet amount. Please try again.");
            }
        }

        // Revalidate and repaint
        boardPanel.revalidate();
        boardPanel.repaint();
    }

    // Function to check if Game is over
    public void checkIfGameOver(){
        if (player_stack == 0 && !game_active){
            // Display game over message
            displayGameOverMessage("Game Over! You lost all your money. Please restart the game.");
            // Disable all buttons
            nextHandButton.setEnabled(false);
            checkButton.setEnabled(false);
            callButton.setEnabled(false);
            minBetButton.setEnabled(false);
            betHalfPotButton.setEnabled(false);
            betPotButton.setEnabled(false);
            allInButton.setEnabled(false);
            peekButton.setEnabled(false);
            betButton.setEnabled(false);
            spinner.setEnabled(false);
        } else if (ai_stack == 0 && !game_active){
            // Display game over message
            displayGameOverMessage("Game Over! You won all the AI's money. Please restart the game.");
            nextHandButton.setEnabled(false);
            checkButton.setEnabled(false);
            callButton.setEnabled(false);
            minBetButton.setEnabled(false);
            betHalfPotButton.setEnabled(false);
            betPotButton.setEnabled(false);
            allInButton.setEnabled(false);
            peekButton.setEnabled(false);
            betButton.setEnabled(false);
            spinner.setEnabled(false);
        }
    }

    public void displayMessage(String msg){
        // Print message
        messagePanel.removeAll();
        JLabel message = new JLabel(msg);
        messagePanel.add(message);
        messagePanel.revalidate();
        messagePanel.repaint();

        // Remove message after 3 seconds
        Timer timer = new Timer(3000, new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                messagePanel.removeAll();
                messagePanel.revalidate();
                messagePanel.repaint();
            }
        });
        timer.setRepeats(false);
        timer.start();
    }

    public void displayGameOverMessage(String msg){
        // Print message
        messagePanel.removeAll();
        JLabel message = new JLabel(msg);
        messagePanel.add(message);
        messagePanel.revalidate();
        messagePanel.repaint();
    }

    public void drawCard(JPanel board, String suit, String value) {

        // Get path of card image
        String path = "PNG-cards-1.3/" + value + "_of_" + suit + ".png";

        // Get image icon from path
        ImageIcon ogImageIcon = createImageIcon(path);

        // Create a label to display the image
        JLabel label = new JLabel();

        // Play around with size
        int width = Math.floorDiv(ogImageIcon.getIconWidth(), 8);
        int height = Math.floorDiv(ogImageIcon.getIconHeight(), 8);

        // Scale the image
        Image scaledImage = ogImageIcon.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT);

        // Create a new image icon
        ImageIcon imageIcon = new ImageIcon(scaledImage);

        // Set the image icon
        label.setIcon(imageIcon);

        // Add the label to the board
        board.add(label);
    }

    /** Returns an ImageIcon, or null if the path was invalid. */
    protected static ImageIcon createImageIcon(String path) {
        java.net.URL imgURL = GUI.class.getResource(path);
        if (imgURL != null) {
            return new ImageIcon(imgURL);
        } else {
            System.err.println("Couldn't find file: " + path);
            return null;
        }
    }

    public void drawBackCard(JPanel board) {

        // Get path of card image
        String path = "PNG-cards-1.3/back.png";

        // Get image icon from path
        ImageIcon ogImageIcon = createImageIcon(path);

        // Create a label to display the image
        JLabel label = new JLabel();

        // Play around with size
        int width = Math.floorDiv(ogImageIcon.getIconWidth(), 8);
        int height = Math.floorDiv(ogImageIcon.getIconHeight(), 8);
        
        // Scale the image
        Image scaledImage = ogImageIcon.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT);

        // Create a new image icon
        ImageIcon imageIcon = new ImageIcon(scaledImage);

        // Set the image icon
        label.setIcon(imageIcon);

        // Set the bounds of the label
        label.setLocation(0, 100);
        
        // Add the label to the board
        board.add(label);
    }

    /**
     * Create the GUI and show it.  For thread safety, 
    * this method should be invoked from the 
    * event-dispatching thread.
    */
    private static void createAndShowGUI() {

        //Create and set up the window.
        JFrame frame = new JFrame("Poker");
        frame.setPreferredSize(new Dimension(1000, 550));
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        //Create and set up the content pane.
        GUI newContentPane = new GUI();
        newContentPane.setOpaque(true);
        frame.setContentPane(newContentPane);

        frame.setResizable(false);

        //Display the window.
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        //creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI(); 
            }
        });
    }
}