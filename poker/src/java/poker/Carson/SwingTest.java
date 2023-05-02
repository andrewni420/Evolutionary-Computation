package poker.Carson;
// import javax.swing.*;
// import java.awt.*;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JFrame;
import javax.swing.ImageIcon;
import javax.swing.JSpinner;
import javax.swing.SpinnerModel;
import javax.swing.SpinnerNumberModel;
import javax.swing.JLabel;
import java.awt.Image;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class SwingTest extends JPanel implements ActionListener {

    // Declare buttons
    protected JButton b1, b2, b3, b4, b5, b6, b7, b8, b9, b10;

     // Create a button panel
     JPanel buttonPanel = new JPanel();

    //  Create a board panel
    JPanel boardPanel = new JPanel();

    // Constructor
    public SwingTest() {

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
            }
        };

        // Next hand button
        b1 = new JButton("Next Hand");
        b1.setActionCommand("next_hand");

        // Fold button
        b2 = new JButton("Fold");
        b2.setActionCommand("fold");

        // Check button
        b3 = new JButton("Check");
        b3.setActionCommand("check");

        // Call button
        b4 = new JButton("Call");
        b4.setActionCommand("call");

        // Min Bet button
        b5 = new JButton("Min Bet");
        b5.setActionCommand("min_bet");

        // Bet Half Pot button 
        b6 = new JButton("Bet Half Pot");
        b6.setActionCommand("bet_half_pot");

        // Bet Pot button
        b7 = new JButton("Bet Pot");
        b7.setActionCommand("bet_pot");

        // All In button
        b8 = new JButton("All In");
        b8.setActionCommand("all_in");

        // Peek button
        b9 = new JButton("Peek");
        b9.setActionCommand("peek");

        // Bet button
        b10 = new JButton("Bet");
        b10.setActionCommand("bet");

        // Spinner for bet amount
        SpinnerModel model = new SpinnerNumberModel(50, 1, 1000, 10);     
        JSpinner spinner = new JSpinner(model);

        //Listen for actions on buttons 1 and 3.
        b1.addActionListener(this);
        b2.addActionListener(this);
        b3.addActionListener(this);
        b4.addActionListener(this);
        b5.addActionListener(this);
        b6.addActionListener(this);
        b7.addActionListener(this);
        b8.addActionListener(this);
        b9.addActionListener(this);
        b10.addActionListener(this);

        // Tooltips (hover text)
        b1.setToolTipText("Click this button to get the next hand.");
        b2.setToolTipText("Click this button to fold the current hand.");
        b3.setToolTipText("Click this button to check.");
        b4.setToolTipText("Click this button to call.");
        b5.setToolTipText("Click this button to bet the minimum.");
        b6.setToolTipText("Click this button to bet half the pot.");
        b7.setToolTipText("Click this button to bet the pot.");
        b8.setToolTipText("Click this button to go all in.");
        b9.setToolTipText("Click this button to peek at your cards.");
        b10.setToolTipText("Click this button to bet a custom amount.");

        // Add buttons to the button panel
        buttonPanel.add(b1);
        buttonPanel.add(b2);
        buttonPanel.add(b3);
        buttonPanel.add(b4);
        buttonPanel.add(b5);
        buttonPanel.add(b6);
        buttonPanel.add(b7);
        buttonPanel.add(b8);
        buttonPanel.add(b9);
        buttonPanel.add(b10);
        buttonPanel.add(spinner);

        drawCard(boardPanel, "spades", "ace");

        // Add the panels to the main panel
        add(boardPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);
    }

    public void actionPerformed(ActionEvent e) {
        if ("next_hand".equals(e.getActionCommand())) {
            drawCard(boardPanel, "spades", "ace");
            boardPanel.repaint();
        } else {
            
        }
    }

    public JLabel getCard(String suit, String value) {

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
        
        // Return the label
        return label;
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

        // Set the bounds of the label
        label.setLocation(0, 100);
        
        // Add the label to the board
        board.add(label);
    }

    /** Returns an ImageIcon, or null if the path was invalid. */
    protected static ImageIcon createImageIcon(String path) {
        java.net.URL imgURL = SwingTest.class.getResource(path);
        if (imgURL != null) {
            return new ImageIcon(imgURL);
        } else {
            System.err.println("Couldn't find file: " + path);
            return null;
        }
    }

    /**
     * Create the GUI and show it.  For thread safety, 
    * this method should be invoked from the 
    * event-dispatching thread.
    */
    private static void createAndShowGUI() {

        //Create and set up the window.
        JFrame frame = new JFrame("SwingTest");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        //Create and set up the content pane.
        SwingTest newContentPane = new SwingTest();
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
