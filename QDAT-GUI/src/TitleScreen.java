import java.awt.Color;
import java.awt.Font;

import javax.swing.ImageIcon;
import javax.swing.JLabel;
import javax.swing.JButton;
import javax.swing.JFrame;



public class TitleScreen extends JFrame{
    
    JButton startbutton;
    JButton helpbutton;
    JButton aboutbutton;

    TunerPage TunerPage;
    HelpPage HelpPage;
    AboutPage AboutPage;

    JLabel TitleScreenLabel;
    ImageIcon logo;

    TitleScreen(){

        // Button Def. & Dim.

        startbutton = new JButton();
        startbutton.setBounds(400,100,100,50);
        startbutton.setText("Start");
        startbutton.setFocusable(false);
        startbutton.setFont(new Font("Times New Roman",Font.BOLD,25));

        aboutbutton = new JButton();
        aboutbutton.setBounds(400,300,100,50);
        aboutbutton.setText("About");
        aboutbutton.setFocusable(false);
        aboutbutton.setFont(new Font("Times New Roman",Font.BOLD,12));

        helpbutton = new JButton();
        helpbutton.setBounds(400,500,100,50);
        helpbutton.setText("Help");
        helpbutton.setFocusable(false);
        helpbutton.setFont(new Font("Times New Roman",Font.BOLD,25));

        // Button Effects

        startbutton.addActionListener(e -> {

        if(e.getSource()==startbutton) { 
            dispose();
            new TunerPage();
        }
        });

        aboutbutton.addActionListener(e -> {
            
        if(e.getSource()==aboutbutton) {
            dispose();
            new AboutPage();
        }
        });

        helpbutton.addActionListener(e -> {
        
        if(e.getSource()==helpbutton) {
            dispose();
            new HelpPage();
        }
        });

        logo = new ImageIcon(getClass().getResource("/logo.png"));
        TitleScreenLabel = new JLabel(logo);
        this.setIconImage(new ImageIcon(getClass().getResource("/logo.png")).getImage());

        this.getContentPane().setBackground(new Color(255,255,255)); 

        this.setTitle("Quantum Dot Auto Tuner"); 
        
        this.setResizable(false);

        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); 
        
        this.setLayout(null);

        this.setSize(900,800); 
        
        this.setVisible(true); 

        this.add(startbutton);
        this.add(aboutbutton);
        this.add(helpbutton);
        this.add(TitleScreenLabel);

    }        

}
