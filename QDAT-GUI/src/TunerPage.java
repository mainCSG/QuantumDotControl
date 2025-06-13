import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.ImageIcon;
import javax.swing.JButton;


public class TunerPage extends JFrame{
    

JButton BackButton = new JButton("Back");


TunerPage(){
    

    BackButton.setBounds(10,400,200,100);
    BackButton.setFocusable(false);
    BackButton.addActionListener(e -> {

        if(e.getSource()==BackButton) {
            dispose();
            new TitleScreen();
        }
    });

    ImageIcon logo = new ImageIcon(getClass().getResource("/logo.png"));
    JLabel TitleScreenLabel = new JLabel(logo);    
    this.setIconImage(new ImageIcon(getClass().getResource("/logo.png")).getImage());


    this.add(BackButton);
    
    this.add(TitleScreenLabel);

    this.setTitle("Quantum Dot Auto Tuner");

    this.setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
    
    this.setSize(500,500);
    
    this.setLayout(null);
    
    this.setVisible(true);


}

}
