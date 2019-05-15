import java.io.*;
import javax.swing.*;
import java.applet.*;
import java.awt.*;
import java.util.*;
import java.awt.event.*;
/*<applet code="p" width=400 height=400> </applet>*/
public class Classifier extends JApplet{
	JTextField input;
	JButton predict;
	JLabel l1;
	static int labels[]=new int[3000];
	static String arr[];
	static int data[][]=new int[3000][];
	static double p1vect[];
	static double p0vect[];
	static int sum(int x[]){
		int sum=0;
		for(int i:x)
			sum+=i;
		return sum;
	}
	static void CreateList()  throws IOException{
		BufferedReader br=new BufferedReader(new FileReader("dataset.txt"));
		String x,t;
		String temp1[],temp2[];
		int i,y=0;
		Set<String> set = new HashSet<String>();
		while((x=br.readLine())!=null){
			temp1=x.split(" ");
			labels[y++]=temp1[temp1.length-1].charAt(temp1[temp1.length-1].length()-1)-'0';
			temp2=new String[temp1.length];
			for(i=0;i<temp1.length;i++){
				if(temp1[i]==null)
					break;
				else if(temp1[i].indexOf(".")!=-1 || temp1[i].indexOf("!")!=-1 || temp1[i].indexOf("  ")!=-1){
					temp2[i]=temp1[i];
					break;
				}
				else
					temp2[i]=temp1[i];
			}
			for(i=0;i<temp2.length;i++)
				set.add(temp2[i]);
		}
		br.close();
		arr=set.toArray(new String[set.size()]);	
		br=new BufferedReader(new FileReader("dataset.txt"));
		y=0;
		while((x=br.readLine())!=null){
			t="";
			temp1=x.split(" ");
			temp2=new String[temp1.length];
			for(i=0;i<temp1.length;i++){
				if(temp1[i]==null)
					break;
				temp2[i]=temp1[i];
				if(temp2[i].indexOf(".")!=-1 || temp2[i].indexOf("!")!=-1)
					break;
			}
			for(i=0;i<temp2.length-2;i++)
				t+=" "+temp2[i];
			data[y++]=setOfWords2Vec(t);
		}
		br.close();
	}
	static int[] setOfWords2Vec(String x){
		int i,n=arr.length;
		int v[]=new int[n];
		String temp[];
		temp=x.split(" ");
		for(String str: temp){
			for(i=0;i<arr.length;i++){
				if(str.equals(arr[i])){
					v[i]+=1;
					break;
				}
			}
		}	
		return v;
	}
	static void trainNB(){
		int x,numtrain=data.length;
		int numwords=arr.length;
		int p0denom=2,p1denom=2;
		int p1num[]=new int[numwords];
		int p0num[]=new int[numwords];
		for(x=0;x<numwords;x++){
			p1num[x]=1;
			p0num[x]=1;
		}
		for(x=0;x<numtrain;x++){
			if(labels[x]==1){
				for(int j=0;j<numwords;j++)
					p1num[j]+=data[x][j];
				p1denom+=sum(data[x]);
			}
			else{
				for(int j=0;j<numwords;j++)
					p0num[j]+=data[x][j];
				p0denom+=sum(data[x]);
			}
		}
		p1vect=new double[numwords];
		p0vect=new double[numwords];
		for(x=0;x<numwords;x++){
			p1vect[x]=Math.log(p1num[x])-Math.log(p1denom);
			p0vect[x]=Math.log(p0num[x])-Math.log(p0denom);	
		}
	}
	static boolean classify(int x[]){
		int p1=0,p0=0,i;
		for(i=0;i<arr.length;i++){
			p0+=x[i]*p0vect[i];
			p1+=x[i]*p1vect[i];
		}
		if(p1>p0)	
			return false;
		return true;		
	}
	public void init(){
		try{
			SwingUtilities.invokeAndWait(new Runnable(){
				public void run(){
					driver();		
				}
			});
		}	
		catch(Exception e){}
	}
	private void driver(){
		input=new JTextField(20);
		predict=new JButton("Predict");
		l1=new JLabel("Prediction:		");
		setLayout(new FlowLayout());
		add(input);
		add(predict);
		add(l1);
		predict.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				try{
					initiate();
				}
				catch(Exception a){}	
			}
		});
	}
	void initiate() throws IOException{
		Scanner sc=new Scanner(System.in);
				String sentence=input.getText();
				int i=0;
				int vect[];
				CreateList();
				trainNB();
				vect=setOfWords2Vec(sentence);
				if(classify(vect))
					l1.setText("Prediction: Statement is negative");	
				else
					l1.setText("Prediction: Statement is positive");
	}
}
