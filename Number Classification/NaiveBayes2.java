import java.io.*;
// classify function is returning proper values but main is getting only 0 and the output to the file is werid
class NaiveBayes2{
	static int labels[]= new int[41900];		//labels of the image
	static int data[][]=new int[41900][784];	//The pixelwise information of the image
	static double pnumerator[][]=new double[10][784];
	static int pdenominator[]=new int[10];
	//static double pvect[][]=new double[10][784];
	static <T extends Number> double sum(T x[]){
		double sum=0;
		for(T i:x)
			sum+=i.doubleValue();
		return sum;
	}
	static int greater(int x){
		if(x>0)
			return 1;
		return 0;
	}
	static void LoadLabels() throws IOException{
		int i=0;
		String y;
		BufferedReader br=new BufferedReader(new FileReader("/home/jenit1/Desktop/train.csv"));
		while((y=br.readLine())!=null){
			if(i==41900)
				break;
			labels[i++]=y.charAt(0)-'0';
		}
		br.close();	
	}
	static void Data() throws IOException{
		FileInputStream fin=new FileInputStream("/home/jenit1/Desktop/train.csv");	//load the file
		int i,num,x,col=0,count=0;		
		for(i=0;i<41900;i++){
			col=0;
			while((x=fin.read())!=-1){
				if(x=='\n'||x==','||x=='\r')
					continue;
				if(col==0){
					col++;
					continue;
				}
				num=x-'0';
				while(((x=fin.read())!=-1) && (x!=','))
					num=num*10+(x-'0');
				num=greater(num);
				data[i][col-1]=num;
				col++;
				if(col==785)
					break;
			}		
		}
		fin.close();
	}
	static void LoadFunctions() throws IOException{
		LoadLabels();
		Data();
	}
	static void trainNB(Integer trainmat[][],int traincat[]){
		int i,j,n=trainmat.length;
		for(i=0;i<10;i++){
			//pdenominator[i]=2;
			for(j=0;j<784;j++)
				pnumerator[i][j]=1;
		}
		pdenominator[0]=586331;
		pdenominator[1]=300443;
		pdenominator[2]=535437;
		pdenominator[3]=530694;
		pdenominator[4]=436246;
		pdenominator[5]=436200;
		pdenominator[6]=489390;
		pdenominator[7]=433875;
		pdenominator[8]=530705;
		pdenominator[9]=451623;
		for(i=0;i<n;i++){
			//pdenominator[traincat[i]]+=sum(trainmat[i]);
			for(j=0;j<784;j++)
				pnumerator[traincat[i]][j]+=trainmat[i][j];
		}
		
		for(i=0;i<10;i++){
			for(j=0;j<784;j++){
				pnumerator[i][j]/=pdenominator[i];
				pnumerator[i][j]=Math.log(pnumerator[i][j]);
			}
		}	
	}
	static int classify(int mat[],double probability[]){
		double prob[]=new double[10];
		Double temp[][]=new Double[10][784];
		int i,j,val=5;
		double max=-100000;
		for(i=0;i<10;i++){
			for(j=0;j<784;j++){
				//System.out.print(pnumerator[i][j]);
				temp[i][j]=mat[j]*pnumerator[i][j];
			}			
		}
		for(i=0;i<10;i++)
			prob[i]=sum(temp[i])+probability[i];
		for(i=0;i<10;i++){
			if(prob[i]>max)
				max=prob[i];
		}
		for(i=0;i<10;i++){
			if(prob[i]==max){
				val=i;
				break;
			}
		}
		return val;
	}
	public static void main(String args[]) throws IOException{
		double start=System.currentTimeMillis();
		LoadFunctions();  			
		double probability[]=new double[10];
		int i,j,error=0,x;
		for(i=0;i<10;i++)
			probability[i]=0;
		for(i=0;i<41900;i++)
			probability[labels[i]]+=1;
		for(i=0;i<10;i++)
			probability[i]/=41900;
		int sublabels[]=new int[31500];
		Integer subdata[][]=new Integer[31500][784];
		for(i=0;i<31500;i++){
			sublabels[i]=labels[i];
			for(j=0;j<784;j++)
				subdata[i][j]=data[i][j];
		}
		trainNB(subdata,sublabels);
		/*for(i=0;i<10;i++){
			for(j=0;j<784;j++)
				System.out.print(pvect[i][j]);
		}*/
		for(i=31500;i<41900;i++){
			x=classify(data[i],probability);
			if(x!=labels[i])
				error+=1;
		}
		System.out.println("Error: "+error);
		double end=System.currentTimeMillis();
		System.out.println("Total time: "+(end-start)/1000 +"s");
	}
}
