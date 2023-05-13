import java.util.Scanner;

public class SJF {
	public static void swap(int [] a, int s1, int s2) {
		int temp = a[s1];
		a[s1] = a[s2];
		a[s2] = temp;
	}
	
	public static void main (String args[])
	{
		Scanner sc=new Scanner(System.in);
		System.out.println("Enter no of process : ");
		int n= sc.nextInt();
		int pid[] = new int[n];
		int at[] = new int[n];
		int bt[] = new int[n];
		int ct[] = new int[n];
		int tat[] = new int[n];
		int wt[] = new int[n];
		int f[] = new int[n];
	    int i,temp, st=0, tot=0;
	    float avg_wt=0, avg_tat=0;
	    
	    for (i=0;i<n;i++)
	    {
	    	pid[i]= i+1;
	    	System.out.println ("Enter arrival time of process " +(i+1)+ ":");
	    	at[i]= sc.nextInt();
	    	System.out.println("Enter burst time of process " +(i+1)+ ":");
	    	bt[i]= sc.nextInt();
	    	f[i]= 0;
	    }
	    
	    for(i = 0 ; i <n; i++)
		{
			for(int  j=0;  j < n-(i+1) ; j++)
			{
				if( at[j] > at[j+1] )
				{
					swap(at,j,j+1);
					swap(bt,j,j+1);
					swap(pid,j,j+1);
				}
			}
		}
	    
	    while(true)
	    {
	    	int min=99,c=n;
	    	if (tot==n)
	    		break;
	    	
	    	for (i=0;i<n;i++)
	    	{
	    		if ((at[i]<=st) && (f[i]==0) && (bt[i]<min))
	    		{	
	    			min=bt[i];
	    			c=i;
	    		}
	    	}
	    	
	    	if (c==n)
	    		st++;
	    	else
	    	{
	    		ct[c]= st+bt[c];
	    		st=st+bt[c];
	    		f[c]=1;
	    		tot++;
	    	}
	    }
	    
	    for(i=0;i<n;i++)
	    {
	    	tat[i] = ct[i] - at[i];
	    	wt[i] = tat[i] - bt[i];
	    	avg_wt+= wt[i];
	    	avg_tat+= tat[i];
	    }
	    
	    System.out.println("Pid\tAT\tBT\tCT\tTAT\tWT\n");
	    for(i=0;i<n;i++)
	    {
	    	System.out.println(pid[i] +"\t"+ at[i]+"\t"+bt[i] +"\t"+ ct[i] +"\t"+ tat[i] +"\t"+ wt[i]);
	    }
	    
	    System.out.println("\nAverage TAT(TURN AROUND TIME) is : "+(avg_tat/n));
	    System.out.println("\nAverage WT(WAITING TIME) is : "+ (avg_wt/n));
	    sc.close();
	}

}
