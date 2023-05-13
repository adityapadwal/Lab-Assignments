import java.util.Scanner;
import java.util.Queue;
import java.util.LinkedList;

public class RoundRobin {
	
	public static void swap(int [] a, int s1, int s2) {
		int temp = a[s1];
		a[s1] = a[s2];
		a[s2] = temp;
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner sc = new Scanner(System.in);
		System.out.println("enter no of process: ");
		int n = sc.nextInt();
		int pid[] = new int[n];
		int at[] = new int[n];   
		int bt[] = new int[n];   
		int ct[] = new int[n]; 
		int tat[] = new int[n];    
		int wt[] = new int[n];
		int vis[] = new int[n];
		int rem_bt[] = new int[n];
		int count=0,time=0;
		float avg_wt=0,avg_tat=0;
		 
		for(int i = 0; i < n; i++)
		{
		System.out.println("Enter PID " + (i+1) + " Arival Time: ");
		at[i] = sc.nextInt();
		System.out.println("Enter PID " + (i+1) + " Burst Time: ");
		bt[i] = sc.nextInt();
		pid[i] = i+1;
		}
		
		for(int i = 0 ; i <n; i++)
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
		
		for(int i=0;i<n;i++) {
			rem_bt[i] = bt[i];
		}
		
		System.out.println("Enter The Time Slice");
		int slice = sc.nextInt();
		
		
		Queue<Integer>q = new LinkedList<Integer>();
		
		int i = 0;
		while(i<n && at[i]<time) {
			q.add(i);
			i++;
		}
		
		if(q.size()==0) {
			time = at[0];
		}
		
		while(q.size() > 0) {
			int size = q.size();
			
			for(int j=0;j<size;j++) {
				int ind = q.remove();
				
//				int temp=0;
				while(i < n && at[i]<=time) {
					if(vis[i]==1) {
						i++;
						continue;
					}
					q.add(i);
				}
				
				if(vis[ind]==0 && rem_bt[ind]>=slice) {
					rem_bt[ind] -= slice;
					time += slice;
				}
				else {
					time += rem_bt[ind];
					rem_bt[ind]=0;
				}
				
				if(vis[ind]==0 && rem_bt[ind]==0) {
					count += 1;
					ct[ind] = time;
					vis[ind] = 1;
					count++;
					System.out.println((i+1) + " " + at[i] + " " + rem_bt[i] + " " + time + " " + count); 
				}
				else if(vis[ind]==1) {
					q.add(ind);
				}
				
			}
			
			if(q.size()==0 && count!=n) {
				time = at[i];
				q.add(i);
			}
			System.out.println((i+1) + " " + at[i] + " " + rem_bt[i] + " " + time + " " + count);
		}
		
		
		
//		while(true) {
//			if(count==n)break;
//			Boolean enter = true;
//			for(int i=0;i<n;i++) {
//				if(vis[i]==1)continue;
//				
//				if(at[i] <= time && rem_bt[i]>0) {
//					enter = false;
//					
//					if(rem_bt[i] >= slice) {
//						time += slice;
//						rem_bt[i] -= slice;
//					}                                                                                                                                                              
//					
//					if(rem_bt[i]==0) {
//						count++;
//						vis[i] = 1;
//					}
//					
//				}
//				else {
//					if(at[i]>time) {
//						time = at[i];
//						break;
//					}
//				}
//				
//				System.out.println((i+1) + " " + at[i] + " " + rem_bt[i] + " " + time + " " + count);
//			}
//		}
		
		for( i=0;i<n;i++)
	    {
	    	tat[i] = ct[i] - at[i];
	    	wt[i] = tat[i] - bt[i];
	    	avg_wt += wt[i];
	    	avg_tat += tat[i];
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
