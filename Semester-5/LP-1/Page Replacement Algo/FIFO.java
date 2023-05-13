import java.util.Scanner;

public class FIFO {

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
		System.out.println("enter no of frames: ");
		int f = sc.nextInt();
		
		int proc[] = new int [n];
		int frames[] = new int [f];
		int freq[] = new int [f];
		int faults=0,hits=0;

		System.out.println("Enter PIDs :- ");
		
		for(int i = 0; i < n; i++)
		{
		proc[i] = sc.nextInt();
		}
		
		for(int i=0;i<n;i++) {
			
			int den = Integer.MIN_VALUE;
			int ind = 0;
			Boolean flag = false;
			
			for(int j=0;j<f;j++) {
				if(frames[j] == 0) {
					frames[j] = proc[i];
					faults++;
					System.out.print("Faults ");
					flag = false;
					
					for(int k=0;k<f;k++) {
						if(frames[k]==0)break;
						freq[k] += 1;
					}
					
					break;
				}
				if(proc[i] == frames[j]) {
					hits++;
					System.out.print("hits ");
					flag = false;
					
					for(int k=0;k<f;k++) {
						if(frames[k]==0)break;
						freq[k] += 1;
					}
					
					break;
				}
				if(freq[j] > den) {
					den = freq[j];
					flag=true;
					ind = j;
				}
			}
			
			if(flag==true) {
				frames[ind] = proc[i];
				freq[ind] = 1;
				System.out.print("Faults ");
				faults++;
			}
			
			for(int k=0;k<f;k++) {
				System.out.print(frames[k]+" ");
			}
			for(int k=0;k<f;k++) {
				System.out.print(freq[k]+" ");
			}
			System.out.println();
		}
		
		System.out.println("No of Faults = "+faults);
		System.out.println("No of Hits = "+hits);
		
	}

}
