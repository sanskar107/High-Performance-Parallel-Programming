#include<iostream>
#include<omp.h>
#include<math.h>

using namespace std;

int isprime(int n)
{

	for(int i = 2; i < sqrt(n) + 1; i++)
		if(n % i == 0)
			return 0;
	return 1;
}

int main()
{
	int count = 0;
	float* time_list = new float[12253]();
	double t;
	t = omp_get_wtime();
	bool is_pr;
	int i = 1;
	#pragma omp parallel for num_threads(4) private(count)
	// {
		for(i = 2; i <= 131072; i++)
		{
			is_pr = isprime(i);
			if(is_pr)
			{
				count++;
				//cout<<count<<"th prime number = "<<i<<" Time : "<<(omp_get_wtime() - t)<<endl;
			}
		}
	// }
	float tot_time1 = omp_get_wtime() - t;
	cout<<"Total Time Taken for 4 threads: "<<tot_time1<<endl;


	t = omp_get_wtime();
	count = 0;
	#pragma omp parallel for num_threads(16) private(count)
	// {
		for(i = 2; i <= 131072; i++)
		{
			is_pr = isprime(i);
			if(is_pr)
			{
				count++;
				//cout<<count<<"th prime number = "<<i<<" Time : "<<(omp_get_wtime() - t)<<endl;
			}
		}
	// }
	float tot_time2 = omp_get_wtime() - t;
	cout<<"Total Time Taken for 16 threads: "<<tot_time2<<endl;

	t = omp_get_wtime();
	count = 0;
	{
		for(i = 2; i <= 131072; i++)
		{
			is_pr = isprime(i);
			if(is_pr)
			{
				count++;
				//cout<<count<<"th prime number = "<<i<<" Time : "<<(omp_get_wtime() - t)<<endl;
			}
		}
	}
	float tot_time3 = omp_get_wtime() - t;
	cout<<"Total Time Taken without threading: "<<tot_time3<<endl;
	cout<<"\nImprovement for 4 threads : "<<(tot_time3 - tot_time1)/(tot_time3)*100<<endl;	
	cout<<"Improvement for 16 threads : "<<(tot_time3 - tot_time2)/(tot_time3)*100<<endl;	
	
	return 1;
}
