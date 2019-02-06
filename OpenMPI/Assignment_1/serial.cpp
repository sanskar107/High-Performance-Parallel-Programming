#include<iostream>
#include<fstream>
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
	ofstream fout;
	fout.open("log.txt", ios::out);
	int count = 0;
	double t;
	t = omp_get_wtime();

	for(int i = 2; i <= 131072; i++)
	{
		bool is_pr = isprime(i);
		if(is_pr)
		{
			count++;
			double time = omp_get_wtime() - t;
			cout<<count<<"th prime number = "<<i<<" Time : "<<time<<endl;
			fout<<count<<"th prime number = "<<i<<" Time : "<<time<<endl;
		}
	}
	return 1;
}