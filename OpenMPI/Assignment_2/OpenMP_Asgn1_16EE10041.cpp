#include<iostream>
#include<omp.h>
#include<time.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<vector>

using namespace std;

void create_database(char** data, int len, int str_len, int num_threads);
void search_string(char** data, char* str, int len, int str_len, int num_threads, bool display);
void redundancy(char** data, int len, int str_len, int num_threads, bool display);

int main()
{
	int len = 64000;
	int str_len = 15;
	char** data= new char*[len];
	for(int i = 0; i < len; i++)
	{
		data[i] = new char[str_len + 1];
		data[i][str_len] = '\0';
	}
	create_database(data, len, str_len, omp_get_max_threads());
	cout<<"Dataset Successfully Created\n";
	cout<<data[100]<<endl;

	cout<<"Enter searching pattern of length "<<str_len<<"\n";
	char str[str_len + 1];
	cin>>str;
	cout<<"Enter number of threads to be used\n";
	int num_threads = 4;
	cin>>num_threads;

	search_string(data, str, len, str_len, num_threads, 1);

	cout<<"\nCalculating Redundancy...\n";
	redundancy(data, len, str_len, num_threads, 1);

	cout<<"\nEvaluating average time with different number of threads\n";
	int threads[] = {1, 2, 4};
	for(int i = 0; i < 3; i++)
	{
		num_threads = threads[i];
		cout<<"\nFor Number of threads = "<<num_threads<<endl;
		
		int time = clock();
		create_database(data, len, str_len, num_threads);
		cout<<"Database created in "<<(clock() - time)*1.0/CLOCKS_PER_SEC<<endl;

		time = clock();
		search_string(data, str, len, str_len, num_threads, 1);
		cout<<"Search completed in "<<(clock() - time)*1.0/CLOCKS_PER_SEC<<endl;

		time = clock();
		redundancy(data, len, str_len, num_threads, 1);
		cout<<"Redundancy checked in "<<(clock() - time)*1.0/CLOCKS_PER_SEC<<endl;
		cout<<endl;
	}

	cout<<"\n\nCalculating Average time over 10 iterations\n";
	num_threads = omp_get_max_threads();
	int time = clock();
	for(int i = 0; i < 10; i++)
	{
		search_string(data, str, len, str_len, num_threads, 0);
	}
	cout<<"Average time for parallel search : "<<((clock() - time)*1.0/CLOCKS_PER_SEC)/10.0<<endl;
	
	time = clock();
	for(int i = 0; i < 10; i++)
	{
		redundancy(data, len, str_len, num_threads, 0);
	}
	cout<<"Average time for checking redundancy : "<<((clock() - time)*1.0/CLOCKS_PER_SEC)/10.0<<"\n\n";



}

void create_database(char** data, int len, int str_len, int num_threads)
{
	char chars[] = "ACTG";
	srand(time(NULL));
	#pragma omp parallel num_threads(num_threads)
	{
		#pragma omp for collapse(2)
		for(int i = 0; i < len; i++)
			for(int j = 0; j < str_len; j++)
				data[i][j] = chars[rand()%4];
	}
}

void search_string(char** data, char* str, int len, int str_len, int num_threads, bool display)
{
	if(display)
	{
		cout<<"Searching...\n";
		cout<<"Index : ";
	}
	vector<int> index;
	int count = 0;
	#pragma omp parallel num_threads(num_threads)
	{
		#pragma omp for
		for(int i = 0; i < len; i++)
		{
			if(strcmp(data[i], str) == 0)
			{
				#pragma omp critical
				{
					index.push_back(i);
					count++;
				}
			}
		}
	}
	if(display)
	{
		for(int i = 0; i < index.size(); i++)
			cout<<index[i]<<' ';
		cout<<endl;
	}

}

void redundancy(char** data, int len, int str_len, int num_threads, bool display)
{
	long max_val = 0;
	for(int i = 0; i < str_len; i++)
		max_val += (4*pow(5, i));
	max_val = 7000001;

	int* hash = new int[max_val]();
	int count = 0;
	long ind = 0;
	#pragma omp parallel num_threads(num_threads)
	{
		#pragma omp for private(ind)
		for(int i = 0; i < len; i++)
		{
			ind = 0;
			for(int j = 0; j < str_len; j++)
			{
				if(data[i][j] == 'A')
					ind += 1*pow(5, j);
				else if(data[i][j] == 'C')
					ind += 2*pow(5, j);
				else if(data[i][j] == 'T')
					ind += 3*pow(5, j);
				else if(data[i][j] == 'G')
					ind += 4*pow(5, j);
			}
			ind = ind%7000001;
			if(hash[ind] == 1)
				continue;
			#pragma omp critical
			{
				hash[ind] = 1;
				count++;
			}
		}
	}
	if(display)
	{
		cout<<"Max length = "<<len<<endl;
		cout<<"Unique = "<<count<<endl;
		cout<<"Redundancy = "<<(len - count)*100.0/len<<" %"<<endl;
	}
}