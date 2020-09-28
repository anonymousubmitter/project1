#include <unistd.h> 
#include <math.h> 
#include <thread>
#include <atomic>
#include<string>
#include <fstream>

class NN
{
public:
    NN(std::string path)
    {
        this->path = path;
        this->width=25;
        atomic_init(&ready_count, 0);
        this->no_threads = width;
        std::ifstream file(path);

        std::string line;

        std::getline(file, line);
        no_layers = std::stoi(line);
        

        sizes = new int[no_layers+1];
        Ws = new float**[no_layers];
        bs = new float*[no_layers];
        use_batch_norm = true;
        if (use_batch_norm)
        {
            mus = new float*[no_layers];
            sigmas = new float*[no_layers];
            gammas = new float*[no_layers];
            betas = new float*[no_layers];
        }

        for (int i = 0; i < no_layers; i++)
        {
            if (use_batch_norm)
                init_batch_norm(&file, mus[i], sigmas[i], gammas[i], betas[i]);

            std::getline(file, line);
            if (i == 0)
            {
                sizes[i] = std::stoi(line.substr(0, line.find(',')));
            }
            sizes[i+1] = std::stoi(line.substr(line.find(',')+1, line.length()));


            init_layer(&file, sizes[i], sizes[i+1], Ws[i], bs[i]);
        }

        mid_res = new float*[no_layers];
        for (int i= 0; i < no_layers; i++)
            mid_res[i] = new float[sizes[i+1]];
    }

    void call_thread(int i, float* x)
    {
        float* in = x;

        for (int j = 0; j < no_layers; j++)
        {
            if (i >= sizes[j+1])
                break;

            float out = 0;

            for (int z = 0; z < sizes[j]; z++)
                out += Ws[j][z][i]*in[z];
            out += bs[j][i];

            if (j == 0)
                out = sin(out);
            else if (j < no_layers - 1)
                out = single_swish(out);

            if (use_batch_norm && j < no_layers - 1)
                out = gammas[j+1][i]*((out-mus[j+1][i])/sqrt(sigmas[j+1][i])) + betas[j+1][i];

            mid_res[j][i] = out;
            if (j == no_layers -1)
                break;

            atomic_fetch_add(&ready_count, 1);
            while (atomic_load(&ready_count)<(j+1)*no_threads && atomic_load(&ready_count)>=j*no_threads);


            in = mid_res[j];
        }
    }


    void call(float* x, float* res)
    {

        ready_count = 0;
        float* in;
        if (use_batch_norm)
        {
            in = new float[sizes[0]];
            batch_norm(mus[0], sigmas[0], betas[0], gammas[0], x, sizes[0], in);
        }
        else
            in = x;

        std::thread** threads = new std::thread*[no_threads];
        for (int i =0; i < no_threads;i++)
            threads[i] = new std::thread(&NN::call_thread, this, i, in);

        for (int i =0; i < no_threads;i++)
        {
            threads[i]->join();
            delete threads[i];
        }
        delete[] threads;

        if (use_batch_norm)
            delete[] in;

        for (int i = 0; i < sizes[no_layers]; i++)
            res[i] = mid_res[no_layers-1][i];
    }

    void init_batch_norm(std::ifstream* file, float*& mu, float*& sigma, float*& gamma, float*& beta)
    {
        std::string line;
        std::getline(*file, line);
        int size = std::stoi(line);

        gamma = new float[size];
        std::getline(*file, line);
        for (int i =0; i < size; i++)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            gamma[i] = std::stof(vals);

            line = line.substr(next_del+1, line.length());
        }

        beta = new float[size];
        std::getline(*file, line);
        std::getline(*file, line);
        for (int i =0; i < size; i++)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            beta[i] = std::stof(vals);

            line = line.substr(next_del+1, line.length());
        }

        mu = new float[size];
        std::getline(*file, line);
        std::getline(*file, line);
        for (int i =0; i < size; i++)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            mu[i] = std::stof(vals);

            line = line.substr(next_del+1, line.length());
        }

        sigma = new float[size];
        std::getline(*file, line);
        std::getline(*file, line);
        for (int i =0; i < size; i++)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            sigma[i] = std::stof(vals);

            line = line.substr(next_del+1, line.length());
        }

    }
    void init_layer(std::ifstream* file, int in_size, int out_size, float**& W, float*& b)
    {
        std::string line;
        std::getline(*file, line);

        W = new float*[in_size];
        for (int i =0; i < in_size; i++)
        {
            W[i] = new float[out_size];
            for (int j = 0; j < out_size; j++)
            {
                int next_del = line.find(',');
                std::string vals = line.substr(0, next_del);
                W[i][j] =  std::stof(vals);

                line = line.substr(next_del+1, line.length());
            }
        }

        b = new float[out_size];
        std::getline(*file, line);
        std::getline(*file, line);
        for (int i =0; i < out_size; i++)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            b[i] = std::stof(vals);

            line = line.substr(next_del+1, line.length());
        }
    }

    float single_swish(float x)
    {
        return x*(1.0/(1.0+exp(-1*x)));
    }

    void swish(float* x, int dim, float* res) 
    {
		for (int i = 0; i < dim; i++) 
            res[i] = single_swish(x[i]);
    }

    void sine(float* x, int dim, float* res) 
    {
		for (int i = 0; i < dim; i++) 
            res[i] = sin(x[i]);
    }


    //W:dim1xdim2, x:dim1, res:dim2
	void multiply(float** w, float* x, int dim1, int dim2, float* res) 
	{ 
		int i, j; 
		for (i = 0; i < dim2; i++) 
		{ 
            res[i] = 0;
			for (j = 0; j < dim1; j++) 
			{ 
                res[i]+= w[j][i] * x[j]; 
			} 
		} 
	} 

	void batch_norm(float* mu, float* sigma, float* beta, float* gamma, float* x, int dim, float* res) 
	{ 
		int i; 
		for (i = 0; i < dim; i++) 
            res[i] = gamma[i]*((x[i]-mu[i])/sqrt(sigma[i])) + beta[i];
	} 

	void elem_add(float* b, float* x, int dim, float* res) 
	{ 
		int i; 
		for (i = 0; i < dim; i++) 
		{ 
            res[i] = b[i]+x[i];
		} 
	} 
    void print_nn()
    {
        for (int z =0; z < no_layers; z++)
        {
            if (use_batch_norm)
            {
                std::cout << "sigma" << z+1;
                for (int i = 0; i < sizes[z]; i++)
                    std::cout << sigmas[z][i] << ',';
                std::cout <<'\n';
                std::cout << "mu" << z+1;
                for (int i = 0; i < sizes[z]; i++)
                    std::cout << mus[z][i] << ',';
                std::cout <<'\n';
                std::cout << "beta" << z+1;
                for (int i = 0; i < sizes[z]; i++)
                    std::cout << betas[z][i] << ',';
                std::cout <<'\n';
                std::cout << "gamma" << z+1;
                for (int i = 0; i < sizes[z]; i++)
                    std::cout << gammas[z][i] << ',';
                std::cout <<'\n';
            }

            std::cout << "W" << z+1;
            for (int i = 0; i < sizes[z]; i++)
            {
                for (int j = 0; j < sizes[z+1]; j++)
                    std::cout << Ws[z][i][j] << ',';
            }
            std::cout <<'\n';

            std::cout << "b" << z+1;
            for (int i = 0; i < sizes[z+1]; i++)
            {
                std::cout << bs[z][i] << ',';
            }
            std::cout <<'\n';
        }

    }

    std::string get_path()
    {
        return path;
    }

private:
    std::string path;

    bool use_batch_norm;
    int no_layers;
    int* sizes;
    float*** Ws;
    float** bs;
    float** mus;
    float** sigmas;
    float** gammas;
    float** betas;

    int no_threads;

    float** mid_res;
    int width;
    std::atomic<int> ready_count;
};
