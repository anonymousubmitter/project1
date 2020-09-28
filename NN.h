#include <thread>
#include <cstring>
#include <math.h> 
#include<string>
#include <fstream>

void init_batch_norm(std::ifstream* file, std::ifstream* file_d, double*& mu, double*& sigma, double*& gamma, double*& beta)
{
    std::string line;
    std::getline(*file, line);
    int size = std::stoi(line);

    gamma = new double[size];
    //std::getline(*file_d, line);
    for (int i =0; i < size; i++)
    {
        file_d->read((char*)&gamma[i], sizeof(double));
        //std::string vals = line.substr(0, sizeof(double));
        //std::memcpy(&gamma[i], vals.c_str(), sizeof(double));
        //line = line.substr(sizeof(double)+1, line.length());

        //int next_del = line.find(',');
        //std::string vals = line.substr(0, next_del);
        //gamma[i] = std::stod(vals);

        //line = line.substr(next_del+1, line.length());
    }

    beta = new double[size];
    std::getline(*file, line);
    //std::getline(*file, line);
    for (int i =0; i < size; i++)
    {
        file_d->read((char*)&beta[i], sizeof(double));
        //std::string vals = line.substr(0, sizeof(double));
        //std::memcpy(&beta[i], vals.c_str(), sizeof(double));
        //line = line.substr(sizeof(double)+1, line.length());
        /*
        int next_del = line.find(',');
        std::string vals = line.substr(0, next_del);
        beta[i] = std::stod(vals);

        line = line.substr(next_del+1, line.length());
        */
    }

    mu = new double[size];
    std::getline(*file, line);
    //std::getline(*file, line);
    for (int i =0; i < size; i++)
    {
        file_d->read((char*)&mu[i], sizeof(double));
        //std::string vals = line.substr(0, sizeof(double));
        //std::memcpy(&mu[i], vals.c_str(), sizeof(double));
        //line = line.substr(sizeof(double)+1, line.length());
        /*
        int next_del = line.find(',');
        std::string vals = line.substr(0, next_del);
        mu[i] = std::stod(vals);

        line = line.substr(next_del+1, line.length());
        */
    }

    sigma = new double[size];
    std::getline(*file, line);
    //std::getline(*file, line);
    for (int i =0; i < size; i++)
    {
        file_d->read((char*)&sigma[i], sizeof(double));
        //std::string vals = line.substr(0, sizeof(double));
        //std::memcpy(&sigma[i], vals.c_str(), sizeof(double));
        //line = line.substr(sizeof(double)+1, line.length());
        /*
        int next_del = line.find(',');
        std::string vals = line.substr(0, next_del);
        sigma[i] = std::stod(vals);

        line = line.substr(next_del+1, line.length());
        */
    }

}

void swish(double* x, int dim, double* res) 
{
    for (int i = 0; i < dim; i++) 
        res[i] = x[i]*(1.0/(1.0+exp(-1*x[i])));
}

void sine(double* x, int dim, double* res) 
{
    for (int i = 0; i < dim; i++) 
        res[i] = sin(x[i]);
}


//W:dim1xdim2, x:dim1, res:dim2
void multiply(double** w, double* x, int dim1, int dim2, double* res) 
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

void batch_norm(double* mu, double* sigma, double* beta, double* gamma, double* x, int dim, double* res) 
{ 
    int i; 
    for (i = 0; i < dim; i++) 
        res[i] = gamma[i]*((x[i]-mu[i])/sqrt(sigma[i])) + beta[i];
} 

void elem_add(double* b, double* x, int dim, double* res) 
{ 
    int i; 
    for (i = 0; i < dim; i++) 
    { 
        res[i] = b[i]+x[i];
    } 
} 

void init_layer(std::ifstream* file, std::ifstream* file_d, int in_size, int out_size, double**& W, double*& b)
{
    std::string line;
    //std::getline(*file, line);

    W = new double*[in_size];
    for (int i =0; i < in_size; i++)
    {
        W[i] = new double[out_size];
        for (int j = 0; j < out_size; j++)
        {
            file_d->read((char*)&W[i][j], sizeof(double));
            //int next_del = line.find(',');
            //std::string vals = line.substr(0, sizeof(double));
            //std::memcpy(&W[i][j], vals.c_str(), sizeof(double));

            //line = line.substr(sizeof(double)+1, line.length());
        }
    }

    b = new double[out_size];
    std::getline(*file, line);
    //std::getline(*file, line);
    for (int i =0; i < out_size; i++)
    {
        file_d->read((char*)&b[i], sizeof(double));
        //std::string vals = line.substr(0, sizeof(double));
        //std::memcpy(&b[i], vals.c_str(), sizeof(double));

        //line = line.substr(sizeof(double)+1, line.length());
        //int next_del = line.find(',');
        //std::string vals = line.substr(0, next_del);
        //b[i] = std::stod(vals);

        //line = line.substr(next_del+1, line.length());
    }
}

class NN
{
public:
    NN(std::string path)
    {
        this->path = path;
        std::cout << path << std::endl;
        std::ifstream file(path);
        std::ifstream file_d(path+"d");

        std::string line;

        std::getline(file, line);
        no_layers = std::stoi(line);
        

        sizes = new int[no_layers+1];
        Ws = new double**[no_layers];
        bs = new double*[no_layers];
        use_batch_norm = true;
        if (use_batch_norm)
        {
            mus = new double*[no_layers];
            sigmas = new double*[no_layers];
            gammas = new double*[no_layers];
            betas = new double*[no_layers];
        }

        for (int i = 0; i < no_layers; i++)
        {
            if (use_batch_norm)
                init_batch_norm(&file, &file_d, mus[i], sigmas[i], gammas[i], betas[i]);

            std::getline(file, line);
            if (i == 0)
            {
                sizes[i] = std::stoi(line.substr(0, line.find(',')));
            }
            sizes[i+1] = std::stoi(line.substr(line.find(',')+1, line.length()));


            init_layer(&file, &file_d, sizes[i], sizes[i+1], Ws[i], bs[i]);
        }
    }

    void call(double* x, double* res)
    {
        double* in = x;
        for (int i = 0; i < no_layers; i++)
        {
            double* res1 = new double[sizes[i+1]];
            double* res2 = new double[sizes[i+1]];
            double* res3 = new double[sizes[i+1]];

            double* res0;
            if (use_batch_norm)
            {
                res0 = new double[sizes[i]];
                batch_norm(mus[i], sigmas[i], betas[i], gammas[i], in, sizes[i], res0);
                if (i!=0)
                    delete[] in;
            }
            else
                res0 = in;

            multiply(Ws[i], res0, sizes[i], sizes[i+1], res1);
            if (use_batch_norm || (!use_batch_norm && i != 0))
                delete[] res0;

            elem_add(bs[i], res1, sizes[i+1], res2);

            delete[] res1;
            if (i == 0)
            {
                sine(res2, sizes[i+1], res3);
                delete[] res2;
            }
            else if (i < no_layers - 1)
            {
                swish(res2, sizes[i+1], res3);
                delete[] res2;
            }
            else
            {
                delete[] res3;
                res3 = res2;
            }

            in = res3;
        }

        for (int i = 0; i < sizes[no_layers]; i++)
            res[i] = in[i];
        delete[] in;
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
    int get_input_dim()
    {
        return sizes[0];
    }

private:
    std::string path;

    bool use_batch_norm;
    int no_layers;
    int* sizes;
    double*** Ws;
    double** bs;
    double** mus;
    double** sigmas;
    double** gammas;
    double** betas;
};


class NNDegree
{
public:
    NNDegree(std::string path, int degree, int out_dim)
    {
        this->degree = degree;
        this->out_dim = out_dim;
        if (out_dim != 1)
        {
            nns = new NN*[out_dim];
            for (int i = 0; i < out_dim; i++)
            {
                nns[i] = new NN(path+std::to_string(i)+"0"+".m");
            }
        }
        else
        {
            nns = new NN*[degree];
            for (int i = 0; i < degree; i++)
            {
                std::cout << path+std::to_string(i)+"0"+".m" << std::endl;
                nns[i] = new NN(path+"0"+std::to_string(i)+".m");
            }
        }

        input_dim = nns[0]->get_input_dim();
        in = new double[input_dim];

        if (out_dim != 1)
        {
            std::thread** threads = new std::thread*[out_dim];
            res = new double[out_dim];
            done = new bool[out_dim];
            for (int i =0; i < out_dim;i++)
            {
                res[i] = 0;
                done[i] = true;
                threads[i] = new std::thread(&NNDegree::start_thread, this, i);
            }
        }
        else if (degree != 1) 
        {

            std::ifstream file(path+"comb.m");
            std::ifstream file_d(path+"comb.md");

            std::string line;
            std::getline(file, line);
            in_dim = std::stoi(line.substr(0, line.find(',')));
            out_dim = std::stoi(line.substr(line.find(',')+1, line.length()));

            init_layer(&file, &file_d, in_dim, out_dim, W, b);

            std::thread** threads = new std::thread*[degree];
            res = new double[degree];
            done = new bool[degree];
            for (int i =0; i < degree;i++)
            {
                done[i] = true;
                threads[i] = new std::thread(&NNDegree::start_thread, this, i);
            }
        }
    }

    void start_thread(int i)
    {
        while(true)
        {
            while(done[i]==true);
            res[i] = 0;
            nns[i]->call(in, &res[i]);
            done[i] = true;
        }
    } 

    void call(double* x, double* res_)
    {
        if (degree == 1 && out_dim == 1)
        {
            nns[0]->call(x, res_);
            return;
        }

        if (degree != 1)
        {
            for (int i = 0; i < input_dim; i++)
                in[i] = x[i];

            for (int i = 0; i < degree;i++)
                done[i] = false;

            for (int i =0; i < degree;i++)
                while(done[i]==false);


            double* res1 = new double[out_dim];
            multiply(W, this->res, in_dim, out_dim, res1);

            elem_add(b, res1, out_dim, res_);
            delete[] res1;
        }
        else if (out_dim != 1)
        {
            for (int i = 0; i < input_dim; i++)
                in[i] = x[i];

            for (int i = 0; i < out_dim;i++)
                done[i] = false;

            for (int i =0; i < out_dim;i++)
            {
                while(done[i]==false);
                res_[i] = this->res[i];
            }
        }

    }

    void print_nn()
    {
        for (int i = 0; i < degree; i++)
            nns[i]->print_nn();

        if (degree == 1)
            return;

        std::cout << "W ";
        for (int i = 0; i < in_dim; i++)
        {
            for (int j = 0; j < out_dim; j++)
                std::cout << W[i][j] << ',';
        }
        std::cout <<'\n';

        std::cout << "b ";
        for (int i = 0; i < out_dim; i++)
        {
            std::cout << b[i] << ',';
        }
        std::cout <<'\n';
    }

private:
    int degree;
    NN** nns;

    double** W;
    double* b;
    int in_dim;
    int out_dim;

    int input_dim;

    double* res;
    bool* done;
    double* in;
};
