#include <iostream>
#include <fstream>
#include <iomanip>

#include <sstream>
#include <cmath>
#include <chrono> 
#include <string>
#include "NN.h"
//#include "NN_Parallel.h"

struct Node
{
    double** centers;
    int dim;
    double* split_points;
    Node** children;
    NNDegree* nn;
};

int fanout = 3;
int dim = 2;
int out_dim = 2;
int depth = 3;
int degree = 3;
bool is_kd_tree = true;

Node* get_new_node()
{
    Node* node = new Node;
    node->nn = NULL;
    node->children = NULL;
    node->centers = NULL;
    return node;
}

void build_kd_tree(Node* root_node, std::ifstream* file, int curr_depth, std::string curr_path, int split_dim)
{
    if (fanout == 1 || curr_depth == depth)
    {
        root_node->nn = new NNDegree(curr_path, degree, out_dim);
        //root_node->nn = new NNDegree*[out_dim];
        //for (int i = 0; i < out_dim; i++)
        //    root_node->nn[i] = new NNDegree(curr_path+std::to_string(i), degree);

        return;
    }

    std::string line;
    std::getline(*file, line);
    line = line.substr(line.find(':')+1, line.length());
    
    root_node->split_points = new double[fanout-1];
    root_node->dim = split_dim;
    root_node->children = new Node*[fanout];
    for (int i = 0; i < fanout; i++)
    {
        if (i != fanout-1)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            std::cout << vals << std::endl;;
            root_node->split_points[i] =  std::stod(vals);

            line = line.substr(next_del+1, line.length());
        }

        root_node->children[i] = get_new_node();
        std::string path = curr_path;
        path.append(std::to_string(i+1));
        build_kd_tree(root_node->children[i], file, curr_depth+1, path, (split_dim+1)%dim);
    }

}

void build_tree(Node* root_node, std::ifstream* file, int curr_depth, std::string curr_path)
{
    if (fanout == 1 || curr_depth == depth)
    {
        root_node->nn = new NNDegree(curr_path, degree, out_dim);
        //for (int i = 0; i < out_dim; i++)
        //root_node->nn = new NNDegree(curr_path, degree, out_dim);

        return;
    }

    std::string line;
    std::getline(*file, line);
    line = line.substr(line.find(':')+1, line.length());
    
    root_node->centers = new double*[fanout];
    root_node->children = new Node*[fanout];
    for (int i = 0; i < fanout; i++)
    {
        root_node->centers[i] = new double[dim];
        for (int j = 0; j < dim; j++)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            root_node->centers[i][j] =  std::stod(vals);

            line = line.substr(next_del+1, line.length());
        }
        root_node->children[i] = get_new_node();
        std::string path = curr_path;
        path.append(std::to_string(i+1));
        build_tree(root_node->children[i], file, curr_depth+1, path);
    }

}

double dist(double* x, double* y, int input_dim)
{
    double sum = 0;
    for (int i = 0; i < input_dim; i++)
        sum += (x[i]-y[i])*(x[i]-y[i]);

    return sqrt(sum);
}

void call_kd_tree(Node* root, double* x, double* res)
{
    if (root->children == NULL)
    {
        root->nn->call(x, res);
        return;
    }

    for (int i = 0; i < fanout-1; i++)
    {
        if (x[root->dim] < root->split_points[i])
            return call_kd_tree(root->children[i], x, res);
    }
    call_kd_tree(root->children[fanout-1], x, res);

}

void call_tree(Node* root, double* x, double* res)
{
    if (root->children == NULL)
    {
        for (int i = 0; i < out_dim; i++)
            root->nn->call(x, res);
        return;
    }

    int min_indx = 0;
    double min = dist(x, root->centers[0], dim);
    for (int i = 1; i < fanout; i++)
    {
        double curr_dist = dist(x, root->centers[i], dim);
        if (curr_dist < min)
        {
            min = curr_dist;
            min_indx = i;
        }
    }

    call_tree(root->children[min_indx], x, res);

}

void print_tree(Node* root)
{
    if (root->children == NULL)
    {
        //std::cout << "LEAF" << root->nn->get_path() << std::endl;
        //for (int i = 0; i < out_dim; i++)
        //    root->nn->print_nn();
        return;
    }
    for (int i = 0; i < fanout-1; i++)
    {
        //for (int j = 0; j < dim; j++)
        std::cout << root->split_points[i] << '\t';
        std::cout << ";;";
    }
    std::cout << root->dim << std::endl;

    for (int i = 0; i < fanout; i++)
    {
        print_tree(root->children[i]);
    }
}

int main(int argc, char** argv)
{
    std::cout << std::setprecision(10);

    fanout = atoi(argv[1]);
    dim = atoi(argv[2]);
    out_dim = atoi(argv[3]);
    depth = atoi(argv[4]);
        
    std::string query_file = argv[5];
    std::string model_file = argv[6];
    int base = atoi(argv[7]);
    degree = atoi(argv[8]);

    int test_size = 100000;
    double** x = new double*[test_size];
    double** y = new double*[test_size];
    std::ifstream file_queries(query_file+"_queries.txt");
    std::ifstream file_res(query_file+"_res.txt");

    std::ofstream file_output(query_file+"_out.txt");

    for (int i = 0; i < test_size; i++)
    {
        x[i] = new double[dim];
        std::string line;
        if (!std::getline(file_queries, line))
        {
            test_size = i;
            break;
        }
        for (int j = 0; j < dim; j++)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            x[i][j] =  std::stod(vals);

            line = line.substr(next_del+1, line.length());
        }
        //x[i][0] = std::stod(line.substr(0, line.find(',')).c_str());
        //x[i][1] = std::stod(line.substr(line.find(',')+1, line.length()).c_str());

        y[i] = new double[out_dim];
        std::getline(file_res, line);
        for (int j = 0; j < out_dim; j++)
        {
            int next_del = line.find(',');
            std::string vals = line.substr(0, next_del);
            y[i][j] =  std::stod(vals);

            line = line.substr(next_del+1, line.length());
        }
        //y[i][0] = std::stod(line.substr(0, line.find(',')).c_str());
        //y[i][1] = std::stod(line.substr(line.find(',')+1, line.length()).c_str());
    }
    


    std::ifstream infile(model_file+"_tree.m");
    if (depth!=0)
    {

        std::string line;
        std::cout << line << std::endl;
        if (std::getline(infile, line))
            is_kd_tree = line[0] == 'K';
        std::cout << "kd "<< is_kd_tree << std::endl;
    }
    else
        is_kd_tree = true;

    Node* root_node = get_new_node();
    if (is_kd_tree)
        build_kd_tree(root_node, &infile, 0, model_file, 0);
    else
        build_tree(root_node, &infile, 0, model_file);

    print_tree(root_node);

    auto start = std::chrono::high_resolution_clock::now(); 

    double threshold = 0.1;
    double count = 0;
    double count_tenth = 0;
    double count_hundredth = 0;
    double mse = 0;
    double rel_acc = 0;
    double max_rel_acc = 0;
    bool output_result = true;
    double total_dist = 0;
    for (int i = 0; i < test_size; i++)
    {
        double* res = new double[out_dim];
        if (is_kd_tree)
            call_kd_tree(root_node, x[i], res);
        else
            call_tree(root_node, x[i], res);

        double acc;
        if (out_dim > 1 && out_dim == dim)
        {
            double true_dist = dist(y[i], x[i], dim);
            double pred_dist = dist(res, x[i], dim);
            acc = std::abs(true_dist-pred_dist)/true_dist;
            total_dist += true_dist;
            mse += std::abs(true_dist-pred_dist);
            //std::cout << true_dist <<  ","<<pred_dist << std::endl;
            //std::cout << "dim1 "<< x[i][0] << ","<< res[0] << "," << y[i][0] << "," << acc << std::endl;
            //std::cout << "dim2 "<< x[i][1] << ","<< res[1] << "," << y[i][1] << "," << acc << std::endl;
            //std::cout << "HERE" << std::endl;


        }
        else
        {
            acc = dist(res, y[i], out_dim)/std::abs(y[i][0]);
            std::cout << acc << std::endl;
            total_dist += std::abs(y[i][0]);
            mse += dist(res, y[i], out_dim);
        }
        if (i < 5)
        //if (acc > 1)
        {
            for (int z = 0; z < out_dim; z++)
                std::cout << "dim " << z << " "<< x[i][z] << ","<< res[z] << "," << y[i][z] << "," << acc << std::endl;
        }
        if (output_result)
        {
            for (int z = 0; z < out_dim; z++)
                file_output <<  res[z] << ",";
            file_output <<  std::endl;
        }
        //std::cout << "dim1 "<< x[i][0] << ","<< x[i][1] << ","<< res[0] << "," << y[i][0] << "," << acc << std::endl;
        //{
         //   std::cout << "why)___________________" << std::endl;
         //   exit(0);
        //}
        //std::cout << "dim2 "<< x[i][1] << ","<< res[1] << "," << y[i][1] << "," << acc << std::endl;
        //double acc = std::abs(res[0]-y[i][0])/y[i][0];
        rel_acc += acc;
        if (acc > max_rel_acc)
            max_rel_acc = acc;
        count += (int)(dist(res, y[i], out_dim) <= threshold);
        count_tenth += (int)(dist(res, y[i], out_dim) <= (threshold/10.0));
        count_hundredth += (int)(dist(res, y[i], out_dim) <= (threshold/100.0));
    }
    
    mse = mse/test_size;
    count = count/test_size;
    count_tenth = count_tenth/test_size;
    count_hundredth = count_hundredth/test_size;


    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
    std::ofstream out_f;
    if (base == -1)
        out_f.open ("res.txt");
    else
        out_f.open ("res_val_"+std::to_string(base)+".txt");
    out_f << "time:" << ((double)duration.count())/((double)test_size) << std::endl; 
    out_f << "avg_rel_acc:" << rel_acc/test_size << std::endl; 
    out_f << "size:" << test_size << std::endl; 
    out_f << "max_rel_acc:" << max_rel_acc << std::endl; 
    out_f << "mse:" << mse << std::endl; 
    out_f << "normalized_mse:" << mse/(total_dist/test_size ) << std::endl; 
    out_f << "avg_dist:" << total_dist/test_size << std::endl; 
    out_f << "count:" << count << std::endl; 
    out_f << "count_tenth:" << count_tenth << std::endl; 
    out_f << "count_hundredth:" << count_hundredth << std::endl; 
    out_f.close();

    return 0;
}
