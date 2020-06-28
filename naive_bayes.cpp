//
// Created by Ahmed Aleshinloye on 2020-06-24.
//

#include "naive_bayes.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <numeric>
#include <time.h>
#include <math.h>

using namespace std;

int Iris_setosa = 0;
int Iris_versicolor = 1;
int Iris_virginica = 2;
int Iris_unknown = 3;

template <typename T>
T calculate_mean(vector <T> data){
    T mean = accumulate(begin(data), end(data), 0.0) / data.size();
    return mean;
}

vector<vector<float> > split_class_by_label(vector<vector<float> > dataset, float class_label){
    vector<vector<float> > output;
    vector <float> sepal_length;
    vector <float> sepal_width;
    vector <float> petal_length;
    vector <float> petal_width;

    for (int i = 0; i < dataset[4].size(); i++){
        if (dataset[4][i] == class_label){
            sepal_length.push_back(dataset[0][i]);
            sepal_width.push_back(dataset[1][i]);
            petal_length.push_back(dataset[2][i]);
            petal_width.push_back(dataset[3][i]);
        }
    }

    output.push_back(sepal_length);
    output.push_back(sepal_width);
    output.push_back(petal_length);
    output.push_back(petal_width);

    return output;
}

template <typename T>
T calculate_std(vector <T> data){
    T mean = calculate_mean(data);
    T total_sum = 0.0;
    for (int i =0; i < data.size(); i++){
        total_sum = total_sum + (data[i] - mean) * (data[i] - mean);
    }
    return sqrt(total_sum / data.size());
}

vector <float> calc_probability(vector<float> values, float mean, float stddev){
    static float inv_sqrt = 0.3989422804;
    vector <float> prob;
    for (int i = 0; i < values.size(); i++){
        float a = (values[i] - mean) / (stddev);
        prob.push_back(inv_sqrt / stddev * exp(-0.5f * a * a));
    }
    return prob;
}

vector <vector <float> > train_split(vector <vector <float> >dataset, float percent){
    float dataset_size = dataset.size();
    int train_dataset_size = floor(percent * dataset_size);
    vector <vector <float> > temp_out;
    for (auto temp = dataset.begin(); temp != dataset.begin() + train_dataset_size; temp++) {
        temp_out.push_back(*temp);
    }
    return transpose_vector(temp_out);
}

vector <vector <float> > test_split(vector <vector <float> >dataset, float percent){
    float dataset_size = dataset.size();
    int32_t train_dataset_size = floor(percent * dataset_size);
    vector <vector <float> > temp_out;
    for (auto temp = dataset.begin() + train_dataset_size; temp != dataset.end(); temp++) {
        temp_out.push_back(*temp);
    }
    return transpose_vector(temp_out);
}

vector<vector<float> > read_iris_dataset(void){
    ifstream myfile("dataset/iris.data");
    string line;
    vector<vector<float> > iris_dataset;

    vector<float> sepal_length;
    vector<float> sepal_width;
    vector<float> petal_length;
    vector<float> petal_width;
    vector<float> species;

    float sepal_length_f, sepal_width_f, petal_length_f, petal_width_f, iris_class_f;
    string species_f;
    int count = 0;
    if (myfile.is_open()){
        cout << "file opened successully!!!" <<endl;
        while (getline(myfile, line)){
            replace(line.begin(), line.end(), '-', '_');
            replace(line.begin(), line.end(), ',', ' ');
            istringstream iss(line);
            iss >> sepal_length_f >> sepal_width_f >> petal_length_f >> petal_width_f >> species_f;
            sepal_length.push_back(sepal_length_f);
            sepal_width.push_back(sepal_width_f);
            petal_length.push_back(petal_length_f);
            petal_width.push_back(petal_width_f);
            if (species_f.compare("Iris_setosa") == 0){
                iris_class_f = Iris_setosa;
            } else if(species_f.compare("Iris_versicolor") == 0){
                iris_class_f = Iris_versicolor;
            } else if(species_f.compare("Iris_virginica") == 0){
                iris_class_f = Iris_virginica;
            } else{
                iris_class_f = Iris_unknown;
            }
            species.push_back(iris_class_f);
            count++;
        }
        iris_dataset.push_back(sepal_length);
        iris_dataset.push_back(sepal_width);
        iris_dataset.push_back(petal_length);
        iris_dataset.push_back(petal_width);
        iris_dataset.push_back(species);

    }
    else {
        cout << "unable to open file " << endl;
    }
    return iris_dataset;
}

vector<vector <float> > transpose_vector(vector<vector <float> > dataset){
    vector<vector <float> > output;
    for (int i = 0; i < dataset[0].size(); i++){
        vector <float> temp_out;
        for (int j = 0; j < dataset.size(); j++) {
            temp_out.push_back(dataset[j][i]);
        }
        output.push_back(temp_out);
    }
    return output;
}

class_summary calculate_class_summary(vector<vector<float> > dataset, float class_label){
    class_summary summary;
    vector<vector<float > > class_dataset = split_class_by_label(dataset, class_label);
    vector <float> temp_mean;
    vector <float> temp_stddev;
    for (auto i = 0; i < class_dataset.size(); i++){
        float mean_class = calculate_mean(class_dataset[i]);
        float std_class = calculate_std(class_dataset[i]);
        summary.mean.push_back(mean_class);
        summary.stddev.push_back(std_class);
    }
    summary.class_prob = float(class_dataset[0].size()) / float(dataset[0].size());
    return summary;
}

void naive_bayes::fit(vector<vector<float> > dataset) {
    for (int i = 0; i < dataset.size()-1; i++){
        naive_bayes::mean.push_back(calculate_mean(dataset[i]));
        naive_bayes::stddev.push_back(calculate_std(dataset[i]));
     }

    vector <vector <float> > norm_dataset;
    for (int i = 0; i < dataset.size() - 1; i++){
        vector<float> temp_dataset;
        for (int j = 0; j < dataset[i].size(); j++){
            temp_dataset.push_back((dataset[i][j] - naive_bayes::mean[i]) / float(naive_bayes::stddev[i]));
        }

        norm_dataset.push_back(temp_dataset);
    }
    norm_dataset.push_back(dataset[4]);
    naive_bayes::unique_label = norm_dataset[4];
    sort(naive_bayes::unique_label.begin(), naive_bayes::unique_label.end());
    auto last = unique(naive_bayes::unique_label.begin(), naive_bayes::unique_label.end());
    naive_bayes::unique_label.erase(last, naive_bayes::unique_label.end());
    for (auto i =0; i< naive_bayes::unique_label.size(); i++){
        naive_bayes::summary.push_back(calculate_class_summary(norm_dataset, naive_bayes::unique_label[i]));
    }
}

vector <int> naive_bayes::predict(vector<vector <float> > test_data) {
    vector <vector <float> > out;
    vector <float> temp_out;
    vector <int> output;
    vector <vector <float> > norm_dataset;
    for (int i = 0; i < naive_bayes::mean.size(); i++){
        vector<float> temp_dataset;
        for (int j = 0; j < test_data[i].size(); j++){
            temp_dataset.push_back((test_data[i][j] - naive_bayes::mean[i]) / (naive_bayes::stddev[i] + 1e-7));
        }
        norm_dataset.push_back(temp_dataset);
    }
    for (auto i = 0; i < naive_bayes::unique_label.size(); i++){
        temp_out = prob_by_summary(norm_dataset, naive_bayes::summary[i]);
        out.push_back(temp_out);
    }
    vector <vector <float> >transposed_out = transpose_vector(out);
    for (int i = 0; i < transposed_out.size(); i++){
        temp_out = transposed_out[i];
        auto result = max_element(temp_out.begin(), temp_out.end());
        int index = distance(temp_out.begin(), result);
        output.push_back(unique_label[index]);
    }
    return output;
}


vector <float> prob_by_summary(vector<vector <float> > dataset, class_summary summary){
    vector <vector <float> > prob;
    vector <float> temp_prob;
    //prob is a vector of size (5, 150);
    for (int i = 0; i < summary.mean.size(); i++){
        temp_prob = calc_probability(dataset[i], summary.mean[i], summary.stddev[i]);
        prob.push_back(temp_prob);
    }
    vector <vector <float> > transposed_prob = transpose_vector(prob);
    vector <float> output(transposed_prob.size());
    for (int i = 0; i < transposed_prob.size(); i++){
        output[i] = 1;
        for (int j = 0; j < transposed_prob[i].size(); j++){
            output[i] *= transposed_prob[i][j];
        }
        output[i] *= summary.class_prob;
    }
    return output;
}


float accuracy_score(vector <int> prediction, vector <float> true_labels){

    int index = 0;
    float result = 0;

    for(float label : true_labels){
        if (int(label) == prediction[index]){
            result++;
        }
        index++;
    }
    float accuracy = result / float(prediction.size());
    return accuracy;
}
