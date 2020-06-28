//
// Created by Ahmed Aleshinloye on 2020-06-24.
//

#ifndef IRIS_ML_NAIVE_BAYES_H
#define IRIS_ML_NAIVE_BAYES_H
#include <vector>

using namespace std;

typedef struct class_summary{
    vector<float> mean;
    vector<float> stddev;
    float class_prob;
}class_summary;

class naive_bayes {

private:
    vector<class_summary> summary;
    vector<float> unique_label;
    vector<float> mean;
    vector<float> stddev;

public:
    void fit(vector<vector<float> > dataset);
    vector <int> predict(vector <vector<float > > test_data);

};

class_summary calculate_class_summary(vector<vector<float> > dataset, float class_label);
vector <float> prob_by_summary(vector <vector <float> > dataset, class_summary summary);
float accuracy_score(vector <int> prediction, vector <float> true_labels);

template <typename T>
T calculate_mean(vector <T> data);

vector<vector<float> > split_class_by_label(vector<vector<float> > dataset, float class_label);

template <typename T>
T calculate_std(vector <T> data);

vector <float> calc_probability(vector<float> value, float mean, float stddev);

vector <vector <float> > test_split(vector <vector <float> >dataset, float percent);
vector <vector <float> > train_split(vector <vector <float> >dataset, float percent);

vector<vector<float> > read_iris_dataset(void);

vector<vector <float> > transpose_vector(vector<vector <float> > dataset);

#endif //IRIS_ML_NAIVE_BAYES_H
