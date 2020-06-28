#include <iostream>
#include "naive_bayes.h"

using namespace std;

int main() {

    vector <vector <float> > dataset = read_iris_dataset();
    naive_bayes nb = naive_bayes();

    dataset = transpose_vector(dataset);
    srand(unsigned(time(0)));
    random_shuffle(dataset.begin(), dataset.end());
    float train_percent = 0.70;
    vector <vector <float> > train_dataset = train_split(dataset, train_percent);
    vector <vector <float> > test_dataset = test_split(dataset, train_percent);
    cout<< "IRIS training Data Size is ( " << train_dataset.size() << " , "  << train_dataset[0].size()<<" )" <<std::endl;

    cout<< "IRIS testing Data Size is ( " << test_dataset.size() << " , "  << test_dataset[0].size()<<" )" <<std::endl;

    nb.fit(train_dataset);
    vector <int> prediction = nb.predict(test_dataset);
    float result = accuracy_score(prediction, test_dataset[4]);
    cout << "Model accuracy is " << result << endl;
    return 0;
}
