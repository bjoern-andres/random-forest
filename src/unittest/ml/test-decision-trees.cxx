#include <stdexcept>
#include <vector>
#include <random>

#include "andres/marray.hxx"
#include "andres/ml/decision-trees.hxx"

inline void test(const bool& x) { 
    if(!x) throw std::logic_error("test failed."); 
}

int main() {
    const size_t numberOfSamples = 10000;
    const size_t numberOfFeatures = 2;
    
    // define random feature matrix
    std::default_random_engine RandomNumberGenerator;
    typedef double Feature;
    std::uniform_real_distribution<double> randomDistribution(0.0, 1.0);
    const size_t shape[] = {numberOfSamples, numberOfFeatures};
    andres::Marray<Feature> features(shape, shape + 2);
    for(size_t sample = 0; sample < numberOfSamples; ++sample)
    for(size_t feature = 0; feature < numberOfFeatures; ++feature) {
        features(sample, feature) = randomDistribution(RandomNumberGenerator);
    }

    // define labels
    typedef unsigned char Label;
    andres::Marray<Label> labels(shape, shape + 1);
    for(size_t sample = 0; sample < numberOfSamples; ++sample) {
        if((features(sample, 0) <= 0.5 && features(sample, 1) <= 0.5)
        || (features(sample, 0) > 0.5 && features(sample, 1) > 0.5)) {
            labels(sample) = 0;
        }
        else {
            labels(sample) = 1;
        }
    }

    // learn decision forest
    typedef double Probability;
    andres::ml::DecisionForest<Feature, Label, Probability> decisionForest;
    const size_t numberOfDecisionTrees = 10;
    decisionForest.learn(features, labels, numberOfDecisionTrees);

    // predict probabilities for every label and every training sample
    andres::Marray<Probability> probabilities(shape, shape + 2);
    decisionForest.predict(features, probabilities);
    // TODO: test formally

    std::stringstream sstream;
    decisionForest.serialize(sstream);

    andres::ml::DecisionForest<Feature, Label, Probability> decisionForest_2;
    decisionForest_2.deserialize(sstream);

    andres::Marray<Probability> probabilities_2(shape, shape + 2);
    decisionForest.predict(features, probabilities_2);

    size_t cnt = 0;
    for (size_t i = 0; i < numberOfSamples; ++i)
        if (fabs(probabilities(i) - probabilities_2(i)) < std::numeric_limits<double>::epsilon())
            ++cnt;

    if (cnt == numberOfSamples)
        std::cout << "two predictions coincide\n";

    return 0;
}
