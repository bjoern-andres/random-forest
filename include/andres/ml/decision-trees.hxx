/// \mainpage
/// andres::ml::DecisionTrees: A Fast C++ Implementation of Random Forests
///
/// \section section_abstract Short Description
/// The header file implements random forests as described in the article:
///
/// Leo Breiman. Random Forests. Machine Learning 45(1):5-32, 2001.
/// http://dx.doi.org/10.1023%2FA%3A1010933404324
/// 
/// \section section_license License
///
/// Copyright (c) 2013 by Steffen Kirchhoff and Bjoern Andres.
///
/// This software was developed by Steffen Kirchhoff and Bjoern Andres.
/// Enquiries shall be directed to bjoern@andres.sc.
///
/// All advertising materials mentioning features or use of this software must
/// display the following acknowledgement: ``This product includes andres::ml 
/// Decision Trees developed by Steffen Kirchhoff and Bjoern Andres. Please 
/// direct enquiries concerning andres::ml Decision Trees to bjoern@andres.sc''.
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
///
/// - Redistributions of source code must retain the above copyright notice,
/// this list of conditions and the following disclaimer.
/// - Redistributions in binary form must reproduce the above copyright notice,
/// this list of conditions and the following disclaimer in the documentation
/// and/or other materials provided with the distribution.
/// - All advertising materials mentioning features or use of this software must
/// display the following acknowledgement: ``This product includes andres::ml 
/// Decision Trees developed by Steffen Kirchhoff and Bjoern Andres. Please 
/// direct enquiries concerning andres::ml Decision Trees to bjoern@andres.sc''.
/// - The names of the authors must not be used to endorse or promote products
/// derived from this software without specific prior written permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR IMPLIED
/// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
/// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
/// EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
/// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
/// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
/// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
/// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
/// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
/// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/// 
#pragma once
#ifndef ANDRES_ML_DECISION_FOREST_HXX
#define ANDRES_ML_DECISION_FOREST_HXX

#include <stdexcept>
#include <random>
#include <vector>
#include <queue>
#include <cmath> // std::ceil, std::sqrt
#include <algorithm> // std::sort
#include <iterator> // std::distance

#include "andres/marray.hxx"

/// The public API.
namespace andres {
    
/// Machine Learning.
namespace ml {

/// A node in a decision tree.
template<class FEATURE, class LABEL>
class DecisionNode {
public:
    typedef FEATURE Feature;
    typedef LABEL Label;

    DecisionNode();
    bool isLeaf() const;
    bool& isLeaf();
    size_t featureIndex() const;
    size_t& featureIndex();
    Feature threshold() const;
    Feature& threshold();
    size_t childNodeIndex(const size_t) const;
    size_t& childNodeIndex(const size_t);
    Label label() const;
    Label& label();
    template<class RandomEngine>
        size_t learn(const andres::View<Feature>&, const andres::View<Label>&, 
            std::vector<size_t>&, const size_t, const size_t, RandomEngine&);

private:
    struct ComparisonByFeature {
        typedef size_t first_argument_type;
        typedef size_t second_argument_type;
        typedef bool result_type;

        ComparisonByFeature(
            const andres::View<Feature>& features,
            const size_t featureIndex
        )
            :   features_(features),
                featureIndex_(featureIndex)
            {
                assert(featureIndex < features.shape(1));
            }
        bool operator()(const size_t j, const size_t k) const 
            { 
                assert(j < features_.shape(0));
                assert(k < features_.shape(0));
                return features_(j, featureIndex_) < features_(k, featureIndex_);
            }

        const andres::View<Feature>& features_;
        const size_t featureIndex_;
    };

    template<class RandomEngine>
        void sampleSubsetWithoutReplacement(const size_t, const size_t, 
            std::vector<size_t>&, RandomEngine&,
            std::vector<size_t>& = std::vector<size_t>()
        );

    size_t featureIndex_;
    Feature threshold_;
    size_t childNodeIndices_[2]; // 0 means <, 1 means >=
    Label label_; 
    bool isLeaf_;
};

/// A decision tree.
template<class FEATURE = double, class LABEL = unsigned char>
class DecisionTree {
public:
    typedef FEATURE Feature;
    typedef LABEL Label;
    typedef DecisionNode<Feature, Label> DecisionNodeType;

    DecisionTree();
    size_t size() const; // number of decision nodes
    void predict(const andres::View<Feature>&, std::vector<Label>&) const;
    const DecisionNodeType& decisionNode(const size_t) const;
    void learn(const andres::View<Feature>&, const andres::View<Label>&, 
        std::vector<size_t>&);
    template<class RandomEngine>
        void learn(const andres::View<Feature>&, const andres::View<Label>&, 
            std::vector<size_t>&, RandomEngine&);

private:    
    struct TreeConstructionQueueEntry {
        TreeConstructionQueueEntry(
            const size_t nodeIndex = 0, 
            const size_t sampleIndexBegin = 0,
            const size_t sampleIndexEnd = 0,
            const size_t thresholdIndex = 0
        )
        :   nodeIndex_(nodeIndex),
            sampleIndexBegin_(sampleIndexBegin),
            sampleIndexEnd_(sampleIndexEnd),
            thresholdIndex_(thresholdIndex)
        {}

        size_t nodeIndex_;
        size_t sampleIndexBegin_;
        size_t sampleIndexEnd_;
        size_t thresholdIndex_;
    };

    std::vector<DecisionNodeType> decisionNodes_;
};

/// A bag of decision trees.
template<
    class FEATURE = double, 
    class LABEL = unsigned char, 
    class PROBABILITY = double
>
class DecisionForest {
public:
    typedef FEATURE Feature;
    typedef LABEL Label;
    typedef PROBABILITY Probability;
    typedef DecisionTree<Feature, Label> DecisionTreeType;

    DecisionForest();
    void clear();    
    size_t size() const;
    const DecisionTreeType& decisionTree(const size_t) const;
    void predict(const andres::View<Feature>&, andres::Marray<Probability>&) const;
    void learn(const andres::View<Feature>&, const andres::View<Label>&,
        const size_t = 255);
    template<class RandomEngine>
        void learn(const andres::View<Feature>&, const andres::View<Label>&,
            const size_t, RandomEngine&);

private:
    template<class RandomEngine>
        void sampleBootstrap(const size_t, std::vector<size_t>&, RandomEngine&);

    std::vector<DecisionTreeType> decisionTrees_;
};

// implementation of DecisionNode

/// Constructs a decision node.
/// 
template<class FEATURE, class LABEL>
inline
DecisionNode<FEATURE, LABEL>::DecisionNode()
:   featureIndex_(), 
    threshold_(), 
    label_(), 
    isLeaf_(false)
{
    childNodeIndices_[0] = 0;
    childNodeIndices_[1] = 0;
}

/// Returns true if the node is a leaf node.
/// 
template<class FEATURE, class LABEL>
inline bool 
DecisionNode<FEATURE, LABEL>::isLeaf() const {
    return isLeaf_;
}

/// Returns true if the node is a leaf node.
/// 
template<class FEATURE, class LABEL>
inline bool& 
DecisionNode<FEATURE, LABEL>::isLeaf() {
    return isLeaf_;
}

/// Returns, for a non-leaf node, the index of a feature wrt which a decision is made.
/// 
template<class FEATURE, class LABEL>
inline size_t 
DecisionNode<FEATURE, LABEL>::featureIndex() const {
    assert(!isLeaf());
    return featureIndex_;
}

/// Returns, for a non-leaf node, the index of a feature wrt which a decision is made.
/// 
template<class FEATURE, class LABEL>
inline size_t& 
DecisionNode<FEATURE, LABEL>::featureIndex() {
    assert(!isLeaf());
    return featureIndex_;
}

/// Returns, for a non-leaf node, a threshold by which a decision is made.
/// 
template<class FEATURE, class LABEL>
inline typename DecisionNode<FEATURE, LABEL>::Feature 
DecisionNode<FEATURE, LABEL>::threshold() const {
    assert(!isLeaf());
    return threshold_;
}

/// Returns, for a non-leaf node, a threshold by which a decision is made.
/// 
template<class FEATURE, class LABEL>
inline typename DecisionNode<FEATURE, LABEL>::Feature& 
DecisionNode<FEATURE, LABEL>::threshold() {
    assert(!isLeaf());
    return threshold_;
}

/// Returns, for a non-leaf node, the index of one of its two child nodes.
///
/// \param j number of the child node (either 0 or 1).
/// 
template<class FEATURE, class LABEL>
inline size_t 
DecisionNode<FEATURE, LABEL>::childNodeIndex(
    const size_t j
) const {
    assert(!isLeaf());
    assert(j < 2);
    return childNodeIndices_[j];
}

/// Returns, for a non-leaf node, the index of one of its two child nodes.
///
/// \param j number of the child node (either 0 or 1).
/// 
template<class FEATURE, class LABEL>
inline size_t& 
DecisionNode<FEATURE, LABEL>::childNodeIndex(
    const size_t j
) {
    assert(!isLeaf());
    assert(j < 2);
    return childNodeIndices_[j];
}

/// Returns, for a leaf node, its label.
///
template<class FEATURE, class LABEL>
inline typename DecisionNode<FEATURE, LABEL>::Label
DecisionNode<FEATURE, LABEL>::label() const {
    assert(isLeaf());
    return label_;
}

/// Returns, for a leaf node, its label.
///
template<class FEATURE, class LABEL>
inline typename DecisionNode<FEATURE, LABEL>::Label& 
DecisionNode<FEATURE, LABEL>::label() {
    assert(isLeaf());
    return label_;
}

/// Learns a decision node from labeled samples as described by Breiman (2001).
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample.
/// \param sampleIndices A sequence of indices of samples to be considered for learning. This vector is used by the function as a scratch-pad for sorting.
/// \param sampleIndexBegin Index of the first element of sampleIndices to be considered.
/// \param sampleIndexEnd Index one greater than that of the last element of sampleIndices to be considered.
/// \param randomEngine C++11 STL-compliant random number generator.
///
template<class FEATURE, class LABEL>
template<class RandomEngine>
size_t
DecisionNode<FEATURE, LABEL>::learn(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    std::vector<size_t>& sampleIndices, // input, will be sorted
    const size_t sampleIndexBegin,
    const size_t sampleIndexEnd,
    RandomEngine& randomEngine
) {
    /*
    std::cout << "learning new node from sample indices";
    for(size_t j = sampleIndexBegin; j < sampleIndexEnd; ++j) {
        std::cout << " " << sampleIndices[j];
    }
    std::cout << std::endl;
    */

    assert(sampleIndices.size() != 0);
    assert(sampleIndexEnd <= sampleIndices.size());
    assert(sampleIndexBegin < sampleIndexEnd);
    assert(features.shape(0) != 0);
    assert(features.shape(0) == labels.size());

    // handle the case in which there is only one unique label (pure node)
    {
        bool isLabelUnique = true;
        const size_t firstLabel = labels(sampleIndices[sampleIndexBegin]);
        for(size_t j = sampleIndexBegin + 1; j < sampleIndexEnd; ++j) {
            if(labels(sampleIndices[j]) != firstLabel) { 
                isLabelUnique = false;
                break;
            }
        }
        if(isLabelUnique) {
            isLeaf_ = true;
            label_ = labels(sampleIndices[sampleIndexBegin]);
            // std::cout << "sample is pure." << std::endl;
            return 0;
        }
    }
    
    const size_t numberOfFeatures = features.shape(1);
    const size_t numberOfFeaturesToBeAssessed = 
        static_cast<size_t>(
            std::ceil(std::sqrt(
                static_cast<double>(numberOfFeatures)
            ))
        );

    std::vector<size_t> featureIndices(numberOfFeaturesToBeAssessed); // expensive!
    std::vector<size_t> buffer; // expensive!
    
    sampleSubsetWithoutReplacement(
        numberOfFeatures, 
        numberOfFeaturesToBeAssessed, 
        featureIndices,
        randomEngine,
        buffer
    );

    std::vector<size_t> numbersOfLabels[2]; 
    numbersOfLabels[0].reserve(10); // expensive!
    numbersOfLabels[1].reserve(10); // expensive!
    double optimalSumOfGiniCoefficients = std::numeric_limits<double>::infinity();
    size_t optimalFeatureIndex;
    size_t optimalThresholdIndex;
    Feature optimalThreshold;
    for(size_t j = 0; j < numberOfFeaturesToBeAssessed; ++j) {
        const size_t fi = featureIndices[j];

        // sort sample indices wrt fi-th feature
        std::sort(
            sampleIndices.begin() + sampleIndexBegin, 
            sampleIndices.begin() + sampleIndexEnd, 
            ComparisonByFeature(features, fi)
        );
        #ifndef NDEBUG
        for(size_t k = sampleIndexBegin; k + 1 < sampleIndexEnd; ++k) {
            assert(
                features(sampleIndices[k], fi) <= features(sampleIndices[k + 1], fi)
            );
        }
        #endif

        // the variables "numbersOfElements" and "numbersOfLabels" are defined 
        // for two sets:
        // [0] all samples with indices {sampleIndexBegin, ..., thresholdIndex - 1}
        // [1] all samples with indices {thresholdIndex, ..., sampleIndexEnd}
        // the initialization is for 0 being the empty set.

        // numbers of elements
        size_t numbersOfElements[] = {0, sampleIndexEnd - sampleIndexBegin};
        
        // numbers of labels
        for(size_t k = sampleIndexBegin; k < sampleIndexEnd; ++k) {
            const Label label = labels(sampleIndices[k]);
            if(label >= numbersOfLabels[1].size()) {
                const size_t newNumberOfLabels = label + 1;
                for(size_t s = 0; s < 2; ++s) {
                    numbersOfLabels[s].resize(newNumberOfLabels); // expensive!
                }
            }
            ++numbersOfLabels[1][label];
        }

        // assess all relevant splits wrt fi-th feature
        size_t thresholdIndex = sampleIndexBegin;
        for(;;) { 
            const size_t thresholdIndexOld = thresholdIndex;

            // skip samples with identical feature value
            while(thresholdIndex + 1 < sampleIndexEnd
            && features(sampleIndices[thresholdIndex], fi) 
            == features(sampleIndices[thresholdIndex + 1], fi)) {                
                const size_t label = labels(sampleIndices[thresholdIndex]);
                ++numbersOfElements[0];
                --numbersOfElements[1];
                ++numbersOfLabels[0][label];
                --numbersOfLabels[1][label];
                ++thresholdIndex;
            }

            // increment
            {
                const size_t label = labels(sampleIndices[thresholdIndex]);
                ++numbersOfElements[0];
                --numbersOfElements[1];
                ++numbersOfLabels[0][label];
                --numbersOfLabels[1][label];
            }
            ++thresholdIndex; 
            if(thresholdIndex == sampleIndexEnd) {
                break;
            }

            // count numbers of distinctly labeled pairs for both sets
            assert(numbersOfLabels[0].size() == numbersOfLabels[1].size());
            size_t numbersOfDistinctPairs[] = {0, 0};
            for(size_t s = 0; s < 2; ++s) // set 0 and 1
            for(size_t k = 0; k < numbersOfLabels[s].size(); ++k)
            for(size_t m = k + 1; m < numbersOfLabels[s].size(); ++m) {
                numbersOfDistinctPairs[s] += 
                    numbersOfLabels[s][k] * numbersOfLabels[s][m];
            }

            // compute Gini coefficients for both sets
            double giniCoefficients[2];
            for(size_t s = 0; s < 2; ++s) {
                if(numbersOfElements[s] < 2) {
                    giniCoefficients[s] = 0;
                }
                else {
                    giniCoefficients[s] = 
                        static_cast<double>(numbersOfDistinctPairs[s])
                        / (numbersOfElements[s] * (numbersOfElements[s] - 1));
                }
            }

            double sumOfginiCoefficients = giniCoefficients[0] + giniCoefficients[1];
            /*
            std::cout << "fi: " << fi
                << ", threshold index: " << thresholdIndex
                << ", threshold: " << features(sampleIndices[thresholdIndex], fi)
                << ", gini: " << sumOfginiCoefficients;
            */
            if(sumOfginiCoefficients < optimalSumOfGiniCoefficients) {
                optimalSumOfGiniCoefficients = sumOfginiCoefficients;
                optimalFeatureIndex = fi;
                optimalThreshold = features(sampleIndices[thresholdIndex], fi); 
                optimalThresholdIndex = thresholdIndex;
                // std::cout << ", new optimum";
            }
            // std::cout << std::endl;
        }
        for(size_t s = 0; s < 2; ++s) {
            std::fill(numbersOfLabels[s].begin(), numbersOfLabels[s].end(), 0);
        }
    }
    threshold_ = optimalThreshold;
    featureIndex_ = optimalFeatureIndex;

    // sort data wrt optimal feature
    std::sort(
        sampleIndices.begin() + sampleIndexBegin, 
        sampleIndices.begin() + sampleIndexEnd, 
        ComparisonByFeature(features, optimalFeatureIndex)
    );
    
    return optimalThresholdIndex;
}

template<class FEATURE, class LABEL>
template<class RandomEngine>
inline void 
DecisionNode<FEATURE, LABEL>::sampleSubsetWithoutReplacement(
    const size_t size,
    const size_t subsetSize,
    std::vector<size_t>& indices, // output
    RandomEngine& randomEngine,
    std::vector<size_t>& candidateIndices // buffer
) {
    assert(subsetSize <= size);
    indices.resize(subsetSize);

    // start with indices {0, ..., size - 1} as candidates
    candidateIndices.resize(size);
    for(size_t j = 0; j < size; ++j) {
        candidateIndices[j] = j;
    }

    // draw "subsetSize" indices without replacement
    #pragma omp critical
    for(size_t j = 0; j < subsetSize; ++j) {
        std::uniform_int_distribution<size_t> distribution(0, size - j - 1);
        const size_t index = distribution(randomEngine);
        indices[j] = candidateIndices[index];
        candidateIndices[index] = candidateIndices[size - j - 1];
        #ifndef NDEBUG
        for(size_t k = 0; k < j; ++k) {
            assert(indices[k] != indices[j]);
        }
        #endif
    }
}

// implementation of DecisionTree

/// Constructs a decision tree.
///
template<class FEATURE, class LABEL>
inline
DecisionTree<FEATURE, LABEL>::DecisionTree()
:   decisionNodes_()
{}

/// Learns a decision tree as described by Leo Breiman (2001).
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample.
/// \param sampleIndices A sequence of indices of samples to be considered for learning. This vector is used by the function as a scratch-pad for sorting.
///
template<class FEATURE, class LABEL>
inline void 
DecisionTree<FEATURE, LABEL>::learn(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    std::vector<size_t>& sampleIndices // input, will be sorted
) {
    typedef std::default_random_engine RandomEngine;
    learn<RandomEngine>(features, labels, sampleIndices, RandomEngine());
}

/// Learns a decision tree as described by Leo Breiman (2001).
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample.
/// \param sampleIndices A sequence of indices of samples to be considered for learning. This vector is used by the function as a scratch-pad for sorting.
/// \param randomEngine C++11 STL-compliant random number generator.
///
template<class FEATURE, class LABEL>
template<class RandomEngine>
void 
DecisionTree<FEATURE, LABEL>::learn(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    std::vector<size_t>& sampleIndices, // input, will be sorted
    RandomEngine& randomEngine
) {
    assert(decisionNodes_.size() == 0);
   
    std::queue<TreeConstructionQueueEntry> queue;
    // learn root node
    {
        decisionNodes_.push_back(DecisionNodeType());
        size_t thresholdIndex = decisionNodes_.back().learn(
            features, 
            labels, 
            sampleIndices, 
            0, sampleIndices.size(),
            randomEngine
        );        
        if(!decisionNodes_[0].isLeaf()) { // if root node is not pure
            queue.push(
                TreeConstructionQueueEntry(
                    0, // node index
                    0, sampleIndices.size(), // range of samples
                    thresholdIndex 
                )
            );
        }
    }
    while(!queue.empty()) {
        const size_t nodeIndex = queue.front().nodeIndex_;
        const size_t sampleIndexBegin = queue.front().sampleIndexBegin_;
        const size_t sampleIndexEnd = queue.front().sampleIndexEnd_;
        const size_t thresholdIndex = queue.front().thresholdIndex_;
        queue.pop();

        size_t nodeIndexNew;
        size_t thresholdIndexNew;

        // learn left child node and put on queue
        nodeIndexNew = decisionNodes_.size();
        decisionNodes_.push_back(DecisionNodeType());
        thresholdIndexNew = decisionNodes_.back().learn(
            features, 
            labels, 
            sampleIndices,
            sampleIndexBegin, thresholdIndex,
            randomEngine
        );
        #ifndef NDEBUG
        if(decisionNodes_[nodeIndexNew].isLeaf()) {
            assert(thresholdIndexNew == 0);
        }
        else {
            assert(
                thresholdIndexNew >= sampleIndexBegin
                && thresholdIndexNew < thresholdIndex
            );
        }
        #endif
        decisionNodes_[nodeIndex].childNodeIndex(0) = nodeIndexNew;
        if(!decisionNodes_[nodeIndexNew].isLeaf()) { // if not pure
            queue.push(
                TreeConstructionQueueEntry(
                    nodeIndexNew,
                    sampleIndexBegin, thresholdIndex,
                    thresholdIndexNew
                )
            );
        }

        // learn right child node and put on queue
        nodeIndexNew = decisionNodes_.size();
        decisionNodes_.push_back(DecisionNodeType());
        thresholdIndexNew = decisionNodes_.back().learn(
            features, 
            labels, 
            sampleIndices,
            thresholdIndex, sampleIndexEnd,
            randomEngine
        );
        #ifndef NDEBUG
        if(decisionNodes_[nodeIndexNew].isLeaf()) {
            assert(thresholdIndexNew == 0);
        }
        else {
            assert(
                thresholdIndexNew >= thresholdIndex
                && thresholdIndexNew < sampleIndexEnd
            );
        }
        #endif
        decisionNodes_[nodeIndex].childNodeIndex(1) = nodeIndexNew;
        if(!decisionNodes_[nodeIndexNew].isLeaf()) { // if not pure
            queue.push(
                TreeConstructionQueueEntry(
                    nodeIndexNew,
                    thresholdIndex, sampleIndexEnd,
                    thresholdIndexNew
                )
            );
        }
    }
}

/// Returns the number of decision nodes.
///
template<class FEATURE, class LABEL>
inline size_t 
DecisionTree<FEATURE, LABEL>::size() const {
    return decisionNodes_.size();
}

/// Returns a decision node.
///
template<class FEATURE, class LABEL>
inline const typename DecisionTree<FEATURE, LABEL>::DecisionNodeType& 
DecisionTree<FEATURE, LABEL>::decisionNode(
    const size_t decisionNodeIndex
) const {
    return decisionNodes_[decisionNodeIndex];
}

/// Predicts the labels of feature vectors.
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample. (output)
///
template<class FEATURE, class LABEL>
inline void 
DecisionTree<FEATURE, LABEL>::predict(
    const andres::View<Feature>& features,
    std::vector<Label>& labels
) const  {
    const size_t numberOfSamples = features.shape(0);
    const size_t numberOfFeatures = features.shape(1);
    labels.resize(numberOfSamples);
    for(size_t j = 0; j < numberOfSamples; ++j) {
        size_t nodeIndex = 0;
        for(;;) {
            const DecisionNodeType& decisionNode = decisionNodes_[nodeIndex];
            if(decisionNode.isLeaf()) {
                labels[j] = decisionNode.label();
                break;
            }
            else {
                const size_t fi = decisionNode.featureIndex();
                const Feature threshold = decisionNode.threshold();
                assert(fi < numberOfFeatures);
                if(features(j, fi) < threshold) {
                    nodeIndex = decisionNode.childNodeIndex(0);
                }
                else {
                    nodeIndex = decisionNode.childNodeIndex(1);
                }
                assert(nodeIndex != 0); // assert that tree is not incomplete
            }
        }
    }
}

// implementation of DecisionForest

/// Constructs a decision forest.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline
DecisionForest<FEATURE, LABEL, PROBABILITY>::DecisionForest()
:   decisionTrees_()
{}

/// Clears a decision forest.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline void
DecisionForest<FEATURE, LABEL, PROBABILITY>::clear() {
    decisionTrees_.clear();
}

/// Returns the number of decision trees.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline size_t
DecisionForest<FEATURE, LABEL, PROBABILITY>::size() const {
    return decisionTrees_.size();
}

/// Learns a decision forest from labeled samples as described by Breiman (2001).
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample.
/// \param numberOfDecisionTrees Number of decision trees to be learned.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline void 
DecisionForest<FEATURE, LABEL, PROBABILITY>::learn(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    const size_t numberOfDecisionTrees
) {
    typedef std::default_random_engine RandomEngine;
    RandomEngine randomEngine;
    learn<RandomEngine>(features, labels, numberOfDecisionTrees, randomEngine);
}

/// Learns a decision forest from labeled samples as described by Breiman (2001).
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample.
/// \param numberOfDecisionTrees Number of decision trees to be learned.
/// \param randomEngine C++11 STL-compatible random number generator.
///
template<class FEATURE, class LABEL, class PROBABILITY>
template<class RandomEngine>
inline void 
DecisionForest<FEATURE, LABEL, PROBABILITY>::learn(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    const size_t numberOfDecisionTrees,
    RandomEngine& randomEngine
) {
    if(features.dimension() != 2) {
        throw std::runtime_error("features.dimension() != 2");
    }
    if(labels.dimension() != 1) {
        throw std::runtime_error("labels.dimension() != 1");
    }
    if(features.shape(0) != labels.size()) {
        throw std::runtime_error("the number of samples does not match the size of the label vector.");
    }
    const size_t numberOfSamples = features.shape(0);

    clear();
    decisionTrees_.resize(numberOfDecisionTrees);

    #pragma omp parallel for schedule(guided)
    for(ptrdiff_t treeIndex = 0; treeIndex < static_cast<ptrdiff_t>(decisionTrees_.size()); ++treeIndex) {
        std::vector<size_t> sampleIndices(numberOfSamples);
        sampleBootstrap(numberOfSamples, sampleIndices, randomEngine);
        decisionTrees_[treeIndex].learn(features, labels, sampleIndices, randomEngine);
    }
}

/// Predict the label probabilities of samples as described by Breiman (2001).
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labelProbabilities A matrix of probabilities in which every rows corresponds to a sample and every column corresponds to a label.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline void 
DecisionForest<FEATURE, LABEL, PROBABILITY>::predict(
    const andres::View<Feature>& features,
    andres::Marray<Probability>& labelProbabilities 
) const  {
    if(size() == 0) {
        throw std::runtime_error("no decision trees.");
    }
    if(features.dimension() != 2) {
        throw std::runtime_error("features.dimension() != 2");
    }
    if(features.shape(0) != labelProbabilities.shape(0)) {
        throw std::runtime_error("labelProbabilities.shape(0) does not match the number of samples.");
    }

    const size_t numberOfSamples = features.shape(0);
    const size_t numberOfFeatures = features.shape(1);
    std::fill(labelProbabilities.begin(), labelProbabilities.end(), Probability());
    #pragma omp parallel for schedule(dynamic)
    for(ptrdiff_t treeIndex = 0; treeIndex < static_cast<ptrdiff_t>(decisionTrees_.size()); ++treeIndex) {
        std::vector<Label> labels(numberOfSamples);
        const DecisionTreeType& decisionTree = decisionTrees_[treeIndex];
        decisionTree.predict(features, labels);
        for(size_t sampleIndex = 0; sampleIndex < numberOfSamples; ++sampleIndex) {
            const Label label = labels[sampleIndex];
            if(label >= labelProbabilities.shape(1)) {
                throw std::runtime_error("labelProbabilities.shape(1) does not match the number of labels.");
            }
            #pragma omp atomic
            ++labelProbabilities(sampleIndex, label);
        }
    }
    #pragma omp parallel for
    for(ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(labelProbabilities.size()); ++j) {
        labelProbabilities(j) /= decisionTrees_.size();
    }
}

/// Returns a decision tree.
///
/// \param treeIndex Index of the decision tree.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline const typename DecisionForest<FEATURE, LABEL, PROBABILITY>::DecisionTreeType& 
DecisionForest<FEATURE, LABEL, PROBABILITY>::decisionTree(
    const size_t treeIndex
) const {
    return decisionTrees_[treeIndex];
}

// draw "size" out of "size", with replacement
template<class FEATURE, class LABEL, class PROBABILITY>
template<class RandomEngine>
inline void 
DecisionForest<FEATURE, LABEL, PROBABILITY>::sampleBootstrap(
    const size_t size,
    std::vector<size_t>& indices,
    RandomEngine& randomEngine
) {
    indices.resize(size);    
    #pragma omp critical
    for(size_t j = 0; j < size; ++j) {
        std::uniform_int_distribution<size_t> distribution(0, size - 1);
        indices[j] = distribution(randomEngine);
        assert(indices[j] < size);
    }
}

} // namespace ml
} // namespace andres

#endif // #ifndef ANDRES_ML_DECISION_FOREST_HXX
