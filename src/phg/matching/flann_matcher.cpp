#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    const int num_trees = 4;
    const int num_checks = 32;
    index_params = flannKdTreeIndexParams(num_trees);
    search_params = flannKsTreeSearchParams(num_checks);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::setRNGSeed(125125);
    cv::Mat indices;
    cv::Mat distances2;
    flann_index->knnSearch(query_desc, indices, distances2, k, *search_params);
    int numMatches = indices.rows;
    for (size_t i = 0; i < numMatches; ++i) {
        std::vector<cv::DMatch> kMatches;
        for (size_t j = 0; j < k; j++) {
            int idx = indices.at<int>(i, j);
            float dist = sqrt(distances2.at<float>(i, j));
            kMatches.emplace_back(i, idx, dist);
        }
        matches.push_back(kMatches);
    }
}
