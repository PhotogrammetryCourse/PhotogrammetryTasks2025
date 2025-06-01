#include "flann_matcher.h"

#include <iostream>

#include "flann_factory.h"

phg::FlannMatcher::FlannMatcher() {
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc) {
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const {
    cv::Mat indices(query_desc.rows, k, CV_32SC1);
    cv::Mat distances2(query_desc.rows, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices, distances2, k, *search_params);

    matches.resize(query_desc.rows);
    for (int query_idx = 0; query_idx < query_desc.rows; ++query_idx) {
        matches[query_idx].reserve(k);
        for (int i = 0; i < k; ++i) {
            matches[query_idx].emplace_back(query_idx, indices.at<int>(query_idx, i), cv::sqrt(distances2.at<float>(query_idx, i)));
        }
    }
}
