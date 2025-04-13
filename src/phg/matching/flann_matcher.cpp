#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"
#include "libutils/rasserts.h"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

#define KD_TREE_INDEX_N_TREES 4
#define KD_TREE_SEARCH_N_CHECKS 32

phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
   index_params = flannKdTreeIndexParams(KD_TREE_INDEX_N_TREES);
   search_params = flannKsTreeSearchParams(KD_TREE_SEARCH_N_CHECKS);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    rassert(k > 1, 92461);
    rassert(query_desc.rows > 0, 924611);
    rassert(query_desc.cols > 0, 9246111);

    cv::Mat indices(query_desc.rows, k, CV_32SC1);
    cv::Mat dists(query_desc.rows, k, CV_32FC1);
    
    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);

    matches.resize(query_desc.rows);
    for (int i = 0; i < query_desc.rows; ++i) {
        matches[i].reserve(k);
        for (int j = 0; j<k; ++j) {
            int idx = indices.at<int>(i, j);
            rassert(j == 0 || dists.at<float>(i, j-1) <= dists.at<float>(i, j), 111222999);

            matches[i].emplace_back(i, idx, sqrt(dists.at<float>(i, j)));
        }
    }
}