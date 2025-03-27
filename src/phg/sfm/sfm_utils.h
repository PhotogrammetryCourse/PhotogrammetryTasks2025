#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

namespace phg {

void randomSample(std::vector<int> &dst, int max_id, int sample_size,
                  uint64_t *state);

bool epipolarTest(const cv::Vec2d &pt0, const cv::Vec2d &pt1,
                  const cv::Matx33d &F, double t);

} // namespace phg
