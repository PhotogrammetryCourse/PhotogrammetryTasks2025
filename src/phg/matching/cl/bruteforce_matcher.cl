#ifdef __CLION_IDE__
// Этот include виден только для CLion парсера, это позволяет IDE "знать" ключевые слова вроде __kernel, __global
// а также уметь подсказывать OpenCL методы, описанные в данном инклюде (такие как get_global_id(...) и get_local_id(...))
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define NDIM 128 // размерность дескриптора, мы полагаемся на то что она совпадает с размером нашей рабочей группы

__attribute__((reqd_work_group_size(NDIM, 1, 1)))
__kernel void bruteforce_matcher(__global const float* train,
                                 __global const float* query,
                                 __global        uint* res_train_idx,
                                 __global        uint* res_query_idx,
                                 __global       float* res_distance,
                                 unsigned int n_train_desc,
                                 unsigned int n_query_desc)
{
    const unsigned int dim_id = get_global_id(0);
    const unsigned int query_id0 = KEYPOINTS_PER_WG * get_global_id(1);

    __local float query_local[KEYPOINTS_PER_WG * NDIM];
    __local uint  res_train_idx_local[KEYPOINTS_PER_WG * 2];
    __local float res_distance2_local[KEYPOINTS_PER_WG * 2];
    if (dim_id < KEYPOINTS_PER_WG * 2) {
        res_distance2_local[dim_id] = FLT_MAX;
    }

    for (int i = 0; i < KEYPOINTS_PER_WG; i++) {
        query_local[i * NDIM + dim_id] = (query_id0 + i < n_query_desc) ? query[(query_id0 + i) * NDIM + dim_id] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __local float dist2_for_reduction[KEYPOINTS_PER_WG * NDIM];

    for (unsigned int train_idx = 0; train_idx < n_train_desc; ++train_idx) {
        float train_value_dim = train[train_idx * NDIM + dim_id];
        for (int query_local_i = 0; query_local_i < KEYPOINTS_PER_WG; ++query_local_i) {
            float diff = train_value_dim - query_local[query_local_i * NDIM + dim_id];
            dist2_for_reduction[dim_id] = diff * diff;

            barrier(CLK_LOCAL_MEM_FENCE);
            unsigned int step = NDIM / 2;
            while (step > 0) {
                if (dim_id < step) {
                    dist2_for_reduction[dim_id] += dist2_for_reduction[dim_id + step];
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                step /= 2;
            }

            if (dim_id == 0) {
                float dist2 = dist2_for_reduction[0];

#define BEST_INDEX        0
#define SECOND_BEST_INDEX 1

                int base = query_local_i * 2;
                if (dist2 <= res_distance2_local[base + BEST_INDEX]) {
                    res_distance2_local[base + SECOND_BEST_INDEX] = res_distance2_local[base + BEST_INDEX];
                    res_train_idx_local[base + SECOND_BEST_INDEX] = res_train_idx_local[base + BEST_INDEX];

                    res_distance2_local[base + BEST_INDEX] = dist2;
                    res_train_idx_local[base + BEST_INDEX] = train_idx;

                } else if (dist2 <= res_distance2_local[base + SECOND_BEST_INDEX]) {
                    res_distance2_local[base + SECOND_BEST_INDEX] = dist2;
                    res_train_idx_local[base + SECOND_BEST_INDEX] = train_idx;
                }
            }
        }
    }

    if (dim_id < KEYPOINTS_PER_WG * 2) {
        const unsigned int query_local_i = dim_id / 2;
        const unsigned int k = dim_id % 2;

        const unsigned int query_id = query_id0 + query_local_i;
        const unsigned int local_idx = query_local_i * 2 + k;
        const unsigned int global_idx = query_id * 2 + k;
        if (query_id < n_query_desc) {
            res_train_idx[global_idx] = res_train_idx_local[local_idx];
            res_query_idx[global_idx] = query_id;
            res_distance [global_idx] = res_distance2_local[local_idx];
        }
    }
}