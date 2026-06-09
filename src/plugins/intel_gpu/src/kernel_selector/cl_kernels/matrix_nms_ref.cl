// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#define NUM_BATCHES INPUT0_BATCH_NUM
#define NUM_BOXES   INPUT0_FEATURE_NUM
#define NUM_CLASSES INPUT1_FEATURE_NUM

typedef struct {
    int batch_idx;
    int class_idx;
    int box_idx;
    INPUT1_TYPE score;
} FUNC(BoxInfo);

#define BOX_INFO FUNC(BoxInfo)

inline INPUT1_TYPE FUNC(decay_gaussian)(INPUT1_TYPE iou, INPUT1_TYPE max_iou) {
    return exp((max_iou * max_iou - iou * iou) * GAUSSIAN_SIGMA);
}

inline INPUT1_TYPE FUNC(decay_linear)(INPUT1_TYPE iou, INPUT1_TYPE max_iou) {
    return (INPUT1_VAL_ONE - iou) / (INPUT1_VAL_ONE - max_iou + TINY);
}

inline void FUNC(swap_boxes)(__global BOX_INFO* a, __global BOX_INFO* b) {
    BOX_INFO temp = *a;
    *a = *b;
    *b = temp;
}

inline COORD_TYPE_4 FUNC(getBoxCoords)(const __global INPUT0_TYPE* boxes, const short batch, const ushort box_idx) {
    COORD_TYPE_4 coords = (COORD_TYPE_4)(boxes[INPUT0_GET_INDEX(batch, box_idx, 0, 0)],
                                         boxes[INPUT0_GET_INDEX(batch, box_idx, 0, 1)],
                                         boxes[INPUT0_GET_INDEX(batch, box_idx, 0, 2)],
                                         boxes[INPUT0_GET_INDEX(batch, box_idx, 0, 3)]);

    // uncomment when flipped coordinates will be fixed in reference impl
    /*
    const INPUT0_TYPE x1 = min(coords[0], coords[2]);
    const INPUT0_TYPE x2 = max(coords[0], coords[2]);
    const INPUT0_TYPE y1 = min(coords[1], coords[3]);
    const INPUT0_TYPE y2 = max(coords[1], coords[3]);
    coords[0] = x1;
    coords[1] = y1;
    coords[2] = x2;
    coords[3] = y2;
    */

    return coords;
}

inline INPUT0_TYPE FUNC(area)(const INPUT0_TYPE w, const INPUT0_TYPE h) {
    return (w + NORM) * (h + NORM);
}

inline INPUT0_TYPE FUNC(areaBox)(const COORD_TYPE_4 box) {
    if (box[2] < box[0] || box[3] < box[1])
        return INPUT0_VAL_ZERO;
    return FUNC_CALL(area)(box[3] - box[1], box[2] - box[0]);
}

inline INPUT0_TYPE FUNC(intersectionOverUnion)(const COORD_TYPE_4 box1, const COORD_TYPE_4 box2) {
    if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] || box2[3] < box1[1])
        return INPUT0_VAL_ZERO;

    const INPUT0_TYPE area = FUNC_CALL(areaBox)(box1);
    const INPUT0_TYPE areaBox = FUNC_CALL(areaBox)(box2);

    const INPUT0_TYPE intersection_xmin = max(box1[0], box2[0]);
    const INPUT0_TYPE intersection_ymin = max(box1[1], box2[1]);
    const INPUT0_TYPE intersection_xmax = min(box1[2], box2[2]);
    const INPUT0_TYPE intersection_ymax = min(box1[3], box2[3]);

    const INPUT0_TYPE intersection_area =
        FUNC_CALL(area)(intersection_xmax - intersection_xmin, intersection_ymax - intersection_ymin);
    const INPUT0_TYPE union_area = area + areaBox - intersection_area;

    return intersection_area / union_area;
}

#ifdef MATRIX_NMS_STAGE_0
KERNEL(matrix_nms_ref_stage_0)
(const __global INPUT0_TYPE* input_boxes,
 const __global INPUT1_TYPE* input_scores,
 __global uchar* buffer0,
 __global int* selected_boxes_num,
 __global INPUT1_TYPE* input_iou_matrix,
 __global INPUT1_TYPE* input_iou_max,
 __global INPUT1_TYPE* input_min_decays) {
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);

    if (classId == BACKGROUND_CLASS)
        return;

    const int offset = batchId * NUM_CLASSES + classId;

    __global BOX_INFO* box_info = (__global BOX_INFO*)buffer0;
    box_info = &box_info[batchId * NUM_CLASSES * MAX_BOXES_PER_CLASS + classId * MAX_BOXES_PER_CLASS];

    // --- Phase 1: Count scores above threshold and find max score ---
    // Single read-only pass over input_scores.
    int count = 0;
    INPUT1_TYPE max_score = SCORE_THRESHOLD;
    for (int idx = 0; idx < NUM_BOXES; idx++) {
        const INPUT1_TYPE score = input_scores[INPUT1_GET_INDEX(batchId, classId, 0, idx)];
        if (score > SCORE_THRESHOLD) {
            count++;
            if (score > max_score)
                max_score = score;
        }
    }

    // --- Phase 2: Collect top candidates into global buffer ---
    // Uses adaptive threshold to select approximately MAX_BOXES_PER_CLASS elements in one pass.
    INPUT1_TYPE adj_threshold = SCORE_THRESHOLD;
    if (count > MAX_BOXES_PER_CLASS) {
        // Estimate threshold assuming roughly uniform distribution in [SCORE_THRESHOLD, max_score].
        INPUT1_TYPE range = max_score - SCORE_THRESHOLD;
        adj_threshold = max_score - range * (INPUT1_TYPE)(MAX_BOXES_PER_CLASS) / (INPUT1_TYPE)count;
    }

    int valid_boxes_num = 0;
    for (int idx = 0; idx < NUM_BOXES; idx++) {
        const INPUT1_TYPE score = input_scores[INPUT1_GET_INDEX(batchId, classId, 0, idx)];
        if (score > adj_threshold && valid_boxes_num < MAX_BOXES_PER_CLASS) {
            box_info[valid_boxes_num].box_idx = idx;
            box_info[valid_boxes_num].score = score;
            valid_boxes_num++;
        }
    }

    if (valid_boxes_num == 0) {
        for (int i = 0; i < MAX_BOXES_PER_CLASS; i++) {
            box_info[i].score = INPUT1_VAL_ZERO;
            box_info[i].batch_idx = 0;
            box_info[i].class_idx = 0;
            box_info[i].box_idx = 0;
        }
        selected_boxes_num[offset] = 0;
        return;
    }

    // --- Phase 3: Sort collected boxes by score (descending) using selection sort ---
    // O(K^2) on at most MAX_BOXES_PER_CLASS elements in global memory.
    for (int i = 0; i < valid_boxes_num - 1; i++) {
        int max_j = i;
        INPUT1_TYPE ms = box_info[i].score;
        int mbi = box_info[i].box_idx;
        for (int j = i + 1; j < valid_boxes_num; j++) {
            if (box_info[j].score > ms ||
                (box_info[j].score == ms && box_info[j].box_idx < mbi)) {
                ms = box_info[j].score;
                mbi = box_info[j].box_idx;
                max_j = j;
            }
        }
        if (max_j != i) {
            BOX_INFO temp = box_info[i];
            box_info[i] = box_info[max_j];
            box_info[max_j] = temp;
        }
    }

    // --- Phase 4: Compute IOU matrix and decay factors ---
    __global INPUT1_TYPE* iou_matrix = input_iou_matrix + offset * MAX_BOXES_PER_CLASS;
    __global INPUT1_TYPE* iou_max = input_iou_max + offset * MAX_BOXES_PER_CLASS;
    __global INPUT1_TYPE* min_decays = input_min_decays + offset * MAX_BOXES_PER_CLASS;

    iou_max[0] = INPUT1_VAL_ZERO;
    for (int i = 1; i < valid_boxes_num; ++i) {
        INPUT1_TYPE max_iou_val = INPUT1_VAL_ZERO;
        INPUT1_TYPE min_decay = INPUT1_VAL_ONE;
        const COORD_TYPE_4 box_i = FUNC_CALL(getBoxCoords)(input_boxes, batchId, box_info[i].box_idx);
        for (int j = 0; j < i; ++j) {
            const COORD_TYPE_4 box_j = FUNC_CALL(getBoxCoords)(input_boxes, batchId, box_info[j].box_idx);
            const INPUT1_TYPE iou = FUNC_CALL(intersectionOverUnion)(box_i, box_j);
            max_iou_val = max(iou, max_iou_val);
            iou_matrix[j] = iou;
        }
        iou_max[i] = max_iou_val;

        for (int j = 0; j < i; ++j) {
            INPUT1_TYPE decay =
                DECAY_FUNC == 0 ? FUNC_CALL(decay_gaussian)(iou_matrix[j], iou_max[j])
                                : FUNC_CALL(decay_linear)(iou_matrix[j], iou_max[j]);
            min_decay = min(min_decay, decay);
        }
        min_decays[i] = min_decay;
    }

    // --- Phase 5: Apply post-threshold filter and write final results ---
    const INPUT1_TYPE first_score = input_scores[INPUT1_GET_INDEX(batchId, classId, 0, box_info[0].box_idx)];
    int box_info_counter = 0;

    if (first_score > POST_THRESHOLD) {
        int first_box_idx = box_info[0].box_idx;
        box_info[0].batch_idx = batchId;
        box_info[0].class_idx = classId;
        box_info[0].box_idx = first_box_idx;
        box_info[0].score = first_score;
        box_info_counter = 1;
    }

    for (int i = 1; i < valid_boxes_num; ++i) {
        int cur_box_idx = box_info[i].box_idx;
        INPUT1_TYPE ds = min_decays[i] * input_scores[INPUT1_GET_INDEX(batchId, classId, 0, cur_box_idx)];

        if (ds <= POST_THRESHOLD)
            continue;

        box_info[box_info_counter].batch_idx = batchId;
        box_info[box_info_counter].class_idx = classId;
        box_info[box_info_counter].box_idx = cur_box_idx;
        box_info[box_info_counter].score = ds;
        ++box_info_counter;
    }

    // Zero out unused positions so Stage 1 partial sort terminates early
    for (int i = box_info_counter; i < MAX_BOXES_PER_CLASS; i++) {
        box_info[i].score = INPUT1_VAL_ZERO;
        box_info[i].batch_idx = 0;
        box_info[i].class_idx = 0;
        box_info[i].box_idx = 0;
    }

    selected_boxes_num[offset] = box_info_counter;
}
#endif /* MATRIX_NMS_STAGE_0 */

#ifdef MATRIX_NMS_STAGE_1
KERNEL(matrix_nms_ref_stage_1)
(__global OUTPUT2_TYPE* valid_outputs, __global uchar* buffer0, __global int* selected_boxes_num) {
    const int batchId = get_global_id(0);

    __global BOX_INFO* box_info = (__global BOX_INFO*)buffer0;

    const int first_idx = batchId * NUM_CLASSES * MAX_BOXES_PER_CLASS;
    const int last_idx = first_idx + NUM_CLASSES * MAX_BOXES_PER_CLASS;

    // Count total valid entries across all classes for this batch
    int total_valid = 0;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        if (i == BACKGROUND_CLASS)
            continue;
        total_valid += selected_boxes_num[batchId * NUM_CLASSES + i];
    }

    // Partial selection sort: find top-K entries from [first_idx, last_idx).
    // Uses O(N*K) instead of O(N^2) bubble sort to avoid GPU TDR timeout.
    const int n = last_idx - first_idx;
    int k = (KEEP_TOP_K > 0 && KEEP_TOP_K < total_valid) ? KEEP_TOP_K : total_valid;
    if (k > n) k = n;

    for (int i = 0; i < k; i++) {
        int max_j = first_idx + i;
        for (int j = first_idx + i + 1; j < last_idx; j++) {
            __global BOX_INFO* cur = &box_info[j];
            __global BOX_INFO* best = &box_info[max_j];
            if ((cur->score > best->score) ||
                (cur->score == best->score && cur->class_idx < best->class_idx) ||
                (cur->score == best->score && cur->class_idx == best->class_idx &&
                 cur->box_idx < best->box_idx)) {
                max_j = j;
            }
        }
        // If best remaining has score 0, no more valid entries
        if (box_info[max_j].score == INPUT1_VAL_ZERO)
            break;
        if (max_j != first_idx + i) {
            FUNC_CALL(swap_boxes)(&box_info[first_idx + i], &box_info[max_j]);
        }
    }

    valid_outputs[OUTPUT2_GET_INDEX(batchId, 0, 0, 0)] = total_valid;
}
#endif /* MATRIX_NMS_STAGE_1 */

#ifdef MATRIX_NMS_STAGE_2
KERNEL(matrix_nms_ref_stage_2)
(const __global INPUT0_TYPE* input_boxes,
 __global OUTPUT_TYPE* output,
 __global OUTPUT1_TYPE* selected_indices,
 __global OUTPUT2_TYPE* valid_outputs,
 __global uchar* buffer0) {
    __global BOX_INFO* box_info = (__global BOX_INFO*)buffer0;

    // Partial selection sort across batches if required.
    // Only sorts the needed top-K entries to avoid O(N^2) timeout.
#if SORT_RESULT_ACROSS_BATCH == 1 && SORT_TYPE != 2
    {
        const int total_size = NUM_BATCHES * NUM_CLASSES * MAX_BOXES_PER_CLASS;
        int total_valid = 0;
        for (int i = 0; i < NUM_BATCHES; ++i)
            total_valid += valid_outputs[OUTPUT2_GET_INDEX(i, 0, 0, 0)];
        int k = (KEEP_TOP_K > 0 && KEEP_TOP_K < total_valid) ? KEEP_TOP_K : total_valid;
        if (k > total_size) k = total_size;

        for (int i = 0; i < k; i++) {
            int max_j = i;
            for (int j = i + 1; j < total_size; j++) {
                __global BOX_INFO* cur = &box_info[j];
                __global BOX_INFO* best = &box_info[max_j];
#if SORT_TYPE == 1
                // Sort by score descending
                if ((cur->score > best->score) ||
                    (cur->score == best->score && cur->batch_idx < best->batch_idx) ||
                    (cur->score == best->score && cur->batch_idx == best->batch_idx &&
                     cur->class_idx < best->class_idx) ||
                    (cur->score == best->score && cur->batch_idx == best->batch_idx &&
                     cur->class_idx == best->class_idx && cur->box_idx < best->box_idx)) {
                    max_j = j;
                }
#elif SORT_TYPE == 0
                // Sort by class id
                if (cur->score != INPUT1_VAL_ZERO &&
                    ((best->score == INPUT1_VAL_ZERO) ||
                     (cur->class_idx < best->class_idx) ||
                     (cur->class_idx == best->class_idx && cur->batch_idx < best->batch_idx) ||
                     (cur->class_idx == best->class_idx && cur->batch_idx == best->batch_idx &&
                      cur->score > best->score) ||
                     (cur->class_idx == best->class_idx && cur->batch_idx == best->batch_idx &&
                      cur->score == best->score && cur->box_idx < best->box_idx))) {
                    max_j = j;
                }
#endif
            }
            if (box_info[max_j].score == INPUT1_VAL_ZERO)
                break;
            if (max_j != i) {
                FUNC_CALL(swap_boxes)(&box_info[i], &box_info[max_j]);
            }
        }
    }
#endif

    int output_idx = 0;
    int box_info_idx = 0;
    for (int i = 0; i < NUM_BATCHES; ++i) {
        if (KEEP_TOP_K != -1 && KEEP_TOP_K < valid_outputs[OUTPUT2_GET_INDEX(i, 0, 0, 0)])
            valid_outputs[OUTPUT2_GET_INDEX(i, 0, 0, 0)] = KEEP_TOP_K;

#if SORT_RESULT_ACROSS_BATCH == 0
        box_info_idx = i * NUM_CLASSES * MAX_BOXES_PER_CLASS;
#endif

        unroll_for(int j = 0; j < valid_outputs[OUTPUT2_GET_INDEX(i, 0, 0, 0)]; ++j) {
            output[OUTPUT_GET_INDEX(output_idx, 0, 0, 0)] = box_info[box_info_idx].class_idx;
            output[OUTPUT_GET_INDEX(output_idx, 1, 0, 0)] = box_info[box_info_idx].score;
            output[OUTPUT_GET_INDEX(output_idx, 2, 0, 0)] =
                input_boxes[INPUT0_GET_INDEX(box_info[box_info_idx].batch_idx, box_info[box_info_idx].box_idx, 0, 0)];
            output[OUTPUT_GET_INDEX(output_idx, 3, 0, 0)] =
                input_boxes[INPUT0_GET_INDEX(box_info[box_info_idx].batch_idx, box_info[box_info_idx].box_idx, 0, 1)];
            output[OUTPUT_GET_INDEX(output_idx, 4, 0, 0)] =
                input_boxes[INPUT0_GET_INDEX(box_info[box_info_idx].batch_idx, box_info[box_info_idx].box_idx, 0, 2)];
            output[OUTPUT_GET_INDEX(output_idx, 5, 0, 0)] =
                input_boxes[INPUT0_GET_INDEX(box_info[box_info_idx].batch_idx, box_info[box_info_idx].box_idx, 0, 3)];

            selected_indices[OUTPUT1_GET_INDEX(output_idx, 0, 0, 0)] =
                box_info[box_info_idx].batch_idx * NUM_BOXES + box_info[box_info_idx].box_idx;

            ++output_idx;
            ++box_info_idx;
        }

        // Paddings
        while (output_idx < (i + 1) * MAX_BOXES_PER_BATCH) {
            unroll_for(int j = 0; j < 6; ++j) {
                output[OUTPUT_GET_INDEX(output_idx, j, 0, 0)] = -OUTPUT_VAL_ONE;
            }
            selected_indices[OUTPUT1_GET_INDEX(output_idx, 0, 0, 0)] = -OUTPUT1_VAL_ONE;
            ++output_idx;
        }
    }
}
#endif /* MATRIX_NMS_STAGE_2 */

#undef NUM_BATCHES
#undef NUM_BOXES
#undef NUM_CLASSES
#undef BOX_INFO
