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

    // For the background class we still need to clear this class's slice of
    // box_info so the Stage 1 partial sort sees score==0 in those slots and
    // can early-exit. Without this, the slice retains stale data from a
    // previous inference (internal buffers are reused, not zero-initialized)
    // and ranked-by-score sorting may pick garbage entries first.
    if (classId == BACKGROUND_CLASS) {
        __global BOX_INFO* bg_info = (__global BOX_INFO*)buffer0;
        bg_info = &bg_info[batchId * NUM_CLASSES * MAX_BOXES_PER_CLASS + classId * MAX_BOXES_PER_CLASS];
        for (int i = 0; i < MAX_BOXES_PER_CLASS; ++i) {
            bg_info[i].batch_idx = 0;
            bg_info[i].class_idx = 0;
            bg_info[i].box_idx = 0;
            bg_info[i].score = INPUT1_VAL_ZERO;
        }
        selected_boxes_num[batchId * NUM_CLASSES + classId] = 0;
        return;
    }

    const int offset = batchId * NUM_CLASSES + classId;

    // Bounded top-K candidate selection (single pass over NUM_BOXES).
    // Maintains sorted_score_indices[0..valid_boxes_num-1] sorted by score (desc),
    // mirrored by sorted_scores[] cached in private memory so comparisons inside the
    // insertion loop never touch global memory.
    // Output is bit-identical to the original "collect-all + bubble sort by score
    // desc (stable) + cap to MAX_BOXES_PER_CLASS" behavior: ties keep input order
    // because the strict `<` comparison stops the shift on equality, so a later
    // candidate with the same score lands at a higher index than the earlier one.
    //
    // Replaces three issues in the previous implementation:
    //   (1) int sorted_score_indices[NUM_BOXES] declared as private memory
    //       (e.g. 22743 * 4B = 91KB per work-item) caused clCreateKernel to fail
    //       with CL_OUT_OF_RESOURCES on devices with tight private memory budgets;
    //   (2) sortIterative() ran an O(N^2) bubble sort over all above-threshold
    //       boxes BEFORE the cap, causing Windows GPU TDR for large NUM_BOXES;
    //   (3) the iou_matrix/iou_max/min_decays typed pointers were over-strided by
    //       sizeof(INPUT1_TYPE), corrupting buffers for class offsets > 0.
    //
    // Complexity: O(NUM_BOXES * MAX_BOXES_PER_CLASS) with private-memory ops only.
    int sorted_score_indices[MAX_BOXES_PER_CLASS];
    INPUT1_TYPE sorted_scores[MAX_BOXES_PER_CLASS];
    sorted_score_indices[0] = 0;   // safe default when no candidate passes the threshold
    sorted_scores[0] = INPUT1_VAL_ZERO;
    int valid_boxes_num = 0;

    for (int idx = 0; idx < NUM_BOXES; ++idx) {
        const INPUT1_TYPE score = input_scores[INPUT1_GET_INDEX(batchId, classId, 0, idx)];
        if (score <= SCORE_THRESHOLD)
            continue;

        int pos;
        if (valid_boxes_num < MAX_BOXES_PER_CLASS) {
            pos = valid_boxes_num;
            ++valid_boxes_num;
        } else {
            // Heap is full; new candidate must beat the current minimum to be kept.
            if (score <= sorted_scores[MAX_BOXES_PER_CLASS - 1])
                continue;
            pos = MAX_BOXES_PER_CLASS - 1;
        }

        // Insertion sort shift using private-memory comparisons only.
        while (pos > 0 && sorted_scores[pos - 1] < score) {
            sorted_score_indices[pos] = sorted_score_indices[pos - 1];
            sorted_scores[pos] = sorted_scores[pos - 1];
            --pos;
        }
        sorted_score_indices[pos] = idx;
        sorted_scores[pos] = score;
    }

    // Typed-pointer arithmetic strides by sizeof(element) automatically;
    // the previous `+ offset * MAX_BOXES_PER_CLASS * sizeof(INPUT1_TYPE)`
    // over-multiplied by sizeof(), causing out-of-bounds writes for offset > 0.
    __global INPUT1_TYPE* iou_matrix = input_iou_matrix + offset * MAX_BOXES_PER_CLASS;
    __global INPUT1_TYPE* iou_max = input_iou_max + offset * MAX_BOXES_PER_CLASS;
    __global INPUT1_TYPE* min_decays = input_min_decays + offset * MAX_BOXES_PER_CLASS;

    iou_max[0] = INPUT1_VAL_ZERO;
    for (int i = 1; i < valid_boxes_num; ++i) {
        INPUT1_TYPE max_iou = INPUT1_VAL_ZERO;
        INPUT1_TYPE min_decay = INPUT1_VAL_ONE;
        const COORD_TYPE_4 box_i = FUNC_CALL(getBoxCoords)(input_boxes, batchId, sorted_score_indices[i]);
        for (int j = 0; j < i; ++j) {
            const COORD_TYPE_4 box_j = FUNC_CALL(getBoxCoords)(input_boxes, batchId, sorted_score_indices[j]);
            const INPUT1_TYPE iou = FUNC_CALL(intersectionOverUnion)(box_i, box_j);

            max_iou = max(iou, max_iou);
            iou_matrix[j] = iou;
        }
        iou_max[i] = max_iou;

        for (int j = 0; j < i; ++j) {
            INPUT1_TYPE decay =
                DECAY_FUNC == 0 ? FUNC_CALL(decay_gaussian)(iou_matrix[j], iou_max[j]) : FUNC_CALL(decay_linear)(iou_matrix[j], iou_max[j]);
            min_decay = min(min_decay, decay);
        }
        min_decays[i] = min_decay;
    }

    const INPUT1_TYPE first_score = input_scores[INPUT1_GET_INDEX(batchId, classId, 0, sorted_score_indices[0])];

    __global BOX_INFO* box_info = (__global BOX_INFO*)buffer0;
    box_info = &box_info[batchId * NUM_CLASSES * MAX_BOXES_PER_CLASS + classId * MAX_BOXES_PER_CLASS];

    int box_info_counter = 0;
    if (first_score > POST_THRESHOLD && valid_boxes_num > 0) {
        box_info[box_info_counter].class_idx = classId;
        box_info[box_info_counter].score = first_score;
        box_info[box_info_counter].box_idx = sorted_score_indices[0];
        box_info[box_info_counter].batch_idx = batchId;
        ++box_info_counter;
    }

    for (int i = 1; i < valid_boxes_num; ++i) {
        INPUT1_TYPE ds = min_decays[i] * input_scores[INPUT1_GET_INDEX(batchId, classId, 0, sorted_score_indices[i])];

        if (ds <= POST_THRESHOLD)
            continue;

        box_info[box_info_counter].batch_idx = batchId;
        box_info[box_info_counter].class_idx = classId;
        box_info[box_info_counter].box_idx = sorted_score_indices[i];
        box_info[box_info_counter].score = ds;
        ++box_info_counter;
    }

    // Zero out the unused tail of this class's slice so the Stage 1 partial sort
    // can terminate early once all valid (score>0) entries have been selected.
    // Without this, the buffer contains uninitialized data from the previous
    // inference, which prevents the early-exit and causes a Windows GPU TDR.
    for (int i = box_info_counter; i < MAX_BOXES_PER_CLASS; ++i) {
        box_info[i].batch_idx = 0;
        box_info[i].class_idx = 0;
        box_info[i].box_idx = 0;
        box_info[i].score = INPUT1_VAL_ZERO;
    }

    selected_boxes_num[batchId * NUM_CLASSES + classId] = box_info_counter;
}
#endif /* MATRIX_NMS_STAGE_0 */

#ifdef MATRIX_NMS_STAGE_1
KERNEL(matrix_nms_ref_stage_1)
(__global OUTPUT2_TYPE* valid_outputs, __global uchar* buffer0, __global int* selected_boxes_num) {
    const int batchId = get_global_id(0);

    __global BOX_INFO* box_info = (__global BOX_INFO*)buffer0;

    const int first_idx = batchId * NUM_CLASSES * MAX_BOXES_PER_CLASS;
    const int last_idx = first_idx + NUM_CLASSES * MAX_BOXES_PER_CLASS;

    // Tally per-class valid counts and derive how many entries we actually need
    // to sort. The previous O(N^2) bubble sort over NUM_CLASSES*MAX_BOXES_PER_CLASS
    // elements (e.g. 80*100 = 8000 for pp-yolo) hit the Windows GPU TDR limit
    // when the buffer tail contained unsorted padding entries.
    int total_valid = 0;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        if (i == BACKGROUND_CLASS)
            continue;
        total_valid += selected_boxes_num[batchId * NUM_CLASSES + i];
    }

    // Partial selection sort: only extract the top-K entries by score, then stop.
    // K is capped by KEEP_TOP_K when provided, otherwise by the total valid count.
    // Stage 0 zeroes out unused per-class slots, so an entry with score==0 means
    // the remaining tail has no more valid items and we can early-exit. This
    // preserves bit-identical ordering with the original full bubble sort.
    int k = total_valid;
    if (KEEP_TOP_K > 0 && KEEP_TOP_K < k)
        k = KEEP_TOP_K;

    for (int i = 0; i < k; ++i) {
        int best = first_idx + i;
        for (int j = first_idx + i + 1; j < last_idx; ++j) {
            __global BOX_INFO* c = &box_info[j];
            __global BOX_INFO* b = &box_info[best];
            if ((c->score > b->score) ||
                (c->score == b->score && c->class_idx < b->class_idx) ||
                (c->score == b->score && c->class_idx == b->class_idx &&
                 c->box_idx < b->box_idx)) {
                best = j;
            }
        }
        if (box_info[best].score == INPUT1_VAL_ZERO)
            break;
        if (best != first_idx + i)
            FUNC_CALL(swap_boxes)(&box_info[first_idx + i], &box_info[best]);
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

    // Cross-batch sort, only required when SORT_RESULT_ACROSS_BATCH is set.
    // Uses the same partial selection sort approach as Stage 1: bounded by
    // KEEP_TOP_K across batches and exits on the first score==0 entry to avoid
    // the O(N^2) bubble sort timeout that the original implementation had.
#if SORT_RESULT_ACROSS_BATCH == 1 && SORT_TYPE != 2
    {
        const int total_size = NUM_BATCHES * NUM_CLASSES * MAX_BOXES_PER_CLASS;
        int total_valid = 0;
        for (int i = 0; i < NUM_BATCHES; ++i)
            total_valid += valid_outputs[OUTPUT2_GET_INDEX(i, 0, 0, 0)];

        int k = total_valid;
        if (KEEP_TOP_K > 0 && KEEP_TOP_K < k)
            k = KEEP_TOP_K;
        if (k > total_size)
            k = total_size;

        for (int i = 0; i < k; ++i) {
            int best = i;
            for (int j = i + 1; j < total_size; ++j) {
                __global BOX_INFO* c = &box_info[j];
                __global BOX_INFO* b = &box_info[best];
#if SORT_TYPE == 1
                // Sort by score descending, stable on (batch_idx, class_idx, box_idx).
                if ((c->score > b->score) ||
                    (c->score == b->score && c->batch_idx < b->batch_idx) ||
                    (c->score == b->score && c->batch_idx == b->batch_idx &&
                     c->class_idx < b->class_idx) ||
                    (c->score == b->score && c->batch_idx == b->batch_idx &&
                     c->class_idx == b->class_idx && c->box_idx < b->box_idx)) {
                    best = j;
                }
#elif SORT_TYPE == 0
                // Sort by class id ascending; empty slots (score==0) are last.
                if (c->score != INPUT1_VAL_ZERO &&
                    ((b->score == INPUT1_VAL_ZERO) ||
                     (c->class_idx < b->class_idx) ||
                     (c->class_idx == b->class_idx && c->batch_idx < b->batch_idx) ||
                     (c->class_idx == b->class_idx && c->batch_idx == b->batch_idx &&
                      c->score > b->score) ||
                     (c->class_idx == b->class_idx && c->batch_idx == b->batch_idx &&
                      c->score == b->score && c->box_idx < b->box_idx))) {
                    best = j;
                }
#endif
            }
            if (box_info[best].score == INPUT1_VAL_ZERO)
                break;
            if (best != i)
                FUNC_CALL(swap_boxes)(&box_info[i], &box_info[best]);
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
