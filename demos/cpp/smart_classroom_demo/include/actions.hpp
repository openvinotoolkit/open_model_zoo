// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

/**
* @brief Class for action info
*/
using Action = int;

/**
* @brief Class for events on a single frame with action info
*/
struct FrameEvent {
    /** @brief Frame index */
    int frame_id;
    /** @brief Action label */
    Action action;

    /**
  * @brief Constructor
  */
    FrameEvent(int frame_id, Action action)
        : frame_id(frame_id), action(action) {}
};
using FrameEventsTrack = std::vector<FrameEvent>;

/**
* @brief Class for range of the same event with action info
*/
struct RangeEvent {
    /** @brief  Start frame index */
    int begin_frame_id;
    /** @brief  Next after the last valid frame index */
    int end_frame_id;
    /** @brief Action label */
    Action action;

    /**
  * @brief Constructor
  */
    RangeEvent(int begin_frame_id, int end_frame_id, Action action)
        : begin_frame_id(begin_frame_id), end_frame_id(end_frame_id), action(action) {}
};
using RangeEventsTrack = std::vector<RangeEvent>;

enum ActionsType { STUDENT, TEACHER, TOP_K };
