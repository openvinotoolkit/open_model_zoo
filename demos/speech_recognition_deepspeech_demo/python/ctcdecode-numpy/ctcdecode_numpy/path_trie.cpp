/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on path_trie.cpp from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#include "path_trie.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "decoder_utils.h"

PathTrie::PathTrie() {
  log_prob_b_prev = -NUM_FLT_INF;
  log_prob_nb_prev = -NUM_FLT_INF;
  log_prob_b_cur = -NUM_FLT_INF;
  log_prob_nb_cur = -NUM_FLT_INF;
  log_prob_c = -NUM_FLT_INF;
  score = -NUM_FLT_INF;
  approx_ctc = -NUM_FLT_INF;

  ROOT_ = -1;
  character = ROOT_;
  timestep = 0;
  exists_ = true;
  parent = nullptr;

  dictionary_ = nullptr;
  has_dictionary_ = false;
  dictionary_state_ = {};
}

PathTrie::~PathTrie() {
  for (auto child : children_) {
    delete child.second;
  }
}

PathTrie* PathTrie::get_path_trie(int new_char, int new_timestep, float cur_log_prob_c, bool reset) {
  auto child = children_.begin();
  for (child = children_.begin(); child != children_.end(); ++child) {
    if (child->first == new_char) {
      if (child->second->log_prob_c < cur_log_prob_c) {
	child->second->log_prob_c = cur_log_prob_c;
	child->second->timestep = new_timestep;
      }
      break;
    }
  }
  if (child != children_.end()) {
    if (!child->second->exists_) {
      child->second->exists_ = true;
      child->second->log_prob_b_prev = -NUM_FLT_INF;
      child->second->log_prob_nb_prev = -NUM_FLT_INF;
      child->second->log_prob_b_cur = -NUM_FLT_INF;
      child->second->log_prob_nb_cur = -NUM_FLT_INF;
    }
    return (child->second);
  } else {
    if (has_dictionary_) {
      WordPrefixSetState new_state = dictionary_state_;
      bool found = dictionary_->append_character(new_char + 1, new_state);
      if (!found) {
        // Adding this character causes word outside dictionary
        bool is_final = dictionary_state_.weight;
        if (is_final && reset) {
          dictionary_state_ = dictionary_->empty_state();
        }
        return nullptr;
      } else {
        PathTrie* new_path = new PathTrie;
        new_path->character = new_char;
        new_path->timestep = new_timestep;
        new_path->parent = this;
        new_path->dictionary_ = dictionary_;
        new_path->has_dictionary_ = true;
	new_path->log_prob_c = cur_log_prob_c;

        // set spell checker state
        // check to see if next state is final
        bool is_final = new_state.weight;
        if (is_final && reset) {
	  // restart spell checker at the start state
          new_path->dictionary_state_ = dictionary_->empty_state();
        } else {
	  // go to next state
          new_path->dictionary_state_ = new_state;
        }

        children_.push_back(std::make_pair(new_char, new_path));
        return new_path;
      }
    } else {
      PathTrie* new_path = new PathTrie;
      new_path->character = new_char;
      new_path->timestep = new_timestep;
      new_path->parent = this;
      new_path->log_prob_c = cur_log_prob_c;
      children_.push_back(std::make_pair(new_char, new_path));
      return new_path;
    }
  }
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output, std::vector<int>& timesteps) {
  return get_path_vec(output, timesteps, ROOT_);
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output,
                                 std::vector<int>& timesteps,
                                 int stop,
                                 size_t max_steps) {
  if (character == stop || character == ROOT_ || output.size() == max_steps) {
    std::reverse(output.begin(), output.end());
    std::reverse(timesteps.begin(), timesteps.end());
    return this;
  } else {
    output.push_back(character);
    timesteps.push_back(timestep);
    return parent->get_path_vec(output, timesteps, stop, max_steps);
  }
}

void PathTrie::iterate_to_vec(std::vector<PathTrie*>& output) {
  if (exists_) {
    log_prob_b_prev = log_prob_b_cur;
    log_prob_nb_prev = log_prob_nb_cur;

    log_prob_b_cur = -NUM_FLT_INF;
    log_prob_nb_cur = -NUM_FLT_INF;

    score = log_sum_exp(log_prob_b_prev, log_prob_nb_prev);
    output.push_back(this);
  }
  for (auto child : children_) {
    child.second->iterate_to_vec(output);
  }
}

void PathTrie::remove() {
  exists_ = false;

  if (children_.size() == 0) {
    auto child = parent->children_.begin();
    for (child = parent->children_.begin(); child != parent->children_.end();
         ++child) {
      if (child->first == character) {
        parent->children_.erase(child);
        break;
      }
    }

    if (parent->children_.size() == 0 && !parent->exists_) {
      parent->remove();
    }

    delete this;
  }
}

void PathTrie::set_dictionary(WordPrefixSet* dictionary) {
  dictionary_ = dictionary;
  dictionary_state_ = dictionary->empty_state();
  has_dictionary_ = true;
}
