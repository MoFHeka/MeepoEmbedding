/* Copyright 2015 The MeepoEmbedding Authors. All Rights Reserved.
Copyright 2024 The MeepoEmbedding Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#ifndef MEEPOEMBEDDING_RENDEZVOUS_AGENT_HPP
#define MEEPOEMBEDDING_RENDEZVOUS_AGENT_HPP

#include <vector>
#include <string>
#include <memory>
#include <yaml-cpp/yaml.h>

namespace meepo_embedding {
struct RankInfo {
    std::string ip;
    int port;
};

struct Topology {
  int num_nodes;
  int gpu_per_node;
  float bandwidth_within_node;
  float bandwidth_between_nodes;
};


class RendezvousAgent {
public:
    virtual RendezvousAgent* create(const YAML::Node* config, int rank) = 0;
    virtual int getRank() const = 0;
    virtual ~Rendezvous() {}
};

} // namespace meepo_embedding
#endif //MEEPOEMBEDDING_RENDEZVOUS_AGENT_HPP
