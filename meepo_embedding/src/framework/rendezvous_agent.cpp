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

#include "meepo_embedding/include/framework/rendezvous_agent.hpp"
#include "meepo_embedding/include/common/logger.h
"
#include <stdexcept>
#include <iostream>
#include <yaml-cpp/yaml.h>


namespace meepo_embedding {
class RendezvousAgentImpl : public RendezvousAgent {
private:
    int tensor_parallel;
    int data_parallel;
    std::vector<RankInfo> ranks;
    Topology topology;
    int rank;

public:
    RendezvousImpl(const YAML::Node* config, int rank): rank(rank) {
      tensor_parallel = config["tensor_parallel"].as<int>();
      data_parallel = config["data_parallel"].as<int>();

      for (const auto& rank_node : config["ranks"]) {
        ranks.push_back({rank_node["ip"].as<std::string>(), rank_node["port"].as<int>()});
      }

      topology.num_nodes = config["topology"]["num_nodes"].as<int>();
      topology.gpu_per_node = config["topology"]["gpu_per_node"].as<int>();
      topology.bandwidth_within_node = config["topology"]["bandwidth_within_node"].as<float>();
      topology.bandwidth_between_nodes = config["topology"]["bandwidth_between_nodes"].as<float>();

      if (tensor_parallel * data_parallel != nodes.size()) {
          throw std::invalid_argument("Tensor parallel * Data parallel must equal the length of nodes.");
      }
    }
    RendezvousAgent* create(const YAML::Node* config, int rank) override {
      return new RendezvousAgentImpl(config, rank);
    }
    int getRank() const override {
        return rank;
    }
};

std::unique_ptr<RendezvousAgent> RendezvousAgent::create(const YAML::Node* config, int rank) {
  return std::make_unique<RendezvousAgentImpl>(config, rank);
}
}


