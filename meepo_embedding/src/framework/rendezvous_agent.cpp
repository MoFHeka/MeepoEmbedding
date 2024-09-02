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
#include <iostream>
#include <stdexcept>

  namespace meepo_embedding {
  class RendezvousAgentImpl : public RendezvousAgent {
   private:
    int tensor_parallel;
    int data_parallel;
    std::vector<RankInfo> rank_infos;
    int rank;

   public:
    RendezvousImpl(int tensor_parallel,
                   int data_parallel,
                   const std::vector<RankInfo>& rank_infos,
                   int rank)
      : tensor_parallel(tensor_parallel),
        data_parallel(data_parallel),
        rank_infos(rank_infos),
        rank(rank) {
      if (tensor_parallel * data_parallel != rank_infos.size()) {
        throw std::invalid_argument(
          "Tensor parallel * Data parallel must equal the length of "
          "rank_infos.");
      }
    }
    RendezvousAgent* create(int tensor_parallel,
                            int data_parallel,
                            const std::vector<RankInfo>& rank_infos,
                            int rank) override {
      return new RendezvousAgentImpl(
        tensor_parallel, data_parallel, rank_infos, rank);
    }
    int getRank() const override { return rank; }
  };

  std::unique_ptr<Rendezvous> Rendezvous::create(
    int tensor_parallel,
    int data_parallel,
    const std::vector<RankInfo>& rank_infos,
    int rank) {
    return std::make_unique<RendezvousImpl>(
      tensor_parallel, data_parallel, rank_infos, rank);
  }
}
