
#include <gtest/gtest.h>
#include "meepo_embedding/include/framework/rendezvous_agent.hpp"

#include <yaml-cpp/yaml.h>

using namespace meepo_embedding;

TEST(RendezvousAgentTest, CorrectInitializationAndRank) {
  YAML::Node config = YAML::Load(R"(
        rendezvous:
          tensor_parallel: 1
          data_parallel: 2
          ranks:
            - ip: "192.168.1.1"
              port: 8080
            - ip: "192.168.1.2"
              port: 8081
          topology:
            num_nodes: 1
            gpu_per_node: 2
            bandwidth_within_node: 12.5
            bandwidth_between_nodes: 6.2
    )");

  const YAML::Node* rendezvous_config = &config["rendezvous"];

  auto rdv = RendezvousAgent::create(rendezvous_config, 0);
  EXPECT_EQ(0, rdv->getRank());
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}