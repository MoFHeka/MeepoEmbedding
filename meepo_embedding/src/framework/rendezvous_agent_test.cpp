
#include <gtest/gtest.h>
#include "meepo_embedding/include/framework/rendezvous_agent.hpp"

using namespace meepo_embedding;

TEST(RendezvousImplTest, CorrectInitializationAndRank) {
  std::vector<RankInfo> rank_infos = {{"192.168.1.1", 8080}, {"192.168.1.2", 8081}};
  auto rdv = Rendezvous::create(1, 2, rank_infos, 0);
  EXPECT_EQ(0, rdv->getRank());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
