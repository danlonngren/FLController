
#include "FLController.h"

#include <gtest/gtest.h>

class FLControllerTests : public ::testing::Test {
protected:
    void SetUp() override {
    }
    void TearDown() override {
    }
};

TEST_F(FLControllerTests, FLControllerResetTest) {
    FLController controller(1.0f, 1.0f, 1.0f, 1.0f, 100.0f);
    controller.reset();
    
    float output = controller.evaluate(0.0f, 0.0f, false);
    
    EXPECT_FLOAT_EQ(output, 0.0f);
}
