/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpRemainderOutTest : public OperatorTest {
 protected:
  Tensor& op_remainder_tensor_out(
      const Tensor& self,
      const Tensor& other,
      Tensor& out) {
    return torch::executor::aten::remainder_outf(context_, self, other, out);
  }

  Tensor& op_remainder_scalar_out(
      const Tensor& self,
      const Scalar& other,
      Tensor& out) {
    return torch::executor::aten::remainder_outf(context_, self, other, out);
  }
};

TEST_F(OpRemainderOutTest, SmokeTest) {
  TensorFactory<ScalarType::Long> tfDouble;
  TensorFactory<ScalarType::Long> tfLong;
  TensorFactory<ScalarType::Int> tfInt;

  Tensor self = tfLong.full({2, 2}, 46);
  Tensor other = tfInt.full({2, 2}, 4);
  Tensor out = tfDouble.zeros({2, 2});
  Tensor out_expected = tfDouble.full({2, 2}, 2.0);
  op_remainder_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpRemainderOutTest, IntegerResultHasSignOfDivisor) {
  TensorFactory<ScalarType::Long> tf;

  Tensor self = tf.make({6}, {7, -7, 7, -7, 6, -6});
  Tensor other = tf.make({6}, {3, 3, -3, -3, 3, -3});
  Tensor out = tf.zeros({6});
  op_remainder_tensor_out(self, other, out);
  EXPECT_TENSOR_EQ(out, tf.make({6}, {1, 2, -2, -1, 0, 0}));

  Tensor scalar_out = tf.zeros({6});
  op_remainder_scalar_out(self, 3, scalar_out);
  EXPECT_TENSOR_EQ(scalar_out, tf.make({6}, {1, 2, 1, 2, 0, 0}));
}

TEST_F(OpRemainderOutTest, DoubleKeepsDoublePrecision) {
  TensorFactory<ScalarType::Double> tf;

  Tensor self = tf.make({2}, {123456789.123456789, -123456789.123456789});
  Tensor other = tf.make({2}, {1000.0, 1000.0});
  Tensor out = tf.zeros({2});
  op_remainder_tensor_out(self, other, out);
  // Rounding to float would be off by about 1.7e-5
  EXPECT_TENSOR_CLOSE_WITH_TOL(
      out, tf.make({2}, {789.123456789, 210.876543211}), 0, 1e-6);
}

TEST_F(OpRemainderOutTest, IntMinRemainderNegOneIsZero) {
  // INT_MIN % -1 overflows because the quotient is not representable;
  // ATen returns 0 instead of computing it.
  TensorFactory<ScalarType::Int> tfInt;
  TensorFactory<ScalarType::Long> tfLong;

  Tensor self_int = tfInt.make({2}, {std::numeric_limits<int32_t>::min(), -7});
  Tensor other_int = tfInt.make({2}, {-1, -1});
  Tensor out_int = tfInt.zeros({2});
  op_remainder_tensor_out(self_int, other_int, out_int);
  EXPECT_TENSOR_EQ(out_int, tfInt.make({2}, {0, 0}));

  Tensor self_long = tfLong.make({1}, {std::numeric_limits<int64_t>::min()});
  Tensor out_long = tfLong.zeros({1});
  op_remainder_scalar_out(self_long, -1, out_long);
  EXPECT_TENSOR_EQ(out_long, tfLong.make({1}, {0}));
}
