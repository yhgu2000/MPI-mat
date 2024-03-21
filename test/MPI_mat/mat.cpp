#include "testutil.hpp"

#include "MPI_mat/mat.hpp"

using namespace MPI_mat;

BOOST_AUTO_TEST_CASE(Mat_1)
{
  Mat a(1, 1), b(1, 1);
  a.zero();
  b(0, 0) = 1;

  auto c = a.dot(b);
  BOOST_TEST(c.size() == 1);
  BOOST_TEST(c[0] == 0);

  c += b;
  b += b;
  a += b;
  BOOST_TEST(c[0] == 1);
  BOOST_TEST(b[0] == 2);
  BOOST_TEST(a == b);

  c = a * b;
  BOOST_TEST(c[0] == 4);

  a.resize(2, 1), b.resize(1, 2);
  a.fill(1), b.fill(2);
  c = a.dot(b);
  BOOST_TEST(c.size() == 4);
  BOOST_TEST(c.sum() == 8);

  a = c;
  a.eye();
  b = a * c;
  BOOST_TEST(a != c);
  BOOST_TEST(b == c);
}

BOOST_AUTO_TEST_CASE(MatFile_1) {}
