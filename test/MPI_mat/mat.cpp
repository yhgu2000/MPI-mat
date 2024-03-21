#include "testutil.hpp"

#include "MPI_mat/mat.hpp"

using namespace MPI_mat;
namespace but = boost::unit_test_framework;

BOOST_AUTO_TEST_CASE(Mat_1)
{
  Mat a(1, 1);
  a(0, 0) = 1;
  Mat b(std::move(a));
  a.resize(1, 1);
  a.zero();

  Mat c;
  BOOST_TEST(!c);
  c = a.dot(b);
  BOOST_TEST(c);

  c = std::move(c);
  BOOST_TEST(c.size() == 1);
  BOOST_TEST(c[0] == 0);

  c = a + b;
  b += b;
  a += b;
  BOOST_TEST(c[0] == 1);
  BOOST_TEST(b[0] == 2);
  BOOST_TEST(a == b);

  c = a * b;
  BOOST_TEST(c[0] == 4);
}

BOOST_AUTO_TEST_CASE(Mat_2)
{
  Mat a(2, 2);
  a.eye();
  a *= a;
  Mat b = std::move(a);
  a.resize(2, 2);
  a.fill(1);
  BOOST_TEST(a != b);

  auto c = a.dot(b);
  BOOST_TEST(c == a);
  BOOST_TEST(a.checksum() == c.checksum());
  BOOST_TEST(a.sum() == c.sum());
}

BOOST_AUTO_TEST_CASE(Mat_3, *but::tolerance(0.00001))
{
  Mat a(100, 100);
  a.rand();
  auto b = a;
  BOOST_TEST(a == b);

  auto c = a.dot(b);
  BOOST_TEST(c == b.dot(b));

  a.fill(1.0 / 100);
  BOOST_TEST(a.sum() == 100);
  for (int i = 0; i < 10; ++i)
    a = a.dot(a);
  BOOST_TEST(a.sum() == 100);
}

BOOST_AUTO_TEST_CASE(MatFile_1)
{
  MatFile af(MPI_mat_BINARY_DIR "/test/MPI_mat/1.mat", 0, 0);
  Mat a(10, 10);
  a.rand();
  af.dump(a);

  Mat b;
  af.load(b);
  BOOST_TEST(a == b);
}

BOOST_AUTO_TEST_CASE(MatFile_2)
{
  MatFile af(MPI_mat_BINARY_DIR "/test/MPI_mat/2.mat", 0, 0);
  Mat a(100, 100);
  a.rand();
  af.dump(a);

  MatFile bf(MPI_mat_BINARY_DIR "/test/MPI_mat/2.mat");
  bf.load_head();
  Mat b(50, 100);

  bf.load(0, b);
  BOOST_TEST(b == a(0, 50, 0, 100));
  bf.load(50, b);
  bf.dump(50, b);
  bf.load(50, b);
  BOOST_TEST(b == a(50, 100, 0, 100));

  b.resize(50, 50);
  bf.load(50, 50, b);
  BOOST_TEST(b == a(50, 100, 50, 100));

  bf.dump_head();
  bf.dump(50, 50, b);
  bf.load(50, 50, b);
  BOOST_TEST(b == a(50, 100, 50, 100));
}
