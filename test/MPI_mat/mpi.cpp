#include "testutil.hpp"

#include "MPI_mat/mpi.hpp"

using namespace MPI_mat;
namespace but = boost::unit_test_framework;

BOOST_AUTO_TEST_CASE(_1, *but::tolerance(0.00001))
{
  mpi::World world;

  MatFile af(MPI_mat_BINARY_DIR "/test/MPI_mat/a.mat", 0, 0);
  Mat m(100, 100);
  m.fill(1 / 100.0);
  af.dump(m);

  MatFile bf(
    MPI_mat_BINARY_DIR "/test/MPI_mat/b.mat", 100, 100, MatFile::CreateOnly());

  mpi::dot_direct_load(af, af, bf);
  bf.load(m);
  BOOST_TEST(m.sum() == 100);

  mpi::dot_grid_bcast(af, af, bf);
  bf.load(m);
  BOOST_TEST(m.sum() == 100);

  mpi::dot_row_bcast(af, af, bf);
  bf.load(m);
  BOOST_TEST(m.sum() == 100);

  mpi::dot_cannon(af, af, bf);
  bf.load(m);
  BOOST_TEST(m.sum() == 100);

  mpi::dot_dns(af, af, bf, 1);
  bf.load(m);
  BOOST_TEST(m.sum() == 100);
}
