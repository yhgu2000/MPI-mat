#include "mpi.hpp"

#include <cassert>
#include <omp.h>
#include <sstream>

#define world (*::MPI_mat::mpi::World::g())

namespace err = My::err;

namespace MPI_mat::mpi {

std::string
Err::info() const noexcept
{
  std::string str(MPI_MAX_ERROR_STRING, '\0');
  int len = 0;
  if (MPI_Error_string(mCode, str.data(), &len) == MPI_SUCCESS)
    str.resize(len);
  else
    str = "MPI_Error_string failed!";
  return str;
}

void
timing_noting(const char* tag, const HRC::duration& dura)
{
}

TimingFunc gTiming = &timing_noting;

void
gen_rand(const MatFile& matFile)
{
  auto prow = matFile.mRown * world.mRank / world.mSize;
  auto prown = matFile.mRown * (world.mRank + 1) / world.mSize - prow;

  Mat mat(prown, matFile.mColn);
  mat.rand();
  matFile.dump(prow, mat);

  if (world.mRank == 0)
    matFile.dump_head();
}

void
dot_direct_load(const MatFile& a, const MatFile& b, MatFile& c)
{
  assert((a.mRown == a.mColn && b.mRown == b.mColn) &&
         "must be square matrix!");
  assert((a.mColn == b.mRown) && "a.mColn != b.mRown!");

  auto n = int(std::sqrt(world.mSize));
  assert((n * n == world.mSize) &&
         "MPI processe amount must be a square number!");

  auto y = world.mRank / n, x = world.mRank % n;

  c.mRown = a.mRown, c.mColn = b.mColn;

  auto prow = c.mRown * y / n;
  auto pcol = c.mColn * x / n;
  auto prown = c.mRown * (y + 1) / n - prow;
  auto pcoln = c.mColn * (x + 1) / n - pcol;

  auto __startLoad = HRC::now();
  Mat pa(prown, a.mColn), pb(b.mRown, pcoln);
  a.load(prow, 0, pa);
  b.load(0, pcol, pb);
  auto __finishLoad = HRC::now();
  gTiming("load", __finishLoad - __startLoad);

  auto __startCalc = HRC::now();
  auto pc = pa * pb;
  auto __finishCalc = HRC::now();
  gTiming("calc", __finishCalc - __startCalc);

  auto __startDump = HRC::now();
  c.dump(prow, pcol, pc);
  if (world.mRank == 0)
    c.dump_head();
  auto __finishDump = HRC::now();
  gTiming("dump", __finishDump - __startDump);
}

void
dot_grid_bcast(const MatFile& a, const MatFile& b, MatFile& c)
{
  assert((a.mRown == a.mColn && b.mRown == b.mColn) &&
         "must be square matrix!");
  assert((a.mColn == b.mRown) && "a.mColn != b.mRown!");

  auto n = int(std::sqrt(world.mSize));
  assert((n * n == world.mSize) &&
         "MPI processe amount must be a square number!");

  //* a.mRown == a.mColn == b.mRown == b.mColn;
  auto side = c.mRown = c.mColn = a.mRown;

  auto y = world.mRank / n, x = world.mRank % n;

  auto prow = side * y / n;
  auto pcol = side * x / n;
  auto prown = side * (y + 1) / n - prow;
  auto pcoln = side * (x + 1) / n - pcol;

  auto __startLoad = HRC::now();

  std::vector<Mat> aRowMats(n);
  std::vector<Mat> bColMats(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    if (i != x)
      aRowMats[i] = Mat(prown, (side * (i + 1) / n) - (side * i / n));
    if (i != y)
      bColMats[i] = Mat((side * (i + 1) / n) - (side * i / n), pcoln);
  }
  a.load(prow, pcol, aRowMats[x]);
  b.load(prow, pcol, bColMats[y]);

  auto __finishLoad = HRC::now();
  gTiming("load", __finishLoad - __startLoad);

  auto __startComm = HRC::now();
  {
    Comm worldComm(MPI_COMM_WORLD);
    auto worldGroup = worldComm.group();

    std::vector<int> ranks;
    for (int i = 0; i < n; ++i)
      ranks.push_back(y * n + i);

    Group::Local rowGroup = worldGroup.incl(ranks);
    Comm::Local rowComm = worldComm.create(rowGroup);

    ranks.clear();
    for (int i = 0; i < n; ++i)
      ranks.push_back(i * n + x);

    Group::Local colGroup = worldGroup.incl(ranks);
    Comm::Local colComm = worldComm.create(colGroup);

    for (std::uint32_t i = 0; i < n; ++i) {
      rowComm.bcast(aRowMats[i].mData, aRowMats[i].size(), MPI_DOUBLE, i);
      colComm.bcast(bColMats[i].mData, bColMats[i].size(), MPI_DOUBLE, i);
    }
  }
  auto __finishComm = HRC::now();
  gTiming("comm", __finishComm - __startLoad);

  auto __startCalc = HRC::now();
  Mat pc = aRowMats[0] * bColMats[0];
  for (std::uint32_t i = 1; i < n; ++i)
    pc += aRowMats[i] * bColMats[i];
  auto __finishCalc = HRC::now();
  gTiming("calc", __finishCalc - __startCalc);

  auto __startDump = HRC::now();
  c.dump(prow, pcol, pc);
  if (world.mRank == 0)
    c.dump_head();
  auto __finishDump = HRC::now();
  gTiming("dump", __finishDump - __startDump);
}

void
dot_row_bcast(const MatFile& a, const MatFile& b, MatFile& c)
{
  assert((a.mRown == a.mColn && b.mRown == b.mColn) &&
         "must be square matrix!");
  assert((a.mColn == b.mRown) && "a.mColn != b.mRown!");

  auto n = int(std::sqrt(world.mSize));
  assert((n * n == world.mSize) &&
         "MPI processe amount must be a square number!");

  // a.mRown == a.mColn == b.mRown == b.mColn;
  auto side = c.mRown = c.mColn = a.mRown;

  auto y = world.mRank / n, x = world.mRank % n;

  auto prow = side * y / n;
  auto pcol = side * x / n;
  auto prown = side * (y + 1) / n - prow;
  auto pcoln = side * (x + 1) / n - pcol;

  Comm worldComm(MPI_COMM_WORLD);

  auto __startLoad = HRC::now();
  std::vector<Mat> aRowMats(n);
  std::vector<Mat> bColMats(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    aRowMats[i] = Mat(prown, (side * (i + 1) / n) - (side * i / n));
    bColMats[i] = Mat((side * (i + 1) / n) - (side * i / n), pcoln);
  }

  {
    // 按行分工加载
    auto grow = prown * x / n; // 组（每行的n个进程）内分配到的起始行号
    auto grown = prown * (x + 1) / n - grow;
    Mat arows(grown, a.mColn), brows(grown, b.mColn);
    a.load(prow + grow, arows);
    b.load(prow + grow, brows);
    auto __finishLoad = HRC::now();
    gTiming("load", __finishLoad - __startLoad);

    // 使用点对点通讯将数据重组为网格状
    auto __startForm = HRC::now();
    std::vector<MPI_Request> aReqs(n * grown);
    std::vector<MPI_Request> bReqs(n * grown);

    for (std::uint32_t i = 0, col = 0; i < n; ++i) {
      auto dest = y * n + i;
      std::uint32_t coln = aRowMats[i].mColn;

      for (std::uint32_t j = 0; j < grown; ++j) {
        aReqs[i * grown + j] = worldComm.isend(
          &arows(j, col), coln, MPI_DOUBLE, dest, (grow + j) << 1);
        bReqs[i * grown + j] = worldComm.isend(
          &brows(j, col), coln, MPI_DOUBLE, dest, ((grow + j) << 1) | 1);
      }

      col += coln;
    }

    for (std::uint32_t i = 0; i < prown; ++i) {
      worldComm.recv(
        &aRowMats[x](i), pcoln, MPI_DOUBLE, MPI_ANY_SOURCE, i << 1);
      worldComm.recv(
        &bColMats[y](i), pcoln, MPI_DOUBLE, MPI_ANY_SOURCE, (i << 1) | 1);
    }

    for (auto i = 0; i < n * grown; ++i) {
      Err::check(MPI_Wait(&aReqs[i], MPI_STATUS_IGNORE));
      Err::check(MPI_Wait(&bReqs[i], MPI_STATUS_IGNORE));
    }

    auto __finishForm = HRC::now();
    gTiming("form", __finishForm - __startForm);
  }

  // {
  //   std::ostringstream sout;
  //   sout << aRowMats[x] << '\n' << bColMats[y] << '\n';
  //   std::cout << '[' << world.mRank << ']' << sout.str() << std::flush;
  // }

  auto __startComm = HRC::now();
  {
    auto worldGroup = worldComm.group();

    std::vector<int> ranks;
    for (int i = 0; i < n; ++i)
      ranks.push_back(y * n + i);

    Group::Local rowGroup = worldGroup.incl(ranks);
    Comm::Local rowComm = worldComm.create(rowGroup);

    ranks.clear();
    for (int i = 0; i < n; ++i)
      ranks.push_back(i * n + x);

    Group::Local colGroup = worldGroup.incl(ranks);
    Comm::Local colComm = worldComm.create(colGroup);

    for (std::uint32_t i = 0; i < n; ++i) {
      rowComm.bcast(aRowMats[i].mData, aRowMats[i].size(), MPI_DOUBLE, i);
      colComm.bcast(bColMats[i].mData, bColMats[i].size(), MPI_DOUBLE, i);
    }
  }
  auto __finishComm = HRC::now();
  gTiming("comm", __finishComm - __startLoad);

  auto __startCalc = HRC::now();
  Mat pc = aRowMats[0] * bColMats[0];
  for (std::uint32_t i = 1; i < n; ++i)
    pc += aRowMats[i] * bColMats[i];
  auto __finishCalc = HRC::now();
  gTiming("calc", __finishCalc - __startCalc);

  auto __startDump = HRC::now();
  c.dump(prow, pcol, pc);
  if (world.mRank == 0)
    c.dump_head();
  auto __finishDump = HRC::now();
  gTiming("dump", __finishDump - __startDump);
}

void
dot_cannon(const MatFile& a, const MatFile& b, MatFile& c)
{
  assert((a.mRown == a.mColn && b.mRown == b.mColn) &&
         "must be square matrix!");
  assert((a.mColn == b.mRown) && "a.mColn != b.mRown!");

  auto n = int(std::sqrt(world.mSize));
  assert((n * n == world.mSize) &&
         "MPI processe amount must be a square number!");

  // a.mRown == a.mColn == b.mRown == b.mColn;
  auto side = c.mRown = c.mColn = a.mRown;

  auto y = world.mRank / n, x = world.mRank % n;

  auto prow = side * y / n;
  auto pcol = side * x / n;
  auto prown = side * (y + 1) / n - prow;
  auto pcoln = side * (x + 1) / n - pcol;

  Comm worldComm(MPI_COMM_WORLD);

  auto __startLoad = HRC::now();
  Mat pa(prown, pcoln), pb(prown, pcoln);
  {
    // 按行分工加载
    auto grow = prown * x / n; // 组（每行的n个进程）内分配到的起始行号
    auto grown = prown * (x + 1) / n - grow;
    Mat arows(grown, a.mColn), brows(grown, b.mColn);
    a.load(prow + grow, arows);
    b.load(prow + grow, brows);

    auto __finishLoad = HRC::now();
    gTiming("load", __finishLoad - __startLoad);

    // 使用点对点通讯将数据重组为网格状
    auto __startForm = HRC::now();
    std::vector<MPI_Request> aReqs(n * grown);
    std::vector<MPI_Request> bReqs(n * grown);

    for (std::uint32_t i = 0, col = 0; i < n; ++i) {
      auto dest = y * n + i;
      auto coln = (side * (i + 1) / n) - (side * i / n);

      for (std::uint32_t j = 0; j < grown; ++j) {
        aReqs[i * grown + j] = worldComm.isend(
          &arows(j, col), coln, MPI_DOUBLE, dest, (grow + j) << 1);
        bReqs[i * grown + j] = worldComm.isend(
          &brows(j, col), coln, MPI_DOUBLE, dest, ((grow + j) << 1) | 1);
      }

      col += coln;
    }

    for (std::uint32_t i = 0; i < prown; ++i) {
      worldComm.recv(&pa(i), pcoln, MPI_DOUBLE, MPI_ANY_SOURCE, i << 1);
      worldComm.recv(&pb(i), pcoln, MPI_DOUBLE, MPI_ANY_SOURCE, (i << 1) | 1);
    }

    for (auto i = 0; i < n * grown; ++i) {
      Err::check(MPI_Wait(&aReqs[i], MPI_STATUS_IGNORE));
      Err::check(MPI_Wait(&bReqs[i], MPI_STATUS_IGNORE));
    }

    auto __finishForm = HRC::now();
    gTiming("form", __finishForm - __startForm);
  }

  // {
  //   std::ostringstream sout;
  //   sout << '[' << world.mRank << ']' << pa << '\n' << pb << '\n';
  //   std::cout << sout.str() << std::flush;
  // }

  // 起始对准
  auto __startComm = HRC::now();
  auto xyRD = (x + y) % n;
  {
    auto xL = (x - y + n) % n, yU = (y - x + n) % n;

    auto reqa =
      worldComm.isend(pa.mData, pa.size(), MPI_DOUBLE, y * n + xL, side << 2);
    auto reqb =
      worldComm.isend(pb.mData, pb.size(), MPI_DOUBLE, yU * n + x, side << 2);

    auto acoln = (side * (xyRD + 1) / n) - (side * xyRD / n);
    Mat ta(prown, acoln), tb(acoln, pcoln);
    worldComm.recv(ta.mData, ta.size(), MPI_DOUBLE, y * n + xyRD, side << 2);
    worldComm.recv(tb.mData, tb.size(), MPI_DOUBLE, xyRD * n + x, side << 2);

    Err::check(MPI_Wait(&reqa, MPI_STATUS_IGNORE));
    Err::check(MPI_Wait(&reqb, MPI_STATUS_IGNORE));

    pa = std::move(ta);
    pb = std::move(tb);
  }
  auto __finishComm = HRC::now();
  gTiming("comm", __finishComm - __startLoad);

  // {
  //   std::ostringstream sout;
  //   sout << '[' << world.mRank << ']' << pa << '\n' << pb << '\n';
  //   std::cout << sout.str() << std::flush;
  // }

  auto __startCalc = HRC::now();
  auto pL = y * n + (x - 1 + n) % n, pR = y * n + (x + 1) % n;
  auto pU = (y - 1 + n) % n * n + x, pD = (y + 1) % n * n + x;

  Mat pc = pa * pb;
  for (std::uint32_t i = 1; i < n; ++i) {
    auto tag = (side << 2) + i;
    auto reqa = worldComm.isend(pa.mData, pa.size(), MPI_DOUBLE, pL, tag);
    auto reqb = worldComm.isend(pb.mData, pb.size(), MPI_DOUBLE, pU, tag);

    auto acoln = (side * (xyRD + i + 1) / n) - (side * (xyRD + i) / n);
    Mat ta(prown, acoln), tb(acoln, pcoln);
    worldComm.recv(ta.mData, ta.size(), MPI_DOUBLE, pR, tag);
    worldComm.recv(tb.mData, tb.size(), MPI_DOUBLE, pD, tag);

    Err::check(MPI_Wait(&reqa, MPI_STATUS_IGNORE));
    Err::check(MPI_Wait(&reqb, MPI_STATUS_IGNORE));

    pa = std::move(ta);
    pb = std::move(tb);
    pc += pa * pb;
  }
  auto __finishCalc = HRC::now();
  gTiming("calc", __finishCalc - __startCalc);

  auto __startDump = HRC::now();
  c.dump(prow, pcol, pc);
  if (world.mRank == 0)
    c.dump_head();
  auto __finishDump = HRC::now();
  gTiming("dump", __finishDump - __startDump);
}

static int
log2(int n)
{
  int x = 0;
  while (n != 1)
    ++x, n >>= 1;
  return x;
}

void
dot_dns(const MatFile& a, const MatFile& b, MatFile& c, int k)
{
  assert((a.mRown == a.mColn && b.mRown == b.mColn) &&
         "must be square matrix!");
  assert((a.mColn == b.mRown) && "a.mColn != b.mRown!");

  // a.mRown == a.mColn == b.mRown == b.mColn;
  auto side = c.mRown = c.mColn = a.mRown;

  int n, n2, x, y, z;
  {
    assert(world.mSize % k == 0 && "k must be a factor of MPI process number!");

    n2 = world.mSize / k;
    n = int(std::sqrt(n2));

    assert(n * n == n2 && "MPI process number must be a square number!");

    x = world.mRank % n;
    y = world.mRank / n % n;
    z = world.mRank / n / n;
  }

  auto prowA = side * x / n, pcolA = side * z / k;
  auto prownA = side * (x + 1) / n - prowA;
  auto pcolnA = side * (z + 1) / k - pcolA;

  auto prowB = side * z / k, pcolB = side * y / n;
  auto prownB = side * (z + 1) / k - prowB;
  auto pcolnB = side * (y + 1) / n - pcolB;

  auto __startLoad = HRC::now();
  Mat pa(prownA, pcolnA), pb(prownB, pcolnB);
  a.load(prowA, pcolA, pa);
  b.load(prowB, pcolB, pb);
  auto __finishLoad = HRC::now();
  gTiming("load", __finishLoad - __startLoad);

  auto __startCalc = HRC::now();
  auto pc = pa * pb;
  auto __finishCalc = HRC::now();
  gTiming("calc", __finishCalc - __startCalc);

  // {
  //   std::ostringstream sout;
  //   sout << '[' << world.mRank << ']' << pc << '\n';
  //   std::cout << sout.str() << std::flush;
  // }

  // 按Z轴归并加
  Comm worldComm(MPI_COMM_WORLD);

  auto __startRedu = HRC::now();
  pa = Mat(pc, Mat::ShapeOnly());
  pb = Mat(pc, Mat::ShapeOnly());

  auto sendTo = (z / 2) * n2 + y * n + x;
  auto recv0 = (z * 2) * n2 + y * n + x;
  auto recv1 = (z * 2 + 1) * n2 + y * n + x;
  auto loopTimes = log2(k) + 1 - log2(z + 1);

  // {
  //   std::ostringstream sout;
  //   sout << '[' << world.mRank << ']' << sendTo << ' ' << recv0 << ' '
  //   << recv1
  //        << ' ' << loopTimes << '\n';
  //   std::cout << sout.str() << std::flush;
  // }

  for (int i = 0;; ++i) {
    auto req = worldComm.isend(pc.mData, pc.size(), MPI_DOUBLE, sendTo, i);
    if (i == loopTimes) {
      if (z != 0)
        Err::check(MPI_Wait(&req, MPI_STATUS_IGNORE));
      break;
    }

    if (recv0 < world.mSize) {
      worldComm.recv(pa.mData, pa.size(), MPI_DOUBLE, recv0, i);
      if (recv1 < world.mSize) {
        worldComm.recv(pb.mData, pb.size(), MPI_DOUBLE, recv1, i);
        pa += pb;
      }
    } else
      pa.zero();

    Err::check(MPI_Wait(&req, MPI_STATUS_IGNORE));

    swap(pc, pa);
  }
  auto __finishRedu = HRC::now();
  gTiming("redu", __finishRedu - __startRedu);

  auto __startDump = HRC::now();
  if (z == 0)
    c.dump(prowA, pcolB, pc);
  if (world.mRank == 0)
    c.dump_head();
  auto __finishDump = HRC::now();
  gTiming("dump", __finishDump - __startDump);
}

} // namespace MPI_mat::mpi
