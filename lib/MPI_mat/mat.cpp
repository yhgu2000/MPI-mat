#include "mat.hpp"

#include <iomanip>
#include <memory>
#include <omp.h>

#define self (*this)

using namespace std::string_literals;

namespace MPI_mat {

bool
Mat::operator==(const Mat& other) const noexcept
{
  if (mRown != other.mRown || mColn != other.mColn)
    return false;

  bool ans = true;

#pragma omp parallel reduction(&& : ans)
  {
    auto tid = omp_get_thread_num();
    auto tnum = omp_get_num_threads();

    auto begin = size() * tid / tnum;
    auto tsize = size() * (tid + 1) / tnum - begin;

    for (auto i = begin, end = begin + tsize; i < end; ++i)
      if (self[i] != other[i]) {
        ans = false;
        break;
      }
  }

  return ans;
}

bool
Mat::operator!=(const Mat& other) const noexcept
{
  if (mRown != other.mRown || mColn != other.mColn)
    return true;

  bool ans = false;

#pragma omp parallel reduction(|| : ans)
  {
    auto tid = omp_get_thread_num();
    auto tnum = omp_get_num_threads();

    auto begin = size() * tid / tnum;
    auto tsize = size() * (tid + 1) / tnum - begin;

    for (auto i = begin, end = begin + tsize; i < end; ++i)
      if (self[i] != other[i]) {
        ans = true;
        break;
      }
  }

  return ans;
}

Mat
Mat::operator+(const Mat& rht) const noexcept
{
  assert(mRown == rht.mRown && mColn == rht.mColn);

  Mat ans(self, ShapeOnly());

#pragma omp parallel for
  for (std::uint64_t i = 0; i < mRown * mColn; ++i)
    ans[i] = self[i] + rht[i];

  return ans;
}

Mat&
Mat::operator+=(const Mat& rht) noexcept
{
  assert(mRown == rht.mRown && mColn == rht.mColn);

#pragma omp parallel for
  for (std::uint64_t i = 0; i < mRown * mColn; ++i)
    self[i] += rht[i];

  return self;
}

Mat
Mat::operator*(const Mat& rht) const noexcept
{
  assert(mRown == rht.mRown && mColn == rht.mColn);

  Mat ans(self, ShapeOnly());

#pragma omp parallel for
  for (std::uint64_t i = 0; i < mRown * mColn; ++i)
    ans[i] = self[i] * rht[i];

  return ans;
}

Mat&
Mat::operator*=(const Mat& rht) noexcept
{
  assert(mRown == rht.mRown && mColn == rht.mColn);

#pragma omp parallel for
  for (std::uint64_t i = 0; i < mRown * mColn; ++i)
    self[i] *= rht[i];

  return self;
}

Mat
Mat::dot(const Mat& rht) const noexcept
{
  assert(mColn == rht.mRown);

  Mat ans(mRown, rht.mColn);

#pragma omp parallel for
  for (std::uint32_t j = 0; j < ans.mColn; ++j) {
    std::unique_ptr<double[]> col(new double[mColn]);
    for (std::uint32_t k = 0; k < mColn; ++k)
      col[k] = rht(k, j);

    for (std::uint32_t i = 0; i < ans.mRown; ++i) {
      double sum = 0;
      for (std::uint32_t k = 0; k < mColn; ++k)
        sum += self(i, k) * col[k];
      ans(i, j) = sum;
    }
  }

  return ans;
}

double
Mat::sum() const noexcept
{
  double sum = 0;
  auto end = mData + size();

#pragma omp parallel for reduction(+ : sum)
  for (auto p = mData; p < end; ++p)
    sum += *p;

  return sum;
}

Mat&
Mat::fill(double value) noexcept
{
  auto end = mData + size();

#pragma omp parallel for
  for (auto p = mData; p < end; ++p)
    *p = value;

  return self;
}

Mat&
Mat::eye() noexcept
{
  zero();

  auto end = mData + size();
  for (auto p = mData; p < end; p += mColn + 1)
    *p = 1.0;

  return self;
}

Mat&
Mat::rand(double min, double max, std::uint32_t seed) noexcept
{
  auto size = this->size();

#pragma omp parallel
  {
    static thread_local std::random_device sRandDev;

    std::minstd_rand rand;
    if (seed != UINT32_MAX)
      rand.seed(seed);
    else
      rand.seed(sRandDev());

    std::uniform_real_distribution<> dis(min, max);

    auto tid = omp_get_thread_num();
    auto tnum = omp_get_num_threads();

    auto begin = size * tid / tnum;
    auto tsize = size * (tid + 1) / tnum - begin;

    for (auto p = mData + begin, end = p + tsize; p < end; ++p)
      *p = dis(rand);
  }

  return self;
}

std::uint64_t
Mat::checksum() const noexcept
{
  std::uint64_t ret = 0xFEDCBA0987654321;
  for (auto p = reinterpret_cast<std::uint64_t*>(mData), end = p + size();
       p < end;
       ++p)
    ret += *p;
  return ret;
}

std::ostream&
operator<<(std::ostream& out, const Mat& mat)
{
  out << "rown=" << mat.mRown << ", coln=" << mat.mColn << '\n';
  for (std::uint32_t r = 0; r < mat.mRown; ++r) {
    for (std::uint32_t c = 0; c < mat.mColn; ++c)
      out << std::setw(10) << std::setprecision(2) << mat(r, c);
    out << '\n';
  }
  return out;
}

static constexpr auto kMatHeadSize = sizeof(Mat::mRown) + sizeof(Mat::mColn);

void
MatFile::load_head()
{
  My::CFile64 file(mPath.c_str(), "r+");
  My::CFile64::Closer closer(file);
  file.read(&mRown, sizeof(mRown), 1);
  file.read(&mColn, sizeof(mColn), 1);
}

void
MatFile::load(Mat& mat)
{
  {
    My::CFile64 file(mPath.c_str(), "r");
    My::CFile64::Closer closer(file);
    file.read(&mRown, sizeof(mRown), 1);
    file.read(&mColn, sizeof(mColn), 1);
  }

  auto size = mat.resize(mRown, mColn);
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto tnum = omp_get_num_threads();

    auto begin = size * tid / tnum;
    auto tsize = size * (tid + 1) / tnum - begin;

    My::CFile64 file(mPath.c_str(), "r+");
    My::CFile64::Closer closer(file);
    file.seek(kMatHeadSize + sizeof(double) * begin, SEEK_SET);
    file.read(mat.mData + begin, sizeof(double), tsize);
  }
}

void
MatFile::load(std::uint32_t row, std::uint32_t col, const Mat& mat) const
{
  if (row >= mRown || row + mat.mRown > mRown)
    throw std::out_of_range("row index out of range.");
  if (col >= mColn || col + mat.mColn > mColn)
    throw std::out_of_range("col index out of range.");
  if (!mat)
    throw std::invalid_argument("empty matrix.");

#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto tnum = omp_get_num_threads();

    auto trow = mat.mRown * tid / tnum;
    auto trown = mat.mRown * (tid + 1) / tnum - trow;

    My::CFile64 file(mPath.c_str(), "r+");
    My::CFile64::Closer closer(file);

    for (long i = 0; i < trown; ++i) {
      auto begin = sizeof(double) * ((row + trow + i) * mColn + col);
      file.seek(kMatHeadSize + begin, SEEK_SET);
      file.read(&mat(trow + i), sizeof(double), mat.mColn);
    }
  }
}

void
MatFile::load(std::uint32_t row, const Mat& mat) const
{
  if (mat.mColn != mColn)
    throw std::out_of_range("col not match.");
  if (row >= mRown || row + mat.mRown > mRown)
    throw std::out_of_range("row index out of range.");
  if (!mat)
    throw std::invalid_argument("empty matrix.");

#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto tnum = omp_get_num_threads();

    auto trow = mat.mRown * tid / tnum;
    auto trown = mat.mRown * (tid + 1) / tnum - trow;

    My::CFile64 file(mPath.c_str(), "r+");
    My::CFile64::Closer closer(file);

    auto begin = sizeof(double) * ((row + trow) * mColn);
    file.seek(kMatHeadSize + begin, SEEK_SET);
    file.read(&mat(trow), sizeof(double), mColn * trown);
  }
}

void
MatFile::dump_head() const
{
  My::CFile64 file(mPath.c_str(), "r+");
  My::CFile64::Closer closer(file);
  file.write(&mRown, sizeof(std::uint32_t), 1);
  file.write(&mColn, sizeof(std::uint32_t), 1);
}

void
MatFile::dump(const Mat& mat)
{
  mRown = mat.mRown, mColn = mat.mColn;
  {
    My::CFile64 file(mPath.c_str(), "w");
    My::CFile64::Closer closer(file);
    file.write(&mRown, sizeof(mRown), 1);
    file.write(&mColn, sizeof(mColn), 1);
  }

  auto size = mat.size();
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto tnum = omp_get_num_threads();

    auto begin = size * tid / tnum;
    auto tsize = size * (tid + 1) / tnum - begin;

    My::CFile64 file(mPath.c_str(), "r+");
    My::CFile64::Closer closer(file);

    file.seek(kMatHeadSize + sizeof(double) * begin, SEEK_SET);
    file.write(mat.mData + begin, sizeof(double), tsize);
  }
}

void
MatFile::dump(std::uint32_t row, std::uint32_t col, const Mat& mat) const
{
  if (row >= mRown || row + mat.mRown > mRown)
    throw std::out_of_range("row index out of range.");
  if (col >= mColn || col + mat.mColn > mColn)
    throw std::out_of_range("col index out of range.");

#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto tnum = omp_get_num_threads();

    auto trow = mat.mRown * tid / tnum;
    auto trown = mat.mRown * (tid + 1) / tnum - trow;

    My::CFile64 file(mPath.c_str(), "r+");
    My::CFile64::Closer closer(file);

    for (long i = 0; i < trown; ++i) {
      auto begin = sizeof(double) * ((row + trow + i) * mColn + col);
      file.seek(kMatHeadSize + begin, SEEK_SET);
      file.write(&mat(trow + i), sizeof(double), mat.mColn);
    }
  }
}

void
MatFile::dump(std::uint32_t row, const Mat& mat) const
{
  if (mat.mColn != mColn)
    throw std::out_of_range("col not match.");
  if (row >= mRown || row + mat.mRown > mRown)
    throw std::out_of_range("row index out of range.");

#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto tnum = omp_get_num_threads();

    auto trow = mat.mRown * tid / tnum;
    auto trown = mat.mRown * (tid + 1) / tnum - trow;

    My::CFile64 file(mPath.c_str(), "r+");
    My::CFile64::Closer closer(file);

    auto begin = sizeof(double) * ((row + trow) * mColn);
    file.seek(kMatHeadSize + begin, SEEK_SET);
    file.write(&mat(trow), sizeof(double), mColn * trown);
  }
}

} // namespace MPI_mat
