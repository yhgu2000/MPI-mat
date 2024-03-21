#pragma once

#include <My/CFile64.hpp>
#include <random>

namespace MPI_mat {

/**
 * @brief 内存矩阵类
 */
struct Mat
{
  std::uint32_t mRown{ 0 }, mColn{ 0 };
  double* mData{ nullptr };

public:
  ~Mat() noexcept
  {
    if (mData)
      delete[] mData;
  }

  Mat() noexcept = default;

  Mat(std::uint32_t rown, std::uint32_t coln) noexcept
    : mRown(rown)
    , mColn(coln)
    , mData(new double[rown * coln])
  {
  }

  struct ShapeOnly
  {};

  Mat(const Mat& other, ShapeOnly) noexcept
    : mRown(other.mRown)
    , mColn(other.mColn)
    , mData(new double[other.mRown * other.mColn])
  {
  }

  Mat(const Mat& other) noexcept
    : Mat(other, ShapeOnly())
  {
    std::memcpy(mData, other.mData, sizeof(*mData) * mRown * mColn);
  }

  Mat(Mat&& other) noexcept
    : Mat()
  {
    swap(*this, other);
  }

  Mat& operator=(Mat other) noexcept
  {
    swap(*this, other);
    return *this;
  }

  friend void swap(Mat& a, Mat& b) noexcept
  {
    using namespace std;

    swap(a.mRown, b.mRown);
    swap(a.mColn, b.mColn);
    swap(a.mData, b.mData);
  }

public:
  operator bool() const noexcept { return mData != nullptr; }

  /// 元素判等
  bool operator==(const Mat& other) const noexcept;
  bool operator!=(const Mat& other) const noexcept;

  /// 元素加
  Mat operator+(const Mat& rht) const noexcept;
  Mat& operator+=(const Mat& rht) noexcept;

  /// 元素乘
  Mat operator*(const Mat& rht) const noexcept;
  Mat& operator*=(const Mat& rht) noexcept;

  /// 矩阵乘
  Mat dot(const Mat& rht) const noexcept;

  /// 求和
  double sum() const noexcept;

public:
  /// 线性访问
  double& operator[](std::uint64_t index) const noexcept
  {
    return mData[index];
  }

  /// 坐标访问
  double& operator()(std::uint32_t row, std::uint32_t col = 0) const noexcept
  {
    assert(row < mRown && col < mColn);
    return mData[row * mColn + col];
  }

  /// 截取子矩阵
  Mat operator()(std::uint32_t row,
                 std::uint32_t rowE,
                 std::uint32_t col,
                 std::uint32_t colE) const noexcept;

  /// 元素数量
  std::uint64_t size() const noexcept { return std::uint64_t(mRown) * mColn; }

  /// 调整矩阵大小
  std::uint64_t resize(std::uint32_t rown, std::uint32_t coln) noexcept
  {
    auto size = std::uint64_t(rown) * coln;
    mData = reinterpret_cast<double*>(realloc(mData, size * sizeof(double)));
    assert(mData);
    mRown = rown, mColn = coln;
    return size;
  }

  /// 置零
  Mat& zero() noexcept
  {
    std::memset(mData, 0, sizeof(double) * size());
    return *this;
  }

  /// 填充
  Mat& fill(double value) noexcept;

  /// 置为单位矩阵
  Mat& eye() noexcept;

  /**
   * @brief 随机数填充
   *
   * @param min 最小值
   * @param max 最大值
   * @param seed 随机数种子，为 UINT32_MAX 时使用硬件熵源。
   * @return Mat& 当前对象自身
   */
  Mat& rand(double min = 0,
            double max = 1,
            std::uint32_t seed = UINT32_MAX) noexcept;

  /// 校验值
  std::uint64_t checksum() const noexcept;
};

/// 输出为文本
std::ostream&
operator<<(std::ostream& out, const Mat& mat);

/**
 * @brief 外存矩阵类
 */
struct MatFile
{
  std::string mPath;
  std::uint32_t mRown, mColn;

public:
  MatFile() noexcept {}

  /**
   * @param path 矩阵文件路径
   * @param rown 矩阵行数
   * @param coln 矩阵列数
   */
  MatFile(std::string path, std::uint32_t rown, std::uint32_t coln)
    : mPath(std::move(path))
    , mRown(rown)
    , mColn(coln)
  {
  }

  /**
   * @param path 矩阵文件路径，从中读取行数和列数
   */
  MatFile(std::string path)
    : mPath(std::move(path))
  {
    My::CFile64 file(mPath.c_str(), "r+");
    My::CFile64::Closer closer(file);
    file.read(&mRown, sizeof(std::uint32_t), 1);
    file.read(&mColn, sizeof(std::uint32_t), 1);
  }

public:
  /// 矩阵元素数量
  std::uint64_t size() const noexcept { return std::uint64_t(mRown) * mColn; }

public:
  /**
   * @brief 加载矩阵文件头（行数和列数）
   */
  void load_head();

  /**
   * @brief 从文件中加载整个矩阵
   *
   * @param[out] mat 输出矩阵
   */
  void load(Mat& mat);

  /**
   * @brief 从文件中加载一个子矩阵
   *
   * @param row 行位置
   * @param col 列位置
   * @param[out] mat 子矩阵，其范围不能超过文件中的矩阵
   */
  void load(std::uint32_t row, std::uint32_t col, const Mat& mat) const;

  /**
   * @brief 从文件中加载若干行，相比于load，这个方法可以减少IO次数
   *
   * @param row 行位置
   * @param[out] mat 子矩阵，列数必须匹配、必须有足够的空间
   */
  void load(std::uint32_t row, const Mat& mat) const;

  /**
   * @brief 转储矩阵文件头（行数和列数）
   */
  void dump_head() const;

  /**
   * @brief 转储整个矩阵，文件不存在时会创建
   *
   * @param mat 矩阵对象，其行数和列数也会被转储
   */
  void dump(const Mat& mat);

  /**
   * @brief 转储内存中的子矩阵到文件中
   *
   * @param row 行位置
   * @param col 列位置
   * @param[in] mat 子矩阵
   */
  void dump(std::uint32_t row, std::uint32_t col, const Mat& mat) const;

  /**
   * @brief 转储若干行到文件中，相比于dump，这个方法可以减少IO次数
   *
   * @param row 行位置
   * @param[in] mat 子矩阵
   */
  void dump(std::uint32_t row, const Mat& mat) const;
};

} // namespace MPI_mat
