#pragma once

#include "cpp"
#include "mat.hpp"
#include <My/Globally.hpp>
#include <My/err.hpp>
#include <chrono>
#include <csignal>
#include <functional>
#include <mpi.h>

namespace MPI_mat::mpi {

struct Err : public My::Err
{
  static inline void check(int n)
  {
    if (n != MPI_SUCCESS)
      throw Err(n);
  }

  static inline void abort(int n)
  {
    if (n != MPI_SUCCESS)
      std::abort();
  }

  int mCode;

  Err(int code)
    : mCode(code)
  {
  }

public:
  ///@name Err interface
  ///@{
  std::string info() const noexcept override;
  ///@}
};

struct World : public My::Globally<World>
{
  int mSize, mRank;
  MPI_Errhandler mErrhandler;

  World(int* argc = nullptr, char*** argv = nullptr)
  {
    Err::check(MPI_Init(argc, argv));
    Err::check(MPI_Comm_size(MPI_COMM_WORLD, &mSize));
    Err::check(MPI_Comm_rank(MPI_COMM_WORLD, &mRank));
  }

  ~World() noexcept { Err::abort(MPI_Finalize()); }
};

class Group
{
public:
  struct Local;

public:
  MPI_Group _;

public:
  Group() noexcept = default;

  Group(MPI_Group _) noexcept
    : _(_)
  {
  }

public:
  Group incl(int n, const int* ranks)
  {
    Group group;
    Err::check(MPI_Group_incl(_, n, ranks, &group._));
    return group;
  }

  Group incl(const std::vector<int>& ranks)
  {
    return incl(ranks.size(), ranks.data());
  }
};

struct Group::Local final : public Group
{
  Local(const Local&) = delete;
  Local(Local&&) = delete;
  Local& operator=(const Local&) = delete;
  Local& operator=(Local&&) = delete;

  Local(Group group)
    : Group(group)
  {
  }

  ~Local() noexcept { MPI_Group_free(&_); }
};

class Comm
{
public:
  struct Local;

public:
  MPI_Comm _;

public:
  Comm() noexcept = default;

  Comm(MPI_Comm _) noexcept
    : _(_)
  {
  }

public:
  Group group() const
  {
    Group group;
    Err::check(MPI_Comm_group(_, &group._));
    return group;
  }

  Comm create(Group group) const
  {
    Comm comm;
    Err::check(MPI_Comm_create(_, group._, &comm._));
    return comm;
  }

  void bcast(void* buffer, int count, MPI_Datatype datatype, int root) const
  {
    Err::check(MPI_Bcast(buffer, count, datatype, root, _));
  }

  void send(void* buffer,
            int count,
            MPI_Datatype datatype,
            int dest,
            int tag) const
  {
    Err::check(MPI_Send(buffer, count, datatype, dest, tag, _));
  }

  MPI_Request isend(void* buffer,
                    int count,
                    MPI_Datatype datatype,
                    int dest,
                    int tag) const
  {
    MPI_Request ret;
    Err::check(MPI_Isend(buffer, count, datatype, dest, tag, _, &ret));
    return ret;
  }

  void recv(void* buffer, int count, MPI_Datatype datatype, int source, int tag)
  {
    Err::check(
      MPI_Recv(buffer, count, datatype, source, tag, _, MPI_STATUS_IGNORE));
  }
};

struct Comm::Local final : public Comm
{
  Local(const Local&) = delete;
  Local(Local&&) = delete;
  Local& operator=(const Local&) = delete;
  Local& operator=(Local&&) = delete;

  Local(Comm comm)
    : Comm(comm)
  {
  }

  ~Local() noexcept { MPI_Comm_free(&_); }
};

using HRC = std::chrono::high_resolution_clock;
using TimingFunc =
  std::function<void(const char* tag, const HRC::duration& dura)>;

/// 等待调试器
inline void
wait_debugger()
{
  std::printf(
    "Rank %d with PID %ld ready for attach\n", World::g()->mRank, get_pid());
  std::raise(SIGSTOP);
}

/// 默认计时函数，什么都不做。
void
timing_noting(const char* tag, const HRC::duration& dura);

/// 全局计时函数。
extern TimingFunc gTiming;

/// 将 \p matFile 全置为 \p fill ， \p matfile 文件必须存在。
void
gen_fill(const MatFile& matFile, double fill);

/// 为 \p matFile 随机赋值， \p matfile 文件必须存在。
void
gen_rand(const MatFile& matFile);

/**
 * @brief 简单矩阵乘，直接加载模式
 */
void
dot_direct_load(const MatFile& a, const MatFile& b, MatFile& c);

/**
 * @brief 简单矩阵乘，网格广播模式
 */
void
dot_grid_bcast(const MatFile& a, const MatFile& b, MatFile& c);

/**
 * @brief 简单矩阵乘，行广播模式
 */
void
dot_row_bcast(const MatFile& a, const MatFile& b, MatFile& c);

/**
 * @brief Cannon 矩阵乘算法
 */
void
dot_cannon(const MatFile& a, const MatFile& b, MatFile& c);

/**
 * @brief DNS 矩阵乘算法
 *
 * @param k 第三维分层数
 */
void
dot_dns(const MatFile& a, const MatFile& b, MatFile& c, int k);

/**
 * @brief 乘幂求和基准测试，使用 Cannon 算法。
 * 
 * @param size 矩阵大小
 * @param pown 乘幂次数
 * @return double 求和结果
 */
double
powsum_benchmark(std::uint32_t size, std::uint32_t pown);

} // namespace MPI_mat::mpi
