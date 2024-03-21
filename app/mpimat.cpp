#include "project.h"
#include "timestamp.h"

static const char kVersionInfo[] =
  "Matrix Algorithm Program (MPI)\n"
  "\n"
  "Built: " MPI_mat_TIMESTAMP "\n"
  "Project: " MPI_mat_VERSION "\n"
  "Copyright (C) 2023-2024 Yuhao Gu. All Rights Reserved.";

#include "po.hpp"

#include <MPI_mat/hpp>
#include <My/Timing.hpp>
#include <My/util.hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>

using namespace My::util;
namespace po = boost::program_options;

namespace {

int
gen(int argc, char* argv[])
{
  std::string oPath;
  std::uint32_t rown, coln;
  std::string method = "rand";
  double fill = 0.0;
  {
    po::options_description od("'gen' Options");
    od.add_options()                                               //
      ("help,h", "print help info")                                //
      ("output,o", povr(oPath), "output file path")                //
      ("rown,r", povr(rown), "row number, height")                 //
      ("coln,c", povr(coln), "column number, width")               //
      ("method,m", povd(method), "generation method: rand / fill") //
      ("fill,f", povd(fill), "fill value")                         //
      ;

    po::positional_options_description pod;
    pod.add("output", 1);
    pod.add("rown", 1);
    pod.add("coln", 1);

    po::variables_map vmap;
    po::store(
      po::command_line_parser(argc, argv).options(od).positional(pod).run(),
      vmap);

    if (vmap.count("help") || argc == 1) {
      std::cout << od << std::endl;
      return 0;
    }
    po::notify(vmap);
  }

  MPI_mat::MatFile mf(oPath, rown, coln, MPI_mat::MatFile::CreateOnly());

  if (method == "rand")
    MPI_mat::mpi::gen_rand(mf);
  else if (method == "fill")
    MPI_mat::mpi::gen_fill(mf, fill);
  else {
    std::cout << "invalid method '" << method << "'." << std::endl;
    return 1;
  }

  return 0;
}

int
dot(int argc, char* argv[])
{
  std::string oPath, lPath, rPath;
  std::string method = "cannon";
  int dnsK = 1;
  {
    po::options_description od("'dot' Options");
    od.add_options()                                           //
      ("help,h", "show help info")                             //
      ("output,o", povr(oPath), "output file path")            //
      ("lft,l", povr(lPath), "left operand matrix file path")  //
      ("rht,r", povr(rPath), "right operand matrix file path") //
      ("method,m",                                             //
       povd(method),                                           //
       "direct_load / grid_bcast / row_bcast / cannon / dns")  //
      ("dns_k", povd(dnsK), "k parameter of dns method")       //
      ;

    po::positional_options_description pod;
    pod.add("output", 1);
    pod.add("lft", 1);
    pod.add("rht", 1);

    po::variables_map vmap;
    po::store(
      po::command_line_parser(argc, argv).options(od).positional(pod).run(),
      vmap);

    if (vmap.count("help") || argc == 1) {
      std::cout << od << std::endl;
      return 0;
    }
    po::notify(vmap);
  }

  auto __startTotal = MPI_mat::mpi::HRC::now();
  MPI_mat::MatFile outmat(oPath, 0, 0, MPI_mat::MatFile::CreateOnly());

  MPI_mat::MatFile lft(lPath), rht(rPath);

  if (method == "direct_load")
    MPI_mat::mpi::dot_direct_load(lft, rht, outmat);
  else if (method == "grid_bcast")
    MPI_mat::mpi::dot_grid_bcast(lft, rht, outmat);
  else if (method == "row_bcast")
    MPI_mat::mpi::dot_row_bcast(lft, rht, outmat);
  else if (method == "cannon")
    MPI_mat::mpi::dot_cannon(lft, rht, outmat);
  else if (method == "dns")
    MPI_mat::mpi::dot_dns(lft, rht, outmat, dnsK);
  else {
    std::cout << "invalid method '" << method << "'." << std::endl;
    return 1;
  }

  auto __finishTotal = MPI_mat::mpi::HRC::now();
  MPI_mat::mpi::gTiming("total", __finishTotal - __startTotal);

  return 0;
}

int
powsum(int argc, char* argv[])
{
  std::uint32_t size = 128, pown = 16;
  {
    po::options_description od("'powsum' Options");
    od.add_options()                                           //
      ("help,h", "show help info")                             //
      ("size,s", povd(size), "matrix size (width and height)") //
      ("pown,p", povd(pown), "power number")                   //
      ;

    po::variables_map vmap;
    po::store(po::command_line_parser(argc, argv).options(od).run(), vmap);

    if (vmap.count("help")) {
      std::cout << od << std::endl;
      return 0;
    }
    po::notify(vmap);
  }

  if (MPI_mat::mpi::World::g()->mRank == 0)
    std::cout << "size = " << size << ", pown = " << pown << std::endl;
  auto __start = My::Timing::Clock::now();
  auto ans = MPI_mat::mpi::powsum_benchmark(size, pown);
  auto __end = My::Timing::Clock::now();
  if (MPI_mat::mpi::World::g()->mRank == 0)
    std::cout << "ans: " << ans << ", cost: " << (__end - __start) << std::endl;

  return 0;
}

} // namespace

// ========================================================================== //
// 主函数
// ========================================================================== //

namespace {

struct SubCmd
{
  const char *mName, *mInfo;
  int (*mFunc)(int argc, char* argv[]);
};

const SubCmd kSubCmds[] = {
  { "gen", "generate random matrix", &gen },
  { "dot", "matrix multiplication", &dot },
  { "powsum", "powsum benchmark", &powsum },
};

void
timing_to_cerr(const char* tag, const MPI_mat::mpi::HRC::duration& dura)
{
  if (MPI_mat::mpi::World::g()->mRank == 0)
    std::cerr << tag << ": " << dura << std::endl;
}

} // namespace

int
main(int argc, char* argv[])
try {
  bool timing = true;

  po::options_description od("Options");
  od.add_options()                                       //
    ("version,v", "print version info")                  //
    ("help,h", "print help info")                        //
    ("timing,t", povd(timing), "enable internal timing") //
    ("...",                                              //
     po::value<std::vector<std::string>>(),              //
     "other arguments")                                  //
    ;

  po::positional_options_description pod;
  pod.add("...", -1);

  std::vector<std::string> opts{ argv[0] };
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '-')
      opts.emplace_back(argv[i]);
    else
      break;
  }

  po::variables_map vmap;
  auto parsed = po::command_line_parser(opts)
                  .options(od)
                  .positional(pod)
                  .allow_unregistered()
                  .run();
  po::store(parsed, vmap);

  if (vmap.count("help") || argc == 1) {
    std::cout << od
              << "\n"
                 "Sub Commands:\n";
    for (auto&& i : kSubCmds)
      std::cout << "  " << std::left << std::setw(12) << i.mName << i.mInfo
                << '\n';
    std::cout << "\n"
                 "[HINT: use '<subcmd> --help' to get help for sub commands.]\n"
              << std::endl;
    return 0;
  }

  if (vmap.count("version")) {
    std::cout << kVersionInfo << std::endl;
    return 0;
  }

  // 如果必须的选项没有指定，在这里会发生错误，因此 version 和 help
  // 选项要放在前面。
  po::notify(vmap);

  if (timing)
    MPI_mat::mpi::gTiming = &timing_to_cerr;

  MPI_mat::mpi::World world;
  if (opts.size() < argc) {
    std::string cmd = argv[opts.size()];
    for (auto&& i : kSubCmds) {
      if (cmd == i.mName)
        return i.mFunc(argc - opts.size(), argv + opts.size());
    }
    std::cout << "invalid sub command '" << cmd << "'." << std::endl;
    return 1;
  }
}

catch (My::Err& e) {
  std::cout << e.what() << ": " << e.info() << std::endl;
  return -3;
}

catch (std::exception& e) {
  std::cout << "Exception: " << e.what() << std::endl;
  return -2;
}

catch (...) {
  std::cout << "UNKNOWN EXCEPTION" << std::endl;
  return -1;
}
