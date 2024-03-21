#include "project.h"
#include "timestamp.h"

static const char kVersionInfo[] =
  "Matrix Algorithm Program (Single Machine)\n"
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
show(int argc, char* argv[])
{
  std::string iPath;

  po::options_description od("'show' Options");
  od.add_options()                                  //
    ("help,h", "show help info")                    //
    ("input,i", povr(iPath), "input mat file path") //
    ;

  po::positional_options_description pod;
  pod.add("input", 1);

  po::variables_map vmap;
  po::store(
    po::command_line_parser(argc, argv).options(od).positional(pod).run(),
    vmap);
  po::notify(vmap);

  if (vmap.count("help") || argc == 1) {
    std::cout << od << std::endl;
    return 0;
  }

  MPI_mat::MatFile mf(iPath);
  MPI_mat::Mat m(mf.mRown, mf.mColn);
  mf.load(m);
  std::cout << m << std::endl;

  return 0;
}

int
gen(int argc, char* argv[])
{
  std::string oPath;
  std::uint32_t rown, coln;
  std::string method = "rand";
  double fill = 0.0;
  {
    po::options_description od("'gen' Options");
    od.add_options()                                  //
      ("help,h", "show help info")                    //
      ("output,o", povr(oPath), "output file path")   //
      ("rown,r", povr(rown), "row number, height")    //
      ("coln,c", povr(coln), "column number, width")  //
      ("method,m",                                    //
       povd(method),                                  //
       "generation method: rand / zero / eye / fill") //
      ("fill,f", povd(fill), "fill value")            //
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

  MPI_mat::Mat mat(rown, coln);
  if (method == "rand")
    mat.rand();
  else if (method == "zero")
    mat.zero();
  else if (method == "eye")
    mat.eye();
  else if (method == "fill")
    mat.fill(fill);
  else {
    std::cout << "invalid method '" << method << "'." << std::endl;
    return 1;
  }

  MPI_mat::MatFile mf(oPath, rown, coln);
  mf.dump(mat);
  return 0;
}

int
add(int argc, char* argv[])
{
  std::string oPath, lPath, rPath;
  {
    po::options_description od("'add' Options");
    od.add_options()                                           //
      ("help,h", "show help info")                             //
      ("output,o", povd(oPath), "output file path")            //
      ("lft,l", povd(lPath), "left operand matrix file path")  //
      ("rht,r", povd(rPath), "right operand matrix file path") //
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

  MPI_mat::MatFile mfLft(lPath), mfRht(rPath);
  MPI_mat::Mat lft, rht;
  mfLft.load(lft), mfRht.load(rht);

  MPI_mat::MatFile mOut(oPath, 0, 0, MPI_mat::MatFile::CreateOnly());
  lft += rht;
  mOut.dump(lft);
  return 0;
}

int
mul(int argc, char* argv[])
{
  std::string oPath, lPath, rPath;
  {
    po::options_description od("'mul' Options");
    od.add_options()                                           //
      ("help,h", "show help info")                             //
      ("output,o", povd(oPath), "output file path")            //
      ("lft,l", povd(lPath), "left operand matrix file path")  //
      ("rht,r", povd(rPath), "right operand matrix file path") //
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

  MPI_mat::MatFile mfLft(lPath), mfRht(rPath);
  MPI_mat::Mat lft, rht;
  mfLft.load(lft), mfRht.load(rht);

  MPI_mat::MatFile mOut(oPath, 0, 0, MPI_mat::MatFile::CreateOnly());
  lft *= rht;
  mOut.dump(lft);
  return 0;
}

int
dot(int argc, char* argv[])
{
  std::string oPath, lPath, rPath;
  {
    po::options_description od("'dot' Options");
    od.add_options()                                           //
      ("help,h", "show help info")                             //
      ("output,o", povr(oPath), "output file path")            //
      ("lft,l", povr(lPath), "left operand matrix file path")  //
      ("rht,r", povr(rPath), "right operand matrix file path") //
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

  MPI_mat::MatFile mfLft(lPath), mfRht(rPath);
  MPI_mat::Mat lft, rht;
  mfLft.load(lft), mfRht.load(rht);

  MPI_mat::MatFile mOut(oPath, 0, 0, MPI_mat::MatFile::CreateOnly());
  auto out = lft.dot(rht);
  mOut.dump(out);
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
  std::cout << "size = " << size << ", pown = " << pown << std::endl;

  auto __start = My::Timing::Clock::now();
  MPI_mat::Mat m(size, size);
  m.fill(1.0 / size);
  for (std::uint32_t i = 0; i < pown; ++i)
    m = m.dot(m);
  auto sum = m.sum();
  auto __end = My::Timing::Clock::now();
  std::cout << "ans: " << sum << ", cost: " << (__end - __start) << std::endl;

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
  { "show", "print mat file in text", &show },
  { "gen", "generate matrix", &gen },
  { "add", "element-wise addition", &add },
  { "mul", "element-wise multiplication ", &mul },
  { "dot", "matrix multiplication", &dot },
  { "powsum", "powsum benchmark", &powsum },
};

} // namespace

int
main(int argc, char* argv[])
try {
  po::options_description od("Options");
  od.add_options()                          //
    ("version,v", "print version info")     //
    ("help,h", "print help info")           //
    ("...",                                 //
     po::value<std::vector<std::string>>(), //
     "other arguments")                     //
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
