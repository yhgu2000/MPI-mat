#include "project.h"
#include <PGEMM/hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std::string_literals;

namespace po = boost::program_options;

/**
 * @brief 以易读形式输出时间间隔
 */
template<typename R, typename P>
static std::ostream&
operator<<(std::ostream& out, const std::chrono::duration<R, P>& dura)
{
  out << std::fixed << std::setprecision(2);
  double count =
    std::chrono::duration_cast<std::chrono::nanoseconds>(dura).count();
  if (count < 10000)
    out << count << "ns";
  else if ((count /= 1000) < 10000)
    out << count << "us";
  else if ((count /= 1000) < 10000)
    out << count << "ms";
  else
    out << count / 1000 << "s";
  return out;
}

int
gen(int argc, char* argv[])
{
  po::options_description od("Options");
  od.add_options()                                                 //
    ("help,h", "print help info")                                  //
    ("output,o", po::value<std::string>(), "output path")          //
    ("rown,r", po::value<std::uint32_t>(), "row number, height")   //
    ("coln,c", po::value<std::uint32_t>(), "column number, width") //
    ("method,m",                                                   //
     po::value<std::string>()->default_value("rand"),              //
     "generation method: rand / zero / eye")                       //
    ;

  po::positional_options_description pod;
  pod.add("output", 1);

  po::variables_map vmap;
  po::store(po::command_line_parser(argc, argv)
              .options(od)
              .positional(pod)
              .allow_unregistered()
              .run(),
            vmap);
  po::notify(vmap);

  if (vmap.count("help") || argc == 1) {
    std::cout << od << std::endl;
    return 0;
  }

  auto outpath = vmap["output"].as<std::string>();
  auto rown = vmap["rown"].as<std::uint32_t>();
  auto coln = vmap["coln"].as<std::uint32_t>();
  auto method = vmap["method"].as<std::string>();

  if (method == "rand")
    // MPI_mat::mpi::gen_rand(outpath.c_str(), rown, coln);
    MPI_mat::mpi::gen_rand(MPI_mat::MatFile(std::move(outpath), rown, coln));
  else
    // TODO
    return 1;
  return 0;
}

int
mul(int argc, char* argv[])
{
  po::options_description od("'mul' Options");
  od.add_options()                                                         //
    ("help,h", "show help info")                                           //
    ("output,o", po::value<std::string>(), "output file path, must exist") //
    ("lft,l", po::value<std::string>(), "left operand matrix file path")   //
    ("rht,r", po::value<std::string>(), "right operand matrix file path")  //
    ("method,m",                                                           //
     po::value<std::string>()->default_value("direct_load"),               //
     "direct_load / grid_bcast / row_bcast / cannon / dns")                //
    ("timing,t", "enable internal timing")                                 //
    ("dns_k",                                                              //
     po::value<int>()->default_value(1),                                   //
     "k parameter of dns method")                                          //
    ;

  po::positional_options_description pod;
  pod.add("output", 1);

  po::variables_map vmap;
  po::store(
    po::command_line_parser(argc, argv).options(od).positional(pod).run(),
    vmap);
  po::notify(vmap);

  if (vmap.count("help") || argc == 1) {
    std::cout << od << std::endl;
    return 0;
  }

  bool timingEnabled = vmap.count("timing");
  std::unique_ptr<std::ostringstream> soutPtr;
  MPI_mat::mpi::TimingFunc timing;
  if (timingEnabled) {
    soutPtr = std::make_unique<std::ostringstream>();
    auto& sout = *soutPtr;
    sout << '[' << MPI_mat::mpi::World::gRank << '/' << MPI_mat::mpi::World::gSize
         << ']';
    timing = [&](const char* tag, const MPI_mat::mpi::HRC::duration& dura) {
      sout << '\t' << tag << '(' << dura << ')';
    };
  } else
    timing = MPI_mat::mpi::timing_noting;

  auto __startTotal = MPI_mat::mpi::HRC::now();

  auto lft = vmap["lft"].as<std::string>();
  auto rht = vmap["rht"].as<std::string>();
  auto output = vmap["output"].as<std::string>();
  MPI_mat::MatFile outmat(std::move(output), 0, 0);

  auto method = vmap["method"].as<std::string>();
  if (method == "direct_load")
    MPI_mat::mpi::mul_direct_load(MPI_mat::MatFile(std::move(lft)),
                                MPI_mat::MatFile(std::move(rht)),
                                outmat,
                                timing);

  else if (method == "grid_bcast")
    MPI_mat::mpi::mul_grid_bcast(MPI_mat::MatFile(std::move(lft)),
                               MPI_mat::MatFile(std::move(rht)),
                               outmat,
                               timing);

  else if (method == "row_bcast")
    MPI_mat::mpi::mul_row_bcast(MPI_mat::MatFile(std::move(lft)),
                              MPI_mat::MatFile(std::move(rht)),
                              outmat,
                              timing);

  else if (method == "cannon")
    MPI_mat::mpi::mul_cannon(MPI_mat::MatFile(std::move(lft)),
                           MPI_mat::MatFile(std::move(rht)),
                           outmat,
                           timing);

  else if (method == "dns")
    MPI_mat::mpi::mul_dns(MPI_mat::MatFile(std::move(lft)),
                        MPI_mat::MatFile(std::move(rht)),
                        outmat,
                        timing,
                        vmap["dns_k"].as<int>());

  else
    return 1;

  auto __finishTotal = MPI_mat::mpi::HRC::now();
  timing("total", __finishTotal - __startTotal);

  if (timingEnabled) {
    *soutPtr << '\n';
    std::cout << soutPtr->str() << std::flush;
  }

  return 0;
}

struct SubCmdFunc
{
  const char *name, *info;
  int (*func)(int argc, char* argv[]);
};

const SubCmdFunc kSubCmdFuncs[] = {
  { "gen", "generate matrix", &gen },
  { "mul", "multiply matrix", &mul },
};

int
main(int argc, char* argv[])
try {
  po::options_description od("Options");
  od.add_options()                          //
    ("version,v", "print version info")     //
    ("help,h", "print help info")           //
    ("...",                                 //
     po::value<std::vector<std::string>>(), //
     "sub arguments")                       //
    ;

  po::positional_options_description pod;
  pod.add("...", -1);

  std::vector<std::string> opts{ argv[0] };
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '-')
      opts.push_back(argv[i]);
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
  po::notify(vmap);

  if (vmap.count("version")) {
    std::cout << "MPI Multi-Machine Matrix Application"
                 "\n"
                 "\nBuilt: " __TIME__ " (" __DATE__ ")"
                 "\nPGEMM: " PGEMM_VERSION "\n"
                 "\nCopyright (C) 2023 Yuhao Gu. All Rights Reserved."
              << std::endl;
    return 0;
  }

  if (vmap.count("help") || argc == 1) {
    std::cout << od
              << "\n"
                 "Sub Commands:\n";
    for (auto&& i : kSubCmdFuncs)
      std::cout << "  " << std::left << std::setw(12) << i.name << i.info
                << '\n';
    std::cout << "\n"
                 "[HINT: use '<subcmd> --help' to get help for sub commands.]\n"
              << std::endl;
    return 0;
  }

  if (opts.size() < argc) {
    std::string cmd = argv[opts.size()];
    for (auto&& i : kSubCmdFuncs) {
      if (cmd == i.name)
        return i.func(argc - opts.size(), argv + opts.size());
    }
    std::cout << "invalid sub command '" << cmd << "'." << std::endl;
    return 1;
  }
}

catch (MPI_mat::Err& e) {
  std::cout << "\nERROR! " << e.what() << "\n" << e.info() << std::endl;
  return -3;
}

catch (std::exception& e) {
  std::cout << "\nERROR! " << e.what() << std::endl;
  return -2;
}

catch (...) {
  return -1;
}
