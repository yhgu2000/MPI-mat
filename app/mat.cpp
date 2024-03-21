#include "project.h"
#include <PGEMM/hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>

using namespace std::string_literals;

namespace po = boost::program_options;

int
show(int argc, char* argv[])
{
  po::options_description od("'show' Options");
  od.add_options()                                               //
    ("help,h", "show help info")                                 //
    ("input,i", po::value<std::string>(), "input mat file path") //
    ;

  po::positional_options_description pod;
  pod.add("input", 1);

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

  auto input = vmap["input"].as<std::string>();
  auto mat = MPI_mat::Mat::load(input.c_str());
  std::cout << mat << std::endl;

  return 0;
}

int
gen(int argc, char* argv[])
{
  po::options_description od("'gen' Options");
  od.add_options()                                                         //
    ("help,h", "show help info")                                           //
    ("output,o", po::value<std::string>(), "output file path, must exist") //
    ("rown,r", po::value<std::uint32_t>(), "row number, height")           //
    ("coln,c", po::value<std::uint32_t>(), "column number, width")         //
    ("method,m",                                                           //
     po::value<std::string>()->default_value("rand"),                      //
     "generation method: rand / zero / eye")                               //
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

  auto rown = vmap["rown"].as<std::uint32_t>();
  auto coln = vmap["coln"].as<std::uint32_t>();
  MPI_mat::Mat mat(rown, coln);

  auto method = vmap["method"].as<std::string>();
  if (method == "rand")
    mat.rand();
  else if (method == "zero")
    mat.zero();
  else if (method == "eye")
    mat.eye();
  else
    return 1;

  auto output = vmap["output"].as<std::string>();
  MPI_mat::Mat::dump(output.c_str(), mat);

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

  auto lft = vmap["lft"].as<std::string>();
  auto matLft = MPI_mat::Mat::load(lft.c_str());

  auto rht = vmap["rht"].as<std::string>();
  auto matRht = MPI_mat::Mat::load(rht.c_str());

  auto output = vmap["output"].as<std::string>();
  MPI_mat::Mat::dump(output.c_str(), matLft * matRht);

  return 0;
}

struct SubCmdFunc
{
  const char *name, *info;
  int (*func)(int argc, char* argv[]);
};

const SubCmdFunc kSubCmdFuncs[] = {
  { "show", "print mat file in text", &show },
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
    std::cout << "Single-Machine Matrix Application"
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
