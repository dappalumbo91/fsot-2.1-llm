#include "fsot_scalar.hpp"

#include <iomanip>
#include <iostream>

int main() {
  using namespace fsot;
  std::cout << std::setprecision(17);
  std::cout << "=== FSOT 2.1 C++ scalar core ===\n";
  std::cout << "version: " << version() << "\n";
  std::cout << "  pi    = " << PI << "\n";
  std::cout << "  e     = " << E << "\n";
  std::cout << "  phi   = " << PHI << "\n";
  std::cout << "  gamma = " << GAMMA << "\n";
  std::cout << "  G_cat = " << G_CAT << "\n";
  std::cout << "  K     = " << K() << "\n";
  std::cout << "  C_f   = " << c_factor() << "\n";

  ScalarInput s0;
  s0.D_eff = 25.0;
  s0.observed = false;
  ScalarInput s1 = s0;
  s1.observed = true;

  std::cout << "  S(D_eff=25, observed=false) = " << compute_scalar(s0) << "\n";
  std::cout << "  S(D_eff=25, observed=true)  = " << compute_scalar(s1) << "\n";
  std::cout << "FSOT fluid medium online (C++).\n";
  return 0;
}
