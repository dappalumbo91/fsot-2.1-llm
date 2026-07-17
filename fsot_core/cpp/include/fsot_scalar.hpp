// FSOT 2.1 — C++ scalar core (double precision hot path)
// Canonical formula mirrors archive vendor/fsot_compute.py compute_scalar.
// Seeds: π, e, φ, γ, G(Catalan). No free parameters.
// Gold standard: Python/mpmath in I:\FSOT-Physical-Archive.

#pragma once

#include <cmath>
#include <string>

namespace fsot {

inline constexpr double PI = 3.14159265358979323846;
inline constexpr double E = 2.71828182845904523536;
inline constexpr double PHI = 1.6180339887498948482;
inline constexpr double GAMMA =
    0.57721566490153286060651209008240243104215933593992;
inline constexpr double G_CAT =
    0.91596559417721901505460351493238411077414937428167;

inline double alpha() {
  return std::log(PI) / (E * std::pow(PHI, 13.0));
}
inline double psi_con() { return 1.0 - std::exp(-1.0); }
inline double eta_eff() { return 1.0 / (PI - 1.0); }
inline double beta() {
  return 1.0 / std::exp(std::pow(PI, PI) + (E - 1.0));
}
inline double gamma_c() { return -std::log(2.0) / PHI; }
inline double omega() { return std::sin(PI / E) * std::sqrt(2.0); }
inline double theta_s() { return std::sin(psi_con() * eta_eff()); }
inline double poof() {
  return std::exp((-std::log(PI) / E) / (eta_eff() * std::log(PHI)));
}
inline double c_eff() {
  return (1.0 - poof() * std::sin(theta_s())) *
         (1.0 + 0.01 * G_CAT / (PI * PHI));
}
inline double a_bleed() {
  return std::sin(PI / E) * PHI / std::sqrt(2.0);
}
inline double p_var() { return -std::cos(theta_s() + PI); }
inline double b_in() {
  return c_eff() * (1.0 - std::sin(theta_s()) / PHI);
}
inline double a_in() {
  return a_bleed() * (1.0 + std::cos(theta_s()) / PHI);
}
inline double suction() {
  return poof() * (-std::cos(theta_s() - PI));
}
inline double chaos() { return gamma_c() / omega(); }
inline double p_base() { return GAMMA / E; }
inline double p_new() { return p_base() * std::sqrt(2.0); }
inline double c_factor() { return c_eff() * p_new(); }
inline double K() {
  return PHI * (GAMMA / E) * std::sqrt(2.0) / std::log(PI) * 0.99;
}

struct ScalarInput {
  double N = 1.0;
  double P = 1.0;
  double D_eff = 25.0;
  double psi_con_v = 0.0;  // 0 => use default psi_con()
  double delta_psi = 1.0;
  double recent_hits = 0.0;
  double rho = 1.0;
  bool observed = false;
  double delta_theta = 1.0;
  double scale = 1.0;
  double amplitude = 1.0;
  double trend_bias = 0.0;
};

// S = K * (T1 + T2 + T3)  — same structure as Python archive engine
inline double compute_scalar(const ScalarInput& s) {
  const double N = s.N;
  const double P = s.P;
  const double D = s.D_eff;
  const double dp = s.delta_psi;
  const double dt = s.delta_theta;
  const double hits = s.recent_hits;
  const double a = alpha();
  const double psi = (s.psi_con_v != 0.0) ? s.psi_con_v : psi_con();
  const double eta = eta_eff();
  const double ce = c_eff();
  const double pn = p_new();
  const double bin = b_in();
  const double ain = a_in();
  const double abl = a_bleed();
  const double pv = p_var();
  const double pf = poof();
  const double su = suction();
  const double ch = chaos();
  const double bt = beta();
  const double th = theta_s();
  const double cf = c_factor();

  // Term 1: Observer-Modulated Base
  const double growth = std::exp(a * (1.0 - hits / N) * GAMMA / PHI);
  double base = (N * P / std::sqrt(D)) * std::cos((psi + dp) / eta) *
                std::exp(-a * hits / N + s.rho + bin * dp) *
                (1.0 + growth * ce);
  double T1 = base * (1.0 + pn * std::log(D / 25.0));
  if (s.observed) {
    T1 = T1 * std::exp(cf * pv) * std::cos(dp + pv);
  }

  // Term 2: Linear Modulation
  const double T2 = s.scale * s.amplitude + s.trend_bias;

  // Term 3: Valve-Acoustic-Phase
  const double valve =
      bt * std::cos(dp) * (N * P / std::sqrt(D)) *
      (1.0 + ch * (D - 25.0) / 25.0) *
      (1.0 + pf * std::cos(th + PI) + su * std::sin(th));
  const double sdt = std::sin(dt);
  const double cdt = std::cos(dt);
  const double acoustic =
      1.0 + (abl * sdt * sdt) / PHI + (ain * cdt * cdt) / PHI;
  const double phase = 1.0 + bin * pv;
  const double T3 = valve * acoustic * phase;

  return K() * (T1 + T2 + T3);
}

inline std::string version() { return "FSOT-2.1-cpp-0.1.0"; }

}  // namespace fsot
